"""Ranker: embedding similarity shortlist plus router-model fragment selection."""

from __future__ import annotations

import json
import logging
import math

from mcps.research.tools.filtering_models import (
    FilterModelError,
    RouterPassageModels,
    build_selection_payload,
)

__all__ = ["LlmRanker"]

logger = logging.getLogger(__name__)

Candidate = tuple[int, str]


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = math.fsum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(math.fsum(a * a for a in left))
    right_norm = math.sqrt(math.fsum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _batches(
    items: list[str], max_items: int, max_chars: int
) -> list[list[str]]:
    """Pack texts into batches bounded by item count and character count."""
    batches: list[list[str]] = []
    current: list[str] = []
    size = 0
    for item in items:
        length = len(item)
        if length > max_chars:
            raise FilterModelError("embedding input exceeds request budget")
        if current and (len(current) >= max_items or size + length > max_chars):
            batches.append(current)
            current = []
            size = 0
        current.append(item)
        size += length
    if current:
        batches.append(current)
    return batches


class LlmRanker:
    """Ranker shortlisting by embedding similarity, then selecting with a model.

    Any model or budget failure raises :class:`FilterModelError`.
    """

    def __init__(
        self,
        models: RouterPassageModels,
        *,
        similarity_threshold: float = 0.5,
        embedding_batch_size: int = 32,
        embedding_request_chars: int = 64_000,
        llm_batch_size: int = 16,
        llm_request_chars: int = 40_000,
        max_llm_batches: int = 128,
    ) -> None:
        if not -1.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in the range [-1.0, 1.0]")
        self._models = models
        self._similarity_threshold = similarity_threshold
        self._embedding_batch_size = embedding_batch_size
        self._embedding_request_chars = embedding_request_chars
        self._llm_batch_size = llm_batch_size
        self._llm_request_chars = llm_request_chars
        self._max_llm_batches = max_llm_batches

    @property
    def similarity_threshold(self) -> float:
        return self._similarity_threshold

    async def __call__(self, fragments: list[str], query: str) -> list[int]:
        if not fragments:
            return []
        vectors, query_vector = await self._embed(fragments, query)
        candidates = self._shortlist(fragments, vectors, query_vector)
        batches = self._selection_batches(candidates, query)
        logger.info(
            "Selecting from %d of %d fragments in %d model batches",
            len(candidates),
            len(fragments),
            len(batches),
        )
        selected: set[int] = set()
        for batch in batches:
            selected |= await self._models.select(query, batch)
        return sorted(selected)

    async def _embed(
        self, fragments: list[str], query: str
    ) -> tuple[list[list[float]], list[float]]:
        query_batch = _batches(
            [query], self._embedding_batch_size, self._embedding_request_chars
        )[0]
        query_vector = (await self._models.embed(query_batch))[0]
        vectors: list[list[float]] = []
        for batch in _batches(
            fragments, self._embedding_batch_size, self._embedding_request_chars
        ):
            for vector in await self._models.embed(batch):
                if len(vector) != len(query_vector):
                    raise FilterModelError("inconsistent embedding width")
                vectors.append(vector)
        return vectors, query_vector

    def _shortlist(
        self,
        fragments: list[str],
        vectors: list[list[float]],
        query_vector: list[float],
    ) -> list[Candidate]:
        """Select candidate fragments matching the similarity threshold."""
        return [
            (index, fragments[index])
            for index, vector in enumerate(vectors)
            if _cosine(query_vector, vector) >= self._similarity_threshold
        ]

    def _selection_batches(
        self, candidates: list[Candidate], query: str
    ) -> list[list[Candidate]]:
        """Pack candidates bounded by item count and serialized payload size."""
        batches: list[list[Candidate]] = []
        current: list[Candidate] = []
        for candidate in candidates:
            if current and (
                len(current) >= self._llm_batch_size
                or self._payload_size(query, [*current, candidate])
                > self._llm_request_chars
            ):
                batches.append(current)
                current = []
            if not current and self._payload_size(query, [candidate]) > (
                self._llm_request_chars
            ):
                raise FilterModelError("selection fragment exceeds request budget")
            current.append(candidate)
        if current:
            batches.append(current)
        if len(batches) > self._max_llm_batches:
            raise FilterModelError("selection batch budget exceeded")
        return batches

    def _payload_size(self, query: str, batch: list[Candidate]) -> int:
        payload = build_selection_payload(query, batch)
        return len(json.dumps(payload, ensure_ascii=False))

