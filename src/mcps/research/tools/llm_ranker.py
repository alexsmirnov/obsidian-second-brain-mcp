"""Hybrid ranker: embedding shortlist plus router-model fragment selection."""

from __future__ import annotations

import asyncio
import json
import math

from mcps.research.tools.bm25_ranker import lexical_scores
from mcps.research.tools.filtering_models import (
    FilterModelError,
    RouterPassageModels,
    build_selection_payload,
)

__all__ = ["LlmRanker"]

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
    items: list[str], max_items: int, max_bytes: int
) -> list[list[str]]:
    """Pack texts into batches bounded by item count and UTF-8 byte size."""
    batches: list[list[str]] = []
    current: list[str] = []
    size = 0
    for item in items:
        length = len(item.encode("utf-8"))
        if length > max_bytes:
            raise FilterModelError("embedding input exceeds request budget")
        if current and (len(current) >= max_items or size + length > max_bytes):
            batches.append(current)
            current = []
            size = 0
        current.append(item)
        size += length
    if current:
        batches.append(current)
    return batches


class LlmRanker:
    """Ranker shortlisting with BM25 and embeddings, then selecting with a model.

    Any model or budget failure raises :class:`FilterModelError`; it never falls
    back to lexical ranking.
    """

    def __init__(
        self,
        models: RouterPassageModels,
        *,
        semantic_candidates: int = 48,
        embedding_batch_size: int = 32,
        embedding_request_chars: int = 64_000,
        llm_batch_size: int = 16,
        llm_request_chars: int = 40_000,
        max_llm_batches: int = 128,
        timeout_seconds: float = 120.0,
    ) -> None:
        self._models = models
        self._semantic_candidates = semantic_candidates
        self._embedding_batch_size = embedding_batch_size
        self._embedding_request_chars = embedding_request_chars
        self._llm_batch_size = llm_batch_size
        self._llm_request_chars = llm_request_chars
        self._max_llm_batches = max_llm_batches
        self._timeout_seconds = timeout_seconds

    async def __call__(self, fragments: list[str], query: str) -> list[int]:
        try:
            async with asyncio.timeout(self._timeout_seconds):
                return await self._rank(fragments, query)
        except TimeoutError:
            raise FilterModelError("model deadline exceeded") from None

    async def _rank(self, fragments: list[str], query: str) -> list[int]:
        vectors, query_vector = await self._embed(fragments, query)
        candidates = self._shortlist(fragments, query, vectors, query_vector)
        selected: set[int] = set()
        for batch in self._selection_batches(candidates, query):
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
        query: str,
        vectors: list[list[float]],
        query_vector: list[float],
    ) -> list[Candidate]:
        """Union every lexical match with the best semantic matches."""
        chosen = {
            index
            for index, score in enumerate(lexical_scores(fragments, query))
            if score > 0.0
        }
        similarities = sorted(
            (
                (-similarity, index)
                for index, vector in enumerate(vectors)
                if (similarity := _cosine(query_vector, vector)) > 0.0
            )
        )
        chosen.update(index for _, index in similarities[: self._semantic_candidates])
        return [(index, fragments[index]) for index in sorted(chosen)]

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
        payload = build_selection_payload(self._models.fetch_model, query, batch)
        return len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
