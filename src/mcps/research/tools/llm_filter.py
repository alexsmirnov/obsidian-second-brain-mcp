"""Hybrid passage filter: embedding shortlist plus router-model block selection."""

from __future__ import annotations

import asyncio
import json
import math

from mcps.research.tools.bm25_filter import lexical_scores
from mcps.research.tools.filtering import FilterLimits
from mcps.research.tools.filtering_content import ContentDocument, ScoringWindow
from mcps.research.tools.filtering_models import (
    FilterModelError,
    RouterPassageModels,
    build_selection_payload,
)

__all__ = ["LlmFilter", "select_hybrid_candidates"]


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = math.fsum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(math.fsum(a * a for a in left))
    right_norm = math.sqrt(math.fsum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def select_hybrid_candidates(
    windows: tuple[ScoringWindow, ...],
    lexical: dict[int, float],
    vectors: list[list[float]],
    query_vector: list[float],
    *,
    semantic_candidates: int,
) -> tuple[ScoringWindow, ...]:
    """Union all positive lexical windows with the best semantic windows."""
    chosen = {
        window.window_id: window
        for window in windows
        if lexical.get(window.window_id, 0.0) > 0.0
    }
    similarities = [
        (_cosine(query_vector, vector), window)
        for window, vector in zip(windows, vectors, strict=False)
    ]
    ranked = sorted(
        (item for item in similarities if item[0] > 0.0),
        key=lambda item: (-item[0], item[1].window_id),
    )
    for _, window in ranked[:semantic_candidates]:
        chosen[window.window_id] = window
    return tuple(sorted(chosen.values(), key=lambda window: window.window_id))


def _embedding_batches(
    items: list[str], limits: FilterLimits
) -> list[list[str]]:
    """Pack texts into batches bounded by item count and UTF-8 byte size."""
    batches: list[list[str]] = []
    current: list[str] = []
    size = 0
    for item in items:
        length = len(item.encode("utf-8"))
        if length > limits.embedding_request_chars:
            raise FilterModelError("embedding input exceeds request budget")
        if current and (
            len(current) >= limits.embedding_batch_size
            or size + length > limits.embedding_request_chars
        ):
            batches.append(current)
            current = []
            size = 0
        current.append(item)
        size += length
    if current:
        batches.append(current)
    return batches


def _payload_size(
    fetch_model: str, query: str, windows: list[ScoringWindow]
) -> int:
    payload = build_selection_payload(fetch_model, query, tuple(windows))
    return len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))


def _selection_batches(
    windows: tuple[ScoringWindow, ...],
    query: str,
    fetch_model: str,
    limits: FilterLimits,
) -> list[tuple[ScoringWindow, ...]]:
    """Pack selection windows bounded by item count and serialized payload size."""
    batches: list[tuple[ScoringWindow, ...]] = []
    current: list[ScoringWindow] = []
    for window in windows:
        if current and (
            len(current) >= limits.llm_batch_size
            or _payload_size(fetch_model, query, [*current, window])
            > limits.llm_request_chars
        ):
            batches.append(tuple(current))
            current = []
        if (
            not current
            and _payload_size(fetch_model, query, [window])
            > limits.llm_request_chars
        ):
            raise FilterModelError("selection window exceeds request budget")
        current.append(window)
    if current:
        batches.append(tuple(current))
    if len(batches) > limits.max_llm_batches:
        raise FilterModelError("selection batch budget exceeded")
    return batches


async def _run_model_selection(
    models: RouterPassageModels,
    windows: tuple[ScoringWindow, ...],
    query: str,
    limits: FilterLimits,
) -> set[int]:
    """Embed all windows, shortlist candidates, and select source block IDs."""
    query_batches = _embedding_batches([query], limits)
    window_batches = _embedding_batches(
        [window.text for window in windows], limits
    )
    query_vector = (await models.embed(query_batches[0]))[0]
    width = len(query_vector)
    vectors: list[list[float]] = []
    for batch in window_batches:
        for vector in await models.embed(batch):
            if len(vector) != width:
                raise FilterModelError("inconsistent embedding width")
            vectors.append(vector)
    lexical = lexical_scores(windows, query)
    candidates = select_hybrid_candidates(
        windows,
        lexical,
        vectors,
        query_vector,
        semantic_candidates=limits.semantic_candidates,
    )
    if not candidates:
        return set()
    selection_batches = _selection_batches(
        candidates, query, models.fetch_model, limits
    )
    selected_window_ids: set[int] = set()
    for batch in selection_batches:
        selected_window_ids |= await models.select(query, batch)
    by_id = {window.window_id: window for window in candidates}
    return {
        by_id[window_id].block_id
        for window_id in selected_window_ids
        if window_id in by_id
    }


class LlmFilter:
    """Passage filter shortlisting with embeddings and selecting with the model."""

    def __init__(
        self,
        models: RouterPassageModels,
        *,
        limits: FilterLimits | None = None,
    ) -> None:
        self._models = models
        self._limits = limits or FilterLimits()

    async def __call__(
        self,
        document: ContentDocument,
        windows: tuple[ScoringWindow, ...],
        query: str,
    ) -> tuple[set[int], bool]:
        limits = self._limits
        try:
            async with asyncio.timeout(limits.model_timeout_seconds):
                selected = await _run_model_selection(
                    self._models, windows, query, limits
                )
        except TimeoutError:
            raise FilterModelError("model deadline exceeded") from None
        return selected, False
