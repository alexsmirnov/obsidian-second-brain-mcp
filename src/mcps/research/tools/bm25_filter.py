"""Local lexical passage filter: BM25L-ranked blocks within the output budget."""

from __future__ import annotations

import asyncio
import re

from rank_bm25 import BM25L

from mcps.research.tools.filtering import FilterLimits
from mcps.research.tools.filtering_content import (
    ContentDocument,
    ScoringWindow,
    render_selection,
)

__all__ = ["Bm25Filter", "lexical_scores"]

_STOPWORDS = frozenset(
    "a an and are as at be been being by can could did do does for from had has "
    "have how i if in into is it its me my of on or our should than that the "
    "their them then there these they this those to was we were what when where "
    "which who why will with would you your".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")


def _normalize_tokens(text: str) -> list[str]:
    """Casefold and tokenize, keeping identifiers and dropping stopwords."""
    return [
        token
        for token in _TOKEN_RE.findall(text.casefold())
        if token not in _STOPWORDS
    ]


def lexical_scores(
    windows: tuple[ScoringWindow, ...], query: str
) -> dict[int, float]:
    """Score every window with BM25L; windows with no query overlap score zero."""
    query_tokens = _normalize_tokens(query)
    corpus = [_normalize_tokens(window.text) for window in windows]
    if not query_tokens or not any(corpus):
        return {window.window_id: 0.0 for window in windows}
    raw_scores = BM25L(corpus).get_scores(query_tokens)
    query_terms = set(query_tokens)
    return {
        window.window_id: 0.0 if query_terms.isdisjoint(tokens) else float(score)
        for window, tokens, score in zip(windows, corpus, raw_scores, strict=True)
    }


def _local_selection(
    document: ContentDocument,
    windows: tuple[ScoringWindow, ...],
    query: str,
    max_output_chars: int,
) -> tuple[set[int], bool]:
    """Greedily keep the best-scoring blocks whose rendering fits the budget.

    Returns the selection and whether any matching block was dropped. The top
    block is always kept, even when oversized; final truncation shortens it.
    Lower-ranked blocks that do not fit are skipped so smaller ones can still
    fill the remaining budget.
    """
    scores = lexical_scores(windows, query)
    ranked = sorted(
        (window for window in windows if scores[window.window_id] > 0.0),
        key=lambda window: (-scores[window.window_id], window.window_id),
    )
    matching = dict.fromkeys(window.block_id for window in ranked)
    selected: set[int] = set()
    # ponytail: re-renders per candidate, O(candidates x blocks); track the
    # rendered size incrementally if huge documents make this slow.
    for block_id in matching:
        candidate = selected | {block_id}
        if selected and len(render_selection(document, candidate)) > max_output_chars:
            continue
        selected = candidate
        if len(render_selection(document, selected)) >= max_output_chars:
            break
    return selected, not matching.keys() <= selected


class Bm25Filter:
    """Passage filter selecting BM25L-ranked blocks within the output budget."""

    def __init__(self, limits: FilterLimits | None = None) -> None:
        self._limits = limits or FilterLimits()

    async def __call__(
        self,
        document: ContentDocument,
        windows: tuple[ScoringWindow, ...],
        query: str,
    ) -> tuple[set[int], bool]:
        return await asyncio.to_thread(
            _local_selection,
            document,
            windows,
            query,
            self._limits.max_output_chars,
        )
