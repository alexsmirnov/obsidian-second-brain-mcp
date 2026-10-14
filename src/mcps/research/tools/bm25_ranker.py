"""Local lexical ranker: fragments with BM25L overlap with the query."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence

from rank_bm25 import BM25L

__all__ = ["Bm25Ranker", "lexical_scores"]

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


def lexical_scores(fragments: Sequence[str], query: str) -> list[float]:
    """Score every fragment with BM25L; fragments with no query overlap score zero."""
    query_tokens = _normalize_tokens(query)
    corpus = [_normalize_tokens(fragment) for fragment in fragments]
    if not query_tokens or not any(corpus):
        return [0.0] * len(corpus)
    raw_scores = BM25L(corpus).get_scores(query_tokens)
    query_terms = set(query_tokens)
    return [
        0.0 if query_terms.isdisjoint(tokens) else float(score)
        for tokens, score in zip(corpus, raw_scores, strict=True)
    ]


def _rank(fragments: list[str], query: str) -> list[int]:
    scores = lexical_scores(fragments, query)
    matching = [index for index, score in enumerate(scores) if score > 0.0]
    return sorted(matching, key=lambda index: (-scores[index], index))


class Bm25Ranker:
    """Ranker selecting fragments by BM25L overlap, best first."""

    async def __call__(self, fragments: list[str], query: str) -> list[int]:
        return await asyncio.to_thread(_rank, fragments, query)
