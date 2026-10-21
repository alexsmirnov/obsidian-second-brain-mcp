"""Local lexical ranker: fragments with BM25L overlap with the query."""

from __future__ import annotations

import asyncio
import logging
import math
import re
import string
from collections.abc import Sequence

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25L

__all__ = ["Bm25Ranker"]

logger = logging.getLogger(__name__)

# Markdown links, URLs, DOIs and years: the marks of citations and reference lists.
_LINK_RE = re.compile(r"\]\(|https?://|doi\.org|\(\d{4}\)|\b(?:19|20)\d{2}\b")


def _smooth(scores: list[float], weight: float) -> list[float]:
    """Blend each score with its neighbours; an edge reuses its own score."""
    return [
        (1 - 2 * weight) * score
        + weight * (scores[index - 1] if index > 0 else score)
        + weight * (scores[index + 1] if index + 1 < len(scores) else score)
        for index, score in enumerate(scores)
    ]


class Bm25Ranker:
    """Rank fragments with BM25L using preinstalled NLTK data; never download it.

    A fragment is kept when its score is above ``threshold`` and at least
    ``relative_cutoff`` times the best score on the page. Before selection,
    ``link_penalty`` scales a score down by the fragment's density of links,
    DOIs and years, and ``smoothing`` blends each score with its neighbours.
    """

    def __init__(
        self,
        *,
        threshold: float = 1.0,
        relative_cutoff: float = 0.0,
        smoothing: float = 0.0,
        link_penalty: float = 0.0,
    ) -> None:
        self._threshold = threshold
        self._relative_cutoff = relative_cutoff
        self._smoothing = smoothing
        self._link_penalty = link_penalty
        self._stop_words = frozenset(stopwords.words("english"))
        self._stemmer = SnowballStemmer("english")

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def relative_cutoff(self) -> float:
        return self._relative_cutoff

    @property
    def smoothing(self) -> float:
        return self._smoothing

    @property
    def link_penalty(self) -> float:
        return self._link_penalty

    def _normalize_tokens(self, text: str) -> list[str]:
        return [
            self._stemmer.stem(token)
            for token in word_tokenize(text.lower())
            if token not in string.punctuation
            and not token.isdigit()
            and token not in self._stop_words
        ]

    def lexical_scores(self, fragments: Sequence[str], query: str) -> list[float]:
        """Score each fragment with BM25L; zero means no query overlap."""
        query_tokens = self._normalize_tokens(query)
        corpus = [self._normalize_tokens(fragment) for fragment in fragments]
        if not query_tokens or not any(corpus):
            return [0.0] * len(corpus)
        raw_scores = BM25L(corpus).get_scores(query_tokens)
        query_terms = set(query_tokens)
        return [
            0.0 if query_terms.isdisjoint(tokens) else float(score)
            for tokens, score in zip(corpus, raw_scores, strict=True)
        ]

    def _link_factor(self, fragment: str) -> float:
        density = len(_LINK_RE.findall(fragment)) / max(
            1, len(self._normalize_tokens(fragment))
        )
        return max(0.0, 1 - self._link_penalty * density)

    def _adjusted_scores(self, fragments: list[str], query: str) -> list[float]:
        scores = self.lexical_scores(fragments, query)
        if self._link_penalty:
            scores = [
                score * self._link_factor(fragment) if score else 0.0
                for fragment, score in zip(fragments, scores, strict=True)
            ]
        if self._smoothing:
            scores = _smooth(scores, self._smoothing)
        return scores

    def _rank(self, fragments: list[str], query: str) -> list[int]:
        scores = self._adjusted_scores(fragments, query)
        cutoff = max(self._threshold, self._relative_cutoff * max(scores, default=0.0))
        matching = [
            index
            for index, score in enumerate(scores)
            if score > self._threshold and score >= cutoff
        ]
        logger.info(
            "Lexical scores: [%s] matched for %d fragments of %d",
            ",".join(f"{score:.2f}" for score in scores),
            len(matching),
            len(fragments),
        )
        return matching

    async def __call__(self, fragments: list[str], query: str) -> list[int]:
        return await asyncio.to_thread(self._rank, fragments, query)
