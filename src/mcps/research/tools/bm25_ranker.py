"""Local lexical ranker: fragments with BM25L overlap with the query."""

from __future__ import annotations

import asyncio
import logging
import string
from collections.abc import Sequence

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25L

__all__ = ["Bm25Ranker"]

logger = logging.getLogger(__name__)


class Bm25Ranker:
    """Rank fragments with BM25L using preinstalled NLTK data; never download it."""

    def __init__(self) -> None:
        self._stop_words = frozenset(stopwords.words("english"))
        self._stemmer = SnowballStemmer("english")

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

    def _rank(self, fragments: list[str], query: str) -> list[int]:
        scores = self.lexical_scores(fragments, query)
        matching = [index for index, score in enumerate(scores) if score > 1.0]
        logger.info(
            "Lexical scores: [%s] matched for %d fragments of %d",
            ",".join(f"{score:.2f}" for score in scores),
            len(matching),
            len(fragments),
        )
        return matching

    async def __call__(self, fragments: list[str], query: str) -> list[int]:
        return await asyncio.to_thread(self._rank, fragments, query)
