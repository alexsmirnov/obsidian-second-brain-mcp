"""Local lexical ranker: fragments with BM25L overlap with the query."""

from __future__ import annotations

import asyncio
import logging
import re
import string
from collections.abc import Sequence

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.stem.snowball import SnowballStemmer
from rank_bm25 import BM25L

__all__ = ["Bm25Ranker", "lexical_scores"]

logger = logging.getLogger(__name__)
_STOPWORDS = frozenset(
    "a an and are as at be been being by can could did do does for from had has "
    "have how i if in into is it its me my of on or our should than that the "
    "their them then there these they this those to was we were what when where "
    "which who why will with would you your".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")

tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
stemmer = SnowballStemmer("english")

# 2. Optimized lookup set for stopwords (O(1) complexity)
STOP_WORDS = set(stopwords.words('english'))

def _normalize_tokens(text: str) -> list[str]:
    """Casefold and tokenize, keeping identifiers and dropping stopwords."""
    # Lowercase and tokenize
    tokens = word_tokenize(text.lower())
    
    clean_tokens = []
    for token in tokens:
        # Filter out punctuation, numbers, and stopwords
        if token in string.punctuation or token.isdigit() or token in STOP_WORDS:
            continue
        # Stem the word to its base form
        clean_tokens.append(stemmer.stem(token))
        
    return clean_tokens


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
    matching = [index for index, score in enumerate(scores) if score > 1.0]
    logger.info("Lexical scores: [%s] matched for %d fragments of %d",",".join([f"{num:.2f}" for num in scores]), len(matching), len(fragments))
    return matching


class Bm25Ranker:
    """Ranker selecting fragments by BM25L overlap, best first."""

    async def __call__(self, fragments: list[str], query: str) -> list[int]:
        return await asyncio.to_thread(_rank, fragments, query)
