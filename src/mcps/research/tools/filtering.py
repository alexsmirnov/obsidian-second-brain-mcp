"""Relevance filtering: chunk the content, rank the fragments, keep the best."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from typing import Protocol

from mcps.research.tools.common import failure, textual_mime
from mcps.research.tools.models import FetchResult, FetchStatus

__all__ = ["Chunker", "Ranker", "RelevanceFilter"]

logger = logging.getLogger(__name__)


class Chunker(Protocol):
    """Split content into fragments, in document order."""

    def __call__(self, text: str) -> list[str]: ...


class Ranker(Protocol):
    """Return the indices of the fragments relevant to the query."""

    async def __call__(self, fragments: list[str], query: str) -> list[int]: ...


class RelevanceFilter:
    """Keep the query-relevant fragments of a textual result, in document order.

    Non-textual media give ``UNSUPPORTED`` and oversized input or query give
    ``FILTER_FAILED``, even for a blank query. Otherwise a blank query returns
    the result unchanged, and any chunker or ranker error gives
    ``FILTER_FAILED``. Output size is bounded by the downstream truncation.
    """

    def __init__(
        self,
        chunker: Chunker,
        ranker: Ranker,
        *,
        max_input_chars: int = 2_000_000,
        max_query_chars: int = 8_192,
    ) -> None:
        self._chunker = chunker
        self._ranker = ranker
        self._max_input_chars = max_input_chars
        self._max_query_chars = max_query_chars

    async def __call__(
        self, result: FetchResult, query: str | None = None, /
    ) -> FetchResult:
        if not result.ok:
            return result
        if textual_mime(result.mime) is None:
            return failure(result.url, FetchStatus.UNSUPPORTED)
        query = (query or "").strip()
        if (
            len(query) > self._max_query_chars
            or len(result.content) > self._max_input_chars
        ):
            return failure(result.url, FetchStatus.FILTER_FAILED)
        if not query:
            return result
        try:
            fragments = await asyncio.to_thread(self._chunker, result.content)
            selected = await self._ranker(fragments, query) if fragments else []
        except Exception as error:
            logger.warning(
                "Relevance filtering failed for %s: %r", result.url, error
            )
            return failure(result.url, FetchStatus.FILTER_FAILED)
        kept = sorted({index for index in selected if 0 <= index < len(fragments)})
        return replace(result, content="\n\n".join(fragments[i] for i in kept))
