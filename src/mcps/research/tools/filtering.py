"""Relevance filtering: chunk the content, rank the fragments, keep the best."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from mcps.research.tools.common import failure, textual_mime
from mcps.research.tools.models import FetchResult, FetchStatus

__all__ = ["Chunker", "FilterModelError", "Fragment", "Ranker", "RelevanceFilter"]

logger = logging.getLogger(__name__)


class FilterModelError(ValueError):
    """A model-backed ranker failed or returned unusable output."""


@dataclass(frozen=True, slots=True)
class Fragment:
    """One contiguous slice of a source text, ranked by its ``text``.

    ``start`` and ``end`` are half-open character offsets into the source, so
    ``text`` equals ``source[start:end]``. Fragments of one source may overlap
    and may arrive in any order.
    """

    text: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid fragment range: [{self.start}, {self.end})")
        if len(self.text) != self.end - self.start:
            raise ValueError(f"fragment text does not match [{self.start}, {self.end})")


class Chunker(Protocol):
    """Split content into fragments and merge the selected ones back into text.

    Fragment offsets address the text given to ``split``. Fragments may overlap
    and arrive in any order. ``merge`` rebuilds the text from the fragments
    alone, so it needs no copy of the source.
    """

    def split(self, text: str) -> list[Fragment]: ...

    def merge(self, fragments: Sequence[Fragment]) -> str: ...


class Ranker(Protocol):
    """Return the indices of the fragments relevant to the query."""

    async def __call__(self, fragments: list[str], query: str) -> list[int]: ...


class RelevanceFilter:
    """Keep the query-relevant fragments of a textual result, merged in document order.

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
            fragments = await asyncio.to_thread(self._chunker.split, result.content)
            selected = (
                await self._ranker([fragment.text for fragment in fragments], query)
                if fragments
                else []
            )
            kept = sorted({index for index in selected if 0 <= index < len(fragments)})
            content = self._chunker.merge([fragments[index] for index in kept])
        except Exception as error:
            logger.warning(
                "Relevance filtering failed for %s: %r", result.url, error
            )
            return failure(result.url, FetchStatus.FILTER_FAILED)
        return replace(result, content=content)
