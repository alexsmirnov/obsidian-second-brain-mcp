"""Relevance filtering pipeline: extract, chunk, then select relevant passages.

:class:`RelevanceFilter` composes three injected steps: an :class:`Extractor`
prepares a fetch result natively, a :class:`Chunker` regroups it into scoring
windows, and a :class:`PassageFilter` selects the source blocks that answer the
query. The extractor and chunker are synchronous CPU work run off the event
loop; the passage filter is asynchronous so it can call models. Implementations
live in their own modules (``default_extractor``, ``semantic_chunker``,
``bm25_filter``, ``llm_filter``) and are combined by
``mcps.research.tools.fetch.create_page_filter``. A blank query returns the
extracted whole document without chunking or any model call.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, replace
from typing import Protocol

from mcps.research.tools.common import TRUNCATION_MARKER, failure
from mcps.research.tools.filtering_content import (
    ContentDocument,
    ScoringWindow,
    render_selection,
)
from mcps.research.tools.models import FetchResult, FetchStatus

__all__ = [
    "Chunker",
    "Extractor",
    "FilterLimits",
    "PassageFilter",
    "RelevanceFilter",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FilterLimits:
    """Constructor tuning for input, window, and model work budgets."""

    max_input_chars: int = 2_000_000
    max_query_chars: int = 8_192
    window_chars: int = 2_000
    overlap_chars: int = 200
    max_windows: int = 8_192
    embedding_batch_size: int = 32
    embedding_request_chars: int = 64_000
    semantic_candidates: int = 48
    llm_batch_size: int = 16
    llm_request_chars: int = 40_000
    max_llm_batches: int = 128
    model_concurrency: int = 4
    model_timeout_seconds: float = 120.0
    max_response_bytes: int = 4_000_000
    # ponytail: mirrors create_fetch's default max_chars; a single shared
    # constant if the fetch budget ever becomes configurable.
    max_output_chars: int = 15_000
    # Selection-unit ceiling and the size below which a section absorbs the
    # following sections (same rule as the RAG SemanticChunker).
    unit_chars: int = 2_000
    min_unit_chars: int = 500

    def __post_init__(self) -> None:
        positive = (
            self.unit_chars,
            self.min_unit_chars,
            self.max_output_chars,
            self.max_input_chars,
            self.max_query_chars,
            self.window_chars,
            self.max_windows,
            self.embedding_batch_size,
            self.embedding_request_chars,
            self.semantic_candidates,
            self.llm_batch_size,
            self.llm_request_chars,
            self.max_llm_batches,
            self.model_concurrency,
            self.model_timeout_seconds,
            self.max_response_bytes,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("filter limits must be positive")
        if not 0 <= self.overlap_chars < self.window_chars:
            raise ValueError("overlap_chars must be below window_chars")
        if self.embedding_request_chars < self.window_chars:
            raise ValueError("embedding request budget cannot hold one window")
        if self.llm_request_chars < self.window_chars:
            raise ValueError("llm request budget cannot hold one window")


class Extractor(Protocol):
    """Step preparing a fetch result into a source-mapped content document."""

    def __call__(self, result: FetchResult) -> ContentDocument: ...


class Chunker(Protocol):
    """Step splitting a prepared document into selection units and windows.

    Returns the (possibly regrouped) document together with scoring windows
    whose ``block_id`` values refer to blocks of the returned document.
    """

    def __call__(
        self, document: ContentDocument
    ) -> tuple[ContentDocument, tuple[ScoringWindow, ...]]: ...


class PassageFilter(Protocol):
    """Step selecting the source block IDs that answer the query.

    Returns the selected block IDs and whether any matching block was dropped.
    """

    async def __call__(
        self,
        document: ContentDocument,
        windows: tuple[ScoringWindow, ...],
        query: str,
    ) -> tuple[set[int], bool]: ...


class RelevanceFilter:
    """Pipeline selecting query-relevant native source passages.

    Composes an injected ``extractor``, ``chunker``, and ``passage_filter``;
    ``limits`` own input, query, output, and model budgets. Runs the two
    synchronous steps off the event loop and maps their failures to
    ``UNSUPPORTED`` (unsupported media) or ``FILTER_FAILED`` (any other error).
    """

    def __init__(
        self,
        extractor: Extractor,
        chunker: Chunker,
        passage_filter: PassageFilter,
        *,
        limits: FilterLimits | None = None,
    ) -> None:
        self._extractor = extractor
        self._chunker = chunker
        self._passage_filter = passage_filter
        self._limits = limits or FilterLimits()

    @property
    def extractor(self) -> Extractor:
        return self._extractor

    @property
    def chunker(self) -> Chunker:
        return self._chunker

    @property
    def passage_filter(self) -> PassageFilter:
        return self._passage_filter

    @property
    def limits(self) -> FilterLimits:
        return self._limits

    async def __call__(
        self, result: FetchResult, query: str | None = None, /
    ) -> FetchResult:
        if not result.ok:
            return result
        limits = self._limits
        query = (query or "").strip()
        if len(query) > limits.max_query_chars:
            return failure(result.url, FetchStatus.FILTER_FAILED)
        try:
            document = await asyncio.to_thread(self._extractor, result)
        except LookupError:
            return failure(result.url, FetchStatus.UNSUPPORTED)
        except Exception:
            logger.warning("Content preparation failed for %s", result.url)
            return failure(result.url, FetchStatus.FILTER_FAILED)
        if not query:
            return replace(result, content=document.text, mime=document.mime)
        try:
            document, windows = await asyncio.to_thread(self._chunker, document)
        except Exception:
            logger.warning("Content windowing failed for %s", result.url)
            return failure(result.url, FetchStatus.FILTER_FAILED)
        if not windows:
            return replace(result, content="", mime=document.mime)
        try:
            selected, dropped = await self._passage_filter(
                document, windows, query
            )
        except Exception:
            logger.warning("Content filtering failed for %s", result.url)
            return failure(result.url, FetchStatus.FILTER_FAILED)
        if not selected:
            return replace(result, content="", mime=document.mime)
        try:
            content = render_selection(document, selected)
        except Exception:
            logger.warning("Content rendering failed for %s", result.url)
            return failure(result.url, FetchStatus.FILTER_FAILED)
        if dropped:
            content += TRUNCATION_MARKER
        return replace(result, content=content, mime=document.mime)
