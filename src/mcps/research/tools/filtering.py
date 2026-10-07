"""Post-retrieval filters selecting query-relevant native source passages.

:class:`MarkdownToHtml` and :class:`PreTextToHtml` are retained legacy
normalizers. :class:`RelevanceFilter` is the shared pipeline: it prepares a
fetch result natively, delegates chunking to an injected :class:`Chunker`, and
asks the subclass to select source blocks. :class:`Bm25RelevanceFilter` scores
source-mapped windows with BM25L; :class:`LlmRelevanceFilter` augments lexical
hits with embedding cosine shortlisting and asks the router model to select
source window IDs. A blank query returns the prepared whole document without
any model call. Implementations are chosen once from configuration (see
``mcps.research.tools.fetch.create_page_filter``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Protocol

import markdown as markdown_lib
from lxml import html as lxml_html
from rank_bm25 import BM25L

from mcps.research.tools.common import MIME_HTML, TRUNCATION_MARKER, failure
from mcps.research.tools.filtering_content import (
    ContentDocument,
    ScoringWindow,
    build_scoring_windows,
    build_selection_units,
    prepare_document,
    render_selection,
)
from mcps.research.tools.filtering_models import (
    FilterModelError,
    RouterPassageModels,
    build_selection_payload,
)
from mcps.research.tools.models import FetchResult, FetchStatus

__all__ = [
    "Bm25RelevanceFilter",
    "Chunker",
    "FilterLimits",
    "LlmRelevanceFilter",
    "MarkdownToHtml",
    "PreTextToHtml",
    "RelevanceFilter",
    "SectionChunker",
    "lexical_scores",
    "markdown_to_html",
    "select_hybrid_candidates",
    "text_page_to_html",
]

logger = logging.getLogger(__name__)

_STOPWORDS = frozenset(
    "a an and are as at be been being by can could did do does for from had has "
    "have how i if in into is it its me my of on or our should than that the "
    "their them then there these they this those to was we were what when where "
    "which who why will with would you your".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")


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


# ---------------------------------------------------------------------------
# Legacy public normalizers
# ---------------------------------------------------------------------------


def markdown_to_html(text: str) -> str:
    """Render Markdown to HTML so crawl4ai's HTML-block filter can process it."""
    return markdown_lib.markdown(text, extensions=["extra"])


def text_page_to_html(page: str) -> str:
    """Convert a browser-rendered plain-text page (sole ``<body><pre>``) to HTML.

    crawl4ai renders plain text, raw Markdown, and JSON URLs as
    ``<html><body><pre>...</pre></body></html>``. HTML pages are returned
    unchanged.
    """
    try:
        document = lxml_html.document_fromstring(page)
        body = document.find("body")
        if body is None:
            return page
        children = [child for child in body if isinstance(child.tag, str)]
        if len(children) != 1 or children[0].tag != "pre":
            return page
        if (body.text or "").strip() or (children[0].tail or "").strip():
            return page
        return markdown_to_html(children[0].text_content())
    except Exception:
        return page


class MarkdownToHtml:
    """Filter rendering a Markdown/plain-text result to HTML for block filters."""

    async def __call__(
        self, result: FetchResult, query: str | None = None, /
    ) -> FetchResult:
        return replace(
            result, content=markdown_to_html(result.content), mime=MIME_HTML
        )


class PreTextToHtml:
    """Filter converting a browser-rendered sole ``<pre>`` page to HTML."""

    async def __call__(
        self, result: FetchResult, query: str | None = None, /
    ) -> FetchResult:
        return replace(result, content=text_page_to_html(result.content))


# ---------------------------------------------------------------------------
# Model-mode helpers
# ---------------------------------------------------------------------------


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


class Chunker(Protocol):
    """Strategy splitting a prepared document into selection units and windows.

    Returns the (possibly regrouped) document together with scoring windows
    whose ``block_id`` values refer to blocks of the returned document.
    """

    def __call__(
        self, document: ContentDocument
    ) -> tuple[ContentDocument, tuple[ScoringWindow, ...]]: ...


class SectionChunker:
    """Chunker grouping sections into units, then windowing each unit."""

    def __init__(self, limits: FilterLimits) -> None:
        self._limits = limits

    def __call__(
        self, document: ContentDocument
    ) -> tuple[ContentDocument, tuple[ScoringWindow, ...]]:
        limits = self._limits
        units = build_selection_units(
            document,
            unit_chars=limits.unit_chars,
            min_unit_chars=limits.min_unit_chars,
        )
        windows = build_scoring_windows(
            units,
            window_chars=limits.window_chars,
            overlap_chars=limits.overlap_chars,
            max_windows=limits.max_windows,
        )
        return units, windows


class RelevanceFilter(ABC):
    """Pipeline selecting query-relevant native source passages.

    The filter's ``limits`` own input, query, output, and model budgets; the
    injected ``chunker`` owns chunking (the default is built from the same
    limits). Subclasses implement only block selection.
    """

    def __init__(
        self,
        *,
        limits: FilterLimits | None = None,
        chunker: Chunker | None = None,
    ) -> None:
        self._limits = limits or FilterLimits()
        self._chunker = chunker or SectionChunker(self._limits)

    @abstractmethod
    async def _select(
        self,
        document: ContentDocument,
        windows: tuple[ScoringWindow, ...],
        query: str,
        url: str,
    ) -> tuple[set[int], bool]:
        """Return the selected block IDs and whether matches were dropped."""

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
            document = await asyncio.to_thread(
                prepare_document,
                result,
                max_input_chars=limits.max_input_chars,
                window_chars=limits.window_chars,
            )
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
            selected, dropped = await self._select(
                document, windows, query, result.url
            )
        except FilterModelError:
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


class Bm25RelevanceFilter(RelevanceFilter):
    """Local lexical filter: BM25L-ranked blocks within the output budget."""

    async def _select(
        self,
        document: ContentDocument,
        windows: tuple[ScoringWindow, ...],
        query: str,
        url: str,
    ) -> tuple[set[int], bool]:
        return await asyncio.to_thread(
            _local_selection,
            document,
            windows,
            query,
            self._limits.max_output_chars,
        )


class LlmRelevanceFilter(RelevanceFilter):
    """Hybrid filter: embedding shortlist plus router-model block selection."""

    def __init__(
        self,
        models: RouterPassageModels,
        *,
        limits: FilterLimits | None = None,
        chunker: Chunker | None = None,
    ) -> None:
        super().__init__(limits=limits, chunker=chunker)
        self._models = models

    async def _select(
        self,
        document: ContentDocument,
        windows: tuple[ScoringWindow, ...],
        query: str,
        url: str,
    ) -> tuple[set[int], bool]:
        limits = self._limits
        try:
            async with asyncio.timeout(limits.model_timeout_seconds):
                selected = await _run_model_selection(
                    self._models, windows, query, limits
                )
        except TimeoutError:
            logger.warning("Content model deadline exceeded for %s", url)
            raise FilterModelError("model deadline exceeded") from None
        except FilterModelError:
            logger.warning("Content model response failed for %s", url)
            raise
        return selected, False
