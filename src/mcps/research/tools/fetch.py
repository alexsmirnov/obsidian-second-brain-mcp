"""Fetch tool composition: routing, restrictions, browser, fallback, filters."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Callable, Sequence
from contextlib import asynccontextmanager
from urllib.parse import urlparse

import httpx

from mcps.common.ai_models import create_chat_model
from mcps.common.decisions import DecisionClient
from mcps.config import ServerConfig
from mcps.research.tools.arxiv import ArxivFetch, is_arxiv_url
from mcps.research.tools.bm25_ranker import Bm25Ranker
from mcps.research.tools.bright_data import BrightDataFetch
from mcps.research.tools.browser import create_browser_fetch
from mcps.research.tools.combinators import (
    Blocked,
    Fallback,
    FilterChain,
    Filtered,
    FilterSelector,
    Truncate,
    UrlSelector,
)
from mcps.research.tools.decision_ranker import DecisionRanker
from mcps.research.tools.default import HttpFetch
from mcps.research.tools.extractors import extractor_for_mime, extractor_for_url
from mcps.research.tools.filtering import LlmRelevanceFilter, Ranker, RelevanceFilter, ChunkedRelevanceFilter, llm_filter_model_limits
from mcps.research.tools.github import (
    GitHubBlobFetch,
    GitHubRepoFetch,
    is_github_blob_url,
    is_github_repo_url,
)
from mcps.research.tools.html_markdown import (
    HtmlToMarkdown,
    create_markdown_generator,
    is_html,
)
from mcps.research.tools.markdown_chunker import MarkdownChunker
from mcps.research.tools.models import Fetch, FetchResult, FetchStatus, Filter
from mcps.research.tools.scrape_do import ScrapeDoFetch
from mcps.research.tools.wikipedia import (
    WIKIPEDIA_HEADERS,
    WikipediaFetch,
    is_wikipedia_url,
)

__all__ = ["build_fetch_tool", "create_fetch", "create_page_filter"]

logger = logging.getLogger(__name__)


def is_restricted(domains: Sequence[str]):
    """Predicate: hostname equals or is a subdomain of a blocked domain."""

    def matches(url: str) -> bool:
        hostname = urlparse(url).hostname or ""
        return any(
            hostname == domain or hostname.endswith(f".{domain}")
            for domain in domains
        )

    return matches


def _has_binary_extension(url: str) -> bool:
    return extractor_for_url(url) is not None


def is_extractable_handoff(result: FetchResult) -> bool:
    """True for a browser result that names a media type ``HttpFetch`` can read."""
    return (
        result.status is FetchStatus.UNSUPPORTED
        and extractor_for_mime(result.mime) is not None
    )


def create_fetch(
    *,
    http_client: httpx.AsyncClient,
    browser: Fetch | None = None,
    provider: Fetch | None = None,
    page_filter: Filter,
    restricted_domains: Sequence[str] = (),
    max_chars: int = 15000,
) -> Fetch:
    """Compose the page fetch returning query-relevant Markdown.

    A restricted host is refused without I/O. GitHub, arXiv and Wikipedia use
    their specialized fetcher and never fall back. URLs with a registered binary
    extension (see :mod:`mcps.research.tools.extractors`) and, when no browser is
    available, every other URL use the httpx extractor; all remaining URLs are
    rendered in the browser. A browser result that reports a registered binary
    media type is retried once through the httpx extractor. A blocked,
    empty, timed-out, or unavailable generic result escalates once to
    ``provider`` when configured. Every successful source is converted to
    Markdown when HTML, passed through ``page_filter`` and truncated to
    ``max_chars``.
    """
    http = HttpFetch(http_client)
    wikipedia_http = HttpFetch(http_client, WIKIPEDIA_HEADERS)
    generic: Fetch = (
        Fallback(browser, http, when=is_extractable_handoff) if browser else http
    )
    generic = UrlSelector([(_has_binary_extension, http)], default=generic)
    if provider is not None:
        generic = Fallback(generic, provider)

    routed = UrlSelector(
        [
            (is_restricted(restricted_domains), Blocked()),
            (is_arxiv_url, ArxivFetch(http)),
            (is_github_blob_url, GitHubBlobFetch(http)),
            (is_github_repo_url, GitHubRepoFetch(http)),
            (is_wikipedia_url, WikipediaFetch(wikipedia_http)),
        ],
        default=generic,
    )
    to_markdown = FilterSelector(
        [(is_html, HtmlToMarkdown(create_markdown_generator()))]
    )
    return Filtered(
        routed, FilterChain(to_markdown, page_filter, Truncate(max_chars))
    )


def _create_provider_fallback(
    config: ServerConfig, http_client: httpx.AsyncClient
) -> Fetch | None:
    """Build the commercial unblocking provider, or None when unconfigured."""
    match config.scraper.provider:
        case "scrape_do":
            return ScrapeDoFetch(
                config.scraper.scrape_do_token,
                http_client=http_client,
            )
        case "bright_data":
            return BrightDataFetch(
                config.scraper.bright_data_api_key,
                config.scraper.bright_data_zone,
                http_client=http_client,
            )
    return None


RankerBuilder = Callable[[ServerConfig, httpx.AsyncClient], Ranker]

# Chunker (max_chars, overlap_chars) per ranker. BM25 scores short windows so
# boilerplate lands in fragments of its own; the decision ranker pays one
# model call per window. Tuning: tasks/relevance-experiments-generated.md.
BM25_WINDOW = (300, 0)
BM25_THRESHOLD = 0.0
BM25_RELATIVE_CUTOFF = 0.075
BM25_SMOOTHING = 0.2
DECISION_WINDOW = (5000, 1500)


def _decision_configured(config: ServerConfig) -> bool:
    """True when DECISION_MODEL and a router URL are set."""
    return bool(config.fetch.decision_model and config.router.api_base)


def _decision_ranker(config: ServerConfig, http_client: httpx.AsyncClient) -> Ranker:
    client = DecisionClient(
        http_client,
        router_url=config.router.api_base,
        router_key=config.router.api_key,
        model=config.fetch.decision_model,
    )
    return DecisionRanker(client)


def _bm25_ranker(config: ServerConfig, http_client: httpx.AsyncClient) -> Ranker:
    return Bm25Ranker(
        threshold=BM25_THRESHOLD,
        relative_cutoff=BM25_RELATIVE_CUTOFF,
        smoothing=BM25_SMOOTHING,
    )


# Ordered registry: the first entry whose predicate holds builds the ranker.
# To add an implementation, append an entry before the BM25 catch-all.
_RANKERS: tuple[
    tuple[Callable[[ServerConfig], bool], RankerBuilder, tuple[int, int]], ...
] = (
    (_decision_configured, _decision_ranker, DECISION_WINDOW),
    (lambda _: True, _bm25_ranker, BM25_WINDOW),
)


def create_page_filter(
    config: ServerConfig, http_client: httpx.AsyncClient
) -> RelevanceFilter:
    """Build the LLM based filter."""
    fetch_model = create_chat_model(config.fetch.fetch_model,config.router,http_client,reasoning_effort="none",**llm_filter_model_limits())
    return LlmRelevanceFilter(fetch_model)


@asynccontextmanager
async def build_fetch_tool(
    config: ServerConfig,
    http_client: httpx.AsyncClient,
    *,
    require_browser: bool = False,
) -> AsyncGenerator[Fetch | None]:
    """Compose the page fetch for ``config`` for the duration of the context.

    Owns the browser lifecycle (see :func:`create_browser_fetch`) so the caller
    only enters a single context manager. With ``require_browser`` the context
    yields ``None`` when no browser is available; otherwise it falls back to
    plain httpx for generic pages. The supplied ``http_client`` is borrowed and
    never closed here.
    """
    async with create_browser_fetch(config) as browser:
        if browser is None and require_browser:
            yield None
            return
        yield create_fetch(
            http_client=http_client,
            browser=browser,
            provider=_create_provider_fallback(config, http_client),
            page_filter=create_page_filter(config, http_client),
            restricted_domains=config.fetch.restricted_domains,
        )
