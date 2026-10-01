"""Fetch tool composition: routing, restrictions, browser, fallback, filters."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from urllib.parse import urlparse

import httpx

from mcps.config import ServerConfig
from mcps.research.tools.arxiv import ArxivFetch, is_arxiv_url
from mcps.research.tools.bright_data import BrightDataFetch
from mcps.research.tools.browser import create_browser_fetch
from mcps.research.tools.combinators import (
    Blocked,
    Fallback,
    FilterChain,
    Filtered,
    FilterSelector,
    Throttled,
    Truncate,
    UrlSelector,
)
from mcps.research.tools.common import MIME_HTML, MIME_MARKDOWN
from mcps.research.tools.default import HttpFetch
from mcps.research.tools.filtering import MarkdownToHtml, PreTextToHtml, RelevanceFilter
from mcps.research.tools.github import (
    GitHubBlobFetch,
    GitHubRepoFetch,
    is_github_blob_url,
    is_github_repo_url,
)
from mcps.research.tools.models import Fetch, FetchResult, Filter
from mcps.research.tools.scrape_do import ScrapeDoFetch

__all__ = ["build_fetch_tool", "create_fetch"]

logger = logging.getLogger(__name__)


def _is_restricted(domains: Sequence[str]):
    """Predicate: hostname equals or is a subdomain of a blocked domain."""

    def matches(url: str) -> bool:
        hostname = urlparse(url).hostname or ""
        return any(
            hostname == domain or hostname.endswith(f".{domain}")
            for domain in domains
        )

    return matches


def _is_pdf_url(url: str) -> bool:
    return urlparse(url).path.lower().endswith(".pdf")


def _is_markdown(result: FetchResult) -> bool:
    return result.mime == MIME_MARKDOWN


def _is_html(result: FetchResult) -> bool:
    return result.mime == MIME_HTML


def create_fetch(
    *,
    http_client: httpx.AsyncClient | None = None,
    browser: Fetch | None = None,
    provider: Fetch | None = None,
    page_filter: Filter,
    restricted_domains: Sequence[str] = (),
    concurrency: int = 2,
    max_chars: int = 15000,
) -> Fetch:
    """Compose the page fetch returning query-relevant Markdown.

    A restricted host is refused without I/O. GitHub and arXiv use their
    specialized fetcher and never fall back. ``.pdf`` and (when no browser is
    available) every other URL use the httpx extractor; all remaining URLs are
    rendered in the browser, at most ``concurrency`` at a time. A blocked,
    empty, timed-out, or unavailable generic result escalates once to
    ``provider`` when configured. Every successful source is normalized to
    HTML, passed through ``page_filter`` and truncated to ``max_chars``.
    """
    http = HttpFetch(http_client)
    generic: Fetch = Throttled(browser, concurrency) if browser else http
    generic = UrlSelector([(_is_pdf_url, http)], default=generic)
    if provider is not None:
        generic = Fallback(generic, provider)

    routed = UrlSelector(
        [
            (_is_restricted(restricted_domains), Blocked()),
            (is_arxiv_url, ArxivFetch(http)),
            (is_github_blob_url, GitHubBlobFetch(http)),
            (is_github_repo_url, GitHubRepoFetch(http)),
        ],
        default=generic,
    )
    normalize = FilterSelector(
        [
            (_is_markdown, MarkdownToHtml()),
            (_is_html, PreTextToHtml()),
        ]
    )
    return Filtered(
        routed, FilterChain(normalize, page_filter, Truncate(max_chars))
    )


def _create_provider_fallback(
    config: ServerConfig, http_client: httpx.AsyncClient
) -> Fetch | None:
    """Build the commercial unblocking provider, or None when unconfigured."""
    match config.scraper_provider:
        case "":
            return None
        case "scrape_do" if config.scrape_do_token:
            return ScrapeDoFetch(
                config.scrape_do_token,
                http_client=http_client,
            )
        case "bright_data" if config.bright_data_api_key and config.bright_data_zone:
            return BrightDataFetch(
                config.bright_data_api_key,
                config.bright_data_zone,
                http_client=http_client,
            )
        case "scrape_do" | "bright_data":
            logger.warning(
                "SCRAPER_PROVIDER=%s is missing credentials; provider fallback "
                "disabled.",
                config.scraper_provider,
            )
        case _:
            logger.warning(
                "Unknown SCRAPER_PROVIDER=%s (expected scrape_do or bright_data); "
                "provider fallback disabled.",
                config.scraper_provider,
            )
    return None


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
            page_filter=RelevanceFilter(
                fetch_model=config.fetch_model,
                router_url=config.router_api_base,
                router_key=config.router_api_key,
            ),
            restricted_domains=config.fetch_restricted_domains,
            concurrency=config.fetch_concurrency,
        )
