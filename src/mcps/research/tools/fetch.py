"""Fetch callable: site routing, restrictions, browser render, fallback."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from urllib.parse import urlparse

import httpx

from mcps.research.tools.arxiv import ArxivFetch, is_arxiv_url
from mcps.research.tools.common import (
    ERROR_EMPTY_RESPONSE,
    ERROR_FETCHER_UNAVAILABLE,
    ERROR_REQUEST_TIMEOUT,
    ERROR_UNSUPPORTED_CONTENT,
    MIME_MARKDOWN,
    Fetch,
    format_source_output,
    http_status_error,
)
from mcps.research.tools.default import HttpFetch
from mcps.research.tools.filtering import (
    PageFilter,
    markdown_to_html,
    text_page_to_html,
)
from mcps.research.tools.github import (
    GitHubBlobFetch,
    GitHubRepoFetch,
    is_github_blob_url,
    is_github_repo_url,
)
from mcps.research.tools.result import Fetch as SourceFetch
from mcps.research.tools.result import FetchResult, FetchStatus

__all__ = ["create_fetch"]

logger = logging.getLogger(__name__)


def _legacy_error(result: FetchResult) -> str:
    """Error string for a failed result (removed once fetch returns results)."""
    match result.status:
        case FetchStatus.HTTP_ERROR:
            return http_status_error(result.http_status or 0)
        case FetchStatus.TIMEOUT:
            return ERROR_REQUEST_TIMEOUT
        case FetchStatus.EMPTY:
            return ERROR_EMPTY_RESPONSE
        case FetchStatus.UNAVAILABLE:
            return ERROR_FETCHER_UNAVAILABLE
        case _:
            return ERROR_UNSUPPORTED_CONTENT


def _is_restricted(url: str, domains: Sequence[str]) -> bool:
    """True when the hostname equals or is a subdomain of a blocked domain."""
    hostname = urlparse(url).hostname or ""
    return any(
        hostname == domain or hostname.endswith(f".{domain}") for domain in domains
    )


def create_fetch(
    *,
    http_client: httpx.AsyncClient | None = None,
    browser: SourceFetch | None = None,
    provider: SourceFetch | None = None,
    page_filter: PageFilter,
    restricted_domains: Sequence[str] = (),
    concurrency: int = 2,
    max_chars: int = 15000,
) -> Fetch:
    """Create an async page fetch returning query-relevant Markdown.

    A restricted host returns ``""`` without I/O. GitHub/arXiv use their
    specialized fetcher and never fall back. ``.pdf`` and (when no browser is
    available) every other URL use the httpx extractor; all remaining URLs are
    rendered in the CDP browser. Every successful source is filtered by
    ``page_filter`` and truncated. A blocked, empty, timed-out, or
    browser-unavailable result escalates once to ``provider`` when configured.
    Concurrent browser renders are capped at ``concurrency``.
    """
    semaphore = asyncio.Semaphore(concurrency)
    http = HttpFetch(http_client)
    specialized: tuple[tuple[Callable[[str], bool], SourceFetch], ...] = (
        (is_arxiv_url, ArxivFetch(http)),
        (is_github_blob_url, GitHubBlobFetch(http_client)),
        (is_github_repo_url, GitHubRepoFetch(http_client)),
    )

    async def _filter(
        source: FetchResult, query: str | None, *, text_page: bool
    ) -> str:
        html = source.content
        if source.mime == MIME_MARKDOWN:
            html = markdown_to_html(html)
        elif text_page:
            html = text_page_to_html(html)
        filtered = await page_filter(html, source.base_url or source.url, query)
        return format_source_output(source.url, filtered, max_chars)

    async def _finish(
        source: FetchResult, query: str | None, *, text_page: bool = False
    ) -> str:
        if not source.ok:
            return _legacy_error(source)
        return await _filter(source, query, text_page=text_page)

    async def fetch(url: str, query: str | None = None) -> str:
        if _is_restricted(url, restricted_domains):
            return ""

        routed = next(
            (route for matches, route in specialized if matches(url)), None
        )
        if routed is not None:
            return await _finish(await routed(url), query)

        is_pdf = urlparse(url).path.lower().endswith(".pdf")
        if browser is None or is_pdf:
            source = await http(url)
            text_page = False
        else:
            async with semaphore:
                source = await browser(url)
            text_page = True
        result = await _finish(source, query, text_page=text_page)

        if provider is not None and source.is_retryable():
            logger.info("Escalating %s after %s", url, source.status)
            candidate = await provider(url)
            if candidate.status is not FetchStatus.UNAVAILABLE:
                result = await _finish(candidate, query, text_page=True)
        return result

    return fetch
