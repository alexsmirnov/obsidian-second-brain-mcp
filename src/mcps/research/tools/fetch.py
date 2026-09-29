"""Fetch callable: site routing, restrictions, browser render, fallback."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from urllib.parse import urlparse

import httpx

from mcps.research.tools.arxiv import fetch_arxiv, is_arxiv_url
from mcps.research.tools.common import (
    ERROR_FETCHER_UNAVAILABLE,
    ERROR_UNSUPPORTED_CONTENT,
    Fetch,
    Retrieve,
    format_source_output,
    is_escalatable,
    to_error_message,
)
from mcps.research.tools.default import fetch_default
from mcps.research.tools.filtering import PageFilter, text_page_to_html
from mcps.research.tools.github import (
    fetch_github_blob,
    fetch_github_repo,
    github_link_base,
    is_github_blob_url,
    is_github_repo_url,
)

__all__ = ["create_fetch"]

logger = logging.getLogger(__file__)

SiteFetcher = Callable[..., Awaitable[str]]

# GitHub and arXiv keep their specialized HTTP fetcher and never fall back.
_SPECIALIZED_ROUTES: tuple[tuple[Callable[[str], bool], SiteFetcher], ...] = (
    (is_arxiv_url, fetch_arxiv),
    (is_github_blob_url, fetch_github_blob),
    (is_github_repo_url, fetch_github_repo),
)

# Intermediate content is not truncated: the caller filters, then truncates.
_NO_TRUNCATION = 10**9


async def _call_fetcher(
    fetcher: SiteFetcher,
    url: str,
    *,
    http_client: httpx.AsyncClient | None,
    max_chars: int,
) -> str:
    try:
        return await fetcher(url, http_client=http_client, max_chars=max_chars)
    except httpx.HTTPError as error:
        message = to_error_message(error)
    except Exception:
        message = ERROR_UNSUPPORTED_CONTENT
    logger.warning("Web fetch failed for %s: %s", url, message)
    return message


def _is_restricted(url: str, domains: Sequence[str]) -> bool:
    """True when the hostname equals or is a subdomain of a blocked domain."""
    hostname = urlparse(url).hostname or ""
    return any(
        hostname == domain or hostname.endswith(f".{domain}") for domain in domains
    )


def create_fetch(
    *,
    http_client: httpx.AsyncClient | None = None,
    browser: Retrieve | None = None,
    provider: Retrieve | None = None,
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

    async def _filter(
        html: str, url: str, query: str | None, base_url: str | None = None
    ) -> str:
        filtered = await page_filter(html, base_url or url, query)
        return format_source_output(url, filtered, max_chars)

    async def fetch(url: str, query: str | None = None) -> str:
        if _is_restricted(url, restricted_domains):
            return ""

        fetcher = next(
            (route for matches, route in _SPECIALIZED_ROUTES if matches(url)), None
        )
        if fetcher is not None:
            result = await _call_fetcher(
                fetcher, url, http_client=http_client, max_chars=_NO_TRUNCATION
            )
            if result.startswith("ERROR"):
                return result
            return await _filter(result, url, query, github_link_base(url))

        is_pdf = urlparse(url).path.lower().endswith(".pdf")
        if browser is None or is_pdf:
            result = await _call_fetcher(
                fetch_default, url, http_client=http_client, max_chars=_NO_TRUNCATION
            )
            if not result.startswith("ERROR"):
                result = await _filter(result, url, query)
        else:
            async with semaphore:
                rendered = await browser(url)
            if rendered.startswith("ERROR"):
                result = rendered
            else:
                result = await _filter(text_page_to_html(rendered), url, query)

        if provider is not None and is_escalatable(result):
            logger.info("Escalating %s after %s", url, result)
            candidate = await provider(url)
            if candidate != ERROR_FETCHER_UNAVAILABLE:
                if candidate.startswith("ERROR"):
                    result = candidate
                else:
                    result = await _filter(text_page_to_html(candidate), url, query)
        return result

    return fetch
