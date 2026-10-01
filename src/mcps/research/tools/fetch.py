"""Fetch tool composition: routing, restrictions, browser, fallback, filters."""

from __future__ import annotations

from collections.abc import Sequence
from urllib.parse import urlparse

import httpx

from mcps.research.tools.arxiv import ArxivFetch, is_arxiv_url
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
from mcps.research.tools.filtering import MarkdownToHtml, PreTextToHtml
from mcps.research.tools.github import (
    GitHubBlobFetch,
    GitHubRepoFetch,
    is_github_blob_url,
    is_github_repo_url,
)
from mcps.research.tools.models import Fetch, FetchResult, Filter

__all__ = ["create_fetch"]


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
