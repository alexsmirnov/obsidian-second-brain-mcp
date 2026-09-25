"""Fetch callable: site-specific routing with an escalation fallback chain."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence

import httpx

from mcps.research.tools.arxiv import fetch_arxiv, is_arxiv_url
from mcps.research.tools.common import (
    ERROR_FETCHER_UNAVAILABLE,
    ERROR_UNSUPPORTED_CONTENT,
    Fetch,
    is_escalatable,
    to_error_message,
)
from mcps.research.tools.default import fetch_default
from mcps.research.tools.github import (
    fetch_github_blob,
    fetch_github_repo,
    is_github_blob_url,
    is_github_repo_url,
)
from mcps.research.tools.reddit import fetch_reddit, is_reddit_url
from mcps.research.tools.wikipedia import fetch_wikipedia, is_wikipedia_url

__all__ = ["create_fetch"]

logger = logging.getLogger(__file__)

SiteFetcher = Callable[..., Awaitable[str]]

# First matching predicate wins; fetch_default handles everything else.
_SITE_ROUTES: tuple[tuple[Callable[[str], bool], SiteFetcher], ...] = (
    (is_arxiv_url, fetch_arxiv),
    (is_wikipedia_url, fetch_wikipedia),
    (is_reddit_url, fetch_reddit),
    (is_github_blob_url, fetch_github_blob),
    (is_github_repo_url, fetch_github_repo),
)


def _select_fetcher(url: str) -> SiteFetcher:
    return next(
        (fetcher for matches, fetcher in _SITE_ROUTES if matches(url)),
        fetch_default,
    )


async def _fetch_direct(
    url: str, *, http_client: httpx.AsyncClient | None, max_chars: int
) -> str:
    fetcher = _select_fetcher(url)
    try:
        return await fetcher(url, http_client=http_client, max_chars=max_chars)
    except httpx.HTTPError as error:
        message = to_error_message(error)
    except Exception:
        message = ERROR_UNSUPPORTED_CONTENT
    logger.warning("Web fetch failed for %s: %s", url, message)
    return message


def create_fetch(
    *,
    http_client: httpx.AsyncClient | None = None,
    max_chars: int = 15000,
    fallbacks: Sequence[Fetch] = (),
) -> Fetch:
    """Create an async webpage fetch callable with markdown extraction.

    When the direct fetch is blocked (401/403/429) or returns an empty
    page, each fallback is tried in order until one yields a result that
    is not escalatable. A fallback reporting itself unavailable is skipped
    and keeps the previous result, so the caller sees the target's error.
    """

    async def fetch(url: str) -> str:
        result = await _fetch_direct(url, http_client=http_client, max_chars=max_chars)
        for fallback in fallbacks:
            if not is_escalatable(result):
                break
            logger.info("Escalating %s after %s", url, result)
            candidate = await fallback(url)
            if candidate != ERROR_FETCHER_UNAVAILABLE:
                result = candidate
        return result

    return fetch
