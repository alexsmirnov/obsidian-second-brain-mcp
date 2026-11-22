"""Yahoo HTML search callable, the free fallback when Google CSE is absent.

Result selectors and redirect unwrapping are adapted from ``ddgs``
(``ddgs/engines/yahoo.py``), MIT License, Copyright (c) 2022 deedy5.
License: https://github.com/deedy5/ddgs/blob/main/LICENSE.md
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from secrets import token_urlsafe
from urllib.parse import unquote_plus, urlsplit

import httpx
from lxml import etree, html

from mcps.research.tools.models import Search, SearchResult

__all__ = ["create_yahoo_search"]

logger = logging.getLogger(__name__)

YAHOO_SEARCH_URL = "https://search.yahoo.com/search"
YAHOO_REDIRECT_HOST = "r.search.yahoo.com"
YAHOO_AD_PREFIX = "https://www.bing.com/aclick?"

# Browser-like headers. Yahoo answers HTTP 500 with an empty body to default
# httpx headers. br/zstd are omitted so httpx can always decompress.
YAHOO_HEADERS = {
    "accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,*/*;q=0.8"
    ),
    "accept-encoding": "gzip, deflate",
    "accept-language": "en-US,en;q=0.9",
    "cache-control": "no-cache",
    "pragma": "no-cache",
    "sec-ch-ua": (
        '"Google Chrome";v="141", "Not?A_Brand";v="8",'
        ' "Chromium";v="141"'
    ),
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "none",
    "sec-fetch-user": "?1",
    "upgrade-insecure-requests": "1",
    "user-agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/141.0.0.0 Safari/537.36"
    ),
}


class _RetryableError(Exception):
    """A failure worth another attempt; the message goes into the warning."""


@dataclass(frozen=True)
class _RetryPolicy:
    max_attempts: int
    backoff_seconds: float
    sleep: Callable[[float], Awaitable[None]]

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.backoff_seconds < 0:
            raise ValueError("backoff_seconds must be >= 0")

    async def wait_before_retry(self, failed_attempt: int) -> None:
        await self.sleep(self.backoff_seconds * 2 ** (failed_attempt - 1))


def _search_url() -> str:
    """Return the search URL with random path tokens, as the Yahoo web UI does."""
    return f"{YAHOO_SEARCH_URL};_ylt={token_urlsafe(18)};_ylu={token_urlsafe(35)}"


def _yahoo_params(query: str, timelimit: str | None) -> dict[str, str]:
    params = {"p": query}
    if timelimit:
        params["btf"] = timelimit
    return params


def _unwrap_redirect(href: str) -> str:
    """Return the destination of a Yahoo redirect link; other links pass through."""
    if urlsplit(href).hostname != YAHOO_REDIRECT_HOST or "/RU=" not in href:
        return href
    target = href.split("/RU=", 1)[1].split("/RK=", 1)[0].split("/RS=", 1)[0]
    return unquote_plus(target)


def _parse_result_item(item: html.HtmlElement) -> SearchResult | None:
    hrefs: list[str] = item.xpath(".//div[contains(@class, 'Title')]//a/@href")
    try:
        url = _unwrap_redirect(hrefs[0]) if hrefs else ""
    except ValueError:  # malformed URL, e.g. an invalid IPv6 host
        return None
    if not url.startswith("http") or url.startswith(YAHOO_AD_PREFIX):
        return None
    return SearchResult(
        url=url,
        title=" ".join(
            "".join(item.xpath(".//div[contains(@class, 'Title')]//h3//text()")).split()
        ),
        snippet=" ".join(
            "".join(item.xpath(".//div[contains(@class, 'Text')]//text()")).split()
        ),
    )


def _parse_yahoo_results(html_text: str) -> list[SearchResult]:
    try:
        tree = html.fromstring(html_text)
    except (etree.ParserError, ValueError) as error:  # empty or non-HTML body
        logger.warning("Yahoo returned an unparseable page: %s", error)
        return []
    items = tree.xpath("//div[contains(@class, 'relsrch')]")
    return [
        result
        for item in items
        if (result := _parse_result_item(item)) is not None
    ]


async def _fetch_once(
    client: httpx.AsyncClient, query: str, timelimit: str | None
) -> str:
    """Return the result page HTML; raise ``_RetryableError`` on transient failures."""
    try:
        response = await client.get(
            _search_url(),
            params=_yahoo_params(query, timelimit),
            headers=YAHOO_HEADERS,
            follow_redirects=True,
        )
    except httpx.TransportError as error:
        raise _RetryableError(repr(error)) from error
    status = response.status_code
    if status == 429 or status >= 500:
        raise _RetryableError(f"HTTP {status}")
    response.raise_for_status()
    return response.text


async def _search_with_retries(
    client: httpx.AsyncClient,
    query: str,
    timelimit: str | None,
    retry: _RetryPolicy,
) -> str:
    """Return the page HTML, or raise the last ``_RetryableError``."""
    for failed_attempt in range(1, retry.max_attempts):
        try:
            return await _fetch_once(client, query, timelimit)
        except _RetryableError:
            await retry.wait_before_retry(failed_attempt)
    return await _fetch_once(client, query, timelimit)


@asynccontextmanager
async def _client_scope(
    http_client: httpx.AsyncClient | None,
) -> AsyncGenerator[httpx.AsyncClient]:
    """Yield the injected client, or a short-lived owned one."""
    if http_client is not None:
        yield http_client
        return
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as owned:
        yield owned


def create_yahoo_search(
    *,
    http_client: httpx.AsyncClient | None = None,
    timelimit: str | None = None,
    max_attempts: int = 3,
    backoff_seconds: float = 1.0,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> Search:
    """Create an async Yahoo HTML search callable.

    Transient failures (HTTP 429, HTTP 5xx, transport errors) are retried up to
    ``max_attempts`` times in total with exponential backoff. Any failure that
    survives returns an empty list, like the other ``Search`` factories.
    """
    retry = _RetryPolicy(max_attempts, backoff_seconds, sleep)

    async def search(query: str, /) -> list[SearchResult]:
        try:
            async with _client_scope(http_client) as client:
                page = await _search_with_retries(client, query, timelimit, retry)
        except (_RetryableError, httpx.HTTPError) as error:
            logger.warning("Yahoo search failed for query %r: %s", query, error)
            return []
        return _parse_yahoo_results(page)

    return search
