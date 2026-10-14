"""Combinators assembling Fetch and Filter instances into new ones."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import replace

from mcps.research.tools.common import TRUNCATION_MARKER
from mcps.research.tools.models import Fetch, FetchResult, FetchStatus, Filter

__all__ = [
    "Blocked",
    "Fallback",
    "FilterChain",
    "FilterSelector",
    "Filtered",
    "Throttled",
    "Truncate",
    "UrlSelector",
]

logger = logging.getLogger(__name__)

UrlPredicate = Callable[[str], bool]
ResultPredicate = Callable[[FetchResult], bool]


class Filtered:
    """Fetch whose successful results pass through a filter."""

    def __init__(self, fetch: Fetch, page_filter: Filter) -> None:
        self._fetch = fetch
        self._filter = page_filter

    async def __call__(self, url: str, query: str | None = None, /) -> FetchResult:
        result = await self._fetch(url, query)
        if not result.ok:
            return result
        return await self._filter(result, query)


class Fallback:
    """Fetch trying ``secondary`` once when ``primary`` fails in a retryable way.

    An unavailable secondary says nothing about the page, so the primary
    result is kept.
    """

    def __init__(
        self,
        primary: Fetch,
        secondary: Fetch,
        when: ResultPredicate = FetchResult.is_retryable,
    ) -> None:
        self._primary = primary
        self._secondary = secondary
        self._when = when

    async def __call__(self, url: str, query: str | None = None, /) -> FetchResult:
        result = await self._primary(url, query)
        if not self._when(result):
            return result
        logger.info("Escalating %s after %s", url, result.status)
        candidate = await self._secondary(url, query)
        if candidate.status is FetchStatus.UNAVAILABLE:
            return result
        return candidate


class UrlSelector:
    """Fetch delegating to the first route whose predicate matches the URL."""

    def __init__(
        self, routes: Sequence[tuple[UrlPredicate, Fetch]], default: Fetch
    ) -> None:
        self._routes = tuple(routes)
        self._default = default

    async def __call__(self, url: str, query: str | None = None, /) -> FetchResult:
        fetch = next((route for matches, route in self._routes if matches(url)), None)
        return await (fetch or self._default)(url, query)


class FilterSelector:
    """Filter delegating to the first route matching the result.

    Without a match the result is returned unchanged unless ``default`` is set.
    """

    def __init__(
        self,
        routes: Sequence[tuple[ResultPredicate, Filter]],
        default: Filter | None = None,
    ) -> None:
        self._routes = tuple(routes)
        self._default = default

    async def __call__(
        self, result: FetchResult, query: str | None = None, /
    ) -> FetchResult:
        page_filter = next(
            (route for matches, route in self._routes if matches(result)),
            self._default,
        )
        if page_filter is None:
            return result
        return await page_filter(result, query)


class FilterChain:
    """Filter applying filters in order, stopping at the first failure."""

    def __init__(self, *filters: Filter) -> None:
        self._filters = filters

    async def __call__(
        self, result: FetchResult, query: str | None = None, /
    ) -> FetchResult:
        for page_filter in self._filters:
            result = await page_filter(result, query)
            if not result.ok:
                break
        return result


class Throttled:
    """Fetch allowing at most ``limit`` concurrent calls."""

    def __init__(self, fetch: Fetch, limit: int) -> None:
        self._fetch = fetch
        self._semaphore = asyncio.Semaphore(limit)

    async def __call__(self, url: str, query: str | None = None, /) -> FetchResult:
        async with self._semaphore:
            return await self._fetch(url, query)


class Blocked:
    """Fetch refusing every URL without I/O."""

    async def __call__(self, url: str, query: str | None = None, /) -> FetchResult:
        return FetchResult(url=url, status=FetchStatus.RESTRICTED, mime="")


_URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)


def _url_safe_cut(content: str, max_chars: int) -> int:
    """Return a cut position that never splits an HTTP(S) URL token."""
    for match in _URL_RE.finditer(content):
        if match.start() >= max_chars:
            break
        if match.end() > max_chars:
            return match.start()
    return max_chars


def _prefix(content: str, max_chars: int) -> str:
    """Longest prefix ending at a word or newline boundary, never splitting a URL."""
    if max_chars <= 0:
        return ""
    candidate = content[: _url_safe_cut(content, max_chars)]
    boundary = max(candidate.rfind(separator) for separator in "\n \t")
    return candidate[:boundary] if boundary > 0 else candidate


class Truncate:
    """Filter cutting content beyond ``max_chars`` at a word boundary."""

    def __init__(self, max_chars: int) -> None:
        if max_chars < 0:
            raise ValueError("max_chars must be nonnegative")
        self._max_chars = max_chars

    async def __call__(
        self, result: FetchResult, query: str | None = None, /
    ) -> FetchResult:
        logger.info("Fetch %d chars from url %s", len(result.content), result.url)
        body = result.content.removesuffix(TRUNCATION_MARKER)
        if len(body) <= self._max_chars:
            return result
        content = _prefix(body, self._max_chars)
        return replace(result, content=content + TRUNCATION_MARKER)
