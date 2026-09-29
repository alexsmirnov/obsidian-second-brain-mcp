"""Fallback fetcher rendering pages in an external browser over CDP (crawl4ai).
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, Protocol

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

from mcps.research.tools.common import (
    ERROR_EMPTY_RESPONSE,
    ERROR_FETCHER_UNAVAILABLE,
    Retrieve,
    http_status_error,
)

__all__ = [
    "LOCAL_CDP_URL",
    "Crawler",
    "CrawlerContext",
    "CrawlerFactory",
    "browser_crawler",
    "browser_endpoint",
    "create_browser_fetch",
    "probe_cdp",
]

logger = logging.getLogger(__file__)

LOCAL_CDP_URL = "ws://127.0.0.1:9222"
REMOTE_CONNECT_SECONDS = 10
LOCAL_STARTUP_SECONDS = 30
_LOCAL_POLL_SECONDS = 0.5
_LOCAL_STOP_SECONDS = 5


class Crawler(Protocol):
    """Minimal crawl4ai crawler contract: one crawl call per URL."""

    async def arun(
        self,
        url: str,
        config: CrawlerRunConfig,
        **kwargs: Any,
    ) -> Any: ...


class CrawlerContext(Protocol):
    """Async context manager that starts a :class:`Crawler` on entry."""

    async def __aenter__(self) -> Crawler: ...

    async def __aexit__(
        self, exc_type: Any, exc_val: Any, exc_tb: Any
    ) -> None: ...


class CrawlerFactory(Protocol):
    """Builds a fresh crawler context; crawl4ai's ``AsyncWebCrawler`` fits."""

    def __call__(self) -> CrawlerContext: ...


def _default_crawler_factory(cdp_url: str) -> CrawlerFactory:
    browser_config = BrowserConfig(
        browser_mode="custom",
        cdp_url=cdp_url,
        # Share one Playwright/CDP connection across crawls (ref-counted) and
        # give each crawl its own context so concurrent fetches don't collide.
        cache_cdp_connection=True,
        create_isolated_context=True,
        verbose=False,
    )
    return lambda: AsyncWebCrawler(config=browser_config)


def _to_fetch_result(url: str, crawl: Any) -> str:
    """Map a crawl4ai ``CrawlResult`` to rendered HTML or an error string."""
    if crawl.status_code is not None and crawl.status_code >= 400:
        return http_status_error(crawl.status_code)
    if not crawl.success:
        logger.warning("Browser fetch failed for %s: %s", url, crawl.error_message)
        return ERROR_FETCHER_UNAVAILABLE
    content = crawl.cleaned_html or ""
    if not content.strip():
        return ERROR_EMPTY_RESPONSE
    return content


def create_browser_fetch(crawler: Crawler) -> Retrieve:
    """Create a callable that renders pages on an already-open crawler.

    The crawler's lifetime is owned by the caller (see :func:`browser_crawler`),
    so fetches never start, close, or replace it. crawl4ai mutates the
    ``CrawlerRunConfig`` it is given, so every call builds a fresh one.
    """
    return _BrowserFetch(crawler)


class _BrowserFetch:
    """Borrowed-crawler retrieval: run one URL, map the crawl to content."""

    def __init__(self, crawler: Crawler) -> None:
        self._crawler = crawler

    async def __call__(self, url: str) -> str:
        try:
            crawl = await self._crawler.arun(
                url, config=CrawlerRunConfig(verbose=False)
            )
        except Exception as error:
            logger.warning("Browser fetch unavailable for %s: %r", url, error)
            return ERROR_FETCHER_UNAVAILABLE
        return _to_fetch_result(url, crawl)


@asynccontextmanager
async def browser_crawler(
    cdp_url: str | None,
    *,
    crawler_factory: CrawlerFactory | None = None,
) -> AsyncGenerator[Crawler | None]:
    """Yield one open crawler for ``cdp_url``, or ``None`` when it cannot start.

    The crawler is created and started once and closed on exit; fetches borrow
    it for their whole lifetime so crawl4ai keeps a single CDP connection.
    Initialization failures are logged and yield ``None`` so the caller can
    degrade gracefully; failures while the crawler is in use propagate.
    """
    if not cdp_url:
        yield None
        return

    factory = crawler_factory or _default_crawler_factory(cdp_url)
    stack = AsyncExitStack()
    try:
        crawler = await stack.enter_async_context(factory())
    except Exception as error:
        logger.warning("Browser crawler unavailable for %s: %r", cdp_url, error)
        await stack.aclose()
        yield None
        return
    async with stack:
        yield crawler


async def probe_cdp(cdp_url: str) -> bool:
    """Return True when a crawl4ai browser session can start at ``cdp_url``."""
    try:
        async with _default_crawler_factory(cdp_url)():
            return True
    except Exception as error:
        logger.warning("CDP probe failed for %s: %r", cdp_url, error)
        return False


async def _wait_for_local(
    process: asyncio.subprocess.Process,
    probe: Callable[[str], Awaitable[bool]],
) -> bool:
    """Poll the local CDP port until it answers, the process exits, or timeout."""
    deadline = time.monotonic() + LOCAL_STARTUP_SECONDS
    while time.monotonic() < deadline and process.returncode is None:
        try:
            async with asyncio.timeout(_LOCAL_POLL_SECONDS):
                if await probe(LOCAL_CDP_URL):
                    return True
        except Exception:
            pass
        await asyncio.sleep(_LOCAL_POLL_SECONDS)
    return False


async def _stop(process: asyncio.subprocess.Process) -> None:
    """Terminate the spawned Obscura, escalating to kill if it lingers."""
    if process.returncode is not None:
        return
    process.terminate()
    try:
        async with asyncio.timeout(_LOCAL_STOP_SECONDS):
            await process.wait()
    except TimeoutError:
        process.kill()
        await process.wait()


@asynccontextmanager
async def browser_endpoint(
    cdp_url: str,
    *,
    probe: Callable[[str], Awaitable[bool]] = probe_cdp,
) -> AsyncGenerator[str | None]:
    """Yield a reachable CDP URL, starting a local Obscura if needed.

    Prefers a configured ``cdp_url`` (probed within ``REMOTE_CONNECT_SECONDS``).
    Otherwise spawns ``obscura serve`` from ``PATH`` and waits up to
    ``LOCAL_STARTUP_SECONDS`` for it to accept CDP. Yields ``None`` when no
    browser is available; any process started here is terminated on exit.
    """
    if cdp_url:
        try:
            async with asyncio.timeout(REMOTE_CONNECT_SECONDS):
                reachable = await probe(cdp_url)
        except TimeoutError:
            reachable = False
        if reachable:
            yield cdp_url
            return

    obscura = shutil.which("obscura")
    if obscura is None:
        logger.warning(
            "No browser available: BROWSER_CDP_URL unset/unreachable and "
            "`obscura` not on PATH; web_research will be disabled."
        )
        yield None
        return

    process = await asyncio.create_subprocess_exec(
        obscura, "serve", "--stealth", "--allow-private-network"
    )
    try:
        ready = await _wait_for_local(process, probe)
        yield LOCAL_CDP_URL if ready else None
    finally:
        await _stop(process)
