"""Fallback fetcher rendering pages in an external browser over CDP (crawl4ai).

The module owns the whole browser lifecycle. :func:`create_browser_fetch`
resolves a reachable CDP endpoint (configured or a locally spawned
``obscura``), starts one crawl4ai crawler, and closes both on exit, so callers
only have to enter a single context manager.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
from contextlib import AsyncExitStack, asynccontextmanager
from typing import AsyncGenerator, cast

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CrawlResult

from mcps.config import ServerConfig
from mcps.research.tools.common import MIME_HTML, failure
from mcps.research.tools.models import FetchResult, FetchStatus

__all__ = [
    "LOCAL_CDP_URL",
    "BrowserFetch",
    "create_browser_fetch",
]

logger = logging.getLogger(__name__)

LOCAL_CDP_URL = "ws://127.0.0.1:9222"
REMOTE_CONNECT_SECONDS = 10
LOCAL_STARTUP_SECONDS = 30
_LOCAL_POLL_SECONDS = 0.5
_LOCAL_STOP_SECONDS = 5


def _new_crawler(cdp_url: str) -> AsyncWebCrawler:
    """Build a crawl4ai crawler bound to ``cdp_url``.

    Shares one Playwright/CDP connection across crawls (ref-counted) and gives
    each crawl its own context so concurrent fetches don't collide.
    """
    browser_config = BrowserConfig(
        browser_mode="custom",
        cdp_url=cdp_url,
        cache_cdp_connection=False,
        create_isolated_context=False,
        enable_stealth=False,
        verbose=False,
        text_mode=True,
        light_mode=True,
        avoid_ads=True,
        memory_saving_mode=True,
        cdp_cleanup_on_close=True,
        # headers={
        #     "Accept-Language": "en-US,en;q=0.9",
        #     "Referer": "https://google.com",
        #     "DNT": "1",
        #     "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        #     "Sec-Ch-Ua": '"Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"',
        #     "Sec-Ch-Ua-Mobile": "?0",
        # }
    )
    return AsyncWebCrawler(config=browser_config)


def _to_fetch_result(url: str, crawl: CrawlResult) -> FetchResult:
    """Map a crawl4ai ``CrawlResult`` to rendered HTML or a failed result."""
    if crawl.status_code is not None and crawl.status_code >= 400:
        return failure(url, FetchStatus.HTTP_ERROR, crawl.status_code)
    if not crawl.success:
        logger.warning("Browser fetch failed for %s: %s", url, crawl.error_message)
        return failure(url, FetchStatus.UNAVAILABLE)
    content = crawl.cleaned_html or ""
    if not content.strip():
        return failure(url, FetchStatus.EMPTY)
    return FetchResult(
        url=url, status=FetchStatus.OK, mime=MIME_HTML, content=content
    )


class BrowserFetch:
    """Render pages on an already-open crawler.

    The crawler's lifetime is owned by :func:`create_browser_fetch`, so fetches
    never start, close, or replace it. crawl4ai mutates the ``CrawlerRunConfig``
    it is given, so every call builds a fresh one.
    """

    def __init__(self, crawler: AsyncWebCrawler) -> None:
        self._crawler = crawler

    async def __call__(self, url: str, query: str | None = None, /) -> FetchResult:
        try:
            crawler_config = CrawlerRunConfig(
                simulate_user=False,  # Add user simulation
                magic=False,  # Enable magic mode
                wait_until="domcontentloaded",
                # Give cloudflare a moment to render the DOM elements, it waits 5 second
                delay_before_return_html=0.2,
                scan_full_page=False,
                exclude_all_images=True,
                process_iframes=False,       # Prevents recursively crawling deep iframe trees
                wait_for_images=False,       # Cuts down on event loops monitoring image states
                # user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            )
            crawl = await self._crawler.arun(url, config=crawler_config)
        except Exception as error:
            logger.warning("Browser fetch unavailable for %s: %r", url, error)
            return failure(url, FetchStatus.UNAVAILABLE)
        return _to_fetch_result(url, cast(CrawlResult, crawl))


@asynccontextmanager
async def create_browser_fetch(
    config: ServerConfig,
) -> AsyncGenerator[BrowserFetch | None]:
    """Yield a browser fetcher on one persistent crawler, or ``None``.

    Owns the whole browser lifecycle: it resolves a reachable CDP endpoint from
    ``config.browser_cdp_url`` (or a locally spawned ``obscura``), starts one
    crawler, and closes both on exit; fetches borrow that crawler for their whole
    lifetime so crawl4ai keeps a single CDP connection. Initialization failures
    are logged and yield ``None`` so the caller can degrade gracefully; failures
    while the fetcher is in use propagate.
    """
    async with _browser_endpoint(config.browser_cdp_url) as cdp_url:
        async with _open_crawler(cdp_url) as crawler:
            yield BrowserFetch(crawler) if crawler is not None else None


@asynccontextmanager
async def _open_crawler(
    cdp_url: str | None,
) -> AsyncGenerator[AsyncWebCrawler | None]:
    """Yield one open crawler for ``cdp_url``, or ``None`` when it cannot start.

    The crawler is created and started once and closed on exit. Startup
    failures (construction or ``__aenter__``) are logged and yield ``None``;
    failures while the crawler is in use propagate.
    """
    if not cdp_url:
        yield None
        return

    stack = AsyncExitStack()
    try:
        crawler = await stack.enter_async_context(_new_crawler(cdp_url))
    except Exception as error:
        logger.warning("Browser crawler unavailable for %s: %r", cdp_url, error)
        await stack.aclose()
        yield None
        return
    async with stack:
        yield crawler


async def _probe_cdp(cdp_url: str) -> bool:
    """Return True when a crawl4ai browser session can start at ``cdp_url``."""
    try:
        async with _new_crawler(cdp_url):
            return True
    except Exception as error:
        logger.warning("CDP probe failed for %s: %r", cdp_url, error)
        return False


async def _wait_for_local(process: asyncio.subprocess.Process) -> bool:
    """Poll the local CDP port until it answers, the process exits, or timeout."""
    deadline = time.monotonic() + LOCAL_STARTUP_SECONDS
    while time.monotonic() < deadline and process.returncode is None:
        try:
            async with asyncio.timeout(_LOCAL_POLL_SECONDS):
                if await _probe_cdp(LOCAL_CDP_URL):
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
async def _browser_endpoint(cdp_url: str) -> AsyncGenerator[str | None]:
    """Yield a reachable CDP URL, starting a local Obscura if needed.

    Prefers a configured ``cdp_url`` (probed within ``REMOTE_CONNECT_SECONDS``).
    Otherwise spawns ``obscura serve`` from ``PATH`` and waits up to
    ``LOCAL_STARTUP_SECONDS`` for it to accept CDP. Yields ``None`` when no
    browser is available; any process started here is terminated on exit.
    """
    if cdp_url:
        try:
            async with asyncio.timeout(REMOTE_CONNECT_SECONDS):
                reachable = await _probe_cdp(cdp_url)
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
        ready = await _wait_for_local(process)
        yield LOCAL_CDP_URL if ready else None
    finally:
        await _stop(process)
