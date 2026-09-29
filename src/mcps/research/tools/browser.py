"""Fallback fetcher rendering pages in an external browser over CDP (crawl4ai).

crawl4ai is an optional extra (``uv sync --extra browser``); when it is not
installed ``CRAWL4AI_AVAILABLE`` is False and the caller skips this fetcher.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from mcps.research.tools.common import (
    ERROR_EMPTY_RESPONSE,
    ERROR_FETCHER_UNAVAILABLE,
    Retrieve,
    http_status_error,
)

try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

    CRAWL4AI_AVAILABLE = True
except ImportError:  # optional extra not installed
    CRAWL4AI_AVAILABLE = False

__all__ = [
    "CRAWL4AI_AVAILABLE",
    "LOCAL_CDP_URL",
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

CrawlerFactory = Callable[[], Any]


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


def create_browser_fetch(
    cdp_url: str,
    *,
    crawler_factory: CrawlerFactory | None = None,
) -> Retrieve:
    """Create a callable rendering pages to cleaned HTML in the CDP browser."""
    factory = crawler_factory or _default_crawler_factory(cdp_url)
    run_config = CrawlerRunConfig(verbose=False) if CRAWL4AI_AVAILABLE else None

    async def fetch(url: str) -> str:
        try:
            async with factory() as crawler:
                crawl = await crawler.arun(url, config=run_config)
        except Exception as error:
            logger.warning("Browser fetch unavailable for %s: %r", url, error)
            return ERROR_FETCHER_UNAVAILABLE
        return _to_fetch_result(url, crawl)

    return fetch


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
) -> AsyncIterator[str | None]:
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
