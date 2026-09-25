"""Fallback fetcher rendering pages in an external browser over CDP (crawl4ai).

crawl4ai is an optional extra (``uv sync --extra browser``); when it is not
installed ``CRAWL4AI_AVAILABLE`` is False and the caller skips this fetcher.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from mcps.research.tools.common import (
    ERROR_EMPTY_RESPONSE,
    ERROR_FETCHER_UNAVAILABLE,
    Fetch,
    format_source_output,
    http_status_error,
)

try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

    CRAWL4AI_AVAILABLE = True
except ImportError:  # optional extra not installed
    CRAWL4AI_AVAILABLE = False

__all__ = ["CRAWL4AI_AVAILABLE", "create_browser_fetch"]

logger = logging.getLogger(__file__)

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


def _to_fetch_result(url: str, crawl: Any, max_chars: int) -> str:
    """Map a crawl4ai ``CrawlResult`` to fetched content or an error string."""
    if crawl.status_code is not None and crawl.status_code >= 400:
        return http_status_error(crawl.status_code)
    if not crawl.success:
        logger.warning("Browser fetch failed for %s: %s", url, crawl.error_message)
        return ERROR_FETCHER_UNAVAILABLE
    content = crawl.markdown.raw_markdown.strip() if crawl.markdown else ""
    if not content:
        return ERROR_EMPTY_RESPONSE
    return format_source_output(url, content, max_chars)


def create_browser_fetch(
    cdp_url: str,
    *,
    max_chars: int = 15000,
    crawler_factory: CrawlerFactory | None = None,
) -> Fetch:
    """Create a fetch callable rendering pages in the browser at ``cdp_url``."""
    factory = crawler_factory or _default_crawler_factory(cdp_url)
    run_config = CrawlerRunConfig(verbose=False) if CRAWL4AI_AVAILABLE else None

    async def fetch(url: str) -> str:
        try:
            async with factory() as crawler:
                crawl = await crawler.arun(url, config=run_config)
        except Exception as error:
            logger.warning("Browser fetch unavailable for %s: %r", url, error)
            return ERROR_FETCHER_UNAVAILABLE
        return _to_fetch_result(url, crawl, max_chars)

    return fetch
