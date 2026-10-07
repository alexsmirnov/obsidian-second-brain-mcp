"""Fallback fetcher rendering pages in an external browser over CDP (crawl4ai).

:func:`create_browser_fetch` opens one crawl4ai crawler on the configured CDP
endpoint and closes it on exit. The browser itself runs outside the server
(docker compose).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, cast

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CrawlResult
from lxml import html as lxml_html

from mcps.config import ServerConfig
from mcps.research.tools.common import (
    MIME_HTML,
    MIME_PLAIN,
    failure,
    textual_mime,
)
from mcps.research.tools.models import FetchResult, FetchStatus

__all__ = [
    "BrowserFetch",
    "create_browser_fetch",
]

logger = logging.getLogger(__name__)


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


def _unwrap_sole_pre(source: str) -> str | None:
    """Return sole ``<body><pre>`` text, or ``None`` for any other shape."""
    if not source:
        return None
    try:
        document = lxml_html.document_fromstring(source)
    except Exception:
        return None
    body = document.find("body")
    if body is None:
        body = document
    children = [child for child in body if isinstance(child.tag, str)]
    if len(children) != 1 or children[0].tag != "pre":
        return None
    pre = children[0]
    if (body.text or "").strip() or (pre.tail or "").strip():
        return None
    return pre.text_content()


def _content_result(url: str, mime: str, content: str) -> FetchResult:
    """Build an OK result, or EMPTY when ``content`` is blank."""
    if not content.strip():
        return failure(url, FetchStatus.EMPTY)
    return FetchResult(url=url, status=FetchStatus.OK, mime=mime, content=content)


def _to_fetch_result(url: str, crawl: CrawlResult) -> FetchResult:
    """Map a crawl4ai ``CrawlResult`` to typed source content or a failure."""
    if not crawl.success:
        logger.warning("Browser fetch failed for %s: %s", url, crawl.error_message)
        return failure(url, FetchStatus.UNAVAILABLE)
    if crawl.status_code is not None and crawl.status_code >= 400:
        return failure(url, FetchStatus.HTTP_ERROR, crawl.status_code)

    headers = {
        key.lower(): value for key, value in (crawl.response_headers or {}).items()
    }
    content_type = headers.get("content-type", "")
    rendered_html = crawl.cleaned_html or ""
    declared = textual_mime(content_type) if content_type else None
    if content_type and declared is None:
        logger.warning("Browser fetch failed for %s: unsupported type", url)
        return failure(url, FetchStatus.UNSUPPORTED)
    if declared == MIME_HTML:
        return _content_result(url, MIME_HTML, rendered_html)

    unwrapped = _unwrap_sole_pre(crawl.html or rendered_html)
    if unwrapped is not None:
        return _content_result(url, declared or MIME_PLAIN, unwrapped)
    if declared is None:
        return _content_result(url, MIME_HTML, rendered_html)
    logger.warning("Browser fetch failed for %s: malformed text wrapper", url)
    return failure(url, FetchStatus.UNSUPPORTED)


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
                wait_until="networkidle",
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

    Yields ``None`` when ``config.browser_cdp_url`` is unset or the crawler
    cannot connect to it. Otherwise the crawler is closed on exit; failures
    while the fetcher is in use propagate.
    """
    cdp_url = config.browser_cdp_url
    if not cdp_url:
        yield None
        return
    started = False
    try:
        async with _new_crawler(cdp_url) as crawler:
            started = True
            yield BrowserFetch(crawler)
    except Exception as error:
        if started:
            raise
        logger.warning("Browser crawler unavailable for %s: %r", cdp_url, error)
        yield None
