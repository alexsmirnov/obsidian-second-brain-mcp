"""Fallback fetcher rendering pages in an external browser over CDP (crawl4ai).

:func:`create_browser_fetch` opens one crawl4ai crawler on the configured CDP
endpoint and closes it on exit. Fetches render in the tabs of a fixed pool of
named sessions, and each tab goes back to ``about:blank`` after its fetch. The
browser itself runs outside the server (docker compose).
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Any, cast

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
    CrawlResult,
)
from lxml import html as lxml_html

from mcps.config import ServerConfig
from mcps.research.tools.common import (
    MIME_HTML,
    MIME_MARKDOWN,
    MIME_PLAIN,
    failure,
    textual_mime,
)
from mcps.research.tools.html_markdown import create_markdown_generator
from mcps.research.tools.models import FetchResult, FetchStatus

__all__ = [
    "BrowserFetch",
    "BrowserSettings",
    "SessionPool",
    "create_browser_fetch",
]

logger = logging.getLogger(__name__)

_BLANK_PAGE = "about:blank"


def _positive(value: float, fallback: float) -> float:
    return value if value > 0 else fallback


@dataclass(frozen=True)
class BrowserSettings:
    """Validated browser pool, reconnect and cleanup settings."""

    pool_size: int
    reconnect_attempts: int
    reconnect_delay_seconds: float
    connect_timeout_seconds: float
    fetch_retries: int
    cleanup_timeout_seconds: float

    @classmethod
    def from_config(cls, config: ServerConfig) -> BrowserSettings:
        """Clamp the configured values; non-positive timeouts use the default."""
        defaults = ServerConfig()
        return cls(
            pool_size=max(1, config.browser_pool_size),
            reconnect_attempts=max(1, config.browser_reconnect_attempts),
            reconnect_delay_seconds=max(0.0, config.browser_reconnect_delay_seconds),
            connect_timeout_seconds=_positive(
                config.browser_connect_timeout_seconds,
                defaults.browser_connect_timeout_seconds,
            ),
            fetch_retries=max(0, config.browser_fetch_retries),
            cleanup_timeout_seconds=_positive(
                config.browser_cleanup_timeout_seconds,
                defaults.browser_cleanup_timeout_seconds,
            ),
        )


class SessionPool:
    """Fixed set of named browser sessions; ``acquire`` waits for a free one."""

    def __init__(self, size: int) -> None:
        self._free: asyncio.Queue[str] = asyncio.Queue()
        for index in range(size):
            self._free.put_nowait(f"obscura_tab_{index}")

    async def acquire(self) -> str:
        return await self._free.get()

    def release(self, name: str) -> None:
        self._free.put_nowait(name)


def _new_crawler(cdp_url: str) -> AsyncWebCrawler:
    """Build a crawl4ai crawler bound to ``cdp_url``.

    One shared Playwright/CDP connection serves every session, and closing the
    crawler leaves the external browser's state alone. All tabs share the
    browser's default context, so they share cookies and storage.
    """
    browser_config = BrowserConfig(
        browser_mode="custom",
        cdp_url=cdp_url,
        cache_cdp_connection=True,
        create_isolated_context=False,
        cdp_cleanup_on_close=False,
        enable_stealth=False,
        verbose=False,
        text_mode=True,
        avoid_ads=True,
        memory_saving_mode=True,
    )
    return AsyncWebCrawler(config=browser_config)


# ---------------------------------------------------------------------------
# crawl4ai internals. They have no public API for session pages, expiry or the
# browser connection, so they are reached through these adapters only. Written
# against crawl4ai 0.9.5; a version without them fails startup (see
# ``_require_internals``) instead of silently skipping the tab cleanup.
# ---------------------------------------------------------------------------


def _browser_manager(crawler: Any) -> Any:
    strategy = getattr(crawler, "crawler_strategy", None)
    manager = getattr(strategy, "browser_manager", None)
    if manager is None:
        raise RuntimeError("incompatible crawl4ai: crawler has no browser manager")
    return manager


def _require_internals(crawler: Any) -> None:
    """Raise ``RuntimeError`` unless every crawl4ai internal we use exists."""
    manager = _browser_manager(crawler)
    if not isinstance(getattr(manager, "sessions", None), dict):
        raise RuntimeError("incompatible crawl4ai: no session store")
    if not hasattr(manager, "session_ttl"):
        raise RuntimeError("incompatible crawl4ai: no session expiry setting")
    if not callable(getattr(getattr(manager, "browser", None), "is_connected", None)):
        raise RuntimeError("incompatible crawl4ai: no browser connection check")
    if not callable(getattr(crawler.crawler_strategy, "kill_session", None)):
        raise RuntimeError("incompatible crawl4ai: no kill_session")
    try:
        from crawl4ai.browser_manager import _CDPConnectionCache  # noqa: F401
    except ImportError:
        raise RuntimeError("incompatible crawl4ai: no shared CDP connection") from None


def _disable_session_expiry(crawler: Any) -> None:
    """Stop crawl4ai from closing sessions after its idle timeout.

    The pool owns session lifetime; an expiry task could close a tab that a
    fetch is using.
    """
    _browser_manager(crawler).session_ttl = math.inf


async def _kill_session(crawler: Any, session_id: str) -> None:
    """Close the session's tab; its next use opens a fresh one."""
    await crawler.crawler_strategy.kill_session(session_id)


def _is_browser_connected(crawler: Any) -> bool:
    browser = _browser_manager(crawler).browser
    return browser is not None and bool(browser.is_connected())


# crawl4ai turns exceptions into failed results carrying ``str(error)``, so a
# lost connection shows up as text. Matched in lower case.
_CONNECTION_LOST_MARKERS = (
    "target page, context or browser has been closed",
    "browser has been closed",
    "no page for session",
    "target closed",
    "connection closed",
    "econnrefused",
)


def _is_connection_lost(crawler: Any, outcome: CrawlResult | Exception) -> bool:
    """Whether a render outcome says the browser connection is gone.

    An exception escaping ``arun`` is a startup or connection failure. A
    successful result is final, even for HTTP errors. A failed result counts
    only when the browser reports a disconnect or the message names one.
    """
    if isinstance(outcome, Exception):
        return True
    if outcome.success:
        return False
    if not _is_browser_connected(crawler):
        return True
    message = (outcome.error_message or "").lower()
    return any(marker in message for marker in _CONNECTION_LOST_MARKERS)


async def _drop_cached_connections() -> None:
    """Force-close crawl4ai's shared CDP connection so the next crawler reconnects.

    crawl4ai keeps one connection per CDP URL for all crawlers. A crawler whose
    close failed never released it, so the next crawler would reuse it.
    """
    from crawl4ai.browser_manager import _CDPConnectionCache

    await _CDPConnectionCache.close_all()


def _session_page(crawler: Any, session_id: str) -> Any | None:
    """Page of ``session_id``, or ``None`` when the session has no tab yet."""
    entry = _browser_manager(crawler).sessions.get(session_id)
    return entry[1] if entry else None


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
    page_url = crawl.redirected_url or url
    if content_type and declared is None:
        logger.warning("Browser fetch failed for %s: unsupported type", url)
        return failure(url, FetchStatus.UNSUPPORTED)
    if crawl.markdown :
        logger.debug("Markdown result, size %d",len(crawl.markdown))
        return _content_result(page_url, MIME_MARKDOWN, crawl.markdown.fit_markdown)
    if declared == MIME_HTML:
        logger.debug("Html result, size %d",len(rendered_html))
        return _content_result(page_url, MIME_HTML, rendered_html)

    unwrapped = _unwrap_sole_pre(crawl.html or rendered_html)
    if unwrapped is not None:
        return _content_result(page_url, declared or MIME_PLAIN, unwrapped)
    if declared is None:
        return _content_result(page_url, MIME_HTML, rendered_html)
    logger.warning("Browser fetch failed for %s: malformed text wrapper", url)
    return failure(url, FetchStatus.UNSUPPORTED)


@dataclass(eq=False)
class _Generation:
    """One crawler, or none after a failed recovery.

    Leases keep a reference to their generation, so the outcome of its recovery
    stays reachable for a fetch that reports its connection loss late.
    """

    crawler: AsyncWebCrawler | None
    recovery: asyncio.Task[bool] | None = None


@dataclass(frozen=True)
class CrawlerLease:
    """A crawler together with the generation it belongs to."""

    crawler: AsyncWebCrawler
    generation: _Generation


class CrawlerConnection:
    """Owns the crawler and replaces it after the connection is lost.

    At most one recovery runs at a time. All fetches that fail on the same
    generation share its outcome, and fetches that arrive during a recovery
    wait for it. Every teardown and start is time-bounded.
    """

    def __init__(
        self, factory: Callable[[], AsyncWebCrawler], settings: BrowserSettings
    ) -> None:
        self._factory = factory
        self._settings = settings
        self._current = _Generation(None)
        self._active_recovery: asyncio.Task[bool] | None = None

    async def open(self) -> None:
        """Start the first crawler; raises when it cannot connect."""
        self._current = _Generation(await self._start())

    async def acquire(self) -> CrawlerLease | None:
        """Lease the current crawler, recovering first if there is none."""
        recovery = self._active_recovery
        if recovery is not None and not recovery.done():
            if not await asyncio.shield(recovery):
                return None
        if self._current.crawler is None and not await self.recover(self._current):
            return None
        generation = self._current
        if generation.crawler is None:
            return None
        return CrawlerLease(generation.crawler, generation)

    async def recover(self, generation: _Generation) -> bool:
        """Replace ``generation``'s crawler; every caller shares the outcome."""
        if generation.recovery is None:
            if generation is not self._current:
                return self._current.crawler is not None
            generation.recovery = asyncio.create_task(self._recover_cycle(generation))
            self._active_recovery = generation.recovery
        return await asyncio.shield(generation.recovery)

    async def invalidate(self, generation: _Generation) -> None:
        """Drop the crawler so the next fetch reconnects.

        A generation that is recovering is left alone: its recovery tears the
        crawler down.
        """
        if generation is not self._current or generation.crawler is None:
            return
        if generation.recovery is not None:
            return
        self._current = _Generation(None)
        await self._teardown(generation.crawler)

    async def close(self) -> None:
        recovery = self._active_recovery
        if recovery is not None and not recovery.done():
            recovery.cancel()
            with suppress(asyncio.CancelledError):
                await recovery
        crawler, self._current = self._current.crawler, _Generation(None)
        if crawler is None:
            return
        try:
            async with asyncio.timeout(self._settings.cleanup_timeout_seconds):
                await crawler.__aexit__(None, None, None)
        except TimeoutError:
            logger.warning("Browser crawler close timed out")
            await self._drop_shared_connection()

    async def _recover_cycle(self, generation: _Generation) -> bool:
        if generation.crawler is not None:
            await self._teardown(generation.crawler)
        attempts = self._settings.reconnect_attempts
        for attempt in range(1, attempts + 1):
            await asyncio.sleep(self._settings.reconnect_delay_seconds)
            try:
                crawler = await self._start()
            except Exception as error:
                logger.warning(
                    "Browser reconnect %d/%d failed: %r", attempt, attempts, error
                )
                continue
            self._current = _Generation(crawler)
            return True
        self._current = _Generation(None)
        return False

    async def _start(self) -> AsyncWebCrawler:
        crawler = self._factory()
        try:
            async with asyncio.timeout(self._settings.connect_timeout_seconds):
                await crawler.__aenter__()
            _require_internals(crawler)
            _disable_session_expiry(crawler)
        except BaseException:
            await self._teardown(crawler)
            raise
        return crawler

    async def _teardown(self, crawler: AsyncWebCrawler) -> None:
        try:
            async with asyncio.timeout(self._settings.cleanup_timeout_seconds):
                await crawler.__aexit__(None, None, None)
        except Exception as error:
            logger.warning("Browser crawler teardown failed: %r", error)
            await self._drop_shared_connection()

    async def _drop_shared_connection(self) -> None:
        try:
            await _drop_cached_connections()
        except Exception as error:
            logger.warning("Dropping the shared browser connection failed: %r", error)


@dataclass
class _Attempt:
    """What the cleanup of a fetch needs to know about its last render."""

    lease: CrawlerLease | None = None
    lost: bool = False


class BrowserFetch:
    """Render pages in the tabs of a pool of named sessions.

    The crawler's lifetime is owned by :func:`create_browser_fetch`. A fetch
    borrows a session, renders in its tab, and puts the tab back to
    ``about:blank`` before the session returns to the pool. That cleanup runs
    in its own task, so cancelling the fetch cannot leave a dirty tab behind.
    crawl4ai mutates the ``CrawlerRunConfig`` it is given, so every render
    builds a fresh one.
    """

    def __init__(
        self,
        connection: CrawlerConnection,
        sessions: SessionPool,
        settings: BrowserSettings,
    ) -> None:
        self._connection = connection
        self._sessions = sessions
        self._settings = settings
        self._cleanups: set[asyncio.Task[None]] = set()

    async def __call__(self, url: str, query: str | None = None, /) -> FetchResult:
        session_id = await self._sessions.acquire()
        attempt = _Attempt()
        try:
            return await self._fetch_in_session(url, session_id, attempt)
        finally:
            cleanup = asyncio.create_task(self._release(attempt, session_id))
            self._cleanups.add(cleanup)
            cleanup.add_done_callback(self._cleanups.discard)
            await asyncio.shield(cleanup)

    async def aclose(self) -> None:
        """Wait (bounded) for pending tab cleanups, then cancel stragglers."""
        pending = {task for task in self._cleanups if not task.done()}
        if not pending:
            return
        timeout = 3 * self._settings.cleanup_timeout_seconds
        _, stragglers = await asyncio.wait(pending, timeout=timeout)
        for task in stragglers:
            task.cancel()
        if stragglers:
            await asyncio.gather(*stragglers, return_exceptions=True)

    async def _fetch_in_session(
        self, url: str, session_id: str, attempt: _Attempt
    ) -> FetchResult:
        """Render ``url``, recovering the connection and retrying when it is lost.

        Only a lost connection is retried. A recovery also runs after the last
        allowed retry, so the next fetch finds a crawler.
        """
        for _ in range(self._settings.fetch_retries + 1):
            attempt.lease = await self._connection.acquire()
            attempt.lost = False
            if attempt.lease is None:
                return failure(url, FetchStatus.UNAVAILABLE)
            outcome = await self._render(attempt.lease.crawler, url, session_id)
            attempt.lost = _is_connection_lost(attempt.lease.crawler, outcome)
            if not attempt.lost:
                return _to_fetch_result(url, cast(CrawlResult, outcome))
            logger.warning("Browser connection lost while fetching %s", url)
            if not await self._connection.recover(attempt.lease.generation):
                return failure(url, FetchStatus.UNAVAILABLE)
        return failure(url, FetchStatus.UNAVAILABLE)

    async def _render(
        self, crawler: AsyncWebCrawler, url: str, session_id: str
    ) -> CrawlResult | Exception:
        try:
            crawler_config = CrawlerRunConfig(
                session_id=session_id,
                cache_mode=CacheMode.BYPASS,
                wait_until="domcontentloaded",
                simulate_user=False,  # Add user simulation
                magic=False,  # Enable magic mode
                # Give cloudflare a moment to render the DOM elements, it waits 5 second
                delay_before_return_html=0.2,
                body_visibility_timeout=15000,
                scan_full_page=False,
                exclude_all_images=True,
                # Prevents recursively crawling deep iframe trees
                process_iframes=False,
                # Cuts down on event loops monitoring image states
                wait_for_images=False,
                markdown_generator=create_markdown_generator(),
                verbose=False,
            )
            return cast(CrawlResult, await crawler.arun(url, config=crawler_config))
        except Exception as error:
            logger.warning("Browser render failed for %s: %r", url, error)
            return error

    async def _release(self, attempt: _Attempt, session_id: str) -> None:
        try:
            if attempt.lease is not None and not attempt.lost:
                await self._reset_tab(attempt.lease, session_id)
        finally:
            self._sessions.release(session_id)

    async def _reset_tab(self, lease: CrawlerLease, session_id: str) -> None:
        """Blank the session's tab; close it, or drop the crawler, if that fails."""
        try:
            await self._blank_tab(lease, session_id)
            return
        except Exception as error:
            logger.warning("Browser tab reset failed for %s: %r", session_id, error)
        try:
            async with asyncio.timeout(self._settings.cleanup_timeout_seconds):
                await _kill_session(lease.crawler, session_id)
        except Exception as error:
            logger.warning("Browser session kill failed for %s: %r", session_id, error)
            await self._connection.invalidate(lease.generation)

    async def _blank_tab(self, lease: CrawlerLease, session_id: str) -> None:
        async with asyncio.timeout(self._settings.cleanup_timeout_seconds):
            page = _session_page(lease.crawler, session_id)
            if page is not None:
                await page.goto(_BLANK_PAGE, wait_until="commit")


async def _open_connection(connection: CrawlerConnection, cdp_url: str) -> bool:
    try:
        await connection.open()
    except Exception as error:
        logger.warning("Browser crawler unavailable for %s: %r", cdp_url, error)
        return False
    return True


@asynccontextmanager
async def create_browser_fetch(
    config: ServerConfig,
) -> AsyncGenerator[BrowserFetch | None]:
    """Yield a browser fetcher on one persistent crawler, or ``None``.

    Yields ``None`` when ``config.browser_cdp_url`` is unset, the crawler
    cannot connect to it, or crawl4ai lacks an internal the fetcher needs. On
    exit it waits for pending tab cleanups, then closes the crawler. An
    exception from the body propagates, and a failing close is only logged.
    """
    cdp_url = config.browser_cdp_url
    if not cdp_url:
        yield None
        return
    settings = BrowserSettings.from_config(config)
    connection = CrawlerConnection(lambda: _new_crawler(cdp_url), settings)
    if not await _open_connection(connection, cdp_url):
        yield None
        return
    fetch = BrowserFetch(connection, SessionPool(settings.pool_size), settings)
    try:
        yield fetch
    except BaseException:
        await fetch.aclose()
        try:
            await connection.close()
        except Exception as error:
            logger.warning("Browser crawler close failed: %r", error)
        raise
    await fetch.aclose()
    await connection.close()
