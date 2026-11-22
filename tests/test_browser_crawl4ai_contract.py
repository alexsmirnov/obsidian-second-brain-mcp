"""Contract tests for the crawl4ai internals the browser fetcher relies on.

These run against the real ``BrowserManager`` (no browser is started), so an
upgrade that changes the session store or the expiry behavior fails here
instead of silently closing tabs in production.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import Any

import pytest
from crawl4ai import BrowserConfig
from crawl4ai.browser_manager import BrowserManager, _CDPConnectionCache

from mcps.research.tools import browser as browser_module

STALE = 10**6  # seconds a session has been idle


class ClosablePage:
    """Page stand-in that records being closed."""

    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def manager() -> BrowserManager:
    manager = BrowserManager(
        browser_config=BrowserConfig(browser_mode="custom", cdp_url="ws://cdp")
    )
    # A started manager holds a Playwright browser; the fixture stands in for it.
    manager.browser = SimpleNamespace(is_connected=lambda: True)  # type: ignore[assignment]
    return manager


@pytest.fixture
def crawler(manager: BrowserManager) -> Any:
    return SimpleNamespace(
        crawler_strategy=SimpleNamespace(
            browser_manager=manager, kill_session=manager.kill_session
        )
    )


async def _run_expiry(manager: BrowserManager) -> None:
    """Run crawl4ai's expiry sweep and wait for the kills it schedules."""
    before = asyncio.all_tasks()
    manager._cleanup_expired_sessions()
    await asyncio.gather(*(asyncio.all_tasks() - before))


def _add_stale_session(manager: BrowserManager) -> ClosablePage:
    page = ClosablePage()
    manager.sessions["s"] = (None, page, time.time() - STALE)  # type: ignore[assignment]
    return page


async def test_drop_cached_connections_closes_the_shared_cdp_connection():
    closed: list[str] = []

    class Browser:
        async def close(self) -> None:
            closed.append("browser")

    class Playwright:
        async def stop(self) -> None:
            closed.append("playwright")

    cdp_url = "ws://contract-test"
    _CDPConnectionCache._cache[cdp_url] = (Playwright(), Browser(), 1)  # type: ignore[assignment]
    try:
        await browser_module._drop_cached_connections()
    finally:
        _CDPConnectionCache._cache.pop(cdp_url, None)

    assert closed == ["browser", "playwright"]
    assert cdp_url not in _CDPConnectionCache._cache


def test_real_manager_passes_internals_check(crawler: Any):
    browser_module._require_internals(crawler)


async def test_disabled_expiry_keeps_pooled_session(
    manager: BrowserManager, crawler: Any
):
    page = _add_stale_session(manager)

    browser_module._disable_session_expiry(crawler)
    await _run_expiry(manager)

    assert "s" in manager.sessions
    assert not page.closed


async def test_default_expiry_kills_stale_session(manager: BrowserManager):
    page = _add_stale_session(manager)

    await _run_expiry(manager)

    assert "s" not in manager.sessions
    assert page.closed


def test_session_page_reads_manager_layout(manager: BrowserManager, crawler: Any):
    page = ClosablePage()
    manager.sessions["s"] = (None, page, time.time())  # type: ignore[assignment]

    assert browser_module._session_page(crawler, "s") is page
    assert browser_module._session_page(crawler, "missing") is None
