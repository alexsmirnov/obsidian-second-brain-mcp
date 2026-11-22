"""Opt-in test of the browser fetcher against a real CDP browser.

Skipped unless ``MCPS_LIVE_CDP_URL`` points at a reachable endpoint that accepts
the connection (Obscura only accepts loopback ``Host`` headers, so run it where
the endpoint is on loopback), for example::

    MCPS_LIVE_CDP_URL=ws://127.0.0.1:9222/devtools/browser \
        test.sh tests/test_browser_live.py

It needs network access to fetch the example page.
"""

from __future__ import annotations

import os

import pytest

from mcps.config import ServerConfig
from mcps.research.tools import browser as browser_module
from mcps.research.tools.browser import create_browser_fetch

LIVE_URL = os.environ.get("MCPS_LIVE_CDP_URL", "")

pytestmark = pytest.mark.skipif(
    not LIVE_URL, reason="set MCPS_LIVE_CDP_URL to run against a live CDP browser"
)


async def test_live_session_reuse_and_reset():
    config = ServerConfig(browser_cdp_url=LIVE_URL, browser_pool_size=1)

    async with create_browser_fetch(config) as fetch:
        assert fetch is not None
        first = await fetch("https://example.com")
        second = await fetch("https://example.com")
        crawler = fetch._connection._current.crawler
        sessions = browser_module._browser_manager(crawler).sessions

        assert first.ok
        assert second.ok
        # One tab served both fetches, and it went back to a blank page.
        assert len(sessions) == 1
        ((_, page, _),) = sessions.values()
        assert page.url == "about:blank"
