"""Shared fakes for research job tests."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

from mcps.research.agent import ProgressReporter, ResearchResponse

_RELEASE_TIMEOUT = 3.0


def fake_crawler_strategy(
    sessions: dict[str, Any] | None = None,
    connected: Callable[[], bool] = lambda: True,
    kill: Callable[[str], Awaitable[None]] | None = None,
) -> SimpleNamespace:
    """Stand-in for the crawl4ai internals the browser fetcher relies on.

    Mirrors ``crawler.crawler_strategy``: ``browser_manager`` holds the session
    store (``session_id -> (context, page, last_used)``), the session expiry and
    the Playwright browser; ``kill_session`` closes a session's tab.
    """
    store = {} if sessions is None else sessions

    async def pop_session(session_id: str) -> None:
        store.pop(session_id, None)

    return SimpleNamespace(
        browser_manager=SimpleNamespace(
            sessions=store,
            browser=SimpleNamespace(is_connected=connected),
            session_ttl=1800.0,
        ),
        kill_session=kill or pop_session,
    )


class FakeResearcher:
    """Reports ``before``, waits for ``release``, then reports ``after``."""

    def __init__(
        self,
        before: list[str] | None = None,
        after: list[str] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.release = asyncio.Event()
        self.before = before or []
        self.after = after or []
        self.error = error
        self.cancelled = False

    async def __call__(
        self, input: str, progress: ProgressReporter | None = None
    ) -> ResearchResponse:
        try:
            for message in self.before:
                if progress:
                    await progress(message, 0.0, None)
            # Safety net: a test that forgets to release must fail, not hang.
            await asyncio.wait_for(self.release.wait(), timeout=_RELEASE_TIMEOUT)
            for message in self.after:
                if progress:
                    await progress(message, 0.0, None)
            if self.error:
                raise self.error
            return {
                "answer": f"answer:{input}",
                "explanation": "e",
                "sources": ["s"],
            }
        except asyncio.CancelledError:
            self.cancelled = True
            raise


class FakeClock:
    """Settable UTC wall clock."""

    def __init__(self) -> None:
        self._now = datetime(2026, 1, 1, tzinfo=UTC)

    def now(self) -> datetime:
        return self._now

    def advance(self, delta: timedelta) -> None:
        self._now += delta
