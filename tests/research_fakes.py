"""Shared fakes for research job tests."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from mcps.research.agent import ProgressReporter, ResearchResponse

_RELEASE_TIMEOUT = 3.0


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
