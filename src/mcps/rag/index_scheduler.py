import asyncio
import logging
from datetime import timedelta

from mcps.rag.interfaces import IVault

logger = logging.getLogger("mcps")


class IndexScheduler:
    """Runs vault index updates periodically and on demand.

    A single loop owns every update, so concurrent requests share one run and
    a caller that stops waiting never cancels an update in progress.
    """

    def __init__(self, vault: IVault, interval: timedelta) -> None:
        self._vault = vault
        self._interval = interval
        self._started = 0
        self._completed = 0
        self._running = False
        self._requested = asyncio.Event()
        self._finished = asyncio.Condition()

    async def run(self) -> None:
        self._running = True
        try:
            while True:
                self._started += 1
                self._requested.clear()
                await self._update_index()
                async with self._finished:
                    self._completed = self._started
                    self._finished.notify_all()
                await self._wait_for_request_or_interval()
        finally:
            self._running = False

    async def wait_for_next_update(self, timeout: timedelta) -> bool:
        """True when an update that started after this call has finished
        (successfully or not)."""
        if not self._running:
            logger.warning("Index scheduler is not running, skipping index refresh")
            return False
        target = self._started + 1
        self._requested.set()
        try:
            async with asyncio.timeout(timeout.total_seconds()):
                async with self._finished:
                    await self._finished.wait_for(lambda: self._completed >= target)
        except TimeoutError:
            return False
        return True

    async def _update_index(self) -> None:
        try:
            await self._vault.update_index()
        except Exception:
            logger.exception("Vault index update failed")

    async def _wait_for_request_or_interval(self) -> None:
        try:
            async with asyncio.timeout(self._interval.total_seconds()):
                await self._requested.wait()
        except TimeoutError:
            pass
