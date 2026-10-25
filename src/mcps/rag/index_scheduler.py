import asyncio
import logging
from contextlib import suppress
from datetime import timedelta

from mcps.rag.interfaces import IVault

logger = logging.getLogger("mcps")


class IndexScheduler:
    """Runs vault index updates periodically and on demand.

    At most one update runs at a time. A caller that arrives while an update is
    running joins it, otherwise it starts a new one. A caller that stops waiting
    never cancels the update in progress.
    """

    def __init__(self, vault: IVault, interval: timedelta) -> None:
        self._vault = vault
        self._interval = interval
        self._update: asyncio.Task[bool] | None = None

    async def run(self) -> None:
        """Update the index every interval until cancelled."""
        try:
            while True:
                await self.refresh(self._interval)
                await asyncio.sleep(self._interval.total_seconds())
        finally:
            await self._cancel_update()

    async def refresh(self, timeout: timedelta) -> bool:
        """Wait until the index update in progress, or a new one, has finished.

        Works whether or not run() is active. Returns True when the update
        succeeded, False when it failed, was cancelled or did not finish in time.
        """
        update = self._current_update()
        done, _ = await asyncio.wait({update}, timeout=timeout.total_seconds())
        return bool(done) and not update.cancelled() and update.result()

    def _current_update(self) -> asyncio.Task[bool]:
        if self._update is None or self._update.done():
            self._update = asyncio.create_task(self._update_index())
        return self._update

    async def _update_index(self) -> bool:
        try:
            await self._vault.update_index()
        except Exception:
            logger.exception("Vault index update failed")
            return False
        return True

    async def _cancel_update(self) -> None:
        if self._update is None:
            return
        self._update.cancel()
        with suppress(asyncio.CancelledError):
            await self._update
