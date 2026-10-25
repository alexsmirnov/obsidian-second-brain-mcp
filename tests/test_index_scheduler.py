import asyncio
from collections.abc import AsyncIterator
from contextlib import suppress
from datetime import timedelta
from typing import cast

import pytest

from mcps.rag.index_scheduler import IndexScheduler
from mcps.rag.interfaces import IVault

HOUR = timedelta(hours=1)
SECOND = timedelta(seconds=1)


class ControlledVault:
    """Vault double whose update_index calls are observable and gateable."""

    def __init__(self) -> None:
        self.calls = 0
        self.entered = asyncio.Event()
        self.gate: asyncio.Event | None = None
        self.fail_on: set[int] = set()
        self.cancelled = False

    async def update_index(self) -> None:
        self.calls += 1
        call_number = self.calls
        gate = self.gate
        self.entered.set()
        try:
            if gate is not None:
                await gate.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        if call_number in self.fail_on:
            raise RuntimeError(f"update_index call {call_number} failed")


async def _yield_to_loop(times: int = 5) -> None:
    for _ in range(times):
        await asyncio.sleep(0)


async def _wait_for_call(vault: ControlledVault, call_number: int) -> None:
    async with asyncio.timeout(1):
        while vault.calls < call_number:
            vault.entered.clear()
            await vault.entered.wait()


def _as_vault(vault: ControlledVault) -> IVault:
    return cast(IVault, vault)


@pytest.fixture
def vault() -> ControlledVault:
    return ControlledVault()


@pytest.fixture
async def start_scheduler() -> AsyncIterator:
    tasks: list[asyncio.Task[None]] = []

    def start(
        vault: ControlledVault, interval: timedelta = HOUR
    ) -> tuple[IndexScheduler, asyncio.Task[None]]:
        scheduler = IndexScheduler(_as_vault(vault), interval)
        task = asyncio.create_task(scheduler.run())
        tasks.append(task)
        return scheduler, task

    yield start

    for task in tasks:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


async def test_run_updates_index_once_at_startup(vault, start_scheduler) -> None:
    start_scheduler(vault)

    await _wait_for_call(vault, 1)
    await _yield_to_loop()

    assert vault.calls == 1


async def test_run_repeats_update_after_interval(vault, start_scheduler) -> None:
    start_scheduler(vault, timedelta(milliseconds=5))

    await _wait_for_call(vault, 3)


async def test_refresh_starts_update_when_idle_and_returns_true(
    vault, start_scheduler
) -> None:
    scheduler, _ = start_scheduler(vault)
    await _wait_for_call(vault, 1)
    await _yield_to_loop()

    fresh = await scheduler.refresh(SECOND)

    assert fresh is True
    assert vault.calls == 2


async def test_refresh_joins_running_update_without_starting_another(
    vault, start_scheduler
) -> None:
    gate = asyncio.Event()
    vault.gate = gate
    scheduler, _ = start_scheduler(vault)
    await _wait_for_call(vault, 1)
    waiter = asyncio.create_task(scheduler.refresh(SECOND))
    await _yield_to_loop()

    assert not waiter.done()
    gate.set()

    assert await waiter is True
    await _yield_to_loop()
    assert vault.calls == 1


async def test_concurrent_requests_share_one_update(vault, start_scheduler) -> None:
    scheduler, _ = start_scheduler(vault)
    await _wait_for_call(vault, 1)
    await _yield_to_loop()
    gate = asyncio.Event()
    vault.gate = gate
    waiters = [
        asyncio.create_task(scheduler.refresh(SECOND)),
        asyncio.create_task(scheduler.refresh(SECOND)),
    ]
    await _yield_to_loop()

    gate.set()
    results = await asyncio.gather(*waiters)

    assert results == [True, True]
    assert vault.calls == 2


async def test_refresh_times_out_without_cancelling_update(
    vault, start_scheduler
) -> None:
    scheduler, _ = start_scheduler(vault)
    await _wait_for_call(vault, 1)
    await _yield_to_loop()
    gate = asyncio.Event()
    vault.gate = gate

    fresh = await scheduler.refresh(timedelta(milliseconds=10))

    assert fresh is False
    assert vault.cancelled is False

    gate.set()
    assert await scheduler.refresh(SECOND) is True
    assert vault.calls == 2


async def test_failed_update_returns_false_and_loop_continues(
    vault, start_scheduler
) -> None:
    vault.fail_on = {2}
    scheduler, task = start_scheduler(vault)
    await _wait_for_call(vault, 1)
    await _yield_to_loop()

    failed_run = await scheduler.refresh(SECOND)
    next_run = await scheduler.refresh(SECOND)

    assert (failed_run, next_run) == (False, True)
    assert vault.calls == 3
    assert not task.done()


async def test_refresh_updates_index_when_scheduler_is_not_running(vault) -> None:
    scheduler = IndexScheduler(_as_vault(vault), HOUR)

    fresh = await scheduler.refresh(SECOND)

    assert fresh is True
    assert vault.calls == 1


async def test_run_cancel_stops_update_in_progress_and_releases_waiters(
    vault, start_scheduler
) -> None:
    vault.gate = asyncio.Event()
    scheduler, task = start_scheduler(vault)
    await _wait_for_call(vault, 1)
    waiter = asyncio.create_task(scheduler.refresh(SECOND))
    await _yield_to_loop()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert vault.cancelled is True
    async with asyncio.timeout(0.1):
        assert await waiter is False
