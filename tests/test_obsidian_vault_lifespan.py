import asyncio
from contextlib import asynccontextmanager
from datetime import timedelta
from unittest.mock import patch

import pytest

from mcps.config import ServerConfig
from mcps.rag.interfaces import IVault
from mcps.tools.obsidian_vault import _periodic_update_index, build_obsidian_lifespan


class FakeVault(IVault):
    def __init__(self) -> None:
        self.update_index_calls = 0
        self.fail_next_n = 0
        self.first_call_event = asyncio.Event()
        self.third_call_event = asyncio.Event()

    async def initialize(self) -> None:
        pass

    async def update_index(self) -> None:
        self.update_index_calls += 1
        if self.update_index_calls == 1:
            self.first_call_event.set()
        if self.fail_next_n > 0:
            self.fail_next_n -= 1
            raise RuntimeError("update_index failed")
        if self.update_index_calls == 3:
            self.third_call_event.set()

    async def search(
        self,
        query: str,
        tags: list[str] | None = None,
        path: str | None = None,
    ) -> list:
        return []

    async def get_file(
        self,
        file_name: str,
        offset: int | None = None,
        limit: int | None = None,
    ) -> str:
        return ""

    async def list_files(self, directory: str) -> list[str]:
        return []


@pytest.fixture
def fast_sleep(monkeypatch):
    real_sleep = asyncio.sleep

    async def _fast_sleep(_delay: float) -> None:
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", _fast_sleep)


async def test_periodic_update_index_calls_update_index_at_interval(fast_sleep) -> None:
    vault = FakeVault()
    task = asyncio.create_task(
        _periodic_update_index(vault, timedelta(seconds=1))
    )

    await vault.third_call_event.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert vault.update_index_calls >= 3


async def test_periodic_update_index_continues_after_failure(fast_sleep) -> None:
    vault = FakeVault()
    vault.fail_next_n = 1
    task = asyncio.create_task(
        _periodic_update_index(vault, timedelta(seconds=1))
    )

    await vault.third_call_event.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert vault.update_index_calls >= 3


async def test_periodic_update_index_exits_on_cancel(fast_sleep) -> None:
    vault = FakeVault()
    task = asyncio.create_task(
        _periodic_update_index(vault, timedelta(seconds=1))
    )

    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


async def test_obsidian_lifespan_starts_and_cancels_periodic_task(
    tmp_path, monkeypatch
) -> None:
    config = ServerConfig(vault_dir=tmp_path)
    fake_vault = FakeVault()

    @asynccontextmanager
    async def fake_create_vault(_config, _http_client):
        yield fake_vault

    with patch("mcps.tools.obsidian_vault.create_vault", fake_create_vault):
        monkeypatch.setattr(
            "mcps.tools.obsidian_vault.UPDATE_INTERVAL",
            timedelta(seconds=0),
        )
        lifespan = build_obsidian_lifespan(config)

        async with lifespan(None) as ctx:
            await asyncio.wait_for(fake_vault.first_call_event.wait(), timeout=1)
            assert ctx["obsidian_vault"] is fake_vault

    assert fake_vault.update_index_calls >= 1
