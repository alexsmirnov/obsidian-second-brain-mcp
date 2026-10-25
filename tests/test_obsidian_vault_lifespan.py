import asyncio
from contextlib import asynccontextmanager
from datetime import timedelta
from unittest.mock import patch

from mcps.config import ServerConfig
from mcps.rag.index_scheduler import IndexScheduler
from mcps.rag.interfaces import IVault, Link, TraversalResult
from mcps.tools.obsidian_vault import build_obsidian_lifespan


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

    async def get_backlinks(
        self, wikilink_names: list[str]
    ) -> dict[str, list[Link]]:
        return {name: [] for name in wikilink_names}

    async def traverse_relations(
        self,
        wikilink_name: str,
        depth: int = 1,
        relation_types: list[str] | None = None,
    ) -> TraversalResult:
        return TraversalResult(origin=wikilink_name)


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
            timedelta(milliseconds=5),
        )
        lifespan = build_obsidian_lifespan(config)

        async with lifespan(None) as ctx:
            await asyncio.wait_for(fake_vault.first_call_event.wait(), timeout=1)
            assert ctx["obsidian_vault"] is fake_vault
            scheduler = ctx["obsidian_index_scheduler"]
            assert isinstance(scheduler, IndexScheduler)

    calls_at_shutdown = fake_vault.update_index_calls
    await asyncio.sleep(0.05)
    assert fake_vault.update_index_calls == calls_at_shutdown
    assert calls_at_shutdown >= 1
