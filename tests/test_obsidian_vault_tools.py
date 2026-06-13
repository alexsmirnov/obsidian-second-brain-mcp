from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import cast

import pytest
from fastmcp import Context, FastMCP

from mcps.config import ServerConfig
from mcps.server import create_server
from mcps.tools import obsidian_vault

OBSIDIAN_TOOL_NAMES = {
    "obsidian_list_files",
    "obsidian_get_content",
    "obsidian_rename_move",
    "obsidian_search",
}


class FakeContext:
    def __init__(self, lifespan_context: dict[str, object]):
        self.lifespan_context = lifespan_context


class FakeVault:
    def __init__(self):
        self.entered = False
        self.exited = False
        self.index_started = False
        self.index_cancelled = False
        self._index_started = asyncio.Event()

    @asynccontextmanager
    async def lifespan(self):
        self.entered = True
        try:
            yield self
        finally:
            self.exited = True

    async def update_index(self) -> None:
        self.index_started = True
        self._index_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.index_cancelled = True
            raise

    async def list_files(self, directory: str) -> list[str]:
        assert directory == ""
        return ["A", "Folder/"]


@pytest.mark.asyncio
async def test_obsidian_tools_are_registered_when_vault_dir_is_configured(
    tmp_path: Path,
):
    config = ServerConfig(vault_dir=tmp_path)
    server = create_server(config)

    tools = await server.mcp.list_tools()
    tool_names = {tool.name for tool in tools}

    assert OBSIDIAN_TOOL_NAMES <= tool_names


@pytest.mark.asyncio
async def test_obsidian_tools_are_not_registered_without_vault_dir():
    config = ServerConfig(vault_dir=None)
    server = create_server(config)

    tools = await server.mcp.list_tools()
    tool_names = {tool.name for tool in tools}

    assert OBSIDIAN_TOOL_NAMES.isdisjoint(tool_names)


@pytest.mark.asyncio
async def test_obsidian_lifespan_enters_vault_lifespan_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    fake_vault = FakeVault()
    monkeypatch.setattr(obsidian_vault, "create_vault", lambda _config: fake_vault)
    config = ServerConfig(vault_dir=tmp_path)
    mcp = FastMCP("test")
    lifespan = obsidian_vault.build_obsidian_lifespan(config)

    async with lifespan(mcp) as context:
        await fake_vault._index_started.wait()
        assert context["obsidian_vault"] is fake_vault
        assert context["obsidian_config"] is config
        assert fake_vault.entered
        assert fake_vault.index_started

    assert fake_vault.exited
    assert fake_vault.index_cancelled


@pytest.mark.asyncio
async def test_obsidian_list_files_uses_lifespan_vault_context():
    fake_vault = FakeVault()
    ctx = cast(Context, FakeContext({"obsidian_vault": fake_vault}))

    result = await obsidian_vault.list_files("/", ctx)

    assert result == "Contents of '/':\nA\nFolder/"
