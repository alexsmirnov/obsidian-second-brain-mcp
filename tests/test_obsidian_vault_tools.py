from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest
from fastmcp import Context
from fastmcp.exceptions import ToolError

from mcps.config import ServerConfig
from mcps.rag.interfaces import Chunk
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
        self.files: dict[str, str] = {}
        self.get_file_calls: list[tuple[str, int | None, int | None]] = []
        self.search_results: list[Chunk] = []

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

    async def get_file(
        self,
        file_name: str,
        offset: int | None = None,
        limit: int | None = None,
    ) -> str:
        self.get_file_calls.append((file_name, offset, limit))
        if file_name not in self.files:
            raise FileNotFoundError(file_name)
        content = self.files[file_name]
        start = offset or 0
        if limit is None:
            return content[start:]
        return content[start:start + limit]

    async def search(
        self,
        query: str,
        tags: list[str] | None = None,
        path: str | None = None,
    ) -> list[Chunk]:
        return self.search_results


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
async def test_obsidian_list_files_uses_lifespan_vault_context():
    fake_vault = FakeVault()
    ctx = cast(Context, FakeContext({"obsidian_vault": fake_vault}))

    result = await obsidian_vault.list_files("/", ctx)

    assert result == "Contents of '/':\nA\nFolder/"


@pytest.mark.asyncio
async def test_get_file_content_validates_and_calls_vault_get_file():
    fake_vault = FakeVault()
    fake_vault.files = {"Note": "content"}
    ctx = cast(Context, FakeContext({"obsidian_vault": fake_vault}))

    result = await obsidian_vault.get_file_content("Note", ctx)

    assert result == "content"
    assert fake_vault.get_file_calls == [("Note", None, None)]


@pytest.mark.asyncio
async def test_get_file_content_reads_path_qualified_wikilink_directly():
    fake_vault = FakeVault()
    fake_vault.files = {"Folder/Note": "content"}
    ctx = cast(Context, FakeContext({"obsidian_vault": fake_vault}))

    result = await obsidian_vault.get_file_content("Folder/Note", ctx)

    assert result == "content"
    assert fake_vault.get_file_calls == [("Folder/Note", None, None)]


@pytest.mark.asyncio
async def test_get_file_content_converts_vault_value_error_to_tool_error():
    fake_vault = FakeVault()
    ctx = cast(Context, FakeContext({"obsidian_vault": fake_vault}))

    with pytest.raises(ToolError, match="Note"):
        await obsidian_vault.get_file_content("Note", ctx)


@pytest.mark.asyncio
async def test_get_file_content_converts_vault_not_found_to_tool_error():
    fake_vault = FakeVault()
    ctx = cast(Context, FakeContext({"obsidian_vault": fake_vault}))

    with pytest.raises(ToolError, match="Missing"):
        await obsidian_vault.get_file_content("Missing", ctx)


@pytest.mark.asyncio
async def test_get_file_content_rejects_relative_traversal_wikilink():
    fake_vault = FakeVault()
    ctx = cast(Context, FakeContext({"obsidian_vault": fake_vault}))

    with pytest.raises(ToolError, match="Invalid wikilink name"):
        await obsidian_vault.get_file_content("../../etc/passwd", ctx)


@pytest.mark.asyncio
async def test_get_file_content_applies_offset_and_limit():
    fake_vault = FakeVault()
    fake_vault.files = {"Note": "a🙂bcdef"}
    ctx = cast(Context, FakeContext({"obsidian_vault": fake_vault}))

    result = await obsidian_vault.get_file_content("Note", ctx, offset=2, limit=3)

    assert result == "bcd"
    assert fake_vault.get_file_calls == [("Note", 2, 3)]


@pytest.mark.asyncio
async def test_get_file_content_rejects_negative_offset():
    fake_vault = FakeVault()
    ctx = cast(Context, FakeContext({"obsidian_vault": fake_vault}))

    with pytest.raises(ToolError, match="Invalid offset"):
        await obsidian_vault.get_file_content("Note", ctx, offset=-1)


@pytest.mark.asyncio
async def test_get_file_content_rejects_negative_limit():
    fake_vault = FakeVault()
    ctx = cast(Context, FakeContext({"obsidian_vault": fake_vault}))

    with pytest.raises(ToolError, match="Invalid limit"):
        await obsidian_vault.get_file_content("Note", ctx, limit=-1)


@pytest.mark.asyncio
async def test_get_file_content_rejects_md_extension_for_wikilink_names():
    fake_vault = FakeVault()
    ctx = cast(Context, FakeContext({"obsidian_vault": fake_vault}))

    with pytest.raises(ToolError, match="Invalid wikilink name"):
        await obsidian_vault.get_file_content("Note.md", ctx)


@pytest.mark.asyncio
async def test_search_returns_wikilink_name_offset_and_size_in_results():
    fake_vault = FakeVault()
    fake_vault.search_results = [
        Chunk(
            id="chunk-1",
            content="content",
            title="Title",
            description="Description",
            source_path="Folder/Note.md",
            wikilink_name="Folder/Note",
            modified_at=datetime.now(UTC),
            position=0,
            offset=12,
            file_size=7,
        )
    ]
    ctx = cast(Context, FakeContext({"obsidian_vault": fake_vault}))

    results = await obsidian_vault.search("query", ctx)

    assert results[0].wikilink_name == "Folder/Note"
    assert results[0].offset == 12
    assert results[0].file_size == 7
