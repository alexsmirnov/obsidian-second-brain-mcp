from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest
from fastmcp import Context
from fastmcp.exceptions import ToolError

from mcps.config import ServerConfig
from mcps.rag.interfaces import Chunk, Link
from mcps.server import create_server
from mcps.tools import obsidian_vault

OBSIDIAN_TOOL_NAMES = {
    "obsidian_list_files",
    "obsidian_read_note",
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
        self.backlinks: dict[str, list[Link]] = {}
        self.get_backlinks_calls: list[list[str]] = []

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
        lines = content.splitlines(keepends=True)
        start = offset or 0
        selected = lines[start:] if limit is None else lines[start:start + limit]
        return "".join(selected)

    async def search(
        self,
        query: str,
        tags: list[str] | None = None,
        path: str | None = None,
    ) -> list[Chunk]:
        return self.search_results

    async def get_backlinks(
        self, wikilink_names: list[str]
    ) -> dict[str, list[Link]]:
        self.get_backlinks_calls.append(wikilink_names)
        return {name: self.backlinks.get(name, []) for name in wikilink_names}


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
    fake_vault.files = {"Note": "line0\nline1\nline2\nline3\n"}
    ctx = cast(Context, FakeContext({"obsidian_vault": fake_vault}))

    result = await obsidian_vault.get_file_content("Note", ctx, offset=1, limit=2)

    assert result == "line1\nline2\n"
    assert fake_vault.get_file_calls == [("Note", 1, 2)]


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
            modified_at=1234.566,
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


def make_result_chunk(
    wikilink_name: str = "Doc",
    links: list[str] | None = None,
    link_types: list[str] | None = None,
) -> Chunk:
    """Build a minimal search-result chunk; links default to none."""
    return Chunk(
        id=f"{wikilink_name}_0",
        content="body",
        title="Doc",
        description="A doc",
        source_path=f"notes/{wikilink_name}.md",
        wikilink_name=wikilink_name,
        modified_at=1234.5,
        position=0,
        offset=0,
        file_size=4,
        tags=[],
        links=links or [],
        link_types=link_types or [],
    )


@pytest.mark.asyncio
async def test_search_exposes_typed_outgoing_links():
    # Arrange
    fake_vault = FakeVault()
    fake_vault.search_results = [
        make_result_chunk(
            links=["RAG Basics", "LanceDB"],
            link_types=["requires", "related"],
        )
    ]
    ctx = cast(Context, FakeContext({"obsidian_vault": fake_vault}))

    # Act
    result = await obsidian_vault.search("query", ctx)

    # Assert
    assert result[0].outgoing_links == [
        Link(type="requires", target="RAG Basics"),
        Link(type="related", target="LanceDB"),
    ]


@pytest.mark.asyncio
async def test_search_exposes_typed_backlinks():
    # Arrange — each backlink's target names the note on the other end of
    # the edge, i.e. the note that points at Doc
    fake_vault = FakeVault()
    fake_vault.search_results = [make_result_chunk()]
    fake_vault.backlinks = {"Doc": [Link(type="refines", target="RAG 2.0")]}
    ctx = cast(Context, FakeContext({"obsidian_vault": fake_vault}))

    # Act
    result = await obsidian_vault.search("query", ctx)

    # Assert
    assert result[0].backlinks == [Link(type="refines", target="RAG 2.0")]


@pytest.mark.asyncio
async def test_search_requests_backlinks_for_all_returned_notes_in_one_call():
    # Arrange
    fake_vault = FakeVault()
    fake_vault.search_results = [
        make_result_chunk(wikilink_name="A"),
        make_result_chunk(wikilink_name="B"),
        make_result_chunk(wikilink_name="A"),
    ]
    ctx = cast(Context, FakeContext({"obsidian_vault": fake_vault}))

    # Act
    await obsidian_vault.search("query", ctx)

    # Assert — one batched call, duplicates collapsed, first-appearance order
    assert fake_vault.get_backlinks_calls == [["A", "B"]]


@pytest.mark.asyncio
async def test_search_caps_backlinks_per_note():
    # Arrange
    fake_vault = FakeVault()
    fake_vault.search_results = [make_result_chunk()]
    fake_vault.backlinks = {
        "Doc": [Link(type="related", target=f"N{i}") for i in range(25)]
    }
    ctx = cast(Context, FakeContext({"obsidian_vault": fake_vault}))

    # Act
    result = await obsidian_vault.search("query", ctx)

    # Assert — capped at 20, the first 20 in the supplied order
    assert len(result[0].backlinks) == 20
    assert result[0].backlinks == [
        Link(type="related", target=f"N{i}") for i in range(20)
    ]


@pytest.mark.asyncio
async def test_search_returns_empty_backlinks_when_note_has_none():
    # Arrange
    fake_vault = FakeVault()
    fake_vault.search_results = [make_result_chunk()]
    ctx = cast(Context, FakeContext({"obsidian_vault": fake_vault}))

    # Act
    result = await obsidian_vault.search("query", ctx)

    # Assert
    assert result[0].backlinks == []


@pytest.mark.asyncio
async def test_search_truncation_warning_item_is_unaffected_by_backlinks():
    # Arrange — 12 chunks over 12 distinct notes
    fake_vault = FakeVault()
    fake_vault.search_results = [
        make_result_chunk(wikilink_name=f"N{i}") for i in range(12)
    ]
    ctx = cast(Context, FakeContext({"obsidian_vault": fake_vault}))

    # Act
    result = await obsidian_vault.search("query", ctx)

    # Assert — 10 full items plus the trailing warning item, and backlinks
    # requested only for the 10 notes actually returned
    assert len(result) == 11
    assert isinstance(result[-1], obsidian_vault.SearchResultItem)
    assert not isinstance(result[-1], obsidian_vault.SearchResultFullItem)
    assert fake_vault.get_backlinks_calls == [[f"N{i}" for i in range(10)]]
