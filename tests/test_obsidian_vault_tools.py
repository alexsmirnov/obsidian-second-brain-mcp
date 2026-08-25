from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock

import pytest
from fastmcp import Context
from fastmcp.exceptions import ToolError, ValidationError

from mcps.config import ServerConfig
from mcps.rag.interfaces import (
    Chunk,
    Link,
    NoteLinks,
    TraversalResult,
)
from mcps.rag.vault import Vault
from mcps.server import create_server
from mcps.tools import obsidian_vault

OBSIDIAN_TOOL_NAMES = {
    "obsidian_list_files",
    "obsidian_read_note",
    "obsidian_search",
    "obsidian_traverse_relations",
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
        self.traversal_result = TraversalResult(origin="Doc")
        self.traverse_calls: list[tuple[str, int, list[str] | None]] = []

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

    async def traverse_relations(
        self,
        wikilink_name: str,
        depth: int = 1,
        relation_types: list[str] | None = None,
    ) -> TraversalResult:
        self.traverse_calls.append((wikilink_name, depth, relation_types))
        return self.traversal_result


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


GRAPH_LINKS = {
    "Alpha": [
        Link(type="requires", target="Beta"),
        Link(type="related", target="Gamma"),
    ],
    "Beta": [Link(type="refines", target="Delta")],
    "Gamma": [Link(type="requires", target="Delta")],
    "Zulu": [Link(type="requires", target="Alpha")],
    "Delta": [],
}


def make_graph_store() -> AsyncMock:
    """Build the vector-store boundary fake used by traversal contracts."""
    async def get_notes_with_links(names: list[str]) -> list[NoteLinks]:
        return [
            NoteLinks(
                note=name,
                title=f"{name} Note",
                description=f"About {name.lower()}",
                links=GRAPH_LINKS[name],
            )
            for name in names
            if name in GRAPH_LINKS
        ]

    async def get_notes_linking_to(
        targets: list[str], relation_types: list[str] | None = None
    ) -> list[NoteLinks]:
        return [
            NoteLinks(
                note=note,
                title=f"{note} Note",
                description=f"About {note.lower()}",
                links=[
                    link
                    for link in links
                    if link.target in targets
                    and (relation_types is None or link.type in relation_types)
                ],
            )
            for note, links in GRAPH_LINKS.items()
            if any(
                link.target in targets
                and (relation_types is None or link.type in relation_types)
                for link in links
            )
        ]

    store = AsyncMock()
    store.get_notes_with_links.side_effect = get_notes_with_links
    store.get_notes_linking_to.side_effect = get_notes_linking_to
    return store


def make_traversal_vault(store: AsyncMock | None = None) -> Vault:
    """Build an initialized vault with the graph-store boundary fake."""
    vault = object.__new__(Vault)
    vault._initialized = True
    vault.vector_store = store or make_graph_store()
    return vault


async def test_traverse_relations_depth_one_returns_both_directions() -> None:
    # Arrange
    vault = make_traversal_vault()

    # Act
    result = await vault.traverse_relations("Alpha", depth=1)

    # Assert
    node_details = {
        (node.note, node.direction, node.relation, node.via)
        for node in result.nodes
    }
    assert node_details == {
        ("Beta", "outgoing", "requires", "Alpha"),
        ("Gamma", "outgoing", "related", "Alpha"),
        ("Zulu", "incoming", "requires", "Alpha"),
    }
    assert {node.depth for node in result.nodes} == {1}
    assert result.truncated is False


async def test_traverse_relations_depth_two_reaches_second_hop() -> None:
    # Arrange
    vault = make_traversal_vault()

    # Act
    result = await vault.traverse_relations("Alpha", depth=2)

    # Assert
    delta_nodes = [node for node in result.nodes if node.note == "Delta"]
    assert len(delta_nodes) == 1
    assert delta_nodes[0].depth == 2
    assert (delta_nodes[0].via, delta_nodes[0].relation) in {
        ("Beta", "refines"),
        ("Gamma", "requires"),
    }


async def test_traverse_relations_keeps_notes_at_shallowest_depth() -> None:
    # Arrange
    vault = make_traversal_vault()

    # Act
    result = await vault.traverse_relations("Zulu", depth=3)

    # Assert
    alpha_nodes = [node for node in result.nodes if node.note == "Alpha"]
    assert len(alpha_nodes) == 1
    assert alpha_nodes[0].depth == 1
    assert "Zulu" not in {node.note for node in result.nodes}


async def test_traverse_relations_filters_requested_relation_types() -> None:
    # Arrange
    vault = make_traversal_vault()

    # Act
    result = await vault.traverse_relations(
        "Alpha", depth=1, relation_types=["requires"]
    )

    # Assert
    assert {node.note for node in result.nodes} == {"Beta", "Zulu"}


async def test_traverse_relations_or_combines_relation_types() -> None:
    # Arrange
    vault = make_traversal_vault()

    # Act
    result = await vault.traverse_relations(
        "Alpha", depth=1, relation_types=["related", "refines"]
    )

    # Assert
    assert {node.note for node in result.nodes} == {"Gamma"}


async def test_traverse_relations_excludes_dangling_targets() -> None:
    # Arrange
    original_links = GRAPH_LINKS["Alpha"]
    GRAPH_LINKS["Alpha"] = [
        *original_links,
        Link(type="requires", target="Ghost"),
    ]
    vault = make_traversal_vault()

    # Act
    try:
        result = await vault.traverse_relations("Alpha", depth=1)
    finally:
        GRAPH_LINKS["Alpha"] = original_links

    # Assert
    assert "Ghost" not in {node.note for node in result.nodes}


async def test_traverse_relations_uses_aggregated_links_for_frontier_note() -> None:
    # Arrange
    store = AsyncMock()

    async def get_notes_with_links(names: list[str]) -> list[NoteLinks]:
        notes = {
            "Alpha": NoteLinks(
                note="Alpha",
                links=[
                    Link(type="requires", target="Beta"),
                    Link(type="related", target="Gamma"),
                ],
            ),
            "Beta": NoteLinks(note="Beta", title="Beta Note"),
            "Gamma": NoteLinks(note="Gamma", title="Gamma Note"),
        }
        return [notes[name] for name in names if name in notes]

    store.get_notes_with_links.side_effect = get_notes_with_links
    store.get_notes_linking_to.return_value = []
    vault = make_traversal_vault(store)

    # Act
    result = await vault.traverse_relations("Alpha", depth=1)

    # Assert
    assert {node.note for node in result.nodes} == {"Beta", "Gamma"}
    assert {node.title for node in result.nodes} == {"Beta Note", "Gamma Note"}
    assert store.get_notes_with_links.await_count == 2


async def test_traverse_relations_caps_distinct_notes_at_limit() -> None:
    # Arrange
    node_count = 150
    hub_links = [
        Link(type="related", target=f"N{index}")
        for index in range(node_count)
    ]
    graph = {
        "Hub": hub_links,
        **{f"N{index}": [] for index in range(node_count)},
    }
    store = AsyncMock()

    async def get_notes_with_links(names: list[str]) -> list[NoteLinks]:
        return [
            NoteLinks(note=name, links=graph[name])
            for name in names
            if name in graph
        ]

    store.get_notes_with_links.side_effect = get_notes_with_links
    store.get_notes_linking_to.return_value = []
    vault = make_traversal_vault(store)

    # Act
    result = await vault.traverse_relations("Hub", depth=2)

    # Assert
    assert len(result.nodes) == 100
    assert result.truncated is True
    assert "100" in result.warning
    assert "depth" in result.warning.lower()


async def test_traverse_relations_batches_queries_per_level() -> None:
    # Arrange
    store = make_graph_store()
    vault = make_traversal_vault(store)

    # Act
    await vault.traverse_relations("Alpha", depth=2)

    # Assert
    assert store.get_notes_with_links.await_count == 3
    assert store.get_notes_linking_to.await_count == 2
    assert store.get_notes_with_links.await_args_list[0].args == (["Alpha"],)
    assert set(store.get_notes_linking_to.await_args_list[0].args[0]) == {"Alpha"}
    assert set(store.get_notes_with_links.await_args_list[1].args[0]) == {
        "Beta",
        "Gamma",
        "Zulu",
    }


async def test_traverse_relations_unknown_origin_raises() -> None:
    # Arrange
    vault = make_traversal_vault()

    # Act / Assert
    with pytest.raises(FileNotFoundError, match="Nonexistent"):
        await vault.traverse_relations("Nonexistent", depth=1)


async def test_obsidian_traverse_relations_tool_rejects_out_of_range_depth(
    tmp_path: Path,
) -> None:
    # Arrange
    server = create_server(ServerConfig(vault_dir=tmp_path))
    tools = await server.mcp.list_tools()
    tool = next(tool for tool in tools if tool.name == "obsidian_traverse_relations")

    # Act / Assert
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        await tool.run({"note": "Doc", "depth": 0})
    with pytest.raises(ValidationError, match="less than or equal to 3"):
        await tool.run({"note": "Doc", "depth": 4})


async def test_obsidian_traverse_relations_tool_delegates_to_vault() -> None:
    # Arrange
    fake_vault = FakeVault()
    fake_vault.traversal_result = TraversalResult(origin="Doc")
    ctx = cast(Context, FakeContext({"obsidian_vault": fake_vault}))

    # Act
    result = await obsidian_vault.traverse_relations(
        "Doc", ctx, depth=2, relation_types=["requires"]
    )

    # Assert
    assert result == TraversalResult(origin="Doc")
    assert fake_vault.traverse_calls == [("Doc", 2, ["requires"])]


async def test_obsidian_traverse_relations_tool_converts_unknown_note() -> None:
    # Arrange
    fake_vault = FakeVault()
    ctx = cast(Context, FakeContext({"obsidian_vault": fake_vault}))

    async def raise_not_found(
        wikilink_name: str,
        depth: int = 1,
        relation_types: list[str] | None = None,
    ) -> TraversalResult:
        raise FileNotFoundError(wikilink_name)

    fake_vault.traverse_relations = raise_not_found  # type: ignore[method-assign]

    # Act / Assert
    with pytest.raises(ToolError, match="Nonexistent"):
        await obsidian_vault.traverse_relations("Nonexistent", ctx)
