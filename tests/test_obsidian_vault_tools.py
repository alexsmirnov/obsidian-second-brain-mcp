from __future__ import annotations

import asyncio
from collections.abc import Iterable
from contextlib import suppress
from datetime import timedelta
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock

import pytest
from fastmcp import Client, Context, FastMCP
from fastmcp.exceptions import ToolError, ValidationError

from mcps.config import ServerConfig
from mcps.rag.index_scheduler import IndexScheduler
from mcps.rag.interfaces import (
    Chunk,
    IVault,
    Link,
    NoteLinks,
    TraversalResult,
)
from mcps.rag.note_paths import select_matching
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


class FakeIndexScheduler:
    def __init__(self, events: list[str], fresh: bool = True):
        self.events = events
        self.fresh = fresh
        self.timeouts: list[timedelta] = []

    async def refresh(self, timeout: timedelta) -> bool:
        self.events.append("wait")
        self.timeouts.append(timeout)
        return self.fresh


class FakeVault:
    def __init__(self):
        self.entered = False
        self.exited = False
        self.events: list[str] = []
        self.files: dict[str, str] = {}
        self.get_file_calls: list[tuple[str, int | None, int | None]] = []
        self.search_results: list[Chunk] = []
        self.backlinks: dict[str, list[Link]] = {}
        self.get_backlinks_calls: list[list[str]] = []
        self.note_paths: list[str] = []
        self.search_calls: list[tuple[str, list[str] | None, str | None]] = []
        self.traversal_result = TraversalResult(origin="Doc")
        self.traverse_calls: list[tuple[str, int, list[str] | None]] = []

    async def update_index(self) -> None:
        pass

    async def list_files(self, directory: str) -> list[str]:
        assert directory == ""
        return ["A", "Folder/"]

    async def resolve_references(
        self, references: Iterable[str]
    ) -> dict[str, list[str]]:
        return {
            reference: select_matching(reference, self.note_paths)
            for reference in references
        }

    async def get_file(
        self,
        file_name: str,
        offset: int | None = None,
        limit: int | None = None,
    ) -> str:
        self.events.append("get_file")
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
        self.events.append("search")
        self.search_calls.append((query, tags, path))
        return self.search_results

    async def get_backlinks(
        self, note_paths: list[str]
    ) -> dict[str, list[Link]]:
        self.get_backlinks_calls.append(note_paths)
        return {path: self.backlinks.get(path, []) for path in note_paths}

    async def traverse_relations(
        self,
        wikilink_name: str,
        depth: int = 1,
        relation_types: list[str] | None = None,
    ) -> TraversalResult:
        self.events.append("traverse")
        self.traverse_calls.append((wikilink_name, depth, relation_types))
        return self.traversal_result


class StaleUntilIndexedVault(FakeVault):
    """Notes in `pending` become readable only after update_index runs."""

    def __init__(self):
        super().__init__()
        self.pending: dict[str, str] = {}

    async def update_index(self) -> None:
        self.files.update(self.pending)
        self.pending.clear()


def make_context(
    vault: FakeVault, scheduler: object | None = None
) -> Context:
    return cast(Context, FakeContext({
        "obsidian_vault": vault,
        "obsidian_index_scheduler": scheduler or FakeIndexScheduler(vault.events),
    }))


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
    ctx = make_context(fake_vault)

    result = await obsidian_vault.list_files("/", ctx)

    assert result == "Contents of '/':\nA\nFolder/"


@pytest.mark.asyncio
async def test_get_file_content_validates_and_calls_vault_get_file():
    fake_vault = FakeVault()
    fake_vault.files = {"Note": "content"}
    ctx = make_context(fake_vault)

    result = await obsidian_vault.get_file_content("Note", ctx)

    assert result == "content"
    assert fake_vault.get_file_calls == [("Note", None, None)]


@pytest.mark.asyncio
@pytest.mark.parametrize("reference", ["/Folder/Note", "Folder/Note", "Note"])
async def test_get_file_content_forwards_every_reference_form_to_vault(reference):
    fake_vault = FakeVault()
    fake_vault.files = {reference: "content"}
    ctx = make_context(fake_vault)

    result = await obsidian_vault.get_file_content(reference, ctx)

    assert result == "content"
    assert fake_vault.get_file_calls == [(reference, None, None)]


@pytest.mark.asyncio
async def test_get_file_content_converts_vault_value_error_to_tool_error():
    fake_vault = FakeVault()
    ctx = make_context(fake_vault)

    with pytest.raises(ToolError, match="Note"):
        await obsidian_vault.get_file_content("Note", ctx)


@pytest.mark.asyncio
async def test_get_file_content_converts_vault_not_found_to_tool_error():
    fake_vault = FakeVault()
    ctx = make_context(fake_vault)

    with pytest.raises(ToolError, match="Missing"):
        await obsidian_vault.get_file_content("Missing", ctx)


@pytest.mark.asyncio
async def test_get_file_content_rejects_relative_traversal_wikilink():
    fake_vault = FakeVault()
    ctx = make_context(fake_vault)

    with pytest.raises(ToolError, match="Invalid note reference"):
        await obsidian_vault.get_file_content("../../etc/passwd", ctx)


@pytest.mark.asyncio
async def test_get_file_content_applies_offset_and_limit():
    fake_vault = FakeVault()
    fake_vault.files = {"Note": "line0\nline1\nline2\nline3\n"}
    ctx = make_context(fake_vault)

    result = await obsidian_vault.get_file_content("Note", ctx, offset=1, limit=2)

    assert result == "line1\nline2\n"
    assert fake_vault.get_file_calls == [("Note", 1, 2)]


@pytest.mark.asyncio
async def test_get_file_content_rejects_negative_offset():
    fake_vault = FakeVault()
    ctx = make_context(fake_vault)

    with pytest.raises(ToolError, match="Invalid offset"):
        await obsidian_vault.get_file_content("Note", ctx, offset=-1)


@pytest.mark.asyncio
async def test_get_file_content_rejects_negative_limit():
    fake_vault = FakeVault()
    ctx = make_context(fake_vault)

    with pytest.raises(ToolError, match="Invalid limit"):
        await obsidian_vault.get_file_content("Note", ctx, limit=-1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reference", ["/", "A//B", "A/B/", "/A/B/", "/../x", "A\\B", " "]
)
async def test_get_file_content_rejects_malformed_references(reference):
    fake_vault = FakeVault()
    ctx = make_context(fake_vault)

    with pytest.raises(ToolError, match="Invalid note reference"):
        await obsidian_vault.get_file_content(reference, ctx)

    assert fake_vault.get_file_calls == []


@pytest.mark.asyncio
async def test_get_file_content_rejects_md_extension_for_wikilink_names():
    fake_vault = FakeVault()
    ctx = make_context(fake_vault)

    with pytest.raises(ToolError, match="Invalid note reference"):
        await obsidian_vault.get_file_content("Note.md", ctx)


@pytest.mark.asyncio
async def test_search_returns_note_path_offset_and_size_in_results():
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
    ctx = make_context(fake_vault)

    response = await obsidian_vault.search("query", ctx)
    results = response.results

    assert results[0].note_path == "/Folder/Note"
    assert results[0].offset == 12
    assert results[0].file_size == 7
    assert "wikilink_name" not in obsidian_vault.SearchResultItem.model_fields
    assert "source_path" not in obsidian_vault.SearchResultItem.model_fields


def make_result_chunk(
    wikilink_name: str = "Doc",
    *,
    content: str = "body",
    title: str | None = "Doc",
    description: str | None = "A doc",
    source_path: str | None = None,
    position: int = 0,
    offset: int = 0,
    file_size: int = 4,
    tags: list[str] | None = None,
    links: list[str] | None = None,
    link_types: list[str] | None = None,
) -> Chunk:
    """Build a minimal search-result chunk; links default to none."""
    return Chunk(
        id=f"{wikilink_name}_{position}",
        content=content,
        title=title,
        description=description,
        source_path=source_path or f"notes/{wikilink_name}.md",
        wikilink_name=wikilink_name,
        modified_at=1234.5,
        position=position,
        offset=offset,
        file_size=file_size,
        tags=tags or [],
        links=links or [],
        link_types=link_types or [],
    )


def result_text_size(result: obsidian_vault.SearchResultItem) -> int:
    values = [
        result.content,
        result.title or "",
        result.description or "",
        *result.tags,
        *(
            value
            for link in result.outgoing_links
            for value in (link.type, link.target)
        ),
        *(value for link in result.backlinks for value in (link.type, link.target)),
        result.note_path,
    ]
    return sum(map(len, values))


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
    ctx = make_context(fake_vault)

    # Act
    response = await obsidian_vault.search("query", ctx)
    result = response.results

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
    fake_vault.backlinks = {"/notes/Doc": [Link(type="refines", target="/RAG 2.0")]}
    ctx = make_context(fake_vault)

    # Act
    response = await obsidian_vault.search("query", ctx)
    result = response.results

    # Assert
    assert result[0].backlinks == [Link(type="refines", target="/RAG 2.0")]


@pytest.mark.asyncio
async def test_search_requests_backlinks_for_all_returned_notes_in_one_call():
    # Arrange
    fake_vault = FakeVault()
    fake_vault.search_results = [
        make_result_chunk(wikilink_name="A"),
        make_result_chunk(wikilink_name="B"),
        make_result_chunk(wikilink_name="A"),
    ]
    ctx = make_context(fake_vault)

    # Act
    await obsidian_vault.search("query", ctx)

    # Assert — one batched call of note paths, duplicates collapsed,
    # first-appearance order
    assert fake_vault.get_backlinks_calls == [["/notes/A", "/notes/B"]]


@pytest.mark.asyncio
async def test_search_caps_backlinks_per_note():
    # Arrange
    fake_vault = FakeVault()
    fake_vault.search_results = [make_result_chunk()]
    fake_vault.backlinks = {
        "/notes/Doc": [Link(type="related", target=f"/N{i}") for i in range(25)]
    }
    ctx = make_context(fake_vault)

    # Act
    response = await obsidian_vault.search("query", ctx)
    result = response.results

    # Assert — capped at 20, the first 20 in the supplied order
    assert len(result[0].backlinks) == 20
    assert result[0].backlinks == [
        Link(type="related", target=f"/N{i}") for i in range(20)
    ]


@pytest.mark.asyncio
async def test_search_returns_empty_backlinks_when_note_has_none():
    # Arrange
    fake_vault = FakeVault()
    fake_vault.search_results = [make_result_chunk()]
    ctx = make_context(fake_vault)

    # Act
    response = await obsidian_vault.search("query", ctx)
    result = response.results

    # Assert
    assert result[0].backlinks == []


@pytest.mark.asyncio
async def test_search_outgoing_links_are_resolved_to_one_note_path_each():
    # Arrange — "Dup" is shared by two notes; "img.png" is not a note
    fake_vault = FakeVault()
    fake_vault.note_paths = ["/P/Dup", "/Q/Dup", "/Solo"]
    fake_vault.search_results = [
        make_result_chunk(
            links=["Dup", "img.png", "Solo"],
            link_types=["uses", "related", "requires"],
        )
    ]
    ctx = make_context(fake_vault)

    # Act
    response = await obsidian_vault.search("query", ctx)
    result = response.results

    # Assert
    assert result[0].outgoing_links == [
        Link(type="uses", target="/P/Dup"),
        Link(type="related", target="img.png"),
        Link(type="requires", target="/Solo"),
    ]


@pytest.mark.asyncio
async def test_search_dedupes_outgoing_links_after_resolution():
    # Arrange — [[Solo]] and [[/Solo]] name the same note
    fake_vault = FakeVault()
    fake_vault.note_paths = ["/Solo"]
    fake_vault.search_results = [
        make_result_chunk(
            links=["Solo", "/Solo"],
            link_types=["related", "uses"],
        )
    ]
    ctx = make_context(fake_vault)

    # Act
    response = await obsidian_vault.search("query", ctx)
    result = response.results

    # Assert
    assert result[0].outgoing_links == [Link(type="uses", target="/Solo")]


@pytest.mark.asyncio
async def test_search_keeps_same_named_notes_in_different_folders_apart():
    # Arrange
    fake_vault = FakeVault()
    fake_vault.search_results = [
        make_result_chunk("Dup", source_path="P/Dup.md", content="p"),
        make_result_chunk("Dup", source_path="Q/Dup.md", content="q"),
    ]
    ctx = make_context(fake_vault)

    # Act
    response = await obsidian_vault.search("query", ctx)
    result = response.results

    # Assert
    assert [(item.note_path, item.content) for item in result] == [
        ("/P/Dup", "p"),
        ("/Q/Dup", "q"),
    ]
    assert fake_vault.get_backlinks_calls == [["/P/Dup", "/Q/Dup"]]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("given", "expected"),
    [("/Projects/", "Projects/"), ("Projects/", "Projects/"), (None, None)],
)
async def test_search_strips_leading_slash_from_path_filter(given, expected):
    # Arrange
    fake_vault = FakeVault()
    ctx = make_context(fake_vault)

    # Act
    await obsidian_vault.search("query", ctx, path=given)

    # Assert
    assert fake_vault.search_calls == [("query", None, expected)]


@pytest.mark.asyncio
async def test_search_combines_isolated_sections_into_one_document_result():
    # Arrange
    fake_vault = FakeVault()
    fake_vault.search_results = [
        make_result_chunk(
            "A",
            content="later",
            title="Later title",
            description="Later description",
            position=2,
            offset=10,
            tags=["later", "shared"],
            links=["Target", "Other"],
            link_types=["related", "uses"],
        ),
        make_result_chunk("B"),
        make_result_chunk(
            "A",
            content="first\nline",
            title="First title",
            description="First description",
            position=0,
            offset=2,
            file_size=100,
            tags=["first", "shared"],
            links=["Target"],
            link_types=["requires"],
        ),
    ]
    fake_vault.backlinks = {"/notes/A": [Link(type="related", target="/Source")]}
    ctx = make_context(fake_vault)

    # Act
    response = await obsidian_vault.search("query", ctx)
    result = response.results

    # Assert
    assert [item.note_path for item in result] == ["/notes/A", "/notes/B"]
    assert result[0].content == "first\nline\n\n<<Gap: 6 lines>>\n\nlater"
    assert result[0].title == "First title"
    assert result[0].description == "First description"
    assert result[0].offset == 2
    assert result[0].file_size == 100
    assert result[0].tags == ["first", "later", "shared"]
    assert result[0].outgoing_links == [
        Link(type="uses", target="Other"),
        Link(type="requires", target="Target"),
    ]
    assert result[0].backlinks == [Link(type="related", target="/Source")]
    assert fake_vault.get_backlinks_calls == [["/notes/A", "/notes/B"]]


@pytest.mark.asyncio
async def test_search_joins_sections_without_gap_when_ranges_touch():
    # Arrange
    fake_vault = FakeVault()
    fake_vault.search_results = [
        make_result_chunk("Doc", content="one\ntwo", offset=0),
        make_result_chunk("Doc", content="three", position=1, offset=2),
    ]
    ctx = make_context(fake_vault)

    # Act
    response = await obsidian_vault.search("query", ctx)
    result = response.results

    # Assert
    assert len(result) == 1
    assert result[0].content == "one\ntwo\n\nthree"
    assert "<<Gap:" not in result[0].content


@pytest.mark.asyncio
async def test_search_returns_all_documents_when_response_fits_budget():
    # Arrange
    fake_vault = FakeVault()
    fake_vault.search_results = [
        make_result_chunk(wikilink_name=f"N{i}") for i in range(11)
    ]
    ctx = make_context(fake_vault)

    # Act
    response = await obsidian_vault.search("query", ctx)
    result = response.results

    # Assert
    assert len(result) == 11
    assert all(isinstance(item, obsidian_vault.SearchResultItem) for item in result)
    assert [item.note_path for item in result] == [
        f"/notes/N{i}" for i in range(11)
    ]
    assert fake_vault.get_backlinks_calls == [
        [f"/notes/N{i}" for i in range(11)]
    ]


@pytest.mark.asyncio
async def test_search_drops_largest_content_until_response_fits_budget():
    # Arrange
    fake_vault = FakeVault()
    fake_vault.search_results = [
        make_result_chunk("Largest", content="a" * 13_000),
        make_result_chunk("Smaller", content="b" * 12_000),
    ]
    ctx = make_context(fake_vault)

    # Act
    response = await obsidian_vault.search("query", ctx)
    result = response.results

    # Assert
    assert len(result) == 2
    assert all(isinstance(item, obsidian_vault.SearchResultItem) for item in result)
    full_results = cast(list[obsidian_vault.SearchResultItem], result)
    assert result[0].content == "<<Dropped: content is too big>>"
    assert result[1].content == "b" * 12_000
    assert sum(result_text_size(item) for item in full_results) <= 25_000


@pytest.mark.asyncio
async def test_search_drops_lower_rank_content_first_when_sizes_are_equal():
    # Arrange
    fake_vault = FakeVault()
    fake_vault.search_results = [
        make_result_chunk("Higher", content="a" * 12_500),
        make_result_chunk("Lower", content="b" * 12_500),
    ]
    ctx = make_context(fake_vault)

    # Act
    response = await obsidian_vault.search("query", ctx)
    result = response.results

    # Assert
    assert result[0].content == "a" * 12_500
    assert result[1].content == "<<Dropped: content is too big>>"


@pytest.mark.asyncio
async def test_search_keeps_short_content_when_marker_would_not_reduce_size():
    # Arrange
    fake_vault = FakeVault()
    fake_vault.search_results = [
        make_result_chunk("Higher", content="short", title="x" * 24_900),
        make_result_chunk("Lower", content="tiny", title="y" * 100),
    ]
    ctx = make_context(fake_vault)

    # Act
    response = await obsidian_vault.search("query", ctx)
    result = response.results

    # Assert
    assert len(result) == 1
    assert result[0].content == "short"
    assert response.truncated is True
    assert response.warning is not None


@pytest.mark.asyncio
async def test_search_removes_lowest_rank_documents_when_metadata_exceeds_budget():
    # Arrange
    fake_vault = FakeVault()
    fake_vault.search_results = [
        make_result_chunk(f"N{i}", content="body", title=str(i) * 9_000)
        for i in range(3)
    ]
    ctx = make_context(fake_vault)

    # Act
    response = await obsidian_vault.search("query", ctx)
    result = response.results

    # Assert
    assert [item.note_path for item in result] == ["/notes/N0", "/notes/N1"]
    assert sum(result_text_size(item) for item in result) <= 25_000
    assert response.total_documents == 3
    assert response.truncated is True
    assert response.warning == (
        "Search result truncated to 2 from 3 documents. "
        "Use more concrete query."
    )


@pytest.mark.asyncio
async def test_search_response_is_not_truncated_when_everything_fits():
    fake_vault = FakeVault()
    fake_vault.search_results = [make_result_chunk("N0")]

    response = await obsidian_vault.search("query", make_context(fake_vault))

    assert response.total_documents == 1
    assert response.truncated is False
    assert response.warning is None


GRAPH_LINKS = {
    "g/Alpha.md": [
        Link(type="requires", target="Beta"),
        Link(type="related", target="Gamma"),
    ],
    "g/Beta.md": [Link(type="refines", target="Delta")],
    "g/Gamma.md": [Link(type="requires", target="Delta")],
    "g/Zulu.md": [Link(type="requires", target="Alpha")],
    "g/Delta.md": [],
}

DUP_LINKS = {
    "Hub/Alpha.md": [Link(type="uses", target="Dup")],
    "P/Dup.md": [],
    "Q/Dup.md": [],
    "Z/Zulu.md": [Link(type="details", target="P/Dup")],
}


def make_graph_store(graph: dict[str, list[Link]] | None = None) -> AsyncMock:
    """Build the vector-store boundary fake used by traversal contracts.

    The graph maps note source paths to their raw outgoing links.
    """
    graph = GRAPH_LINKS if graph is None else graph

    def note_title(path: str) -> str:
        return f"{Path(path).stem} Note"

    async def get_sources_by_name(name: str) -> list[str]:
        return sorted(path for path in graph if Path(path).stem == name)

    async def get_notes_with_links(paths: list[str]) -> list[NoteLinks]:
        return [
            NoteLinks(
                note=path,
                title=note_title(path),
                description=f"About {Path(path).stem.lower()}",
                links=graph[path],
            )
            for path in paths
            if path in graph
        ]

    async def get_notes_linking_to(
        targets: list[str], relation_types: list[str] | None = None
    ) -> list[NoteLinks]:
        return [
            NoteLinks(
                note=path,
                title=note_title(path),
                description=f"About {Path(path).stem.lower()}",
                links=[
                    link
                    for link in links
                    if link.target in targets
                    and (relation_types is None or link.type in relation_types)
                ],
            )
            for path, links in graph.items()
            if any(
                link.target in targets
                and (relation_types is None or link.type in relation_types)
                for link in links
            )
        ]

    store = AsyncMock()
    store.get_sources_by_name.side_effect = get_sources_by_name
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
    result = await vault.traverse_relations("/g/Alpha", depth=1)

    # Assert
    assert result.origin == "/g/Alpha"
    node_details = {
        (node.note, node.direction, node.relation, node.via)
        for node in result.nodes
    }
    assert node_details == {
        ("/g/Beta", "outgoing", "requires", "/g/Alpha"),
        ("/g/Gamma", "outgoing", "related", "/g/Alpha"),
        ("/g/Zulu", "incoming", "requires", "/g/Alpha"),
    }
    assert {node.depth for node in result.nodes} == {1}
    assert result.truncated is False


async def test_traverse_relations_depth_two_reaches_second_hop() -> None:
    # Arrange
    vault = make_traversal_vault()

    # Act
    result = await vault.traverse_relations("/g/Alpha", depth=2)

    # Assert
    delta_nodes = [node for node in result.nodes if node.note == "/g/Delta"]
    assert len(delta_nodes) == 1
    assert delta_nodes[0].depth == 2
    assert (delta_nodes[0].via, delta_nodes[0].relation) in {
        ("/g/Beta", "refines"),
        ("/g/Gamma", "requires"),
    }


async def test_traverse_relations_keeps_notes_at_shallowest_depth() -> None:
    # Arrange
    vault = make_traversal_vault()

    # Act
    result = await vault.traverse_relations("/g/Zulu", depth=3)

    # Assert
    alpha_nodes = [node for node in result.nodes if node.note == "/g/Alpha"]
    assert len(alpha_nodes) == 1
    assert alpha_nodes[0].depth == 1
    assert "/g/Zulu" not in {node.note for node in result.nodes}


async def test_traverse_relations_filters_requested_relation_types() -> None:
    # Arrange
    vault = make_traversal_vault()

    # Act
    result = await vault.traverse_relations(
        "/g/Alpha", depth=1, relation_types=["requires"]
    )

    # Assert
    assert {node.note for node in result.nodes} == {"/g/Beta", "/g/Zulu"}


async def test_traverse_relations_or_combines_relation_types() -> None:
    # Arrange
    vault = make_traversal_vault()

    # Act
    result = await vault.traverse_relations(
        "/g/Alpha", depth=1, relation_types=["related", "refines"]
    )

    # Assert
    assert {node.note for node in result.nodes} == {"/g/Gamma"}


async def test_traverse_relations_excludes_dangling_targets() -> None:
    # Arrange
    graph = {
        **GRAPH_LINKS,
        "g/Alpha.md": [
            *GRAPH_LINKS["g/Alpha.md"],
            Link(type="requires", target="Ghost"),
        ],
    }
    vault = make_traversal_vault(make_graph_store(graph))

    # Act
    result = await vault.traverse_relations("/g/Alpha", depth=1)

    # Assert
    assert {node.note for node in result.nodes} == {
        "/g/Beta",
        "/g/Gamma",
        "/g/Zulu",
    }


async def test_traverse_relations_uses_aggregated_links_for_frontier_note() -> None:
    # Arrange
    graph = {
        "Alpha.md": [
            Link(type="requires", target="Beta"),
            Link(type="related", target="Gamma"),
        ],
        "Beta.md": [],
        "Gamma.md": [],
    }
    store = make_graph_store(graph)
    vault = make_traversal_vault(store)

    # Act
    result = await vault.traverse_relations("/Alpha", depth=1)

    # Assert
    assert {node.note for node in result.nodes} == {"/Beta", "/Gamma"}
    assert {node.title for node in result.nodes} == {"Beta Note", "Gamma Note"}
    assert store.get_notes_with_links.await_count == 2


async def test_traverse_relations_caps_distinct_notes_at_limit() -> None:
    # Arrange
    node_count = 150
    graph = {
        "Hub.md": [
            Link(type="related", target=f"N{index}")
            for index in range(node_count)
        ],
        **{f"N{index}.md": [] for index in range(node_count)},
    }
    vault = make_traversal_vault(make_graph_store(graph))

    # Act
    result = await vault.traverse_relations("/Hub", depth=2)

    # Assert
    assert len(result.nodes) == 100
    assert result.truncated is True
    assert result.warning is not None
    assert "100" in result.warning
    assert "depth" in result.warning.lower()


async def test_traverse_relations_batches_queries_per_level() -> None:
    # Arrange
    store = make_graph_store()
    vault = make_traversal_vault(store)

    # Act
    await vault.traverse_relations("/g/Alpha", depth=2)

    # Assert
    assert store.get_notes_with_links.await_count == 3
    assert store.get_notes_linking_to.await_count == 2
    assert store.get_notes_with_links.await_args_list[0].args == (["g/Alpha.md"],)
    assert set(store.get_notes_linking_to.await_args_list[0].args[0]) == {
        "Alpha",
        "g/Alpha",
        "/g/Alpha",
    }
    assert set(store.get_notes_with_links.await_args_list[1].args[0]) == {
        "g/Beta.md",
        "g/Gamma.md",
        "g/Zulu.md",
    }


async def test_traverse_relations_unknown_origin_raises() -> None:
    # Arrange
    vault = make_traversal_vault()

    # Act / Assert
    with pytest.raises(FileNotFoundError, match="Nonexistent"):
        await vault.traverse_relations("Nonexistent", depth=1)


async def test_traverse_relations_accepts_bare_name_and_partial_path() -> None:
    # Arrange
    vault = make_traversal_vault()

    # Act
    by_name = await vault.traverse_relations("Alpha", depth=1)
    by_partial = await vault.traverse_relations("g/Alpha", depth=1)

    # Assert
    assert by_name.origin == by_partial.origin == "/g/Alpha"
    assert by_name.nodes == by_partial.nodes


async def test_traverse_relations_ambiguous_origin_is_picked_alphabetically() -> None:
    # Arrange
    vault = make_traversal_vault(make_graph_store(DUP_LINKS))

    # Act
    result = await vault.traverse_relations("Dup", depth=1)

    # Assert
    assert result.origin == "/P/Dup"


async def test_traverse_relations_follows_ambiguous_outgoing_link_to_picked_note() -> None:
    # Arrange
    vault = make_traversal_vault(make_graph_store(DUP_LINKS))

    # Act
    result = await vault.traverse_relations("/Hub/Alpha", depth=1)

    # Assert
    assert {(n.note, n.direction, n.via) for n in result.nodes} == {
        ("/P/Dup", "outgoing", "/Hub/Alpha"),
    }


async def test_traverse_relations_incoming_via_is_the_frontier_path() -> None:
    # Arrange
    vault = make_traversal_vault(make_graph_store(DUP_LINKS))

    # Act
    result = await vault.traverse_relations("/P/Dup", depth=1)

    # Assert
    assert {(n.note, n.relation, n.via) for n in result.nodes} == {
        ("/Hub/Alpha", "uses", "/P/Dup"),
        ("/Z/Zulu", "details", "/P/Dup"),
    }
    assert {n.direction for n in result.nodes} == {"incoming"}


async def test_traverse_relations_ignores_link_resolving_to_other_same_named_note() -> None:
    # Arrange
    vault = make_traversal_vault(make_graph_store(DUP_LINKS))

    # Act
    result = await vault.traverse_relations("/Q/Dup", depth=1)

    # Assert
    assert result.nodes == []


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
    ctx = make_context(fake_vault)

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
    ctx = make_context(fake_vault)

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


async def test_obsidian_traverse_relations_tool_reports_ambiguous_note() -> None:
    # Arrange
    fake_vault = FakeVault()
    ctx = make_context(fake_vault)

    async def raise_ambiguous(
        wikilink_name: str,
        depth: int = 1,
        relation_types: list[str] | None = None,
    ) -> TraversalResult:
        raise ValueError(
            f"Ambiguous note reference '{wikilink_name}'. "
            "Matches: /P/Dup, /Q/Dup"
        )

    fake_vault.traverse_relations = raise_ambiguous  # type: ignore[method-assign]

    # Act / Assert
    with pytest.raises(ToolError, match=r"/P/Dup, /Q/Dup"):
        await obsidian_vault.traverse_relations("Dup", ctx)


async def test_search_waits_for_index_update_before_querying() -> None:
    fake_vault = FakeVault()
    fake_vault.search_results = [make_result_chunk()]
    scheduler = FakeIndexScheduler(fake_vault.events)

    await obsidian_vault.search("query", make_context(fake_vault, scheduler))

    assert fake_vault.events[:2] == ["wait", "search"]
    assert scheduler.timeouts == [obsidian_vault.INDEX_REFRESH_TIMEOUT]


async def test_get_file_content_waits_for_index_update_before_reading() -> None:
    fake_vault = FakeVault()
    fake_vault.files = {"Note": "content"}

    await obsidian_vault.get_file_content("Note", make_context(fake_vault))

    assert fake_vault.events == ["wait", "get_file"]


async def test_traverse_relations_waits_for_index_update_before_traversing() -> None:
    fake_vault = FakeVault()

    await obsidian_vault.traverse_relations("Doc", make_context(fake_vault))

    assert fake_vault.events == ["wait", "traverse"]


async def test_search_answers_from_current_index_when_update_is_late() -> None:
    fake_vault = FakeVault()
    fake_vault.search_results = [make_result_chunk()]
    scheduler = FakeIndexScheduler(fake_vault.events, fresh=False)

    response = await obsidian_vault.search(
        "query", make_context(fake_vault, scheduler)
    )
    results = response.results

    assert len(results) == 1


async def test_get_file_content_rejects_invalid_name_without_waiting() -> None:
    fake_vault = FakeVault()

    with pytest.raises(ToolError):
        await obsidian_vault.get_file_content("../x", make_context(fake_vault))

    assert fake_vault.events == []


async def test_list_files_does_not_wait_for_index_update() -> None:
    fake_vault = FakeVault()

    await obsidian_vault.list_files("/", make_context(fake_vault))

    assert fake_vault.events == []


async def test_read_note_finds_note_created_after_last_index_update() -> None:
    vault = StaleUntilIndexedVault()
    scheduler = IndexScheduler(cast(IVault, vault), timedelta(hours=1))
    task = asyncio.create_task(scheduler.run())
    try:
        await asyncio.sleep(0)
        vault.pending = {"New": "content"}
        ctx = cast(Context, FakeContext({
            "obsidian_vault": vault,
            "obsidian_index_scheduler": scheduler,
        }))

        result = await obsidian_vault.get_file_content("New", ctx)

        assert result == "content"
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


async def test_tool_schemas_document_note_path_contract() -> None:
    # Arrange
    mcp = FastMCP("test")
    obsidian_vault.register_tools(mcp)

    # Act
    async with Client(mcp) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}

    # Assert
    search_schema = tools["obsidian_search"].output_schema
    item_schema = search_schema["properties"]["results"]["items"]
    assert "anyOf" not in item_schema
    assert {"content", "note_path"} <= set(item_schema["required"])
    assert set(search_schema["required"]) >= {"results", "total_documents"}
    read_schema = tools["obsidian_read_note"].input_schema
    traverse_schema = tools["obsidian_traverse_relations"].input_schema
    read_doc = read_schema["properties"]["note"]["description"]
    traverse_doc = traverse_schema["properties"]["note"]["description"]
    assert "full vault path starting with '/'" in read_doc
    assert read_doc == traverse_doc
    search_output = str(tools["obsidian_search"].output_schema)
    assert "note_path" in search_output
    assert "wikilink_name" not in search_output
    assert "source_path" not in search_output
