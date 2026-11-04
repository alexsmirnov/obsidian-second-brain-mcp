from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import frontmatter
import pytest
from fastmcp import Client, Context, FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.lifespan import lifespan

from mcps.config import ServerConfig
from mcps.rag.note_paths import (
    is_malformed_reference,
    pick_note,
    select_matching,
    to_note_path,
    to_source_path,
)
from mcps.server import create_server
from mcps.tools import vault_memory

VALID_ARGS: dict[str, Any] = {
    "name": "Fact",
    "description": "A fact.",
    "content": "x",
}


class FakeContext:
    def __init__(self, lifespan_context: dict[str, object]):
        self.lifespan_context = lifespan_context


class FakeScheduler:
    async def refresh(self, timeout: object) -> bool:
        return True


class FakeSearchVault:
    """Search boundary: returns canned chunks and records the calls."""

    def __init__(self, source_paths: list[str] | None = None, fail: bool = False):
        self.source_paths = source_paths or []
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    async def search(
        self, query: str, tags: list[str] | None = None, path: str | None = None
    ) -> list[Any]:
        self.calls.append({"query": query, "path": path})
        if self.fail:
            raise RuntimeError("index is down")
        return [SimpleNamespace(source_path=path) for path in self.source_paths]


class FakeVault:
    """Resolution boundary: resolves references over the real files in a folder."""

    def __init__(self, root: Path):
        self.root = root

    async def resolve_source_path(self, reference: str) -> str:
        if reference.startswith("/"):
            if is_malformed_reference(reference):
                raise FileNotFoundError(f"No file found matching: {reference}")
            return to_source_path(reference)
        notes = [
            to_note_path(path.relative_to(self.root).as_posix())
            for path in self.root.rglob("*.md")
        ]
        matches = select_matching(reference, notes)
        if not matches:
            raise FileNotFoundError(f"No file found matching: {reference}")
        return to_source_path(pick_note(matches))


def make_context(
    vault: Path,
    memory_dir: str = "Memory",
    search_vault: FakeSearchVault | FakeVault | None = None,
) -> Context:
    config = ServerConfig(vault_dir=vault, memory_dir=memory_dir)
    lifespan_context: dict[str, object] = {
        "obsidian_config": config,
        "obsidian_index_scheduler": FakeScheduler(),
    }
    if search_vault is not None:
        lifespan_context["obsidian_vault"] = search_vault
    return cast(Context, FakeContext(lifespan_context))


def make_memory_server(vault: Path) -> FastMCP:
    """Bare server with only the memory tools: no RAG lifespan, no index."""
    config = ServerConfig(vault_dir=vault)

    @lifespan
    async def memory_lifespan(server: FastMCP):
        yield {"obsidian_config": config}

    mcp = FastMCP("memory-test", lifespan=memory_lifespan)
    vault_memory.register_tools(mcp)
    return mcp


def read_memory(vault: Path, note_path: str) -> frontmatter.Post:
    return frontmatter.load(vault / f"{note_path.removeprefix('/')}.md")


async def add(vault: Path, name: str = "Fact", **kwargs: Any) -> str:
    kwargs.setdefault("content", "A fact. More.")
    kwargs.setdefault("description", "A fact.")
    result = await vault_memory.add_memory(
        name=name, ctx=make_context(vault), **kwargs
    )
    return result.note_path


async def delete(vault: Path, note: str) -> str:
    ctx = make_context(vault, search_vault=FakeVault(vault))
    return (await vault_memory.delete_memory(note, ctx)).deleted


class TestAddMemory:
    async def test_default_project_is_general(self, tmp_path: Path):
        note_path = await add(tmp_path, "UV Lock Pitfall")

        assert note_path == "/Memory/general/UV Lock Pitfall"
        assert (tmp_path / "Memory/general/UV Lock Pitfall.md").is_file()

    async def test_project_is_the_subfolder(self, tmp_path: Path):
        note_path = await add(tmp_path, "Pin Versions", project="my-app")

        assert note_path == "/Memory/my-app/Pin Versions"
        assert (tmp_path / "Memory/my-app/Pin Versions.md").is_file()

    async def test_unicode_names_are_kept(self, tmp_path: Path):
        note_path = await add(tmp_path, "Мой факт", project="Проект")

        assert note_path == "/Memory/Проект/Мой факт"

    async def test_custom_memory_dir(self, tmp_path: Path):
        result = await vault_memory.add_memory(
            name="T",
            description="d",
            content="x",
            ctx=make_context(tmp_path, memory_dir="Agent/Notes"),
        )

        assert result.note_path == "/Agent/Notes/general/T"
        assert (tmp_path / "Agent/Notes/general/T.md").is_file()

    async def test_frontmatter_and_body(self, tmp_path: Path):
        note_path = await add(
            tmp_path,
            "T",
            project="p",
            description="  Short \n summary ",
            content="\nThe fact.\n",
        )

        post = read_memory(tmp_path, note_path)
        assert post["description"] == "Short summary"
        assert post["project"] == "p"
        assert post["source"] == "agent"
        assert isinstance(
            datetime.fromisoformat(str(post["created"])), datetime
        )
        assert post.content == "The fact."

    async def test_name_and_project_whitespace_is_collapsed(self, tmp_path: Path):
        note_path = await add(tmp_path, "  A \n B ", project=" My   App ")

        assert note_path == "/Memory/My App/A B"

    @pytest.mark.parametrize(
        "name",
        ["", "   ", "a/b", "/a", "a\\b", ".", "..", "x.md", "a#b", "a|b", "[a]", "a^b"],
    )
    async def test_unusable_name_is_rejected(self, tmp_path: Path, name: str):
        with pytest.raises(ToolError, match="name"):
            await add(tmp_path, name)

        assert not (tmp_path / "Memory").exists()

    @pytest.mark.parametrize(
        "project",
        ["", "a/b", "../x", "..", "a\\b", "x.md"],
    )
    async def test_unusable_project_is_rejected(self, tmp_path: Path, project: str):
        with pytest.raises(ToolError, match="project"):
            await add(tmp_path, project=project)

        assert not (tmp_path / "Memory").exists()

    async def test_blank_content_is_rejected(self, tmp_path: Path):
        with pytest.raises(ToolError, match="content"):
            await add(tmp_path, "T", content="  \n ")

        assert not (tmp_path / "Memory").exists()

    async def test_existing_note_is_never_overwritten(self, tmp_path: Path):
        first = await add(tmp_path, "T", content="first")

        with pytest.raises(ToolError, match=first):
            await add(tmp_path, "T", content="second")

        assert read_memory(tmp_path, first).content == "first"

    async def test_unusable_memory_folder_is_not_reported_as_existing_note(
        self, tmp_path: Path
    ):
        (tmp_path / "Memory").write_text("a file, not a folder")

        with pytest.raises(ToolError) as error:
            await add(tmp_path, "T")

        assert "already exists" not in str(error.value)

    async def test_project_folder_symlinked_outside_memory_is_rejected(
        self, tmp_path: Path
    ):
        vault = tmp_path / "vault"
        outside = tmp_path / "outside"
        outside.mkdir()
        (vault / "Memory").mkdir(parents=True)
        (vault / "Memory/escape").symlink_to(outside, target_is_directory=True)

        with pytest.raises(ToolError, match="memory folder"):
            await add(vault, "T", project="escape")

        assert list(outside.iterdir()) == []

    async def test_project_folder_symlinked_inside_memory_is_accepted(
        self, tmp_path: Path
    ):
        (tmp_path / "Memory/real").mkdir(parents=True)
        (tmp_path / "Memory/alias").symlink_to(
            tmp_path / "Memory/real", target_is_directory=True
        )

        note_path = await add(tmp_path, "T", project="alias")

        assert note_path == "/Memory/alias/T"
        assert (tmp_path / "Memory/real/T.md").is_file()


class TestAddMemoryResult:
    async def test_result_holds_only_the_note_path(self, tmp_path: Path):
        result = await vault_memory.add_memory(
            name="T",
            description="d",
            content="x",
            ctx=make_context(tmp_path, search_vault=FakeSearchVault(["Memory/a.md"])),
        )

        assert result.model_dump() == {"note_path": "/Memory/general/T"}


class TestDeleteMemory:
    @pytest.mark.parametrize(
        "reference",
        ["/Memory/my-app/X", "my-app/X", "X"],
    )
    async def test_deletes_note_by_full_partial_or_bare_reference(
        self, tmp_path: Path, reference: str
    ):
        note_path = await add(tmp_path, "X", project="my-app")

        deleted = await delete(tmp_path, reference)

        assert deleted == note_path
        assert not (tmp_path / "Memory/my-app").exists()
        assert (tmp_path / "Memory").is_dir()

    async def test_keeps_folder_with_sibling_note(self, tmp_path: Path):
        first = await add(tmp_path, "A", project="p")
        await add(tmp_path, "B", project="p")

        await delete(tmp_path, first)

        assert not (tmp_path / "Memory/p/A.md").exists()
        assert (tmp_path / "Memory/p/B.md").is_file()

    async def test_refreshes_the_index_before_resolving(self, tmp_path: Path):
        events: list[str] = []
        note_path = await add(tmp_path, "X")

        class RecordingScheduler(FakeScheduler):
            async def refresh(self, timeout: object) -> bool:
                events.append("refresh")
                return True

        class RecordingVault(FakeVault):
            async def resolve_source_path(self, reference: str) -> str:
                events.append("resolve")
                return await super().resolve_source_path(reference)

        ctx = make_context(tmp_path, search_vault=RecordingVault(tmp_path))
        cast(FakeContext, ctx).lifespan_context["obsidian_index_scheduler"] = (
            RecordingScheduler()
        )

        await vault_memory.delete_memory(note_path, ctx)

        assert events == ["refresh", "resolve"]

    @pytest.mark.parametrize(
        ("reference", "reason"),
        [
            ("/Notes/Other", "memory folder"),
            ("Other", "memory folder"),
            ("/Memory", "memory folder"),
            ("/Memory/../Notes/Other", "No file found"),
            ("/Memory/general/x.md", "No file found"),
            ("/Memory/general", "No file found"),
            ("/Memory/general/missing", "No file found"),
            ("Missing", "No file found"),
        ],
    )
    async def test_invalid_references_are_rejected(
        self, tmp_path: Path, reference: str, reason: str
    ):
        (tmp_path / "Notes").mkdir()
        (tmp_path / "Notes/Other.md").write_text("keep")
        kept = await add(tmp_path, "x")

        with pytest.raises(ToolError, match=reason):
            await delete(tmp_path, reference)

        assert (tmp_path / "Notes/Other.md").read_text() == "keep"
        assert (tmp_path / f"{kept.removeprefix('/')}.md").is_file()
        assert await delete(tmp_path, kept) == kept

    async def test_symlinked_note_pointing_outside_memory_is_rejected(
        self, tmp_path: Path
    ):
        outside = tmp_path / "outside.md"
        outside.write_text("keep")
        vault = tmp_path / "vault"
        (vault / "Memory/general").mkdir(parents=True)
        (vault / "Memory/general/link.md").symlink_to(outside)

        with pytest.raises(ToolError, match="memory folder"):
            await delete(vault, "/Memory/general/link")

        assert outside.read_text() == "keep"


class TestRegistration:
    async def test_tools_are_registered_with_annotations(self, tmp_path: Path):
        server = create_server(ServerConfig(vault_dir=tmp_path))

        tools = {tool.name: tool for tool in await server.mcp.list_tools()}

        assert "update_memory" not in tools
        add_tool = tools["add_memory"]
        delete_tool = tools["delete_memory"]
        assert add_tool.annotations is not None
        assert add_tool.annotations.read_only_hint is False
        assert add_tool.annotations.destructive_hint is False
        assert delete_tool.annotations is not None
        assert delete_tool.annotations.read_only_hint is False
        assert delete_tool.annotations.destructive_hint is True

    async def test_add_memory_takes_only_four_arguments(self, tmp_path: Path):
        server = create_server(ServerConfig(vault_dir=tmp_path))

        tools = {tool.name: tool for tool in await server.mcp.list_tools()}

        schema = tools["add_memory"].parameters
        assert set(schema["properties"]) == {
            "name",
            "description",
            "project",
            "content",
        }
        assert set(schema["required"]) == {"name", "description", "content"}
        assert schema["properties"]["project"]["default"] == "general"

    async def test_delete_memory_takes_a_note_reference(self, tmp_path: Path):
        server = create_server(ServerConfig(vault_dir=tmp_path))

        tools = {tool.name: tool for tool in await server.mcp.list_tools()}

        schema = tools["delete_memory"].parameters
        assert set(schema["properties"]) == {"note"}
        assert "partial path" in schema["properties"]["note"]["description"]

    async def test_tools_are_absent_without_vault(self):
        server = create_server(ServerConfig(vault_dir=None))

        names = {tool.name for tool in await server.mcp.list_tools()}

        assert {"add_memory", "update_memory", "delete_memory"}.isdisjoint(names)


class TestSchemaValidation:
    async def test_valid_call_writes_note(self, tmp_path: Path):
        async with Client(make_memory_server(tmp_path)) as client:
            result = await client.call_tool("add_memory", VALID_ARGS)

        assert result.structured_content is not None
        assert result.structured_content["note_path"] == "/Memory/general/Fact"

    @pytest.mark.parametrize(
        "arguments",
        [
            {**VALID_ARGS, "content": ""},
            {
                **VALID_ARGS,
                "content": "x" * (vault_memory.MAX_CONTENT_LENGTH + 1),
            },
            {**VALID_ARGS, "name": ""},
            {**VALID_ARGS, "name": "n" * 121},
            {**VALID_ARGS, "description": "d" * 201},
            {**VALID_ARGS, "project": "p" * 101},
            {"name": "Fact", "content": "x"},
            {"description": "d", "content": "x"},
            {**VALID_ARGS, "kind": "reference"},
        ],
    )
    async def test_out_of_limit_arguments_are_rejected(
        self, tmp_path: Path, arguments: dict[str, Any]
    ):
        async with Client(make_memory_server(tmp_path)) as client:
            with pytest.raises(ToolError):
                await client.call_tool("add_memory", arguments)

        assert not (tmp_path / "Memory").exists()
