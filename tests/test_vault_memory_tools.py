from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, cast

import frontmatter
import pytest
from fastmcp import Client, Context, FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.lifespan import lifespan

from mcps.config import ServerConfig
from mcps.server import create_server
from mcps.tools import vault_memory


class FakeContext:
    def __init__(self, lifespan_context: dict[str, object]):
        self.lifespan_context = lifespan_context


def make_context(vault: Path, memory_dir: str = "Memory") -> Context:
    config = ServerConfig(vault_dir=vault, memory_dir=memory_dir)
    return cast(Context, FakeContext({"obsidian_config": config}))


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


async def add(vault: Path, content: str = "A fact. More.", **kwargs: Any) -> str:
    kwargs.setdefault("scope", "general")
    result = await vault_memory.add_memory(
        content=content, ctx=make_context(vault), **kwargs
    )
    return result.note_path


class TestSlugify:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("My App", "my-app"),
            ("  UV__Lock -- Pitfall!  ", "uv-lock-pitfall"),
            ("Мой проект", "мой-проект"),
            ("!!!", ""),
        ],
    )
    def test_slugify(self, text: str, expected: str):
        assert vault_memory.slugify(text) == expected

    def test_slug_is_capped_and_not_left_with_trailing_dash(self):
        slug = vault_memory.slugify("word " * 100)

        assert len(slug) <= 80
        assert not slug.endswith("-")


class TestAddMemoryPlacement:
    async def test_general_memory_with_topic(self, tmp_path: Path):
        note_path = await add(
            tmp_path,
            "uv lock must be committed. More.",
            topic="Python",
            title="UV Lock Pitfall",
        )

        assert note_path == "/Memory/General/python/uv-lock-pitfall"
        assert (tmp_path / "Memory/General/python/uv-lock-pitfall.md").is_file()

    async def test_general_memory_without_topic(self, tmp_path: Path):
        note_path = await add(tmp_path, title="Plain Fact")

        assert note_path == "/Memory/General/plain-fact"

    async def test_project_memory_keeps_topic_in_frontmatter_only(
        self, tmp_path: Path
    ):
        note_path = await add(
            tmp_path,
            scope="project",
            project="My App",
            topic="python",
            title="Pin Versions",
        )

        assert note_path == "/Memory/Projects/my-app/pin-versions"
        post = read_memory(tmp_path, note_path)
        assert post["topic"] == "python"
        assert post["project"] == "My App"
        assert post["scope"] == "project"

    async def test_general_memory_does_not_store_project(self, tmp_path: Path):
        note_path = await add(tmp_path, project="Ignored", title="T")

        assert "project" not in read_memory(tmp_path, note_path).metadata

    async def test_unusable_memory_folder_is_not_reported_as_existing_note(
        self, tmp_path: Path
    ):
        (tmp_path / "Memory").write_text("a file, not a folder")

        with pytest.raises(ToolError) as error:
            await add(tmp_path, title="T")

        assert "already exists" not in str(error.value)

    async def test_unicode_names_are_kept(self, tmp_path: Path):
        note_path = await add(
            tmp_path, scope="project", project="Мой проект", title="Заметка"
        )

        assert note_path == "/Memory/Projects/мой-проект/заметка"

    async def test_custom_memory_dir(self, tmp_path: Path):
        result = await vault_memory.add_memory(
            content="Fact.",
            scope="general",
            ctx=make_context(tmp_path, memory_dir="Notes/Mem"),
            title="T",
        )

        assert result.note_path == "/Notes/Mem/General/t"
        assert (tmp_path / "Notes/Mem/General/t.md").is_file()

    async def test_derived_title_uses_first_eight_words(self, tmp_path: Path):
        note_path = await add(
            tmp_path, "Always run tests before commit because CI is slow today ok"
        )

        assert note_path == (
            "/Memory/General/always-run-tests-before-commit-because-ci-is"
        )
        post = read_memory(tmp_path, note_path)
        assert post["title"] == "Always run tests before commit because CI is"

    async def test_long_title_gives_capped_file_name(self, tmp_path: Path):
        note_path = await add(tmp_path, title="x" * 300)

        assert len(note_path.rsplit("/", 1)[1]) <= 80
        assert (tmp_path / f"{note_path.removeprefix('/')}.md").is_file()


class TestAddMemoryContent:
    async def test_frontmatter_fields(self, tmp_path: Path):
        note_path = await add(
            tmp_path,
            "uv lock must be committed. More.",
            topic="Python",
            title="UV Lock Pitfall",
            tags=["python", "git"],
        )

        post = read_memory(tmp_path, note_path)
        assert set(post.metadata) == {
            "title",
            "description",
            "tags",
            "scope",
            "topic",
            "created",
            "source",
        }
        assert post["title"] == "UV Lock Pitfall"
        assert post["description"] == "uv lock must be committed."
        assert post["tags"] == ["python", "git"]
        assert post["scope"] == "general"
        assert post["source"] == "agent"
        assert isinstance(post["created"], str)
        datetime.fromisoformat(str(post["created"]))

    async def test_body_has_heading_context_line_and_content(
        self, tmp_path: Path
    ):
        note_path = await add(
            tmp_path, "Commit it. Always.", topic="Python", title="UV Lock Pitfall"
        )

        post = read_memory(tmp_path, note_path)
        assert post.content == (
            "# UV Lock Pitfall\n\nGeneral, topic Python.\n\nCommit it. Always."
        )

    @pytest.mark.parametrize(
        ("kwargs", "context_line"),
        [
            ({"scope": "project", "project": "My App"}, "Project My App."),
            (
                {"scope": "project", "project": "My App", "topic": "Py"},
                "Project My App, topic Py.",
            ),
            ({}, "General."),
        ],
    )
    async def test_context_line_variants(
        self, tmp_path: Path, kwargs: dict[str, Any], context_line: str
    ):
        note_path = await add(tmp_path, "Fact.", title="T", **kwargs)

        assert read_memory(tmp_path, note_path).content.split("\n")[2] == context_line

    async def test_multiline_title_is_collapsed(self, tmp_path: Path):
        note_path = await add(tmp_path, title="a\nb")

        post = read_memory(tmp_path, note_path)
        assert post["title"] == "a b"
        assert post.content.startswith("# a b\n")

    async def test_description_skips_markdown_marker(self, tmp_path: Path):
        note_path = await add(tmp_path, "## Heading text. Rest.", title="T")

        assert read_memory(tmp_path, note_path)["description"] == "Heading text."


class TestAddMemoryRejections:
    @pytest.mark.parametrize("project", [None, "  ", "!!!"])
    async def test_project_scope_requires_a_usable_project(
        self, tmp_path: Path, project: str | None
    ):
        with pytest.raises(ToolError, match="project"):
            await add(tmp_path, scope="project", project=project, title="T")

        assert not (tmp_path / "Memory").exists()
        assert await add(tmp_path, scope="project", project="ok", title="T")

    async def test_unusable_title_is_rejected(self, tmp_path: Path):
        with pytest.raises(ToolError, match="title"):
            await add(tmp_path, "!!!")

        assert not (tmp_path / "Memory").exists()

    async def test_existing_note_is_never_overwritten(self, tmp_path: Path):
        first = await add(tmp_path, "First.", title="Same")
        before = (tmp_path / f"{first.removeprefix('/')}.md").read_text()

        with pytest.raises(ToolError, match=first):
            await add(tmp_path, "Second.", title="Same")

        assert (tmp_path / f"{first.removeprefix('/')}.md").read_text() == before

    @pytest.mark.parametrize("link_at", ["Memory", "Memory/General"])
    async def test_symlinked_folder_is_rejected(
        self, tmp_path: Path, link_at: str
    ):
        vault = tmp_path / "vault"
        outside = tmp_path / "outside"
        vault.mkdir()
        outside.mkdir()
        link = vault / link_at
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(outside, target_is_directory=True)

        with pytest.raises(ToolError, match="symlink"):
            await add(vault, title="T")

        assert list(outside.iterdir()) == []

    async def test_a_plain_folder_in_the_same_vault_is_accepted(
        self, tmp_path: Path
    ):
        (tmp_path / "Memory/General").mkdir(parents=True)

        assert await add(tmp_path, title="T") == "/Memory/General/t"


class TestDeleteMemory:
    async def test_deletes_note_and_prunes_empty_folders(self, tmp_path: Path):
        note_path = await add(
            tmp_path, scope="project", project="My App", title="X"
        )

        result = await vault_memory.delete_memory(note_path, make_context(tmp_path))

        assert result.deleted == note_path
        assert not (tmp_path / "Memory/Projects").exists()
        assert (tmp_path / "Memory").is_dir()

    async def test_keeps_folder_with_sibling_note(self, tmp_path: Path):
        first = await add(tmp_path, scope="project", project="p", title="A")
        await add(tmp_path, scope="project", project="p", title="B")

        await vault_memory.delete_memory(first, make_context(tmp_path))

        assert not (tmp_path / "Memory/Projects/p/a.md").exists()
        assert (tmp_path / "Memory/Projects/p/b.md").is_file()

    @pytest.mark.parametrize(
        ("note_path", "reason"),
        [
            ("/Notes/Other", "under"),
            ("/Memory/../Notes/Other", "malformed|invalid"),
            ("/Memory/General/x.md", "malformed|invalid"),
            ("Memory/General/x", "leading"),
            ("/Memory", "under"),
            ("/Memory/General", "not a note|not found"),
            ("/Memory/General/missing", "not found"),
        ],
    )
    async def test_invalid_paths_are_rejected(
        self, tmp_path: Path, note_path: str, reason: str
    ):
        (tmp_path / "Notes").mkdir()
        (tmp_path / "Notes/Other.md").write_text("keep")
        kept = await add(tmp_path, title="x")
        ctx = make_context(tmp_path)

        with pytest.raises(ToolError, match=reason):
            await vault_memory.delete_memory(note_path, ctx)

        assert (tmp_path / "Notes/Other.md").read_text() == "keep"
        assert (tmp_path / f"{kept.removeprefix('/')}.md").is_file()
        assert (await vault_memory.delete_memory(kept, ctx)).deleted == kept

    async def test_symlinked_note_is_rejected(self, tmp_path: Path):
        outside = tmp_path / "outside.md"
        outside.write_text("keep")
        vault = tmp_path / "vault"
        (vault / "Memory/General").mkdir(parents=True)
        (vault / "Memory/General/link.md").symlink_to(outside)

        with pytest.raises(ToolError, match="symlink"):
            await vault_memory.delete_memory(
                "/Memory/General/link", make_context(vault)
            )

        assert outside.read_text() == "keep"


class TestRegistration:
    async def test_tools_are_registered_with_annotations(self, tmp_path: Path):
        server = create_server(ServerConfig(vault_dir=tmp_path))

        tools = {tool.name: tool for tool in await server.mcp.list_tools()}

        add_tool = tools["add_memory"]
        delete_tool = tools["delete_memory"]
        assert add_tool.annotations is not None
        assert add_tool.annotations.read_only_hint is False
        assert add_tool.annotations.destructive_hint is False
        assert delete_tool.annotations is not None
        assert delete_tool.annotations.read_only_hint is False
        assert delete_tool.annotations.destructive_hint is True

    async def test_tools_are_absent_without_vault(self):
        server = create_server(ServerConfig(vault_dir=None))

        names = {tool.name for tool in await server.mcp.list_tools()}

        assert {"add_memory", "delete_memory"}.isdisjoint(names)


class TestSchemaValidation:
    async def test_valid_call_writes_note(self, tmp_path: Path):
        async with Client(make_memory_server(tmp_path)) as client:
            result = await client.call_tool(
                "add_memory", {"content": "Fact.", "scope": "general", "title": "T"}
            )

        assert result.structured_content == {"note_path": "/Memory/General/t"}

    @pytest.mark.parametrize(
        "arguments",
        [
            {"content": "", "scope": "general"},
            {"content": "x" * 20_001, "scope": "general"},
            {"content": "x", "scope": "other"},
            {"content": "x", "scope": "general", "title": "t" * 121},
            {"content": "x", "scope": "general", "tags": ["t"] * 11},
            {"content": "x", "scope": "project", "project": "p" * 101},
        ],
    )
    async def test_out_of_limit_arguments_are_rejected(
        self, tmp_path: Path, arguments: dict[str, Any]
    ):
        async with Client(make_memory_server(tmp_path)) as client:
            with pytest.raises(ToolError):
                await client.call_tool("add_memory", arguments)

        assert not (tmp_path / "Memory").exists()
