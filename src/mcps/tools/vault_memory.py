"""Agent memory tools: markdown notes stored under the vault's memory folder.

Memories are plain vault notes, so the existing search, read and traversal
tools retrieve them. The indexer picks up new and deleted files on its own.
"""

import asyncio
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

import frontmatter
from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from mcps.config import ServerConfig
from mcps.rag.note_paths import is_malformed_reference

MAX_SLUG_LENGTH = 80
MAX_DESCRIPTION_LENGTH = 200
DERIVED_TITLE_WORDS = 8

_NON_WORD_RUNS = re.compile(r"[\W_]+")
_LEADING_MARKERS = re.compile(r"^\s*(?:(?:#{1,6}|[-*>])\s+|```\w*\s*)+")
_FIRST_SENTENCE = re.compile(r"^.*?[.!?](?=\s|$)")

MemoryScope = Literal["general", "project"]

ContentArg = Annotated[
    str,
    Field(
        min_length=1,
        max_length=20_000,
        description=(
            "The fact, decision or lesson to remember. Write it so a reader "
            "who has not seen this session understands it."
        ),
    ),
]
ScopeArg = Annotated[
    MemoryScope,
    Field(
        description=(
            "'project' for knowledge tied to one codebase (requires project); "
            "'general' for knowledge that holds across projects."
        )
    ),
]
ProjectArg = Annotated[
    str | None,
    Field(
        max_length=100,
        description="Project or repository name. Required when scope is 'project'.",
    ),
]
TopicArg = Annotated[
    str | None,
    Field(
        max_length=100,
        description=(
            "Subject such as 'python'. For general memories it also becomes a "
            "subfolder."
        ),
    ),
]
TitleArg = Annotated[
    str | None,
    Field(
        max_length=120,
        description="Note title; defaults to the first words of the content.",
    ),
]
TagsArg = Annotated[
    list[str] | None,
    Field(
        max_length=10,
        description="Tags from the vault's Tags note, stored as given.",
    ),
]
NotePathArg = Annotated[
    str,
    Field(
        min_length=1,
        max_length=500,
        description=(
            "Full vault path of the memory, as returned by add_memory or "
            "obsidian_search, for example '/Memory/General/python/uv-lock'."
        ),
    ),
]


class AddMemoryResult(BaseModel):
    note_path: str = Field(
        description="Vault path of the new note; pass it to obsidian_read_note."
    )


class DeleteMemoryResult(BaseModel):
    deleted: str = Field(description="Vault path of the deleted note.")


def slugify(text: str) -> str:
    """Lowercase file-name form of text; Unicode letters are kept."""
    slug = _NON_WORD_RUNS.sub("-", text.casefold()).strip("-")
    return slug[:MAX_SLUG_LENGTH].strip("-")


def _collapse_whitespace(text: str) -> str:
    return " ".join(text.split())


def _derive_title(content: str) -> str:
    return " ".join(content.split()[:DERIVED_TITLE_WORDS])


def _describe(content: str) -> str:
    text = _collapse_whitespace(_LEADING_MARKERS.sub("", content))
    sentence = _FIRST_SENTENCE.match(text)
    return (sentence.group() if sentence else text)[:MAX_DESCRIPTION_LENGTH]


def _context_line(scope: MemoryScope, project: str | None, topic: str | None) -> str:
    subject = f"Project {project}" if scope == "project" else "General"
    return f"{subject}, topic {topic}." if topic else f"{subject}."


def _clean_name(value: str | None, label: str) -> tuple[str, str] | None:
    """Return (display name, slug) for an optional name; None when blank."""
    if value is None or not value.strip():
        return None
    display = _collapse_whitespace(value)
    slug = slugify(display)
    if not slug:
        raise ToolError(f"{label} has no letters or digits: {value!r}")
    return display, slug


def _vault_and_memory_dir(ctx: Context) -> tuple[Path, str]:
    config: ServerConfig = ctx.lifespan_context["obsidian_config"]
    if config.vault_dir is None:
        raise ToolError("No vault is configured")
    return config.vault_dir, config.memory_dir


def _ensure_inside_vault_without_symlinks(vault: Path, target: Path) -> None:
    current = vault
    for part in target.relative_to(vault).parts:
        current = current / part
        if current.is_symlink():
            raise ToolError(f"Refusing to follow symlink: {current.name}")
    if not target.resolve().is_relative_to(vault.resolve()):
        raise ToolError("Path escapes the vault")


def _note_path_of(vault: Path, target: Path) -> str:
    return "/" + target.relative_to(vault).with_suffix("").as_posix()


def _render(
    title: str,
    content: str,
    scope: MemoryScope,
    project: str | None,
    topic: str | None,
    tags: list[str],
) -> str:
    metadata: dict[str, Any] = {
        "title": title,
        "description": _describe(content) or title,
        "tags": tags,
        "scope": scope,
    }
    if project:
        metadata["project"] = project
    if topic:
        metadata["topic"] = topic
    metadata["created"] = datetime.now(UTC).isoformat(timespec="seconds")
    metadata["source"] = "agent"
    body = (
        f"# {title}\n\n{_context_line(scope, project, topic)}\n\n{content.strip()}"
    )
    return frontmatter.dumps(frontmatter.Post(body, **metadata)) + "\n"


def _write_memory(
    vault: Path,
    memory_dir: str,
    content: str,
    scope: MemoryScope,
    project: str | None,
    topic: str | None,
    title: str | None,
    tags: list[str] | None,
) -> str:
    project_name = _clean_name(project, "project")
    topic_name = _clean_name(topic, "topic")
    if scope == "project" and project_name is None:
        raise ToolError("scope 'project' requires a project name")
    title_name = _clean_name(title, "title") or _clean_name(
        _derive_title(content), "title"
    )
    if title_name is None:
        raise ToolError("title is empty and the content gives no usable title")

    folder = vault / memory_dir
    if scope == "project" and project_name:
        folder = folder / "Projects" / project_name[1]
    else:
        folder = folder / "General"
        if topic_name:
            folder = folder / topic_name[1]
    target = folder / f"{title_name[1]}.md"
    _ensure_inside_vault_without_symlinks(vault, target)

    note = _render(
        title_name[0],
        content,
        scope,
        project_name[0] if project_name and scope == "project" else None,
        topic_name[0] if topic_name else None,
        list(tags or []),
    )
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ToolError(f"Could not create folder: {error.strerror or error}") from None
    try:
        with target.open("x", encoding="utf-8") as file:
            try:
                file.write(note)
            except OSError:
                target.unlink(missing_ok=True)
                raise
    except FileExistsError:
        raise ToolError(
            f"A memory already exists at {_note_path_of(vault, target)}. Read it, "
            "delete it first, or choose another title."
        ) from None
    except OSError as error:
        raise ToolError(f"Could not write memory: {error.strerror or error}") from None
    return _note_path_of(vault, target)


def _memory_target(vault: Path, memory_dir: str, note_path: str) -> Path:
    if not note_path.startswith("/"):
        raise ToolError("note_path must have a leading '/'")
    if is_malformed_reference(note_path):
        raise ToolError(f"note_path is malformed: {note_path}")
    parts = note_path.removeprefix("/").split("/")
    root_parts = memory_dir.split("/")
    if parts[: len(root_parts)] != root_parts or len(parts) == len(root_parts):
        raise ToolError(f"note_path must be a note under '/{memory_dir}/'")
    target = vault / f"{note_path.removeprefix('/')}.md"
    _ensure_inside_vault_without_symlinks(vault, target)
    if not target.is_file():
        raise ToolError(f"Memory note not found: {note_path}")
    return target


def _remove_memory(vault: Path, memory_dir: str, note_path: str) -> str:
    target = _memory_target(vault, memory_dir, note_path)
    memory_root = vault / memory_dir
    try:
        target.unlink()
    except OSError as error:
        raise ToolError(f"Could not delete memory: {error.strerror or error}") from None
    for parent in target.parents:
        if parent == memory_root:
            break
        try:
            parent.rmdir()
        except OSError:
            break
    return note_path


async def add_memory(
    content: ContentArg,
    scope: ScopeArg,
    ctx: Context,
    project: ProjectArg = None,
    topic: TopicArg = None,
    title: TitleArg = None,
    tags: TagsArg = None,
) -> AddMemoryResult:
    vault, memory_dir = _vault_and_memory_dir(ctx)
    note_path = await asyncio.to_thread(
        _write_memory,
        vault,
        memory_dir,
        content,
        scope,
        project,
        topic,
        title,
        tags,
    )
    return AddMemoryResult(note_path=note_path)


async def delete_memory(note_path: NotePathArg, ctx: Context) -> DeleteMemoryResult:
    vault, memory_dir = _vault_and_memory_dir(ctx)
    deleted = await asyncio.to_thread(_remove_memory, vault, memory_dir, note_path)
    return DeleteMemoryResult(deleted=deleted)


def register_tools(mcp: FastMCP) -> None:
    mcp.tool(
        add_memory,
        name="add_memory",
        description=(
            "Save a durable fact, decision or lesson as a new markdown note in "
            "the vault's memory folder, so later sessions can find it with "
            "obsidian_search. Project knowledge goes under the project, "
            "cross-project knowledge under general. Fails if a note with the "
            "same title exists; to change a memory, delete it and add it again."
        ),
        annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False),
    )
    mcp.tool(
        delete_memory,
        name="delete_memory",
        description=(
            "Permanently delete one memory note by its vault path, for example "
            "a stale or wrong fact. Only notes in the memory folder can be "
            "deleted."
        ),
        annotations=ToolAnnotations(
            read_only_hint=False, destructive_hint=True, idempotent_hint=False
        ),
    )
