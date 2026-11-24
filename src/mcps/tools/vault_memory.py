"""Agent memory tools: markdown notes stored under the vault's memory folder.

Memories are plain vault notes, so the existing search, read and traversal
tools retrieve them. The indexer picks up new and deleted files on its own.
"""

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import frontmatter
from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from mcps.config import ServerConfig
from mcps.rag.note_paths import is_malformed_reference
from mcps.tools.obsidian_vault import (
    NoteReference,
    vault_from_context,
    wait_for_fresh_index,
)

DEFAULT_PROJECT = "general"
MAX_NAME_LENGTH = 120
MAX_DESCRIPTION_LENGTH = 200
MAX_PROJECT_LENGTH = 100
MAX_CONTENT_LENGTH = 4_000

NameArg = Annotated[
    str,
    Field(
        min_length=1,
        max_length=MAX_NAME_LENGTH,
        description=(
            "Note name; it becomes the file name, so it cannot contain '/'. "
            "Make it short and specific."
        ),
    ),
]
DescriptionArg = Annotated[
    str,
    Field(
        min_length=1,
        max_length=MAX_DESCRIPTION_LENGTH,
        description="One-line summary used to decide relevance in search.",
    ),
]
ProjectArg = Annotated[
    str,
    Field(
        max_length=MAX_PROJECT_LENGTH,
        description=(
            "Project or repository name; it becomes the memory subfolder. "
            "Use 'general' for knowledge that holds across projects."
        ),
    ),
]
ContentArg = Annotated[
    str,
    Field(
        min_length=1,
        max_length=MAX_CONTENT_LENGTH,
        description=(
            "One fact, decision or lesson to remember. Write it so a reader "
            "who has not seen this session understands it. One fact per "
            "memory: split longer knowledge into several memories and link "
            "them with [[note]]."
        ),
    ),
]


class AddMemoryResult(BaseModel):
    note_path: str = Field(
        description="Vault path of the new note; pass it to obsidian_read_note."
    )


class DeleteMemoryResult(BaseModel):
    deleted: str = Field(description="Vault path of the deleted note.")


_WIKILINK_BREAKERS = frozenset("#|[]^")


def _collapse_whitespace(text: str) -> str:
    return " ".join(text.split())


def _path_segment(value: str, label: str) -> str:
    """Return value as a single safe file or folder name."""
    segment = _collapse_whitespace(value)
    if (
        not segment
        or "/" in segment
        or _WIKILINK_BREAKERS.intersection(segment)
        or is_malformed_reference(segment)
    ):
        raise ToolError(f"{label} is not a usable file name: {value!r}")
    return segment


def _vault_and_memory_dir(ctx: Context) -> tuple[Path, str]:
    config: ServerConfig = ctx.lifespan_context["obsidian_config"]
    if config.vault.dir is None:
        raise ToolError("No vault is configured")
    return config.vault.dir, config.vault.memory_dir


def _ensure_in_memory_folder(memory_root: Path, target: Path) -> None:
    """Refuse a target that, after following symlinks, leaves the memory folder."""
    if not target.resolve().is_relative_to(memory_root.resolve()):
        raise ToolError(f"Not a note under the memory folder: {target.name}")


def _note_path_of(vault: Path, target: Path) -> str:
    return "/" + target.relative_to(vault).with_suffix("").as_posix()


def _render(description: str, project: str, content: str) -> str:
    metadata: dict[str, Any] = {
        "description": _collapse_whitespace(description),
        "project": project,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": "agent",
        "tags": ["generated","memory"]
    }
    post = frontmatter.Post(content.strip(), **metadata)
    return frontmatter.dumps(post) + "\n"


def _write_memory(
    vault: Path,
    memory_dir: str,
    name: str,
    description: str,
    project: str,
    content: str,
) -> str:
    if not content.strip():
        raise ToolError("content is empty")
    file_name = _path_segment(name, "name")
    folder_name = _path_segment(project, "project")
    memory_root = vault / memory_dir
    folder = memory_root / folder_name
    target = folder / f"{file_name}.md"
    _ensure_in_memory_folder(memory_root, target)

    note = _render(description, folder_name, content)
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
            "delete it first, or choose another name."
        ) from None
    except OSError as error:
        raise ToolError(f"Could not write memory: {error.strerror or error}") from None
    return _note_path_of(vault, target)


def _remove_memory(vault: Path, memory_dir: str, source_path: str) -> str:
    memory_root = vault / memory_dir
    target = vault / source_path
    _ensure_in_memory_folder(memory_root, target)
    if not target.is_file():
        raise ToolError(f"No file found matching: {_note_path_of(vault, target)}")
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
    return _note_path_of(vault, target)


async def add_memory(
    name: NameArg,
    description: DescriptionArg,
    content: ContentArg,
    ctx: Context,
    project: ProjectArg = DEFAULT_PROJECT,
) -> AddMemoryResult:
    vault, memory_dir = _vault_and_memory_dir(ctx)
    note_path = await asyncio.to_thread(
        _write_memory, vault, memory_dir, name, description, project, content
    )
    return AddMemoryResult(note_path=note_path)


async def delete_memory(note: NoteReference, ctx: Context) -> DeleteMemoryResult:
    vault, memory_dir = _vault_and_memory_dir(ctx)
    await wait_for_fresh_index(ctx)
    try:
        source_path = await vault_from_context(ctx).resolve_source_path(note)
    except FileNotFoundError as error:
        raise ToolError(str(error)) from None
    deleted = await asyncio.to_thread(_remove_memory, vault, memory_dir, source_path)
    return DeleteMemoryResult(deleted=deleted)


def register_tools(mcp: FastMCP) -> None:
    mcp.tool(
        add_memory,
        name="add_memory",
        description=(
            "Save one durable fact, decision or lesson as a new markdown note "
            "in the vault's memory folder, so later sessions can find it with "
            "obsidian_search. Do not save what the code or git history "
            "already records. Write absolute dates, not relative ones. Link "
            "related memories with [[note]]. Project knowledge goes under the "
            "project name, cross-project knowledge under 'general'. Fails if a "
            "note with the same name exists in the project."
        ),
        annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False),
    )
    mcp.tool(
        delete_memory,
        name="delete_memory",
        description=(
            "Permanently delete one memory note, for example a stale or wrong "
            "fact. Accepts the same note references as obsidian_read_note. "
            "Only notes in the memory folder can be deleted."
        ),
        annotations=ToolAnnotations(
            read_only_hint=False, destructive_hint=True, idempotent_hint=False
        ),
    )
