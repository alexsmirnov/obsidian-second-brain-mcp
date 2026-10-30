"""Agent memory tools: markdown notes stored under the vault's memory folder.

Memories are plain vault notes, so the existing search, read and traversal
tools retrieve them. The indexer picks up new and deleted files on its own.
"""

import asyncio
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal, get_args

import frontmatter
import yaml
from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from mcps.config import ServerConfig
from mcps.rag.note_paths import is_malformed_reference, to_note_path

logger = logging.getLogger("mcps")

MAX_SLUG_LENGTH = 80
MAX_DESCRIPTION_LENGTH = 200
MAX_CONTENT_LENGTH = 4_000
MAX_REASON_LENGTH = 500
DERIVED_TITLE_WORDS = 8
MAX_SIMILAR_MEMORIES = 3
SIMILAR_SEARCH_TIMEOUT_SECONDS = 5
MAX_SEARCH_QUERY_LENGTH = 500
KINDS_REQUIRING_WHY = frozenset({"feedback", "project"})

_NON_WORD_RUNS = re.compile(r"[\W_]+")
_LEADING_MARKERS = re.compile(r"^\s*(?:(?:#{1,6}|[-*>])\s+|```\w*\s*)+")
_FIRST_SENTENCE = re.compile(r"^.*?[.!?](?=\s|$)")

MemoryScope = Literal["general", "project"]
MemoryKind = Literal["user", "feedback", "project", "reference"]

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
KindArg = Annotated[
    MemoryKind,
    Field(
        description=(
            "'user': who the user is and how they work. 'feedback': guidance "
            "on how to work, with a reason. 'project': ongoing work, goals or "
            "constraints not visible in the code (not the same as scope). "
            "'reference': pointer to an external resource."
        )
    ),
]
DescriptionArg = Annotated[
    str | None,
    Field(
        max_length=MAX_DESCRIPTION_LENGTH,
        description=(
            "One-line summary used to decide relevance in search. Always "
            "provide it; defaults to the first sentence of the content."
        ),
    ),
]
WhyArg = Annotated[
    str | None,
    Field(
        max_length=MAX_REASON_LENGTH,
        description="Why this matters. Required for kinds 'feedback' and 'project'.",
    ),
]
HowToApplyArg = Annotated[
    str | None,
    Field(
        max_length=MAX_REASON_LENGTH,
        description="When and how to apply this memory.",
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
    similar: list[str] = Field(
        default_factory=list,
        description=(
            "Vault paths of existing memories on the same subject. Read them "
            "and use update_memory or delete_memory if the new note duplicates one."
        ),
    )


class UpdateMemoryResult(BaseModel):
    note_path: str = Field(description="Vault path of the updated note.")


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


@dataclass(frozen=True)
class MemoryDraft:
    """Everything that makes up the text of one memory note."""

    title: str
    content: str
    kind: MemoryKind
    scope: MemoryScope
    project: str | None
    topic: str | None
    description: str | None
    why: str | None
    how_to_apply: str | None


def _require_why(kind: MemoryKind, why: str | None) -> None:
    if kind in KINDS_REQUIRING_WHY and not (why and why.strip()):
        raise ToolError(f"kind '{kind}' requires why")


def _render_body(draft: MemoryDraft) -> str:
    body = (
        f"# {draft.title}\n\n"
        f"{_context_line(draft.scope, draft.project, draft.topic)}\n\n"
        f"{draft.content.strip()}"
    )
    if draft.why:
        body += f"\n\n**Why:** {_collapse_whitespace(draft.why)}"
    if draft.how_to_apply:
        separator = "\n" if draft.why else "\n\n"
        how_to_apply = _collapse_whitespace(draft.how_to_apply)
        body += f"{separator}**How to apply:** {how_to_apply}"
    return body


def _description_of(draft: MemoryDraft) -> str:
    explicit = _collapse_whitespace(draft.description or "")
    return explicit or _describe(draft.content) or draft.title


def _render(draft: MemoryDraft, tags: list[str]) -> str:
    metadata: dict[str, Any] = {
        "title": draft.title,
        "description": _description_of(draft),
        "tags": tags,
        "scope": draft.scope,
        "type": draft.kind,
    }
    if draft.project:
        metadata["project"] = draft.project
    if draft.topic:
        metadata["topic"] = draft.topic
    metadata["created"] = datetime.now(UTC).isoformat(timespec="seconds")
    metadata["source"] = "agent"
    post = frontmatter.Post(_render_body(draft), **metadata)
    return frontmatter.dumps(post) + "\n"


def _write_memory(
    vault: Path,
    memory_dir: str,
    content: str,
    kind: MemoryKind,
    scope: MemoryScope,
    project: str | None,
    topic: str | None,
    title: str | None,
    tags: list[str] | None,
    description: str | None,
    why: str | None,
    how_to_apply: str | None,
) -> str:
    _require_why(kind, why)
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

    draft = MemoryDraft(
        title=title_name[0],
        content=content,
        kind=kind,
        scope=scope,
        project=project_name[0] if project_name and scope == "project" else None,
        topic=topic_name[0] if topic_name else None,
        description=description,
        why=why,
        how_to_apply=how_to_apply,
    )
    note = _render(draft, list(tags or []))
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


def _optional_text(value: object) -> str | None:
    return str(value) if value else None


def _replace_memory(
    vault: Path,
    memory_dir: str,
    note_path: str,
    content: str,
    description: str | None,
    why: str | None,
    how_to_apply: str | None,
) -> str:
    target = _memory_target(vault, memory_dir, note_path)
    try:
        post = frontmatter.load(target)
    except (OSError, ValueError, yaml.YAMLError) as error:
        raise ToolError(f"Could not read memory: {error}") from None
    metadata = post.metadata
    if metadata.get("source") != "agent":
        raise ToolError(
            "Only memories written by the agent can be updated; the note may "
            "have been edited by hand. Delete it and add a new memory instead."
        )
    kind: MemoryKind = next(
        (known for known in get_args(MemoryKind) if known == metadata.get("type")),
        "reference",
    )
    _require_why(kind, why)
    draft = MemoryDraft(
        title=str(metadata.get("title") or target.stem),
        content=content,
        kind=kind,
        scope="project" if metadata.get("scope") == "project" else "general",
        project=_optional_text(metadata.get("project")),
        topic=_optional_text(metadata.get("topic")),
        description=description,
        why=why,
        how_to_apply=how_to_apply,
    )
    post.content = _render_body(draft)
    post["description"] = _description_of(draft)
    post["updated"] = datetime.now(UTC).isoformat(timespec="seconds")
    _replace_file(target, frontmatter.dumps(post) + "\n")
    return note_path


def _replace_file(target: Path, text: str) -> None:
    """Swap target for text in one step, keeping its permissions."""
    staging: Path | None = None
    try:
        mode = target.stat().st_mode & 0o777
        descriptor, name = tempfile.mkstemp(
            dir=target.parent, prefix=".", suffix=".tmp"
        )
        staging = Path(name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            os.fchmod(file.fileno(), mode)
            file.write(text)
        os.replace(staging, target)
    except OSError as error:
        if staging is not None:
            staging.unlink(missing_ok=True)
        raise ToolError(f"Could not update memory: {error.strerror or error}") from None


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


async def _find_similar(
    ctx: Context, memory_dir: str, query: str, exclude: str
) -> list[str]:
    """Best effort: paths of other memories close to query; empty on any failure."""
    search_vault = ctx.lifespan_context.get("obsidian_vault")
    if search_vault is None:
        return []
    try:
        async with asyncio.timeout(SIMILAR_SEARCH_TIMEOUT_SECONDS):
            chunks = await search_vault.search(
                _collapse_whitespace(query)[:MAX_SEARCH_QUERY_LENGTH],
                path=f"{memory_dir}/",
            )
        found = dict.fromkeys(to_note_path(chunk.source_path) for chunk in chunks)
    except Exception as error:
        logger.warning("Similar memory search failed: %s", error)
        return []
    found.pop(exclude, None)
    return list(found)[:MAX_SIMILAR_MEMORIES]


async def add_memory(
    content: ContentArg,
    kind: KindArg,
    scope: ScopeArg,
    ctx: Context,
    description: DescriptionArg = None,
    why: WhyArg = None,
    how_to_apply: HowToApplyArg = None,
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
        kind,
        scope,
        project,
        topic,
        title,
        tags,
        description,
        why,
        how_to_apply,
    )
    similar = await _find_similar(ctx, memory_dir, description or content, note_path)
    return AddMemoryResult(note_path=note_path, similar=similar)


async def update_memory(
    note_path: NotePathArg,
    content: ContentArg,
    ctx: Context,
    description: DescriptionArg = None,
    why: WhyArg = None,
    how_to_apply: HowToApplyArg = None,
) -> UpdateMemoryResult:
    vault, memory_dir = _vault_and_memory_dir(ctx)
    updated = await asyncio.to_thread(
        _replace_memory,
        vault,
        memory_dir,
        note_path,
        content,
        description,
        why,
        how_to_apply,
    )
    return UpdateMemoryResult(note_path=updated)


async def delete_memory(note_path: NotePathArg, ctx: Context) -> DeleteMemoryResult:
    vault, memory_dir = _vault_and_memory_dir(ctx)
    deleted = await asyncio.to_thread(_remove_memory, vault, memory_dir, note_path)
    return DeleteMemoryResult(deleted=deleted)


def register_tools(mcp: FastMCP) -> None:
    mcp.tool(
        add_memory,
        name="add_memory",
        description=(
            "Save one durable fact, decision or lesson as a new markdown note "
            "in the vault's memory folder, so later sessions can find it with "
            "obsidian_search. Search first and use update_memory when a memory "
            "on the subject exists. Do not save what the code or git history "
            "already records. Write absolute dates, not relative ones. Link "
            "related memories with [[note]]. Always give a description. "
            "Project knowledge goes under the project, cross-project knowledge "
            "under general. Fails if a note with the same title exists."
        ),
        annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False),
    )
    mcp.tool(
        update_memory,
        name="update_memory",
        description=(
            "Replace the text of an existing memory note by its vault path, "
            "keeping its title, location, tags and creation date. Use it when "
            "a fact changed instead of adding a duplicate."
        ),
        annotations=ToolAnnotations(
            read_only_hint=False, destructive_hint=True, idempotent_hint=True
        ),
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
