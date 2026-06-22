import asyncio
import logging
import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated, Any

import httpx
from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.lifespan import Lifespan, lifespan
from pydantic import BaseModel, Field

from mcps.config import ServerConfig
from mcps.rag.interfaces import IVault
from mcps.rag.vault import create_vault

logger = logging.getLogger("mcps")

FolderPath = Annotated[
    str,
    Field(
        description=(
            "Path to the folder within the Obsidian Vault to list contents from. "
            "Use forward slashes for path separation. Examples: '/', 'Projects/', "
            "Daily Notes/2024/', 'Resources/Documentation/'. Use '/' for vault "
            "root directory"
        ),
        min_length=1,
        max_length=500,
    ),
]

WikilinkName = Annotated[
    str,
    Field(
        description=(
            "File name used in [[Wikilinks]] to retrieve content from. "
            "Exclude the file extension (.md) for markdown files. Examples: "
            "'Meeting Notes', 'README', '2024-01-15'."
        ),
        min_length=1,
        max_length=500,
    ),
]

CurrentNotePath = Annotated[
    str,
    Field(
        description=(
            "Current path of the note within the Obsidian Vault, including file "
            "extension. Examples: 'Old Note.md', 'Archive/Project Notes.md', "
            "'Daily/2024-01-15.md'. Use forward slashes for path separation"
        ),
        min_length=1,
        max_length=500,
    ),
]

NewNotePath = Annotated[
    str,
    Field(
        description=(
            "New path for the note within the Obsidian Vault, including file "
            "extension. Can be used to rename (same folder) or move (different "
            "folder). Examples: 'New Note.md', 'Projects/Renamed Note.md', "
            "'Archive/2024/Old Project.md'. Use forward slashes for path "
            "separation"
        ),
        min_length=1,
        max_length=500,
    ),
]

SearchQuery = Annotated[
    str,
    Field(
        description=(
            "Natural language search query describing what to find in the user's "
            "notes. Can be a topic, concept, question, or keywords. Examples: "
            "'machine learning algorithms', 'meeting notes about Q3 roadmap', "
            "'python debugging tips', 'what I know about distributed systems'"
        ),
        min_length=1,
        max_length=500,
    ),
]

Tags = Annotated[
    list[str] | None,
    Field(
        default=None,
        description=(
            "Optional list of tags to narrow results. Only notes with ALL "
            "specified tags are returned. Use when the user mentions specific "
            "categories or labels. Examples: ['project', 'python'], "
            "['meeting', 'q3']"
        ),
    ),
]

PathFilter = Annotated[
    str | None,
    Field(
        default=None,
        description=(
            "Optional folder path prefix to limit search scope. Use when the "
            "user specifies a particular section or folder of their vault. "
            "Examples: 'Projects/', 'Daily Notes/2024/', 'Research/'"
        ),
        max_length=500,
    ),
]

ReadOffset = Annotated[
    int | None,
    Field(
        default=None,
        description=(
            "Optional decoded character offset to start reading from. Use offsets "
            "returned by obsidian_search chunk results when reading a specific "
            "section. Must be greater than or equal to 0."
        ),
        ge=0,
    ),
]

ReadLimit = Annotated[
    int | None,
    Field(
        default=None,
        description=(
            "Optional maximum number of decoded characters to return from the "
            "requested note. Must be greater than or equal to 0."
        ),
        ge=0,
    ),
]


class SearchResultItem(BaseModel):
    """Structured search result returned by the obsidian_search tool."""

    title: str | None
    description: str | None
    content: str
    tags: list[str]
    source_path: str
    wikilink_name: str
    offset: int
    file_size: int


def build_obsidian_lifespan(config: ServerConfig) -> Lifespan:
    assert config.vault_dir is not None

    if not config.vault_dir.exists():
        raise FileNotFoundError(
            f"Vault dir does not exist: {config.vault_dir}"
        )

    if not config.vault_dir.is_dir():
        raise ValueError(
            f"Vault dir is not a directory: {config.vault_dir}"
                    )
    @lifespan
    async def obsidian_lifespan(server: FastMCP) -> AsyncIterator[dict[str, Any]]:
        async with httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
        ) as http_client:
            async with create_vault(
                config,
                http_client
            ) as vault:
                index_task = asyncio.create_task(vault.update_index())
                try:
                    logger.info("Obsidian vault initialized and indexing started")
                    yield {"obsidian_vault": vault, "obsidian_config": config}
                finally:
                    if not index_task.done():
                        index_task.cancel()
                    try:
                        await index_task
                    except asyncio.CancelledError:
                        logger.debug("Obsidian index task cancelled during shutdown")

    return obsidian_lifespan


def register_tools(mcp: FastMCP) -> None:
    mcp.tool(
        list_files,
        name="obsidian_list_files",
        description=(
            "Gets a list of files and subfolders in the specified folder within "
            "the Obsidian Vault."
        ),
    )
    mcp.tool(
        get_file_content,
        name="obsidian_get_content",
        description="Get file content from Obsidian Vault",
    )
    mcp.tool(
        rename_move_note,
        name="obsidian_rename_move",
        description="Rename or move Obsidian note and update Wikilinks",
    )
    mcp.tool(
        search,
        name="obsidian_search",
        description=(
            "Search the user's personal knowledge base (Obsidian Vault) to find "
            "notes, ideas, and information they have previously written or saved. "
            "Use this tool when the user asks what they know about a topic, wants "
            "to recall something they wrote, asks to find or look up their notes "
            "on a subject, refers to their knowledge base or vault, or asks "
            "'do I have anything on...' / 'what did I write about...'. Returns "
            "relevant note excerpts ranked by relevance. Does NOT list files or "
            "read a specific file by path — use obsidian_list_files or "
            "obsidian_get_content for those."
        ),
    )

    logger.info("Obsidian tools registered successfully with MCP")


async def list_files(folder_path: FolderPath, ctx: Context) -> str:
    """Gets a list of files and subfolders in the specified folder."""
    try:
        logger.info(f"Listing files in Obsidian Vault folder: {folder_path}")
        normalized_path = folder_path.strip("/")
        files = await _vault_from_context(ctx).list_files(normalized_path)

        if not files:
            return f"No files found in folder: {folder_path}"

        result = f"Contents of '{folder_path}':\n" + "\n".join(files)
        logger.info(f"Found {len(files)} items in {folder_path}")
        return result

    except Exception as e:
        logger.error(f"Failed to list files in {folder_path}: {e}")
        return f"Error listing files in {folder_path}: {e!s}"


async def get_file_content(
    wikilink_name: WikilinkName,
    ctx: Context,
    offset: ReadOffset = None,
    limit: ReadLimit = None,
) -> str:
    """Gets note content by Obsidian wikilink name."""
    logger.info(f"Getting content of Obsidian Vault file: {wikilink_name}")
    invalid_message = _validate_content_read_request(wikilink_name, offset, limit)
    if invalid_message is not None:
        raise ToolError(invalid_message)

    try:
        return await _vault_from_context(ctx).get_file(
            wikilink_name,
            offset=offset,
            limit=limit,
        )
    except Exception as e:
        logger.error(f"Failed to get content for {wikilink_name}: {e}")
        raise ToolError(str(e)) from e


async def rename_move_note(
    old_path: CurrentNotePath,
    new_path: NewNotePath,
    ctx: Context,
) -> str:
    """Renames or moves an Obsidian note and updates Wikilink references."""
    try:
        logger.info(f"Renaming/moving Obsidian note from {old_path} to {new_path}")
        config = _config_from_context(ctx)
        vault = _vault_from_context(ctx)

        if config.vault_dir is None:
            error_msg = "Vault directory is not configured"
            logger.error(error_msg)
            return error_msg

        vault_path = Path(config.vault_dir).resolve()
        old_full_path = (vault_path / old_path).resolve()
        new_full_path = (vault_path / new_path).resolve()

        if not old_full_path.is_relative_to(vault_path):
            error_msg = f"Source path escapes vault: {old_path}"
            logger.error(error_msg)
            return error_msg

        if not new_full_path.is_relative_to(vault_path):
            error_msg = f"Destination path escapes vault: {new_path}"
            logger.error(error_msg)
            return error_msg

        if not old_full_path.exists():
            error_msg = f"Source file does not exist: {old_path}"
            logger.error(error_msg)
            return error_msg

        new_full_path.parent.mkdir(parents=True, exist_ok=True)

        if new_full_path.exists():
            error_msg = f"Destination file already exists: {new_path}"
            logger.error(error_msg)
            return error_msg

        old_full_path.rename(new_full_path)
        logger.info(f"File moved from {old_path} to {new_path}")

        await _update_wikilinks(vault_path, old_path, new_path)

        logger.info("Updating vault index after file move")
        await vault.update_index()

        success_msg = (
            f"Successfully moved '{old_path}' to '{new_path}' "
            "and updated all references"
        )
        logger.info(success_msg)
        return success_msg

    except Exception as e:
        logger.error(f"Failed to rename/move note from {old_path} to {new_path}: {e}")
        return f"Error renaming/moving note: {e!s}"


async def search(
    query: SearchQuery,
    ctx: Context,
    tags: Tags = None,
    path: PathFilter = None,
) -> list[SearchResultItem]:
    """Search for content within the Obsidian Vault using semantic search."""
    try:
        logger.info(f"Searching Obsidian Vault for: {query}")
        chunks = await _vault_from_context(ctx).search(
            query, tags=tags, path=path
        )
        logger.info(f"Search completed for query: {query}")
        return [
            SearchResultItem(
                title=c.title,
                description=c.description,
                content=c.content,
                tags=c.tags,
                source_path=c.source_path,
                wikilink_name=c.wikilink_name,
                offset=c.offset,
                file_size=c.file_size,
            )
            for c in chunks
        ]

    except Exception as e:
        logger.error(f"Failed to search vault for query '{query}': {e}")
        return []


def _vault_from_context(ctx: Context) -> IVault:
    return ctx.lifespan_context["obsidian_vault"]


def _config_from_context(ctx: Context) -> ServerConfig:
    return ctx.lifespan_context["obsidian_config"]


def _validate_content_read_request(
    wikilink_name: str,
    offset: int | None,
    limit: int | None,
) -> str | None:
    if offset is not None and offset < 0:
        return "Invalid offset: must be greater than or equal to 0"
    if limit is not None and limit < 0:
        return "Invalid limit: must be greater than or equal to 0"
    if _is_invalid_wikilink_name(wikilink_name):
        return f"Invalid wikilink name: {wikilink_name}"
    return None


def _is_invalid_wikilink_name(wikilink_name: str) -> bool:
    if not wikilink_name.strip():
        return True
    if wikilink_name.endswith(".md"):
        return True
    if "\\" in wikilink_name:
        return True
    path = Path(wikilink_name)
    return path.is_absolute() or any(part in {"..", ""} for part in path.parts)


async def _update_wikilinks(vault_path: Path, old_path: str, new_path: str) -> None:
    try:
        old_name = Path(old_path).stem
        new_name = Path(new_path).stem

        if old_name == new_name:
            logger.info("File name unchanged, no wikilink updates needed")
            return

        logger.info(f"Updating wikilinks from '{old_name}' to '{new_name}'")
        wikilink_pattern = re.compile(
            rf"\[\[{re.escape(old_name)}(\|[^\]]+)?\]\]",
            re.IGNORECASE,
        )
        updated_files = 0

        for md_file in vault_path.rglob("*.md"):
            try:
                if md_file.is_symlink():
                    logger.debug(f"Skipping symlinked file: {md_file}")
                    continue

                resolved_md_file = md_file.resolve()
                if not resolved_md_file.is_relative_to(vault_path):
                    logger.warning(f"Skipping file outside vault: {md_file}")
                    continue

                content = md_file.read_text(encoding="utf-8")
                if not wikilink_pattern.search(content):
                    continue

                updated_content = wikilink_pattern.sub(
                    lambda match: f"[[{new_name}{match.group(1) or ''}]]",
                    content,
                )
                md_file.write_text(updated_content, encoding="utf-8")
                updated_files += 1
                logger.debug(f"Updated wikilinks in: {md_file.relative_to(vault_path)}")

            except Exception as e:
                logger.warning(f"Failed to update wikilinks in {md_file}: {e}")
                continue

        logger.info(f"Updated wikilinks in {updated_files} files")

    except Exception as e:
        logger.error(f"Error updating wikilinks: {e}")
        raise
