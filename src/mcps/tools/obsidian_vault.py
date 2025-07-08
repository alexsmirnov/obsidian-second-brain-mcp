import logging
import re
from pathlib import Path

from fastmcp import FastMCP
from pydantic import Field

from mcps.config import ServerConfig
from mcps.rag.vault import Vault, create_vault
from mcps.common import Tools
import asyncio

logger = logging.getLogger("mcps")


class ObsidianTools(Tools):
    """
    Obsidian Vault tools class that provides file operations and content management
    for Obsidian vaults using the integrated Vault RAG system.
    """
    
    def __init__(self, mcp: FastMCP, config: ServerConfig):
        """
        Initialize the ObsidianTools with MCP instance and configuration.
        
        Args:
            mcp: FastMCP instance for tool registration
            config: Server configuration containing vault directory and other settings
        """
        self.mcp = mcp
        self.config = config
        self.vault = create_vault(config.vault_dir)
        logger.info(f"ObsidianTools initialized for vault: {config.vault_dir}")
    
    def register(self) -> None:
        """
        Register all Obsidian tools with the MCP instance.
        """
        # Register bound methods directly with FastMCP
        self.mcp.tool(
            self.list_files,
            name="obsidian_list_files",
            description="Gets a list of files and subfolders in the specified folder within the Obsidian Vault."
        )
        
        self.mcp.tool(
            self.get_file_content,
            name="obsidian_get_content", 
            description="Get file content from Obsidian Vault"
        )
        
        self.mcp.tool(
            self.rename_move_note,
            name="obsidian_rename_move",
            description="Rename or move Obsidian note and update Wikilinks"
        )
        
        self.mcp.tool(
            self.search,
            name="obsidian_search",
            description="Search for content within the Obsidian Vault using semantic search"
        )
        
        logger.info("ObsidianTools registered successfully with MCP")
    
    async def __aenter__(self) -> 'ObsidianTools':
        """
        Async context manager entry point.
        Initialize the Vault instance and prepare all resources for operation.
        """
        await self.vault.initialize()
        asyncio.create_task(self.vault.update_index())
        logger.info("ObsidianTools vault initialized and indexed")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Async context manager exit point.
        Perform cleanup of all resources.
        """
        await self.vault.cleanup()
        logger.info("ObsidianTools vault cleanup completed")
    
    async def list_files(
        self,
        folder_path: str = Field(
            description="Path to the folder within the Obsidian Vault to list contents from. Use forward slashes for path separation. Examples: '/', 'Projects/', 'Daily Notes/2024/', 'Resources/Documentation/'. Use '/' for vault root directory",
            min_length=1,
            max_length=500
        )
    ) -> str:
        """
        Gets a list of files and subfolders in the specified folder within the Obsidian Vault.
        """
        try:
            logger.info(f"Listing files in Obsidian Vault folder: {folder_path}")
            
            # Normalize folder path
            normalized_path = folder_path.strip('/')
            
            # Use vault's list_files method
            files = await self.vault.list_files(normalized_path)
            
            if not files:
                return f"No files found in folder: {folder_path}"
            
            result = f"Contents of '{folder_path}':\n" + "\n".join(files)
            logger.info(f"Found {len(files)} items in {folder_path}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to list files in {folder_path}: {e}")
            return f"Error listing files in {folder_path}: {e!s}"
    
    async def get_file_content(
        self,
        file_path: str = Field(
            description="Path to the file within the Obsidian Vault to retrieve content from. Include the file extension (.md for markdown files). Examples: 'Meeting Notes.md', 'Projects/Web App/README.md', 'Daily Notes/2024-01-15.md'. Use forward slashes for path separation",
            min_length=1,
            max_length=500
        )
    ) -> str:
        """
        Gets the content of a file within the Obsidian Vault.
        """
        try:
            logger.info(f"Getting content of Obsidian Vault file: {file_path}")
            
            # Remove .md extension if present for vault.get_file method
            file_name = file_path
            if file_name.endswith('.md'):
                file_name = file_name[:-3]
            
            # Use vault's get_file method
            content = await self.vault.get_file(file_name)
            
            logger.info(f"Successfully retrieved content for {file_path}")
            return content
            
        except FileNotFoundError:
            error_msg = f"File not found: {file_path}"
            logger.warning(error_msg)
            return error_msg
        except Exception as e:
            logger.error(f"Failed to get content for {file_path}: {e}")
            return f"Error retrieving file content: {e!s}"
    
    async def rename_move_note(
        self,
        old_path: str = Field(
            description="Current path of the note within the Obsidian Vault, including file extension. Examples: 'Old Note.md', 'Archive/Project Notes.md', 'Daily/2024-01-15.md'. Use forward slashes for path separation",
            min_length=1,
            max_length=500
        ),
        new_path: str = Field(
            description="New path for the note within the Obsidian Vault, including file extension. Can be used to rename (same folder) or move (different folder). Examples: 'New Note.md', 'Projects/Renamed Note.md', 'Archive/2024/Old Project.md'. Use forward slashes for path separation",
            min_length=1,
            max_length=500
        )
    ) -> str:
        """
        Renames or moves an Obsidian note and updates [[Wikilinks]] references if needed.
        """
        try:
            logger.info(f"Renaming/moving Obsidian note from {old_path} to {new_path}")
            
            # Resolve full paths
            vault_path = Path(self.config.vault_dir)
            old_full_path = vault_path / old_path
            new_full_path = vault_path / new_path
            
            # Validate source file exists
            if not old_full_path.exists():
                error_msg = f"Source file does not exist: {old_path}"
                logger.error(error_msg)
                return error_msg
            
            # Create destination directory if needed
            new_full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if destination already exists
            if new_full_path.exists():
                error_msg = f"Destination file already exists: {new_path}"
                logger.error(error_msg)
                return error_msg
            
            # Move the file
            old_full_path.rename(new_full_path)
            logger.info(f"File moved from {old_path} to {new_path}")
            
            # Update wikilinks throughout the vault
            await self._update_wikilinks(old_path, new_path)
            
            # Update vault index to reflect changes
            logger.info("Updating vault index after file move")
            await self.vault.update_index()
            
            success_msg = f"Successfully moved '{old_path}' to '{new_path}' and updated all references"
            logger.info(success_msg)
            return success_msg
            
        except Exception as e:
            logger.error(f"Failed to rename/move note from {old_path} to {new_path}: {e}")
            return f"Error renaming/moving note: {e!s}"
    
    async def search(
        self,
        query: str = Field(
            description="Search query to find relevant content within the Obsidian Vault. Use natural language or specific terms to search for notes, concepts, or information. Examples: 'machine learning algorithms', 'project meeting notes', 'python debugging tips'",
            min_length=1,
            max_length=500
        )
    ) -> str:
        """
        Search for content within the Obsidian Vault using semantic search.
        
        This tool performs semantic search across all notes in the vault, returning
        relevant content chunks with context and metadata.
        """
        try:
            logger.info(f"Searching Obsidian Vault for: {query}")
            
            # Use vault's search method which performs semantic search
            search_results = await self.vault.search(query)
            
            logger.info(f"Search completed for query: {query}")
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to search vault for query '{query}': {e}")
            return f"Error searching vault: {e!s}"
    
    async def _update_wikilinks(self, old_path: str, new_path: str) -> None:
        """
        Update [[Wikilink]] references throughout the vault after a file move/rename.
        """
        try:
            # Extract file names without extensions for wikilink matching
            old_name = Path(old_path).stem
            new_name = Path(new_path).stem
            
            if old_name == new_name:
                logger.info("File name unchanged, no wikilink updates needed")
                return
            
            logger.info(f"Updating wikilinks from '{old_name}' to '{new_name}'")
            
            # Pattern to match [[old_name]] and [[old_name|display text]]
            wikilink_pattern = re.compile(
                rf'\[\[{re.escape(old_name)}(\|[^\]]+)?\]\]',
                re.IGNORECASE
            )
            
            vault_path = Path(self.config.vault_dir)
            updated_files = 0
            
            # Search through all markdown files in the vault
            for md_file in vault_path.rglob('*.md'):
                try:
                    content = md_file.read_text(encoding='utf-8')
                    
                    # Check if file contains the old wikilink
                    if wikilink_pattern.search(content):
                        # Replace wikilinks
                        def replace_wikilink(match):
                            display_part = match.group(1) or ''
                            return f'[[{new_name}{display_part}]]'
                        
                        updated_content = wikilink_pattern.sub(replace_wikilink, content)
                        
                        # Write updated content back to file
                        md_file.write_text(updated_content, encoding='utf-8')
                        updated_files += 1
                        logger.debug(f"Updated wikilinks in: {md_file.relative_to(vault_path)}")
                
                except Exception as e:
                    logger.warning(f"Failed to update wikilinks in {md_file}: {e}")
                    continue
            
            logger.info(f"Updated wikilinks in {updated_files} files")
            
        except Exception as e:
            logger.error(f"Error updating wikilinks: {e}")
            raise