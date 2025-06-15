from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Dict
from pydantic import BaseModel, Field

from mcp import ClientCapabilities, RootsCapability
from mcp.server.session import ServerSession
from fastmcp import FastMCP, Context

import mcps.prompts as prompts_module
import mcps.resources.url_resource as url_resource
import mcps.resources.doc_resource as doc_resource
import mcps.resources.project_resource as project_resource
import mcps.tools.internet_search as internet_search
import mcps.tools.perplexity_search as perplexity_search
import mcps.tools.rag_search as rag_search
import mcps.tools.obsidian_vault as obsidian_vault
from mcps.config import ServerConfig, create_config  # Import from config module


logger = logging.getLogger("mcps")
@dataclass
class AppContext:
    config: ServerConfig


class DevAutomationServer:
    def __init__(self, config: ServerConfig):
        self.config = config
        self.mcp = FastMCP(
            "Development Automation Server",
            # dependencies=["pytest", "httpx", "beautifulsoup4"],  # dependencies for resources/tools
        )
        self._setup_resources()
        self._setup_tools()
        self._setup_prompts()


    def _setup_resources(self):
        @self.mcp.resource("url://{encoded_url}")
        async def url_resource_handler(encoded_url: str) -> str:
            return await url_resource.get_resource(encoded_url, self.config)

        @self.mcp.resource("doc://{library_name}")
        async def doc_resource_handler(library_name: str) -> str:
            return await doc_resource.get_resource(library_name, self.config)

        @self.mcp.resource("project://{project_name}")
        async def project_resource_handler(project_name: str) -> str:
            return await project_resource.get_resource(project_name, self.config)
        @self.mcp.resource("resource://test", name="test/resource", description="Test project resource")
        async def test_resource_handler(context: Context) -> str:
            try:
                session: ServerSession = context.session
                if session.check_client_capability(ClientCapabilities(roots=RootsCapability())) :
                    result = await session.list_roots()
                    logger.info(f"Result: {result}")
                    for root in result.roots:
                        logger.info(f"Root: {root.name} , {root.uri}")
            except Exception as e:
                logger.error(f"Error listing roots: {e}")
            return "Test project resource"
        @self.mcp.resource("documentation://test/docs")
        async def test_docs_handler() -> str:
            return "Test project documentation"

    def _setup_tools(self):
        @self.mcp.tool(name="web_search")
        async def web_search(
            context: Context,
            query: str = Field(
                description="The search query string to find relevant web pages. Should be specific and descriptive to get the best results. Examples: 'Python async programming best practices', 'React hooks tutorial', 'machine learning algorithms comparison'",
                min_length=1,
                max_length=500
            )
        ) -> str:
            """
            Performs a web search using serper and tivily APIs. Find the most relevant pages
            and return summary result.
            
            Returns:
                The summary of the most relevant search results.
            """
            try:
                session: ServerSession = context.session
                if session.check_client_capability(ClientCapabilities(roots=RootsCapability())) :
                    result = await session.list_roots()
                    logger.info(f"Result: {result}")
                    for root in result.roots:
                        logger.info(f"Root: {root.name} , location: {root.uri}")
                else:
                    logger.info("Client does not support roots capability")
                    # Try to get the roots from the environment variable ROOT
                    root_value = os.getenv("ROOT")
                    logger.info(f"ROOT environment variable: {root_value}")
            except Exception as e:
                logger.error(f"Error listing roots: {e}")
            return await internet_search.do_search(query, self.config)

        @self.mcp.tool(name="perplexity_summary_search")
        async def perplexity_summary_search(
            query: str = Field(
                description="The search query string for perplexity.io research. Should be a clear, specific question or topic. Examples: 'What are the latest developments in AI?', 'How does quantum computing work?', 'Best practices for microservices architecture'",
                min_length=1,
                max_length=500
            ),
            research_level: str = Field(
                default="SIMPLE",
                description="Level of research depth to perform. Valid options: 'SIMPLE' for basic overview, 'DETAIL' for comprehensive analysis, 'DEEP' for extensive research with multiple sources and perspectives",
                pattern="^(SIMPLE|DETAIL|DEEP)$"
            )
        ) -> str:
            """
            Performs a web search with summary using perplexity.io.
            
            Returns:
                The summary of search results with the specified research level.
            """
            return await perplexity_search.do_search(query, self.config)

        @self.mcp.tool(name="rag_search", description="RAG search in markdown files")
        async def rag_search_tool(
            query: str = Field(
                description="The search query to find relevant content in markdown files. Use natural language questions or keywords. Examples: 'How to configure authentication?', 'deployment procedures', 'API documentation for user management'",
                min_length=1,
                max_length=500
            ),
            start_folder: str | None = Field(
                default=None,
                description="Optional starting folder path to limit the search scope. If not provided, searches all available markdown files. Use relative paths like 'docs/', 'notes/projects/', or absolute paths. Leave empty to search entire knowledge base"
            )
        ) -> str:
            """
            Performs a RAG search in markdown files for content related to the query.
            
            Returns:
                Content from markdown files that may contain answers to the query.
            """
            return await rag_search.search_markdown_files(query, start_folder, self.config)

        @self.mcp.tool(name="obsidian_list_files")
        async def obsidian_list_files(
            folder_path: str = Field(
                description="Path to the folder within the Obsidian Vault to list contents from. Use forward slashes for path separation. Examples: '/', 'Projects/', 'Daily Notes/2024/', 'Resources/Documentation/'. Use '/' for vault root directory",
                min_length=1,
                max_length=500
            )
        ) -> str:
            """
            Gets a list of files and subfolders in the specified folder within the Obsidian Vault.
            
            Returns:
                A formatted string containing the list of files and subfolders.
            """
            return await obsidian_vault.list_files(folder_path, self.config)

        @self.mcp.tool(name="obsidian_get_content", description="Get file content from Obsidian Vault")
        async def obsidian_get_content(
            file_path: str = Field(
                description="Path to the file within the Obsidian Vault to retrieve content from. Include the file extension (.md for markdown files). Examples: 'Meeting Notes.md', 'Projects/Web App/README.md', 'Daily Notes/2024-01-15.md'. Use forward slashes for path separation",
                min_length=1,
                max_length=500
            )
        ) -> str:
            """
            Gets the content of a file within the Obsidian Vault.
            
            Returns:
                The content of the specified file.
            """
            return await obsidian_vault.get_file_content(file_path, self.config)

        @self.mcp.tool(name="obsidian_rename_move", description="Rename or move Obsidian note and update Wikilinks")
        async def obsidian_rename_move(
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
            
            Returns:
                A message indicating the result of the operation.
            """
            return await obsidian_vault.rename_move_note(old_path, new_path, self.config)

    def _setup_prompts(self):
        # Dynamically register prompts from the prompts directory
        prompts_module.setup_prompts(self.mcp, self.config)

    def start(self):
        self.mcp.run()


def create_server(config: ServerConfig) -> DevAutomationServer:
    """
    Creates and configures the Development Automation Server.

    Args:
        config: The server configuration.

    Returns:
        The configured FastMCP server instance.
    """
    server = DevAutomationServer(config)
    return server


if __name__ == "__main__":
    # Example usage with configuration from the config module
    config = create_config()  # Use the factory method
    server = create_server(config)
    server.start()