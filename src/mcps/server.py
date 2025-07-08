import asyncio
import logging
import os
from dataclasses import dataclass

from fastmcp import Context, FastMCP
from mcp import ClientCapabilities, RootsCapability
from mcp.server.session import ServerSession
from pydantic import Field

import mcps.prompts as prompts_module
import mcps.resources.doc_resource as doc_resource
import mcps.resources.project_resource as project_resource
import mcps.resources.url_resource as url_resource
from mcps.common import Tools
import mcps.tools.internet_search as internet_search
import mcps.tools.obsidian_vault as obsidian_vault
import mcps.tools.perplexity_search as perplexity_search
import mcps.tools.rag_search as rag_search
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
        self.obsidian: Tools = obsidian_vault.ObsidianTools(self.mcp, self.config) if config.vault_dir else Tools()


    def register(self):
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
        self.obsidian.register()

        # Dynamically register prompts from the prompts directory
        prompts_module.setup_prompts(self.mcp, self.config)

    async def start(self):
        async with self.obsidian as o:
            await self.mcp.run_async()


def create_server(config: ServerConfig) -> DevAutomationServer:
    """
    Creates and configures the Development Automation Server.

    Args:
        config: The server configuration.

    Returns:
        The configured FastMCP server instance.
    """
    server = DevAutomationServer(config)
    server.register()
    return server


if __name__ == "__main__":
    # Example usage with configuration from the config module
    config = create_config()  # Use the factory method
    server = create_server(config)
    asyncio.run( server.start() )