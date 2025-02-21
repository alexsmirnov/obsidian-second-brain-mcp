from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from mcp.server.fastmcp import FastMCP

import mcps.prompts as prompts_module
import mcps.resources.url_resource as url_resource
import mcps.resources.doc_resource as doc_resource
import mcps.resources.project_resource as project_resource
import mcps.tools.internet_search as internet_search
import mcps.tools.perplexity_search as perplexity_search
from mcps.config import ServerConfig, create_config  # Import from config module


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
        async def test_resource_handler() -> str:
            return "Test project resource"
        @self.mcp.resource("docs://test/docs")
        async def test_docs_handler() -> str:
            return "Test project documentation"

    def _setup_tools(self):
        @self.mcp.tool()
        async def web_search(query: str) -> str:
            return await internet_search.do_search(query, self.config)

        @self.mcp.tool()
        async def perplexity_summary_search(query: str) -> str:
            return await perplexity_search.do_search(query, self.config)

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