import asyncio
import logging
from dataclasses import dataclass

from fastmcp import Context, FastMCP
from mcp import ClientCapabilities, RootsCapability
from mcp.server.session import ServerSession

import mcps.prompts as prompts_module
import mcps.resources.doc_resource as doc_resource
import mcps.resources.project_resource as project_resource
import mcps.resources.url_resource as url_resource
import mcps.tools.obsidian_vault as obsidian_vault
from mcps.config import ServerConfig, create_config
from mcps.tools.research.lifespan import build_research_lifespan

logger = logging.getLogger("mcps")


@dataclass
class AppContext:
    config: ServerConfig


_WEB_RESEARCH_DESCRIPTION = (
    "Use this tool to find information from the internet — "
    "including recent news, current pricing, live documentation, "
    "and anything that requires up-to-date data beyond your "
    "training knowledge.\n"
    "\n"
    "Trigger this tool when the user asks you to:\n"
    "- Look up, check, or fetch information from a specific "
    "website or URL\n"
    "- Find recent or current information (news, releases, "
    "changelogs, CVEs, announcements)\n"
    "- Compare or research topics that require browsing "
    "the web\n"
    "- Answer questions where the answer may have changed "
    "recently\n"
    "\n"
    "Provide a full sentence describing what to find, "
    "with as much context as possible (timeframe, specific "
    "site, etc.). Example: \"Find the current Claude API "
    "pricing on Anthropic's website\" rather than "
    "\"claude pricing\".\n"
    "\n"
    "Do NOT use this tool for tasks that only require "
    "reading local files, running code, or answering from "
    "existing context.\n"
    "\n"
    "Args:\n"
    "    query (str): question to research\n"
    "\n"
    "Returns:\n"
    "    str: answer found by the researcher agent"
)


class DevAutomationServer:
    def __init__(self, config: ServerConfig):
        self.config = config
        server_lifespan = build_research_lifespan(self.config)
        if config.vault_dir:
            server_lifespan = server_lifespan | obsidian_vault.build_obsidian_lifespan(
                self.config
            )
        self.mcp = FastMCP(
            "Development Automation Server",
            lifespan=server_lifespan,
        )

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

        @self.mcp.resource(
            "resource://test",
            name="test/resource",
            description="Test project resource",
        )
        async def test_resource_handler(context: Context) -> str:
            try:
                session: ServerSession = context.session
                if session.check_client_capability(
                    ClientCapabilities(roots=RootsCapability())
                ):
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

        @self.mcp.tool(name="web_research", description=_WEB_RESEARCH_DESCRIPTION)
        async def web_research(query: str, ctx: Context) -> str:
            researcher = ctx.lifespan_context["researcher"]
            result = await researcher(query)
            answer = result["answer"]
            explanation = result["explanation"]
            sources = result["sources"]

            text = answer
            if explanation:
                text = f"{text}\n\n{explanation}"
            if sources:
                return (
                    f"{text}\n\nSources:\n"
                    + "\n".join(f"- {s}" for s in sources)
                )
            return text

        if self.config.vault_dir:
            obsidian_vault.register_tools(self.mcp)

        prompts_module.setup_prompts(self.mcp, self.config)

    async def start(self):
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
    config = create_config()
    server = create_server(config)
    asyncio.run(server.start())
