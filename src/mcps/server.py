import argparse
import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

import httpx
from fastmcp import Context, FastMCP

# from mcp import ClientCapabilities, RootsCapability
# from mcp.server.session import ServerSession
# import mcps.prompts as prompts_module
# import mcps.resources.doc_resource as doc_resource
# import mcps.resources.project_resource as project_resource
# import mcps.resources.url_resource as url_resource
import mcps.tools.obsidian_vault as obsidian_vault
from mcps.config import ServerConfig, create_config
from mcps.logs import setup_logging
from mcps.rag.vault import create_vault
from mcps.research.agent import ResearchResponse
from mcps.research.lifespan import build_research_lifespan

logger = logging.getLogger("mcps")


@dataclass
class AppContext:
    config: ServerConfig


_SERVER_INSTRUCTIONS = (
    "This server provides knowledge base tools that search the user's "
    "personal wiki notes (Obsidian Vault) and the internet.\n"
    "\n"
    "Use these tools proactively to verify your assumptions and ground "
    "your responses in the user's own knowledge and current web sources "
    "rather than relying solely on training data.\n"
    "\n"
    "Knowledge base conventions:\n"
    "- Tag taxonomy is documented in the `Tags.md` note — read it with "
    "`obsidian_get_content` to understand available tags before filtering.\n"
    "- Folder structure is documented in the `Folders.md` note — read it "
    "with `obsidian_get_content` to understand how notes are organized."
)

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
            instructions=_SERVER_INSTRUCTIONS,
            lifespan=server_lifespan,
        )

    def register(self):
        # @self.mcp.resource("url://{encoded_url}")
        # async def url_resource_handler(encoded_url: str) -> str:
        #     return await url_resource.get_resource(encoded_url, self.config)

        # @self.mcp.resource("doc://{library_name}")
        # async def doc_resource_handler(library_name: str) -> str:
        #     return await doc_resource.get_resource(library_name, self.config)

        # @self.mcp.resource("project://{project_name}")
        # async def project_resource_handler(project_name: str) -> str:
        #     return await project_resource.get_resource(project_name, self.config)


        @self.mcp.tool(name="web_research", description=_WEB_RESEARCH_DESCRIPTION)
        async def web_research(query: str, ctx: Context) -> ResearchResponse:
            researcher = ctx.lifespan_context["researcher"]
            return await researcher(query)

        if self.config.vault_dir:
            obsidian_vault.register_tools(self.mcp)

        # prompts_module.setup_prompts(self.mcp, self.config)

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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the MCP server."""
    parser = argparse.ArgumentParser(description="Development Automation Server")
    parser.add_argument(
        "--vault",
        type=Path,
        default=None,
        help="Path to the Obsidian vault (overrides VAULT env var)",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Re-index the vault and exit instead of starting the server",
    )
    return parser.parse_args(argv)


async def _run_reindex(config: ServerConfig, http_client: httpx.AsyncClient) -> int:
    """Re-index the configured vault using the factory context manager."""
    async with create_vault(config, http_client) as vault:
        await vault.update_index()
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and either re-index the vault or start the server."""
    args = parse_args(argv)
    config = create_config(vault_dir=args.vault)

    if args.reindex:
        if config.vault_dir is None:
            logger.error(
                "--reindex requires a vault path via --vault or the VAULT env var"
            )
            return 1
        if not config.vault_dir.exists() or not config.vault_dir.is_dir():
            logger.error(
                f"Vault path does not exist or is not a directory: {config.vault_dir}"
            )
            return 1

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        async def _reindex_flow() -> int:
            async with httpx.AsyncClient(
                timeout=30.0,
                follow_redirects=True,
            ) as http_client:
                return await _run_reindex(config, http_client)

        try:
            return asyncio.run(_reindex_flow())
        except Exception as e:
            logger.error(f"Reindex failed: {e}")
            return 1

    setup_logging()
    server = create_server(config)
    asyncio.run(server.start())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
