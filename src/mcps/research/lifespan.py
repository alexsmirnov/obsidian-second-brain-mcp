from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastmcp import FastMCP
from fastmcp.server.lifespan import Lifespan, lifespan

from mcps.config import ServerConfig
from mcps.research.agent import create_researcher
from mcps.research.config import build_research_config


def build_research_lifespan(config: ServerConfig) -> Lifespan:
    @lifespan
    async def research_lifespan(server: FastMCP) -> AsyncIterator[dict[str, Any]]:
        async with httpx.AsyncClient(
            timeout=30.0, follow_redirects=True
        ) as http_client:
            async with build_research_config(config, http_client) as research_config:
                if research_config is None:
                    # Only web_research depends on a browser; keep every other
                    # tool available.
                    server.disable(names={"web_research"})
                    yield {"researcher": None, "http_client": http_client}
                    return
                server.enable(names={"web_research"})
                researcher = create_researcher(
                    research_config, implementation="deep_research"
                )
                yield {"researcher": researcher, "http_client": http_client}

    return research_lifespan
