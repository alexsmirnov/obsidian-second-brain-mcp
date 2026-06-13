from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastmcp import FastMCP
from fastmcp.server.lifespan import Lifespan, lifespan

from mcps.config import ServerConfig
from mcps.tools.research.agent import create_researcher
from mcps.tools.research.config import build_research_config


def build_research_lifespan(config: ServerConfig) -> Lifespan:
    @lifespan
    async def research_lifespan(server: FastMCP) -> AsyncIterator[dict[str, Any]]:
        async with httpx.AsyncClient(
            timeout=30.0, follow_redirects=True
        ) as http_client:
            research_config = build_research_config(
                router_url=config.litellm_router,
                router_key=config.litellm_router_key,
                http_client=http_client,
            )
            researcher = create_researcher(
                research_config, implementation="deep_research"
            )
            yield {"researcher": researcher, "http_client": http_client}

    return research_lifespan
