import dataclasses
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastmcp import FastMCP
from fastmcp.server.lifespan import Lifespan, lifespan

from mcps.config import ServerConfig
from mcps.research.agent import create_researcher
from mcps.research.config import build_research_config
from mcps.research.tools.browser import browser_endpoint


def build_research_lifespan(config: ServerConfig) -> Lifespan:
    @lifespan
    async def research_lifespan(server: FastMCP) -> AsyncIterator[dict[str, Any]]:
        async with httpx.AsyncClient(
            timeout=30.0, follow_redirects=True
        ) as http_client:
            async with browser_endpoint(config.browser_cdp_url) as cdp_url:
                if cdp_url is None:
                    # Only web_research depends on a browser; keep every other
                    # tool available.
                    server.disable(names={"web_research"})
                    yield {"researcher": None, "http_client": http_client}
                    return
                server.enable(names={"web_research"})
                research_config = build_research_config(
                    dataclasses.replace(config, browser_cdp_url=cdp_url),
                    http_client=http_client,
                )
                researcher = create_researcher(
                    research_config, implementation="deep_research"
                )
                yield {"researcher": researcher, "http_client": http_client}

    return research_lifespan
