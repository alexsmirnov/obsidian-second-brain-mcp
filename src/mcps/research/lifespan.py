from collections.abc import AsyncIterator
from datetime import timedelta
from typing import Any

import httpx
from fastmcp import FastMCP
from fastmcp.server.lifespan import Lifespan, lifespan

from mcps.config import ServerConfig
from mcps.research.agent import create_researcher
from mcps.research.config import build_research_config
from mcps.research.jobs import InMemoryJobStore, ResearchJobs

_RESEARCH_TOOLS = {"web_research", "web_research_status"}


def build_research_lifespan(config: ServerConfig) -> Lifespan:
    @lifespan
    async def research_lifespan(server: FastMCP) -> AsyncIterator[dict[str, Any]]:
        async with httpx.AsyncClient(
            timeout=30.0, follow_redirects=True
        ) as http_client:
            async with build_research_config(config, http_client) as research_config:
                if research_config is None:
                    # Only the research tools depend on a browser; keep every
                    # other tool available.
                    server.disable(names=_RESEARCH_TOOLS)
                    yield {
                        "researcher": None,
                        "research_jobs": None,
                        "http_client": http_client,
                    }
                    return
                server.enable(names=_RESEARCH_TOOLS)
                researcher = create_researcher(
                    research_config, implementation="deep_research"
                )
                jobs = ResearchJobs(
                    researcher,
                    InMemoryJobStore(
                        ttl=timedelta(seconds=config.research_job_ttl_seconds)
                    ),
                    wait=timedelta(seconds=config.research_wait_seconds),
                )
                try:
                    yield {
                        "researcher": researcher,
                        "research_jobs": jobs,
                        "http_client": http_client,
                    }
                finally:
                    # Cancel running jobs while the browser is still open.
                    await jobs.aclose()

    return research_lifespan
