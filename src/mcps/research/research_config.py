"""Research configuration factory for LangChain models and tools.

The module exposes a lifespan-friendly async context manager
``build_research_config`` that accepts a pre-constructed ``ServerConfig`` and an
``httpx.AsyncClient`` so the FastMCP lifespan owns the connection pool. It
enters the fetch tool (which owns the browser) and yields ``None`` when no
browser is available.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx
from langchain_core.language_models import BaseChatModel

from mcps.common.ai_models import create_chat_model
from mcps.config import ServerConfig
from mcps.research.tools import (
    Fetch,
    Search,
    SearchResult,
    create_google_search,
    create_yahoo_search,
)
from mcps.research.tools.fetch import build_fetch_tool

__all__ = [
    "ResearchConfig",
    "SearchResult",
    "build_research_config",
]

logger = logging.getLogger(__name__)


@dataclass
class ResearchConfig:
    """Configuration containing models and tools for research operations."""

    fast: BaseChatModel
    small: BaseChatModel
    search: Search
    fetch: Fetch
    # Hostnames (and subdomains) the fetch tool refuses; not offered for triage.
    restricted_domains: tuple[str, ...] = ()


def _is_google_cse_configured(config: ServerConfig) -> bool:
    """Return True when Google Custom Search credentials are present."""
    return bool(config.google.api_key and config.google.search_id)


def create_search_tool(
    *,
    config: ServerConfig,
    http_client: httpx.AsyncClient | None = None,
) -> Search:
    """Return the Google search when configured, else Yahoo."""
    if _is_google_cse_configured(config):
        return create_google_search(
            config.google.api_key,
            config.google.search_id,
            http_client=http_client,
        )
    return create_yahoo_search(http_client=http_client)


@asynccontextmanager
async def build_research_config(
    config: ServerConfig,
    http_client: httpx.AsyncClient,
) -> AsyncGenerator[ResearchConfig | None]:
    """Build a ResearchConfig for the duration of the context, or ``None``.

    Enters :func:`build_fetch_tool` in required-browser mode first: when no
    browser is available the context yields ``None`` without constructing any
    models. Otherwise the models and search tool are built once and yielded for
    the lifetime of the context. The lifespan provides the pooled
    ``httpx.AsyncClient``; it is borrowed and never closed here.
    """
    async with build_fetch_tool(
        config, http_client, require_browser=True
    ) as fetch:
        if fetch is None:
            yield None
            return
        yield ResearchConfig(
            fast=create_chat_model(
                config.research.fast_model,
                config.router,
                http_client,
            ),
            small=create_chat_model(
                config.research.infer_model,
                config.router,
                http_client,
            ),
            search=create_search_tool(config=config, http_client=http_client),
            fetch=fetch,
            restricted_domains=config.fetch.restricted_domains,
        )
