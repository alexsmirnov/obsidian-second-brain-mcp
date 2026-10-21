"""Research configuration factory for LangChain models and tools.

The module exposes a lifespan-friendly async context manager
``build_research_config`` that accepts a pre-constructed ``ServerConfig`` and an
``httpx.AsyncClient`` so the FastMCP lifespan owns the connection pool. It
enters the fetch tool (which owns the browser) and yields ``None`` when no
browser is available.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncGenerator

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from mcps.config import ServerConfig
from mcps.research.tools import (
    Fetch,
    Search,
    SearchResult,
    create_duckduckgo_search,
    create_google_search,
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


def _is_google_cse_configured(config: ServerConfig) -> bool:
    """Return True when Google Custom Search credentials are present."""
    return bool(config.google_api_key and config.google_search_id)


def _create_chat_model(
    *,
    model_name: str,
    router_url: str,
    router_key: str,
    http_client: httpx.AsyncClient | None = None,
) -> BaseChatModel:
    """Instantiate a router-backed chat model.

    Gemini models are instantiated the router URL
    Args:
        model_name: Model identifier (e.g. "gemini-flash", "gpt-4o").
        router_url: Base URL of the OpenAI-compatible model router.
        router_key: Auth token for the model router.
        http_client: Shared async httpx client for connection pooling
    """
    if "gemini" in model_name:
        return ChatGoogleGenerativeAI(
            model=model_name,
            base_url=router_url,
            google_api_key=SecretStr(router_key),
        )
    return ChatOpenAI(
        model=model_name,
        base_url=router_url,
        api_key=SecretStr(router_key),
        http_async_client=http_client,
    )


def create_search_tool(
    *,
    config: ServerConfig,
    http_client: httpx.AsyncClient | None = None,
) -> Search:
    """Return the Google search when configured, else DuckDuckGo."""
    if _is_google_cse_configured(config):
        return create_google_search(
            config.google_api_key,
            config.google_search_id,
            http_client=http_client,
        )
    return create_duckduckgo_search(http_client=http_client)


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
            fast=_create_chat_model(
                model_name=config.research_fast_model,
                router_url=config.router_api_base,
                router_key=config.router_api_key,
                http_client=http_client,
            ),
            small=_create_chat_model(
                model_name=config.research_infer_model,
                router_url=config.router_api_base,
                router_key=config.router_api_key,
                http_client=http_client,
            ),
            search=create_search_tool(config=config, http_client=http_client),
            fetch=fetch,
        )
