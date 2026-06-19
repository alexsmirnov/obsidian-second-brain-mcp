"""Research configuration factory for LangChain models and tools.

The module exposes a lifespan-friendly builder `build_research_config` that
accepts a pre-constructed `ServerConfig` instance and an ``httpx.AsyncClient``
so the FastMCP lifespan owns the connection pool.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from mcps.config import ServerConfig
from mcps.research.tools import (
    SearchResult,
    create_duckduckgo_search,
    create_fetch,
    create_google_search,
)

__all__ = [
    "ResearchConfig",
    "SearchResult",
    "build_research_config",
]


@dataclass
class ResearchConfig:
    """Configuration containing models and tools for research operations."""

    fast: BaseChatModel
    small: BaseChatModel
    search: Callable[[str], Awaitable[list[SearchResult]]]
    fetch: Callable[[str], Awaitable[str]]


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
    """Instantiate an OpenAI-compatible router model.

    Args:
        model_name: Model identifier passed to ChatOpenAI.
        router_url: Base URL of the litellm proxy (without /v1 suffix).
        router_key: Auth token for the litellm proxy.
        http_client: Shared async httpx client for connection pooling.
    """
    kwargs: dict[str, Any] = {
        "model": model_name,
        "base_url": f"{router_url.rstrip('/')}/v1",
        "api_key": router_key,
    }
    if http_client is not None:
        kwargs["http_async_client"] = http_client
    return ChatOpenAI(**kwargs)


def create_search_tool(
    *,
    config: ServerConfig,
    http_client: httpx.AsyncClient | None = None,
) -> Callable[[str], Awaitable[list[SearchResult]]]:
    """Return GoogleSearchTool or DuckDuckGoSearchTool fallback."""
    if _is_google_cse_configured(config):
        return create_google_search(
            config.google_api_key,
            config.google_search_id,
            http_client=http_client,
        )
    return create_duckduckgo_search(http_client=http_client)


def create_fetch_tool(
    *, http_client: httpx.AsyncClient | None = None
) -> Callable[[str], Awaitable[str]]:
    """Return FetchTool for web content extraction."""
    return create_fetch(http_client=http_client)


def build_research_config(
    config: ServerConfig,
    *,
    http_client: httpx.AsyncClient,
) -> ResearchConfig:
    """Build a ResearchConfig using injected ServerConfig and HTTP client.

    The FastMCP lifespan is expected to provide a pooled
    ``httpx.AsyncClient``. It is threaded through to models and the
    HTTP-speaking tools so they reuse a single connection pool.

    Args:
        config: Populated ServerConfig instance.
        http_client: Shared httpx.AsyncClient for connection pooling.
    """
    return ResearchConfig(
        fast=_create_chat_model(
            model_name=config.research_fast_model,
            router_url=config.litellm_router,
            router_key=config.litellm_router_key,
            http_client=http_client,
        ),
        small=_create_chat_model(
            model_name=config.research_infer_model,
            router_url=config.litellm_router,
            router_key=config.litellm_router_key,
            http_client=http_client,
        ),
        search=create_search_tool(config=config, http_client=http_client),
        fetch=create_fetch_tool(http_client=http_client),
    )
