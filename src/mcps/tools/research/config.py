"""Research configuration factory for LangChain models and tools.

The module exposes a lifespan-friendly builder `build_research_config` that
accepts pre-constructed resource clients (an ``httpx.AsyncClient`` and an
optional boto3 ``bedrock-runtime`` client) so the FastMCP lifespan owns
their lifetime.
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx
from langchain_aws import ChatBedrockConverse
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from mcps.tools.research.tools import (
    SearchResult,
    create_duckduckgo_search,
    create_fetch,
    create_google_search,
)

__all__ = [
    "ResearchConfig",
    "SearchResult",
    "build_research_config",
    "is_google_cse_configured",
]


@dataclass
class ResearchConfig:
    """Configuration containing models and tools for research operations."""

    smart: BaseChatModel
    small: BaseChatModel
    fast: BaseChatModel
    evaluation: BaseChatModel
    search: Callable[[str], Awaitable[list[SearchResult]]]
    fetch: Callable[[str], Awaitable[str]]



def is_google_cse_configured() -> bool:
    """Return True when Google Custom Search credentials are present."""
    return bool(
        os.environ.get("GOOGLE_API_KEY") and os.environ.get("GOOGLE_SEARCH_ID")
    )


ROUTER_MODELS: dict[str, dict[str, Any]] = {
    "smart": {"model": "gemini-pro", "temperature": 1.0},
    "small": {"model": "gemini-flash", "temperature": 1.0},
    "fast": {"model": "gemini-flash-lite", "temperature": 1.0},
    "evaluation": {"model": "gemini-flash-lite", "temperature": 1.0},
}



def _create_model(
    role: str,
    *,
    router_url: str,
    router_key: str,
    http_client: httpx.AsyncClient | None = None,
) -> BaseChatModel:
    """Instantiate an OpenAI-compatible router model for the given role.

    Args:
        role: Logical role key (smart/small/fast/evaluation).
        router_url: Base URL of the litellm proxy (without /v1 suffix).
        router_key: Auth token for the litellm proxy.
        http_client: Shared async httpx client for connection pooling.
    """
    kwargs: dict[str, Any] = {
        **ROUTER_MODELS[role],
        "base_url": f"{router_url.rstrip('/')}/v1",
        "api_key": router_key,
    }
    if http_client is not None:
        kwargs["http_async_client"] = http_client
    return ChatOpenAI(**kwargs)



def create_search_tool(
    *, http_client: httpx.AsyncClient | None = None
) -> Callable[[str], Awaitable[list[SearchResult]]]:
    """Return GoogleSearchTool or DuckDuckGoSearchTool fallback."""
    if is_google_cse_configured():
        return create_google_search(
            os.environ["GOOGLE_API_KEY"],
            os.environ["GOOGLE_SEARCH_ID"],
            http_client=http_client,
        )
    return create_duckduckgo_search(http_client=http_client)


def create_fetch_tool(
    *, http_client: httpx.AsyncClient | None = None
) -> Callable[[str], Awaitable[str]]:
    """Return FetchTool for web content extraction."""
    return create_fetch(http_client=http_client)


def build_research_config(
    *,
    router_url: str,
    router_key: str,
    http_client: httpx.AsyncClient,
) -> ResearchConfig:
    """Build a ResearchConfig using injected resource clients and credentials.

    The FastMCP lifespan is expected to provide a pooled
    ``httpx.AsyncClient`` and — when Bedrock is configured — a boto3
    ``bedrock-runtime`` client. Both are threaded through to models and
    HTTP-speaking tools so they reuse a single connection pool.

    Args:
        router_url: LiteLLM proxy base URL (e.g. http://localhost:4000).
        router_key: LiteLLM proxy auth token.
        http_client: Shared httpx.AsyncClient for connection pooling.
        bedrock_client: Optional boto3 bedrock-runtime client.

    Environment Variables:
        Bedrock: CLAUDE_CODE_USE_BEDROCK=true, AWS credentials
        Google Search: GOOGLE_API_KEY, GOOGLE_SEARCH_ID
    """
    return ResearchConfig(
        smart=_create_model(
            "smart",
            router_url=router_url,
            router_key=router_key,
            http_client=http_client,
        ),
        small=_create_model(
            "small",
            router_url=router_url,
            router_key=router_key,
            http_client=http_client,
        ),
        fast=_create_model(
            "fast",
            router_url=router_url,
            router_key=router_key,
            http_client=http_client,
        ),
        evaluation=_create_model(
            "evaluation",
            router_url=router_url,
            router_key=router_key,
            http_client=http_client,
        ),
        search=create_search_tool(http_client=http_client),
        fetch=create_fetch_tool(http_client=http_client),
    )
