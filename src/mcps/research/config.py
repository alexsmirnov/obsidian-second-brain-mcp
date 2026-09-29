"""Research configuration factory for LangChain models and tools.

The module exposes a lifespan-friendly builder `build_research_config` that
accepts a pre-constructed `ServerConfig` instance and an ``httpx.AsyncClient``
so the FastMCP lifespan owns the connection pool.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from mcps.config import ServerConfig
from mcps.research.tools import (
    Fetch,
    Retrieve,
    SearchResult,
    create_duckduckgo_search,
    create_fetch,
    create_google_search,
)
from mcps.research.tools.bright_data import create_bright_data_fetch
from mcps.research.tools.browser import CRAWL4AI_AVAILABLE, create_browser_fetch
from mcps.research.tools.filtering import create_page_filter
from mcps.research.tools.scrape_do import create_scrape_do_fetch

__all__ = [
    "ResearchConfig",
    "SearchResult",
    "build_research_config",
    "create_fetch_tool",
]

logger = logging.getLogger(__file__)

# The provider returns untruncated Markdown; fetch filters then truncates once.
_NO_TRUNCATION = 10**9


@dataclass
class ResearchConfig:
    """Configuration containing models and tools for research operations."""

    fast: BaseChatModel
    small: BaseChatModel
    search: Callable[[str], Awaitable[list[SearchResult]]]
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
) -> Callable[[str], Awaitable[list[SearchResult]]]:
    """Return GoogleSearchTool or DuckDuckGoSearchTool fallback."""
    if _is_google_cse_configured(config):
        return create_google_search(
            config.google_api_key,
            config.google_search_id,
            http_client=http_client,
        )
    return create_duckduckgo_search(http_client=http_client)


def _create_provider_fallback(
    config: ServerConfig, http_client: httpx.AsyncClient
) -> Retrieve | None:
    """Build the commercial unblocking provider, or None when unconfigured.

    The provider returns untruncated Markdown: the fetch callable filters and
    truncates the result once, on the shared path.
    """
    match config.scraper_provider:
        case "":
            return None
        case "scrape_do" if config.scrape_do_token:
            return create_scrape_do_fetch(
                config.scrape_do_token,
                http_client=http_client,
                max_chars=_NO_TRUNCATION,
            )
        case "bright_data" if config.bright_data_api_key and config.bright_data_zone:
            return create_bright_data_fetch(
                config.bright_data_api_key,
                config.bright_data_zone,
                http_client=http_client,
                max_chars=_NO_TRUNCATION,
            )
        case "scrape_do" | "bright_data":
            logger.warning(
                "SCRAPER_PROVIDER=%s is missing credentials; provider fallback "
                "disabled.",
                config.scraper_provider,
            )
        case _:
            logger.warning(
                "Unknown SCRAPER_PROVIDER=%s (expected scrape_do or bright_data); "
                "provider fallback disabled.",
                config.scraper_provider,
            )
    return None


def _create_browser(config: ServerConfig) -> Retrieve | None:
    if not config.browser_cdp_url:
        return None
    if not CRAWL4AI_AVAILABLE:
        logger.warning(
            "BROWSER_CDP_URL is set but crawl4ai is not installed; "
            "browser fetch disabled."
        )
        return None
    return create_browser_fetch(config.browser_cdp_url)


def create_fetch_tool(
    *, config: ServerConfig, http_client: httpx.AsyncClient
) -> Fetch:
    """Return the web fetch callable with browser, filtering, and fallback."""
    return create_fetch(
        http_client=http_client,
        browser=_create_browser(config),
        provider=_create_provider_fallback(config, http_client),
        page_filter=create_page_filter(
            fetch_model=config.fetch_model,
            router_url=config.router_api_base,
            router_key=config.router_api_key,
        ),
        restricted_domains=config.fetch_restricted_domains,
        concurrency=config.fetch_concurrency,
    )


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
        fetch=create_fetch_tool(config=config, http_client=http_client),
    )
