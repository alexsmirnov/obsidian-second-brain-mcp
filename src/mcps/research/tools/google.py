"""Google Custom Search Engine search callable."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from mcps.research.tools.models import SearchResult

__all__ = ["create_google_search"]

logger = logging.getLogger(__file__)

GOOGLE_CSE_URL = "https://www.googleapis.com/customsearch/v1"


def _google_search_params(
    api_key: str,
    cse_id: str,
    query: str,
) -> dict[str, Any]:
    return {
        "key": api_key,
        "cx": cse_id,
        "q": query,
        "num": 10,
        "safe": "off",
    }


def _parse_google_results(data: dict[str, Any]) -> list[SearchResult]:
    return [
        SearchResult(
            url=url,
            title=str(item.get("title", "N/A")),
            snippet=str(item.get("snippet", "N/A")),
        )
        for item in data.get("items", [])
        if (url := str(item.get("link", "")).strip())
    ]


def create_google_search(
    api_key: str,
    cse_id: str,
    *,
    http_client: httpx.AsyncClient | None = None,
) -> Callable[[str], Awaitable[list[SearchResult]]]:
    """Create an async Google CSE search callable."""

    async def search(query: str) -> list[SearchResult]:
        if not api_key or not cse_id:
            return []
        try:
            if http_client is not None:
                response = await http_client.get(
                    GOOGLE_CSE_URL,
                    params=_google_search_params(api_key, cse_id, query),
                )
                response.raise_for_status()
                data = response.json()
            else:
                async with httpx.AsyncClient(timeout=30.0) as owned:
                    response = await owned.get(
                        GOOGLE_CSE_URL,
                        params=_google_search_params(api_key, cse_id, query),
                    )
                    response.raise_for_status()
                    data = response.json()
        except httpx.HTTPError:
            logger.warning("Google search failed for query: %s", query)
            return []
        return _parse_google_results(data)

    return search
