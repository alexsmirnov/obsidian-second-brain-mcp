"""Reddit fetcher using the public ``.json`` endpoint."""

from __future__ import annotations

import logging

import httpx

from mcps.research.tools.common import (
    CHROME_HEADERS,
    ERROR_EMPTY_RESPONSE,
    extract_hostname,
    format_source_output,
    request_get,
)

__all__ = ["fetch_reddit", "is_reddit_url"]

logger = logging.getLogger(__file__)


def is_reddit_url(url: str) -> bool:
    return extract_hostname(url).endswith("reddit.com")


async def fetch_reddit(
    url: str,
    *,
    http_client: httpx.AsyncClient | None,
    max_chars: int,
) -> str:
    response = await request_get(
        f"{url.rstrip('/')}.json",
        http_client=http_client,
        headers=CHROME_HEADERS,
    )
    children = response.json()[0]["data"]["children"]
    post_data = children[0]["data"] if children else {}
    title = str(post_data.get("title", "")).strip()
    selftext = str(post_data.get("selftext", "")).strip()
    if not title and not selftext:
        logger.warning("Reddit fetch failed for %s: %s", url, ERROR_EMPTY_RESPONSE)
        return ERROR_EMPTY_RESPONSE
    markdown = f"# {title}\n\n{selftext}".strip()
    return format_source_output(url, markdown, max_chars)
