"""Fallback fetcher using the Bright Data Web Unlocker API (HTML output)."""

from __future__ import annotations

import logging

import httpx

from mcps.research.tools.common import (
    ERROR_EMPTY_RESPONSE,
    ERROR_FETCHER_UNAVAILABLE,
    Retrieve,
    format_source_output,
    http_status_error,
)

__all__ = ["BRIGHT_DATA_URL", "create_bright_data_fetch"]

logger = logging.getLogger(__file__)

BRIGHT_DATA_URL = "https://api.brightdata.com/request"


def _to_fetch_result(url: str, response: httpx.Response, max_chars: int) -> str:
    """Map a ``format=json`` response ``{status_code, headers, body}``.

    A non-2xx API status or a malformed envelope is a provider failure; the
    envelope's ``status_code`` is the target's.
    """
    try:
        response.raise_for_status()
        payload = response.json()
        target_status = int(payload["status_code"])
        body = str(payload.get("body") or "")
    except (httpx.HTTPStatusError, ValueError, KeyError, TypeError) as error:
        logger.warning("Bright Data failed for %s: %r", url, error)
        return ERROR_FETCHER_UNAVAILABLE
    if target_status >= 400:
        return http_status_error(target_status)
    content = body.strip()
    if not content:
        return ERROR_EMPTY_RESPONSE
    return format_source_output(url, content, max_chars)


def create_bright_data_fetch(
    api_key: str,
    zone: str,
    *,
    http_client: httpx.AsyncClient,
    max_chars: int = 15000,
    api_url: str = BRIGHT_DATA_URL,
) -> Retrieve:
    """Create a fetch callable routing requests through a Web Unlocker zone."""
    headers = {"Authorization": f"Bearer {api_key}"}

    async def fetch(url: str) -> str:
        body = {"zone": zone, "url": url, "format": "json"}
        try:
            response = await http_client.post(api_url, headers=headers, json=body)
        except httpx.HTTPError as error:
            logger.warning("Bright Data unavailable for %s: %r", url, error)
            return ERROR_FETCHER_UNAVAILABLE
        return _to_fetch_result(url, response, max_chars)

    return fetch
