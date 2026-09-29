"""Fallback fetcher using the Scrape.do unblocking API (markdown output)."""

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

__all__ = ["SCRAPE_DO_URL", "create_scrape_do_fetch"]

logger = logging.getLogger(__file__)

SCRAPE_DO_URL = "https://api.scrape.do/"

# Scrape.do's own failures, not the target's: 401 = no credits/suspended,
# 429 = concurrency limit, 5xx = request failed after retries (not charged).
_PROVIDER_FAILURE_CODES = frozenset([401, 429])


def _is_provider_failure(status_code: int) -> bool:
    return status_code in _PROVIDER_FAILURE_CODES or status_code >= 500


def _to_fetch_result(url: str, response: httpx.Response, max_chars: int) -> str:
    if _is_provider_failure(response.status_code):
        logger.warning(
            "Scrape.do failed for %s: %d %s",
            url,
            response.status_code,
            response.text[:200],
        )
        return ERROR_FETCHER_UNAVAILABLE
    if response.status_code >= 400:
        return http_status_error(response.status_code)
    content = response.text.strip()
    if not content:
        return ERROR_EMPTY_RESPONSE
    return format_source_output(url, content, max_chars)


def create_scrape_do_fetch(
    token: str,
    *,
    http_client: httpx.AsyncClient,
    max_chars: int = 15000,
    api_url: str = SCRAPE_DO_URL,
) -> Retrieve:
    """Create a fetch callable routing requests through Scrape.do.

    Uses premium proxies (``super``) and headless rendering (``render``) since
    it only runs for pages that already blocked a direct request.
    """

    async def fetch(url: str) -> str:
        params = {
            "token": token,
            "url": url,
            "super": "true",
            "render": "true",
            "output": "markdown",
        }
        try:
            response = await http_client.get(api_url, params=params)
        except httpx.HTTPError as error:
            logger.warning("Scrape.do unavailable for %s: %r", url, error)
            return ERROR_FETCHER_UNAVAILABLE
        return _to_fetch_result(url, response, max_chars)

    return fetch
