"""Fallback fetcher using the Scrape.do unblocking API (rendered HTML output)."""

from __future__ import annotations

import logging

import httpx

from mcps.research.tools.common import MIME_HTML, failure
from mcps.research.tools.result import FetchResult, FetchStatus

__all__ = ["SCRAPE_DO_URL", "ScrapeDoFetch"]

logger = logging.getLogger(__name__)

SCRAPE_DO_URL = "https://api.scrape.do/"

# Scrape.do's own failures, not the target's: 401 = no credits/suspended,
# 429 = concurrency limit, 5xx = request failed after retries (not charged).
_PROVIDER_FAILURE_CODES = frozenset([401, 429])


def _is_provider_failure(status_code: int) -> bool:
    return status_code in _PROVIDER_FAILURE_CODES or status_code >= 500


def _to_fetch_result(url: str, response: httpx.Response) -> FetchResult:
    if _is_provider_failure(response.status_code):
        logger.warning(
            "Scrape.do failed for %s: %d %s",
            url,
            response.status_code,
            response.text[:200],
        )
        return failure(url, FetchStatus.UNAVAILABLE)
    if response.status_code >= 400:
        return failure(url, FetchStatus.HTTP_ERROR, response.status_code)
    content = response.text.strip()
    if not content:
        return failure(url, FetchStatus.EMPTY)
    return FetchResult(
        url=url,
        status=FetchStatus.OK,
        mime=MIME_HTML,
        content=content,
    )


class ScrapeDoFetch:
    """Fetch routing requests through Scrape.do.

    Uses premium proxies (``super``) and headless rendering (``render``) since
    it only runs for pages that already blocked a direct request.
    """

    def __init__(
        self,
        token: str,
        *,
        http_client: httpx.AsyncClient,
        api_url: str = SCRAPE_DO_URL,
    ) -> None:
        self._token = token
        self._http_client = http_client
        self._api_url = api_url

    async def __call__(self, url: str, query: str | None = None, /) -> FetchResult:
        params = {
            "token": self._token,
            "url": url,
            "super": "true",
            "render": "true",
        }
        try:
            response = await self._http_client.get(self._api_url, params=params)
        except httpx.HTTPError as error:
            logger.warning("Scrape.do unavailable for %s: %r", url, error)
            return failure(url, FetchStatus.UNAVAILABLE)
        return _to_fetch_result(url, response)
