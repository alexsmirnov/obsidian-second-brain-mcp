"""Fallback fetcher using the Bright Data Web Unlocker API (HTML output)."""

from __future__ import annotations

import logging

import httpx

from mcps.research.tools.common import MIME_HTML, failure
from mcps.research.tools.result import FetchResult, FetchStatus

__all__ = ["BRIGHT_DATA_URL", "BrightDataFetch"]

logger = logging.getLogger(__name__)

BRIGHT_DATA_URL = "https://api.brightdata.com/request"


def _to_fetch_result(url: str, response: httpx.Response) -> FetchResult:
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
        return failure(url, FetchStatus.UNAVAILABLE)
    if target_status >= 400:
        return failure(url, FetchStatus.HTTP_ERROR, target_status)
    content = body.strip()
    if not content:
        return failure(url, FetchStatus.EMPTY)
    return FetchResult(
        url=url,
        status=FetchStatus.OK,
        mime=MIME_HTML,
        content=content,
    )


class BrightDataFetch:
    """Fetch routing requests through a Bright Data Web Unlocker zone."""

    def __init__(
        self,
        api_key: str,
        zone: str,
        *,
        http_client: httpx.AsyncClient,
        api_url: str = BRIGHT_DATA_URL,
    ) -> None:
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._zone = zone
        self._http_client = http_client
        self._api_url = api_url

    async def __call__(self, url: str, query: str | None = None, /) -> FetchResult:
        body = {"zone": self._zone, "url": url, "format": "json"}
        try:
            response = await self._http_client.post(
                self._api_url, headers=self._headers, json=body, timeout=60.0
            )
        except httpx.HTTPError as error:
            logger.warning("Bright Data unavailable for %s: %r", url, error)
            return failure(url, FetchStatus.UNAVAILABLE)
        return _to_fetch_result(url, response)
