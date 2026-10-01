"""Generic httpx page fetcher with content-type based extraction."""

from __future__ import annotations

import logging

import httpx

from mcps.research.tools.common import (
    CHROME_HEADERS,
    failure,
    request_get,
    safe_fetch,
)
from mcps.research.tools.extract import (
    CONTENT_TYPE_EXTRACTORS,
    Extracted,
    extract_without_content_type,
    normalize_content_type,
)
from mcps.research.tools.models import FetchResult, FetchStatus

__all__ = ["HttpFetch"]

logger = logging.getLogger(__name__)


def _extract(response: httpx.Response) -> Extracted:
    """Return extracted text, or raise LookupError for unsupported types."""
    content_type = normalize_content_type(response.headers.get("content-type", ""))
    if not content_type:
        return extract_without_content_type(response)
    extractor = CONTENT_TYPE_EXTRACTORS.get(content_type)
    if extractor is None:
        raise LookupError(content_type)
    return extractor(response)


class HttpFetch:
    """Fetch ``url`` with browser-like headers and return its HTML."""

    def __init__(self, http_client: httpx.AsyncClient | None) -> None:
        self._http_client = http_client

    async def __call__(self, url: str, query: str | None = None, /) -> FetchResult:
        return await safe_fetch(url, self._fetch(url))

    async def _fetch(self, url: str) -> FetchResult:
        response = await request_get(
            url, http_client=self._http_client, headers=CHROME_HEADERS
        )
        try:
            extracted = _extract(response)
        except Exception:
            logger.warning("Web fetch failed for %s: unsupported content", url)
            return failure(url, FetchStatus.UNSUPPORTED)
        if extracted is None:
            logger.warning("Web fetch failed for %s: empty response", url)
            return failure(url, FetchStatus.EMPTY)
        content, mime = extracted
        return FetchResult(url=url, status=FetchStatus.OK, mime=mime, content=content)
