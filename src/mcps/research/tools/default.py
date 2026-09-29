"""Generic httpx page fetcher with content-type based extraction."""

from __future__ import annotations

import logging

import httpx

from mcps.research.tools.common import (
    CHROME_HEADERS,
    ERROR_EMPTY_RESPONSE,
    ERROR_UNSUPPORTED_CONTENT,
    format_source_output,
    request_get,
)
from mcps.research.tools.extract import (
    CONTENT_TYPE_EXTRACTORS,
    extract_without_content_type,
    normalize_content_type,
)

__all__ = ["fetch_default"]

logger = logging.getLogger(__file__)


def _extract(response: httpx.Response) -> str | None:
    """Return extracted text, or raise LookupError for unsupported types."""
    content_type = normalize_content_type(response.headers.get("content-type", ""))
    if not content_type:
        return extract_without_content_type(response)
    extractor = CONTENT_TYPE_EXTRACTORS.get(content_type)
    if extractor is None:
        raise LookupError(content_type)
    return extractor(response)


async def fetch_default(
    url: str,
    *,
    http_client: httpx.AsyncClient | None,
    max_chars: int,
) -> str:
    """Fetch ``url`` with browser-like headers and return its HTML.

    HTTP and transport errors propagate as ``httpx`` exceptions.
    """
    response = await request_get(
        url,
        http_client=http_client,
        headers=CHROME_HEADERS,
    )
    try:
        extracted = _extract(response)
    except Exception:
        logger.warning("Web fetch failed for %s: %s", url, ERROR_UNSUPPORTED_CONTENT)
        return ERROR_UNSUPPORTED_CONTENT
    if extracted is None:
        logger.warning("Web fetch failed for %s: %s", url, ERROR_EMPTY_RESPONSE)
        return ERROR_EMPTY_RESPONSE
    return format_source_output(url, extracted, max_chars)
