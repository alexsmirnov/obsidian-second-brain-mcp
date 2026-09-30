"""arXiv fetcher: tries HTML, then PDF, then abstract page."""

from __future__ import annotations

import logging
from urllib.parse import urlparse

import httpx

from mcps.research.tools.common import (
    ERROR_UNSUPPORTED_CONTENT,
    extract_hostname,
    to_error_message,
)
from mcps.research.tools.default import fetch_default

__all__ = ["fetch_arxiv", "is_arxiv_url"]

logger = logging.getLogger(__name__)


def is_arxiv_url(url: str) -> bool:
    return extract_hostname(url).endswith("arxiv.org")


def _extract_arxiv_id(url: str) -> str | None:
    path_parts = [part for part in urlparse(url).path.split("/") if part]
    if len(path_parts) < 2 or path_parts[0] not in {"abs", "pdf", "html"}:
        return None
    paper_id = path_parts[1].removesuffix(".pdf")
    return paper_id or None


async def _try_fetch(
    url: str, *, http_client: httpx.AsyncClient | None, max_chars: int
) -> str:
    try:
        return await fetch_default(url, http_client=http_client, max_chars=max_chars)
    except httpx.HTTPError as error:
        return to_error_message(error)
    except Exception:
        return ERROR_UNSUPPORTED_CONTENT


async def fetch_arxiv(
    url: str,
    *,
    http_client: httpx.AsyncClient | None,
    max_chars: int,
) -> str:
    arxiv_id = _extract_arxiv_id(url)
    if arxiv_id is None:
        return await fetch_default(url, http_client=http_client, max_chars=max_chars)
    result = ERROR_UNSUPPORTED_CONTENT
    for kind in ("html", "pdf", "abs"):
        result = await _try_fetch(
            f"https://arxiv.org/{kind}/{arxiv_id}",
            http_client=http_client,
            max_chars=max_chars,
        )
        if not result.startswith("ERROR:"):
            return result
    logger.warning("arXiv fetch failed for %s: %s", url, result)
    return result
