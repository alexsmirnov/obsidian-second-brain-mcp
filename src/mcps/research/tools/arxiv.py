"""arXiv fetcher: tries HTML, then PDF, then abstract page."""

from __future__ import annotations

import logging
from dataclasses import replace
from urllib.parse import urlparse

from mcps.research.tools.common import extract_hostname, failure
from mcps.research.tools.models import Fetch, FetchResult, FetchStatus

__all__ = ["ArxivFetch", "is_arxiv_url"]

logger = logging.getLogger(__name__)


def is_arxiv_url(url: str) -> bool:
    return extract_hostname(url).endswith("arxiv.org")


def _extract_arxiv_id(url: str) -> str | None:
    path_parts = [part for part in urlparse(url).path.split("/") if part]
    if len(path_parts) < 2 or path_parts[0] not in {"abs", "pdf", "html"}:
        return None
    paper_id = path_parts[1].removesuffix(".pdf")
    return paper_id or None


class ArxivFetch:
    """Tries the HTML, then PDF, then abstract page of an arXiv paper."""

    def __init__(self, http: Fetch) -> None:
        self._http = http

    async def __call__(self, url: str, query: str | None = None, /) -> FetchResult:
        arxiv_id = _extract_arxiv_id(url)
        if arxiv_id is None:
            return await self._http(url, query)
        result = failure(url, FetchStatus.UNSUPPORTED)
        for kind in ("html", "pdf", "abs"):
            attempt = await self._http(f"https://arxiv.org/{kind}/{arxiv_id}", query)
            # Links resolve against the requested URL, not the variant fetched.
            result = replace(attempt, url=url)
            if result.ok:
                return result
        logger.warning("arXiv fetch failed for %s: %s", url, result.status)
        return result
