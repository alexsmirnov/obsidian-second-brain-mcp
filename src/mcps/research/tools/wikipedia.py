"""Wikipedia fetcher: raw wikitext served with ``action=raw``."""

from __future__ import annotations

import logging
from dataclasses import replace
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from mcps.research.tools.common import failure
from mcps.research.tools.models import Fetch, FetchResult, FetchStatus

__all__ = ["WIKIPEDIA_HEADERS", "WikipediaFetch", "is_wikipedia_url"]

logger = logging.getLogger(__name__)

_DOMAIN = "wikipedia.org"

# Wikimedia requires an identifying User-Agent
WIKIPEDIA_HEADERS = {"User-Agent": "WebResearchBot/1.0 (asmirnov@alexsmirnov.link)"}


def is_wikipedia_url(url: str) -> bool:
    hostname = urlparse(url).hostname or ""
    return hostname == _DOMAIN or hostname.endswith(f".{_DOMAIN}")


def _with_raw_action(url: str) -> str:
    """Request the page source: set ``action=raw`` and drop the fragment."""
    parsed = urlparse(url)
    params = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key != "action"
    ]
    params.append(("action", "raw"))
    return urlunparse(parsed._replace(query=urlencode(params), fragment=""))


class WikipediaFetch:
    """Raw article source for ``*.wikipedia.org`` URLs, as Markdown."""

    def __init__(self, http: Fetch) -> None:
        self._http = http

    async def __call__(self, url: str, query: str | None = None, /) -> FetchResult:
        result = await self._http(_with_raw_action(url), query)
        if not result.ok:
            return replace(result, url=url)
        content = result.content.strip()
        if not content:
            logger.warning("Wikipedia fetch failed for %s: empty", url)
            return failure(url, FetchStatus.EMPTY)
        return replace(result, url=url, content=content)
