"""Wikipedia fetcher using the raw wikitext action."""

from __future__ import annotations

import logging
import re
from urllib.parse import urlparse

import httpx

from mcps.research.tools.common import (
    CHROME_HEADERS,
    ERROR_EMPTY_RESPONSE,
    extract_hostname,
    format_source_output,
    request_get,
)

__all__ = ["fetch_wikipedia", "is_wikipedia_url"]

logger = logging.getLogger(__file__)


def is_wikipedia_url(url: str) -> bool:
    return ".wikipedia.org" in extract_hostname(url)


def _convert_heading(match: re.Match[str]) -> str:
    level = max(1, min(6, len(match.group(1)) - 1))
    return f"{'#' * level} {match.group(2).strip()}"


def mediawiki_to_markdown(raw_wikitext: str) -> str:
    content = raw_wikitext
    content = re.sub(r"<ref[^>]*>.*?</ref>", "", content, flags=re.DOTALL)
    content = re.sub(r"<ref[^>]*/>", "", content)
    content = re.sub(r"\{\{[^{}]*\}\}", "", content)
    content = re.sub(
        r"^(={2,6})\s*(.*?)\s*\1\s*$",
        _convert_heading,
        content,
        flags=re.MULTILINE,
    )
    content = re.sub(r"'''(.*?)'''", r"**\1**", content)
    content = re.sub(r"''(.*?)''", r"*\1*", content)
    content = re.sub(r"\[\[(?:[^\]|]+\|)?([^\]]+)\]\]", r"\1", content)
    content = re.sub(r"^\*(?!\*)\s?", "- ", content, flags=re.MULTILINE)
    return content.strip()


def _wikipedia_raw_url(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}?action=raw"


async def fetch_wikipedia(
    url: str,
    *,
    http_client: httpx.AsyncClient | None,
    max_chars: int,
) -> str:
    response = await request_get(
        _wikipedia_raw_url(url),
        http_client=http_client,
        headers=CHROME_HEADERS,
    )
    markdown = mediawiki_to_markdown(response.text.strip())
    if not markdown:
        logger.warning("Wikipedia fetch failed for %s: %s", url, ERROR_EMPTY_RESPONSE)
        return ERROR_EMPTY_RESPONSE
    return format_source_output(url, markdown, max_chars)
