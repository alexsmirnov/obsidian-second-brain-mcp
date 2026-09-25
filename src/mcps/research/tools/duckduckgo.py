"""DuckDuckGo HTML search callable."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

import httpx
from lxml import html

from mcps.research.tools.models import SearchResult

__all__ = ["create_duckduckgo_search"]

logger = logging.getLogger(__file__)

DDG_URL = "https://html.duckduckgo.com/html/"

# DDG POST headers — omit br/zstd so httpx can always decompress the
# response using its built-in gzip/deflate support.  origin + referer
# signal a same-origin form submission to DuckDuckGo.
DDG_HEADERS = {
    "accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,*/*;q=0.8"
    ),
    "accept-encoding": "gzip, deflate",
    "accept-language": "en-US,en;q=0.9",
    "cache-control": "no-cache",
    "origin": "https://html.duckduckgo.com",
    "pragma": "no-cache",
    "referer": "https://html.duckduckgo.com/html/",
    "sec-ch-ua": (
        '"Google Chrome";v="141", "Not?A_Brand";v="8",'
        ' "Chromium";v="141"'
    ),
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "same-origin",
    "sec-fetch-user": "?1",
    "upgrade-insecure-requests": "1",
    "user-agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/141.0.0.0 Safari/537.36"
    ),
}


def _duckduckgo_payload(
    query: str,
    *,
    region: str,
    timelimit: str | None,
) -> dict[str, str]:
    payload: dict[str, str] = {"q": query, "b": "", "l": region}
    if timelimit:
        payload["df"] = timelimit
    return payload


def _parse_result_item(item: html.HtmlElement) -> SearchResult | None:
    title = " ".join("".join(item.xpath(".//h2//text()")).split())
    hrefs: list[str] = item.xpath("./a/@href")
    href = hrefs[0] if hrefs else ""
    snippet = " ".join("".join(item.xpath("./a//text()")).split())
    if not href or href.startswith("https://duckduckgo.com/y.js?"):
        return None
    return SearchResult(url=href, title=title, snippet=snippet)


def _parse_duckduckgo_results(html_text: str) -> list[SearchResult]:
    tree = html.fromstring(html_text)
    items = tree.xpath("//div[contains(@class, 'body')]")
    return [
        result
        for item in items
        if (result := _parse_result_item(item)) is not None
    ]


def create_duckduckgo_search(
    *,
    http_client: httpx.AsyncClient | None = None,
    region: str = "us-en",
    timelimit: str | None = None,
) -> Callable[[str], Awaitable[list[SearchResult]]]:
    """Create an async DuckDuckGo HTML search callable."""

    async def search(query: str) -> list[SearchResult]:
        payload = _duckduckgo_payload(
            query,
            region=region,
            timelimit=timelimit,
        )
        try:
            if http_client is not None:
                response = await http_client.post(
                    DDG_URL,
                    data=payload,
                    headers=DDG_HEADERS,
                )
            else:
                async with httpx.AsyncClient(timeout=30.0) as owned:
                    response = await owned.post(
                        DDG_URL,
                        data=payload,
                        headers=DDG_HEADERS,
                    )
            response.raise_for_status()
        except httpx.HTTPError:
            logger.warning("DuckDuckGo search failed for query: %s", query)
            return []
        return _parse_duckduckgo_results(response.text)

    return search
