"""Contract tests for fallback fetchers: CDP browser, Scrape.do, Bright Data."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from pytest_httpx import HTTPXMock

from mcps.config import ServerConfig
from mcps.research.config import create_fetch_tool
from mcps.research.tools.bright_data import create_bright_data_fetch
from mcps.research.tools.browser import create_browser_fetch
from mcps.research.tools.scrape_do import create_scrape_do_fetch

TARGET = "https://blocked.example/article"


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(follow_redirects=True) as http_client:
        yield http_client


# ---------------------------------------------------------------------------
# Browser (crawl4ai over CDP)
# ---------------------------------------------------------------------------


@dataclass
class FakeCrawler:
    """Async-context crawler stub returning a canned crawl result."""

    result: Any = None
    error: Exception | None = None
    urls: list[str] = field(default_factory=list)

    async def __aenter__(self) -> FakeCrawler:
        if self.error is not None:
            raise self.error
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    async def arun(self, url: str, **_kwargs: Any) -> Any:
        self.urls.append(url)
        return self.result


def crawl_result(
    *,
    success: bool = True,
    status_code: int | None = 200,
    cleaned_html: str = "",
) -> SimpleNamespace:
    return SimpleNamespace(
        success=success,
        status_code=status_code,
        cleaned_html=cleaned_html,
        error_message="" if success else "boom",
    )


async def test_browser_fetch_returns_cleaned_html():
    crawler = FakeCrawler(result=crawl_result(cleaned_html="<h1>Rendered</h1>"))
    fetch = create_browser_fetch("ws://cdp", crawler_factory=lambda: crawler)

    result = await fetch(TARGET)

    assert result == "<h1>Rendered</h1>"
    assert crawler.urls == [TARGET]


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (crawl_result(status_code=403, cleaned_html="denied"), "ERROR: http code 403"),
        (crawl_result(status_code=404, cleaned_html="gone"), "ERROR: http code 404"),
        (crawl_result(cleaned_html="   "), "ERROR: empty response"),
        (crawl_result(success=False, status_code=None), "ERROR: fetcher unavailable"),
    ],
)
async def test_browser_fetch_maps_crawl_outcomes(result: Any, expected: str):
    fetch = create_browser_fetch(
        "ws://cdp", crawler_factory=lambda: FakeCrawler(result=result)
    )

    assert await fetch(TARGET) == expected


async def test_browser_fetch_connection_failure_is_unavailable():
    crawler = FakeCrawler(error=ConnectionError("cdp down"))
    fetch = create_browser_fetch("ws://cdp", crawler_factory=lambda: crawler)

    assert await fetch(TARGET) == "ERROR: fetcher unavailable"


@pytest.mark.parametrize(
    ("cleaned_html", "expected"),
    [
        ("", "ERROR: empty response"),
        ("   ", "ERROR: empty response"),
    ],
)
async def test_browser_fetch_blank_cleaned_html_is_empty(
    cleaned_html: str, expected: str
):
    crawler = FakeCrawler(result=crawl_result(cleaned_html=cleaned_html))
    fetch = create_browser_fetch("ws://cdp", crawler_factory=lambda: crawler)

    assert await fetch(TARGET) == expected


# ---------------------------------------------------------------------------
# Scrape.do
# ---------------------------------------------------------------------------


async def test_scrape_do_requests_rendered_html_with_unblocking(
    httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    httpx_mock.add_response(text="<h1>Article</h1>")
    fetch = create_scrape_do_fetch("tok", http_client=client)

    result = await fetch(TARGET)

    request = httpx_mock.get_request()
    assert request is not None
    assert (request.method, request.url.host) == ("GET", "api.scrape.do")
    assert dict(request.url.params) == {
        "token": "tok",
        "url": TARGET,
        "super": "true",
        "render": "true",
    }
    assert result == "<h1>Article</h1>"


@pytest.mark.parametrize(
    ("status_code", "text", "expected"),
    [
        (200, "  ", "ERROR: empty response"),
        (404, "not found", "ERROR: http code 404"),
        (400, "bad target", "ERROR: http code 400"),
        (401, "no credits", "ERROR: fetcher unavailable"),
        (429, "concurrency", "ERROR: fetcher unavailable"),
        (502, "failed", "ERROR: fetcher unavailable"),
    ],
)
async def test_scrape_do_maps_api_status(
    httpx_mock: HTTPXMock,
    client: httpx.AsyncClient,
    status_code: int,
    text: str,
    expected: str,
):
    httpx_mock.add_response(status_code=status_code, text=text)
    fetch = create_scrape_do_fetch("tok", http_client=client)

    assert await fetch(TARGET) == expected


async def test_scrape_do_transport_error_is_unavailable(
    httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    httpx_mock.add_exception(httpx.ConnectError("refused"))
    fetch = create_scrape_do_fetch("tok", http_client=client)

    assert await fetch(TARGET) == "ERROR: fetcher unavailable"


# ---------------------------------------------------------------------------
# Bright Data Web Unlocker
# ---------------------------------------------------------------------------


async def test_bright_data_posts_zone_request_for_html(
    httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    httpx_mock.add_response(json={"status_code": 200, "body": "<h1>Article</h1>"})
    fetch = create_bright_data_fetch("key", "unlocker", http_client=client)

    result = await fetch(TARGET)

    request = httpx_mock.get_request()
    assert request is not None
    assert (request.method, str(request.url)) == (
        "POST",
        "https://api.brightdata.com/request",
    )
    assert request.headers["authorization"] == "Bearer key"
    assert json.loads(request.content) == {
        "zone": "unlocker",
        "url": TARGET,
        "format": "json",
    }
    assert result == "<h1>Article</h1>"


@pytest.mark.parametrize(
    ("status_code", "payload", "expected"),
    [
        (200, {"status_code": 403, "body": "denied"}, "ERROR: http code 403"),
        (200, {"status_code": 200, "body": ""}, "ERROR: empty response"),
        (200, {"unexpected": True}, "ERROR: fetcher unavailable"),
        (401, {"error": "bad key"}, "ERROR: fetcher unavailable"),
        (502, {"error": "upstream"}, "ERROR: fetcher unavailable"),
    ],
)
async def test_bright_data_maps_outcomes(
    httpx_mock: HTTPXMock,
    client: httpx.AsyncClient,
    status_code: int,
    payload: dict[str, Any],
    expected: str,
):
    httpx_mock.add_response(status_code=status_code, json=payload)
    fetch = create_bright_data_fetch("key", "unlocker", http_client=client)

    assert await fetch(TARGET) == expected


# ---------------------------------------------------------------------------
# Config wiring
# ---------------------------------------------------------------------------


async def test_fetch_tool_escalates_blocked_page_to_configured_provider(
    httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    httpx_mock.add_response(url=TARGET, status_code=403)
    httpx_mock.add_response(
        url=httpx.URL(
            "https://api.scrape.do/",
            params={
                "token": "t",
                "url": TARGET,
                "super": "true",
                "render": "true",
            },
        ),
        text="<html><body><h1>Unblocked</h1></body></html>",
    )
    config = ServerConfig(
        scraper_provider="scrape_do", scrape_do_token="t", browser_cdp_url=""
    )
    fetch = create_fetch_tool(config=config, http_client=client)

    assert await fetch(TARGET, None) == "# Unblocked"
