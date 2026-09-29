"""Contract tests for fallback fetchers: CDP browser, Scrape.do, Bright Data."""

from __future__ import annotations

import asyncio
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
from mcps.research.tools.browser import browser_crawler, create_browser_fetch
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
    """Crawler stub over a connection that can be opened/closed."""

    result: Any = None
    error: Exception | None = None
    open: bool = True
    opens: int = 0
    closes: int = 0
    urls: list[str] = field(default_factory=list)
    configs: list[Any] = field(default_factory=list)

    async def arun(self, url: str, config: Any = None, **_kwargs: Any) -> Any:
        if not self.open:
            raise RuntimeError("crawler used while its connection is closed")
        self.urls.append(url)
        self.configs.append(config)
        if self.error is not None:
            raise self.error
        return self.result


@dataclass
class FakeCrawlerContext:
    """Async context manager starting and closing a :class:`FakeCrawler`."""

    crawler: FakeCrawler
    enter_error: Exception | None = None
    entered: bool = False
    exited: bool = False

    async def __aenter__(self) -> FakeCrawler:
        if self.enter_error is not None:
            raise self.enter_error
        self.entered = True
        self.crawler.open = True
        self.crawler.opens += 1
        return self.crawler

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.exited = True
        self.crawler.open = False
        self.crawler.closes += 1


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


class GatedCrawler:
    """Crawler whose per-URL waits overlap until released, recording configs."""

    def __init__(self, gates: dict[str, asyncio.Event]):
        self.gates = gates
        self.entered = {url: asyncio.Event() for url in gates}
        self.configs: list[Any] = []

    async def arun(self, url: str, config: Any = None, **_kwargs: Any) -> Any:
        config.url = url
        self.configs.append(config)
        self.entered[url].set()
        async with asyncio.timeout(5):
            await self.gates[url].wait()
        return crawl_result(cleaned_html=f"<h1>{config.url}</h1>")


async def test_browser_fetch_returns_cleaned_html():
    crawler = FakeCrawler(result=crawl_result(cleaned_html="<h1>Rendered</h1>"))
    fetch = create_browser_fetch(crawler)

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
    fetch = create_browser_fetch(FakeCrawler(result=result))

    assert await fetch(TARGET) == expected


async def test_browser_fetch_crawl_failure_is_unavailable():
    crawler = FakeCrawler(error=ConnectionError("cdp down"))
    fetch = create_browser_fetch(crawler)

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
    fetch = create_browser_fetch(
        FakeCrawler(result=crawl_result(cleaned_html=cleaned_html))
    )

    assert await fetch(TARGET) == expected


# ---------------------------------------------------------------------------
# Browser crawler connection lifetime
# ---------------------------------------------------------------------------


async def test_browser_fetch_repeated_requests_reuse_open_connection():
    crawler = FakeCrawler(result=crawl_result(cleaned_html="<h1>X</h1>"))
    context = FakeCrawlerContext(crawler)

    async with browser_crawler(
        "ws://cdp", crawler_factory=lambda: context
    ) as open_crawler:
        assert open_crawler is not None
        assert open_crawler is crawler
        fetch = create_browser_fetch(open_crawler)
        first = await fetch("https://a.example")
        second = await fetch("https://b.example")
        assert crawler.opens == 1 and crawler.closes == 0 and crawler.open

    assert first == second == "<h1>X</h1>"
    assert crawler.urls == ["https://a.example", "https://b.example"]
    assert crawler.closes == 1 and crawler.open is False


async def test_browser_fetch_overlapping_requests_share_connection():
    gates = {"https://a.example": asyncio.Event(), "https://b.example": asyncio.Event()}
    crawler = GatedCrawler(gates)
    fetch = create_browser_fetch(crawler)

    tasks = [asyncio.ensure_future(fetch(url)) for url in gates]
    async with asyncio.timeout(5):
        await asyncio.gather(*(crawler.entered[url].wait() for url in gates))
    assert len(crawler.configs) == 2
    assert crawler.configs[0] is not crawler.configs[1]
    for gate in gates.values():
        gate.set()
    results = await asyncio.gather(*tasks)

    assert results == ["<h1>https://a.example</h1>", "<h1>https://b.example</h1>"]


async def test_browser_fetch_failure_keeps_connection_for_next_request():
    crawler = FakeCrawler(error=ConnectionError("cdp hiccup"))
    context = FakeCrawlerContext(crawler)

    async with browser_crawler(
        "ws://cdp", crawler_factory=lambda: context
    ) as open_crawler:
        assert open_crawler is not None
        fetch = create_browser_fetch(open_crawler)
        assert await fetch("https://a.example") == "ERROR: fetcher unavailable"
        crawler.error = None
        crawler.result = crawl_result(cleaned_html="<h1>Recovered</h1>")
        assert await fetch("https://b.example") == "<h1>Recovered</h1>"
        assert crawler.opens == 1 and crawler.closes == 0

    assert crawler.closes == 1


def _entering_failure() -> FakeCrawlerContext:
    return FakeCrawlerContext(FakeCrawler(), enter_error=ConnectionError("cdp down"))


def _constructing_failure() -> FakeCrawlerContext:
    raise ConnectionError("bad config")


def _missing_endpoint_factory() -> FakeCrawlerContext:
    raise AssertionError("factory must not be called without a CDP URL")


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(_entering_failure, id="enter"),
        pytest.param(_constructing_failure, id="construct"),
    ],
)
async def test_browser_crawler_startup_failure_yields_none(factory: Any):
    async with browser_crawler("ws://cdp", crawler_factory=factory) as crawler:
        assert crawler is None


async def test_browser_crawler_without_endpoint_opens_no_connection():
    async with browser_crawler(
        None, crawler_factory=_missing_endpoint_factory
    ) as crawler:
        assert crawler is None


async def test_browser_crawler_exceptional_exit_closes_connection():
    crawler = FakeCrawler()
    context = FakeCrawlerContext(crawler)

    with pytest.raises(ValueError, match="body boom"):
        async with browser_crawler("ws://cdp", crawler_factory=lambda: context):
            raise ValueError("body boom")

    assert context.exited and crawler.open is False and crawler.closes == 1


async def test_browser_crawler_cancellation_closes_connection():
    crawler = FakeCrawler()
    context = FakeCrawlerContext(crawler)
    entered = asyncio.Event()

    async def consume() -> None:
        async with browser_crawler("ws://cdp", crawler_factory=lambda: context):
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(consume())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert context.exited and crawler.open is False and crawler.closes == 1


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
