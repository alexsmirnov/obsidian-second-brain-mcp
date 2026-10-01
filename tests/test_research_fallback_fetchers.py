"""Contract tests for fallback fetchers: CDP browser, Scrape.do, Bright Data."""

from __future__ import annotations

import asyncio
import json
import shutil
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, ClassVar, cast

import httpx
import pytest
from crawl4ai import AsyncWebCrawler
from pytest_httpx import HTTPXMock

from mcps.config import ServerConfig
from mcps.research.tools import browser as browser_module
from mcps.research.tools.bright_data import BrightDataFetch
from mcps.research.tools.browser import BrowserFetch, create_browser_fetch
from mcps.research.tools.fetch import build_fetch_tool
from mcps.research.tools.models import FetchResult, FetchStatus
from mcps.research.tools.scrape_do import ScrapeDoFetch

TARGET = "https://blocked.example/article"


def outcome(result: FetchResult) -> tuple[FetchStatus, int | None]:
    return result.status, result.http_status


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


@dataclass
class _ConstructFailure:
    """Script item making ``FakeAsyncWebCrawler`` raise while constructing."""

    error: Exception


class FakeAsyncWebCrawler:
    """Boundary fake for ``crawl4ai.AsyncWebCrawler``.

    Each construction consumes the next item of :attr:`script`: ``None`` uses a
    fresh successful :class:`FakeCrawler`; a :class:`FakeCrawler` is used as-is;
    a :class:`_ConstructFailure` raises while constructing; any exception is
    raised from ``__aenter__``. This lets a test script the probe (first
    construction) separately from the persistent crawler.
    """

    script: ClassVar[list[Any]] = []
    constructed: ClassVar[list[FakeAsyncWebCrawler]] = []

    def __init__(self, config: Any = None) -> None:
        self.config = config
        self.crawler = FakeCrawler()
        item = type(self).script.pop(0) if type(self).script else None
        if isinstance(item, _ConstructFailure):
            raise item.error
        if isinstance(item, FakeCrawler):
            self.crawler = item
        self.enter_error = item if isinstance(item, BaseException) else None
        type(self).constructed.append(self)

    async def __aenter__(self) -> FakeAsyncWebCrawler:
        if self.enter_error is not None:
            raise self.enter_error
        self.crawler.open = True
        self.crawler.opens += 1
        return self

    async def __aexit__(self, *_exc: object) -> None:
        self.crawler.open = False
        self.crawler.closes += 1

    async def arun(self, url: str, config: Any = None, **_kwargs: Any) -> Any:
        return await self.crawler.arun(url, config=config, **_kwargs)


def _script_crawlers(monkeypatch, *items: Any) -> None:
    monkeypatch.setattr(browser_module, "AsyncWebCrawler", FakeAsyncWebCrawler)
    FakeAsyncWebCrawler.script = list(items)
    FakeAsyncWebCrawler.constructed = []


def _browser_config(**overrides: Any) -> ServerConfig:
    return ServerConfig(browser_cdp_url="ws://cdp", **overrides)


def _browser(crawler: FakeCrawler) -> BrowserFetch:
    return BrowserFetch(cast(AsyncWebCrawler, crawler))


async def test_browser_fetch_returns_cleaned_html():
    crawler = FakeCrawler(result=crawl_result(cleaned_html="<h1>Rendered</h1>"))
    fetch = _browser(crawler)

    result = await fetch(TARGET)

    assert outcome(result) == (FetchStatus.OK, None)
    assert (result.url, result.content) == (TARGET, "<h1>Rendered</h1>")
    assert crawler.urls == [TARGET]


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (
            crawl_result(status_code=403, cleaned_html="denied"),
            (FetchStatus.HTTP_ERROR, 403),
        ),
        (
            crawl_result(status_code=404, cleaned_html="gone"),
            (FetchStatus.HTTP_ERROR, 404),
        ),
        (crawl_result(cleaned_html="   "), (FetchStatus.EMPTY, None)),
        (
            crawl_result(success=False, status_code=None),
            (FetchStatus.UNAVAILABLE, None),
        ),
    ],
)
async def test_browser_fetch_maps_crawl_outcomes(
    result: Any, expected: tuple[FetchStatus, int | None]
):
    fetch = _browser(FakeCrawler(result=result))

    assert outcome(await fetch(TARGET)) == expected


async def test_browser_fetch_crawl_failure_is_unavailable():
    crawler = FakeCrawler(error=ConnectionError("cdp down"))
    fetch = _browser(crawler)

    assert outcome(await fetch(TARGET)) == (FetchStatus.UNAVAILABLE, None)


@pytest.mark.parametrize("cleaned_html", ["", "   "])
async def test_browser_fetch_blank_cleaned_html_is_empty(cleaned_html: str):
    fetch = _browser(FakeCrawler(result=crawl_result(cleaned_html=cleaned_html)))

    assert outcome(await fetch(TARGET)) == (FetchStatus.EMPTY, None)


# ---------------------------------------------------------------------------
# Browser crawler connection lifetime
# ---------------------------------------------------------------------------


async def test_browser_fetch_repeated_requests_reuse_open_connection(monkeypatch):
    crawler = FakeCrawler(result=crawl_result(cleaned_html="<h1>X</h1>"))
    _script_crawlers(monkeypatch, None, crawler)

    async with create_browser_fetch(_browser_config()) as browser:
        assert browser is not None
        first = await browser("https://a.example")
        second = await browser("https://b.example")
        assert crawler.opens == 1 and crawler.closes == 0 and crawler.open

    assert first.ok and second.ok
    assert first.content == second.content == "<h1>X</h1>"
    assert crawler.urls == ["https://a.example", "https://b.example"]
    assert crawler.closes == 1 and crawler.open is False


async def test_browser_fetch_overlapping_requests_share_connection():
    gates = {"https://a.example": asyncio.Event(), "https://b.example": asyncio.Event()}
    crawler = GatedCrawler(gates)
    fetch = _browser(cast(FakeCrawler, crawler))

    tasks = [asyncio.ensure_future(fetch(url)) for url in gates]
    async with asyncio.timeout(5):
        await asyncio.gather(*(crawler.entered[url].wait() for url in gates))
    assert len(crawler.configs) == 2
    assert crawler.configs[0] is not crawler.configs[1]
    for gate in gates.values():
        gate.set()
    results = await asyncio.gather(*tasks)

    assert [result.content for result in results] == [
        "<h1>https://a.example</h1>",
        "<h1>https://b.example</h1>",
    ]


async def test_browser_fetch_failure_keeps_connection_for_next_request(monkeypatch):
    crawler = FakeCrawler(error=ConnectionError("cdp hiccup"))
    _script_crawlers(monkeypatch, None, crawler)

    async with create_browser_fetch(_browser_config()) as browser:
        assert browser is not None
        failure = await browser("https://a.example")
        crawler.error = None
        crawler.result = crawl_result(cleaned_html="<h1>Recovered</h1>")
        recovered = await browser("https://b.example")
        assert outcome(failure) == (FetchStatus.UNAVAILABLE, None)
        assert recovered.content == "<h1>Recovered</h1>"
        assert crawler.opens == 1 and crawler.closes == 0

    assert crawler.closes == 1


@pytest.mark.parametrize(
    "item",
    [
        pytest.param(ConnectionError("cdp down"), id="enter"),
        pytest.param(_ConstructFailure(ConnectionError("bad config")), id="construct"),
    ],
)
async def test_browser_fetch_startup_failure_yields_none(monkeypatch, item: Any):
    _script_crawlers(monkeypatch, None, item)

    async with create_browser_fetch(_browser_config()) as browser:
        assert browser is None


async def test_browser_fetch_without_endpoint_opens_no_connection(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    _script_crawlers(monkeypatch)

    async with create_browser_fetch(ServerConfig(browser_cdp_url="")) as browser:
        assert browser is None

    assert FakeAsyncWebCrawler.constructed == []


async def test_browser_fetch_exceptional_exit_closes_connection(monkeypatch):
    crawler = FakeCrawler()
    _script_crawlers(monkeypatch, None, crawler)

    with pytest.raises(ValueError, match="body boom"):
        async with create_browser_fetch(_browser_config()):
            raise ValueError("body boom")

    assert crawler.open is False and crawler.closes == 1


async def test_browser_fetch_cancellation_closes_connection(monkeypatch):
    crawler = FakeCrawler()
    _script_crawlers(monkeypatch, None, crawler)
    entered = asyncio.Event()

    async def consume() -> None:
        async with create_browser_fetch(_browser_config()):
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(consume())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert crawler.open is False and crawler.closes == 1


# ---------------------------------------------------------------------------
# Scrape.do
# ---------------------------------------------------------------------------


async def test_scrape_do_requests_rendered_html_with_unblocking(
    httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    httpx_mock.add_response(text="<h1>Article</h1>")
    fetch = ScrapeDoFetch("tok", http_client=client)

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
    assert outcome(result) == (FetchStatus.OK, None)
    assert (result.url, result.content) == (TARGET, "<h1>Article</h1>")


@pytest.mark.parametrize(
    ("status_code", "text", "expected"),
    [
        (200, "  ", (FetchStatus.EMPTY, None)),
        (404, "not found", (FetchStatus.HTTP_ERROR, 404)),
        (400, "bad target", (FetchStatus.HTTP_ERROR, 400)),
        (401, "no credits", (FetchStatus.UNAVAILABLE, None)),
        (429, "concurrency", (FetchStatus.UNAVAILABLE, None)),
        (502, "failed", (FetchStatus.UNAVAILABLE, None)),
    ],
)
async def test_scrape_do_maps_api_status(
    httpx_mock: HTTPXMock,
    client: httpx.AsyncClient,
    status_code: int,
    text: str,
    expected: tuple[FetchStatus, int | None],
):
    httpx_mock.add_response(status_code=status_code, text=text)
    fetch = ScrapeDoFetch("tok", http_client=client)

    assert outcome(await fetch(TARGET)) == expected


async def test_scrape_do_transport_error_is_unavailable(
    httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    httpx_mock.add_exception(httpx.ConnectError("refused"))
    fetch = ScrapeDoFetch("tok", http_client=client)

    assert outcome(await fetch(TARGET)) == (FetchStatus.UNAVAILABLE, None)


# ---------------------------------------------------------------------------
# Bright Data Web Unlocker
# ---------------------------------------------------------------------------


async def test_bright_data_posts_zone_request_for_html(
    httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    httpx_mock.add_response(json={"status_code": 200, "body": "<h1>Article</h1>"})
    fetch = BrightDataFetch("key", "unlocker", http_client=client)

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
    assert outcome(result) == (FetchStatus.OK, None)
    assert (result.url, result.content) == (TARGET, "<h1>Article</h1>")


@pytest.mark.parametrize(
    ("status_code", "payload", "expected"),
    [
        (
            200,
            {"status_code": 403, "body": "denied"},
            (FetchStatus.HTTP_ERROR, 403),
        ),
        (200, {"status_code": 200, "body": ""}, (FetchStatus.EMPTY, None)),
        (200, {"unexpected": True}, (FetchStatus.UNAVAILABLE, None)),
        (401, {"error": "bad key"}, (FetchStatus.UNAVAILABLE, None)),
        (502, {"error": "upstream"}, (FetchStatus.UNAVAILABLE, None)),
    ],
)
async def test_bright_data_maps_outcomes(
    httpx_mock: HTTPXMock,
    client: httpx.AsyncClient,
    status_code: int,
    payload: dict[str, Any],
    expected: tuple[FetchStatus, int | None],
):
    httpx_mock.add_response(status_code=status_code, json=payload)
    fetch = BrightDataFetch("key", "unlocker", http_client=client)

    assert outcome(await fetch(TARGET)) == expected


# ---------------------------------------------------------------------------
# Config wiring
# ---------------------------------------------------------------------------


async def test_fetch_tool_escalates_blocked_page_to_configured_provider(
    monkeypatch, httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    monkeypatch.setattr(shutil, "which", lambda _name: None)
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

    async with build_fetch_tool(config, client) as fetch:
        assert fetch is not None
        result = await fetch(TARGET, None)

    assert result.ok
    assert result.content == "# Unblocked"


async def test_fetch_tool_escalates_blocked_page_to_bright_data(
    monkeypatch, httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    httpx_mock.add_response(url=TARGET, status_code=403)
    httpx_mock.add_response(
        url="https://api.brightdata.com/request",
        json={
            "status_code": 200,
            "body": "<html><body><h1>Unblocked</h1></body></html>",
        },
    )
    config = ServerConfig(
        scraper_provider="bright_data",
        bright_data_api_key="key",
        bright_data_zone="unlocker",
        browser_cdp_url="",
    )

    async with build_fetch_tool(config, client) as fetch:
        assert fetch is not None
        result = await fetch(TARGET, None)

    assert result.ok
    assert result.content == "# Unblocked"


@pytest.mark.parametrize(
    "config_kwargs",
    [
        pytest.param({}, id="unset"),
        pytest.param(
            {"scraper_provider": "unknown", "scrape_do_token": "t"}, id="unknown"
        ),
        pytest.param({"scraper_provider": "scrape_do"}, id="scrape-do-no-token"),
        pytest.param(
            {"scraper_provider": "bright_data", "bright_data_api_key": "k"},
            id="bright-data-no-zone",
        ),
        pytest.param(
            {"scraper_provider": "bright_data", "bright_data_zone": "z"},
            id="bright-data-no-key",
        ),
    ],
)
async def test_unusable_provider_config_keeps_target_failure(
    monkeypatch,
    httpx_mock: HTTPXMock,
    client: httpx.AsyncClient,
    config_kwargs: dict[str, Any],
):
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    httpx_mock.add_response(url=TARGET, status_code=403)
    config = ServerConfig(browser_cdp_url="", **config_kwargs)

    async with build_fetch_tool(config, client) as fetch:
        assert fetch is not None
        result = await fetch(TARGET, None)

    assert outcome(result) == (FetchStatus.HTTP_ERROR, 403)
