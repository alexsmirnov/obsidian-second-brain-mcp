"""Contract tests for CDP browser startup and web_research availability."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, ClassVar

from fastmcp import Client

from mcps.config import ServerConfig, create_config
from mcps.research.lifespan import build_research_lifespan
from mcps.research.tools.browser import create_browser_fetch
from mcps.research.tools.models import FetchStatus
from mcps.server import create_server


class FakeAsyncWebCrawler:
    """Stand-in for crawl4ai.AsyncWebCrawler tracking connection lifetime.

    ``reachable`` is the set of CDP URLs the crawler can enter.
    """

    started: ClassVar[list[FakeAsyncWebCrawler]] = []
    closed: ClassVar[list[FakeAsyncWebCrawler]] = []
    reachable: ClassVar[set[str]] = set()

    def __init__(self, config: Any = None):
        self.config = config
        self.urls: list[str] = []
        self.is_open = False
        type(self).started.append(self)

    def _cdp_url(self) -> str:
        return getattr(self.config, "cdp_url", "") or ""

    async def __aenter__(self) -> FakeAsyncWebCrawler:
        if self._cdp_url() not in type(self).reachable:
            raise ConnectionError(f"cdp unavailable at {self._cdp_url()!r}")
        self.is_open = True
        return self

    async def __aexit__(self, *_exc: object) -> None:
        self.is_open = False
        type(self).closed.append(self)

    async def arun(self, url: str, config: Any = None) -> Any:
        assert self.is_open, "crawler used while closed"
        self.urls.append(url)
        return SimpleNamespace(
            success=True,
            status_code=200,
            cleaned_html="<html><body><p>hello world</p></body></html>",
            html="<html><body><p>hello world</p></body></html>",
            response_headers=None,
            error_message="",
        )


def _install_crawler(monkeypatch, *, reachable: set[str]) -> None:
    FakeAsyncWebCrawler.started = []
    FakeAsyncWebCrawler.closed = []
    FakeAsyncWebCrawler.reachable = set(reachable)
    monkeypatch.setattr(
        "mcps.research.tools.browser.AsyncWebCrawler", FakeAsyncWebCrawler
    )


def _crawled() -> list[FakeAsyncWebCrawler]:
    """Crawlers that actually fetched a URL."""
    return [crawler for crawler in FakeAsyncWebCrawler.started if crawler.urls]


class FakeServer:
    """Minimal FastMCP stand-in recording tool enable/disable calls."""

    def __init__(self) -> None:
        self.enabled: set[str] = set()
        self.disabled: set[str] = set()

    def enable(self, *, names: set[str]) -> None:
        self.enabled |= names
        self.disabled -= names

    def disable(self, *, names: set[str]) -> None:
        self.disabled |= names
        self.enabled -= names


async def test_configured_endpoint_is_used(monkeypatch):
    _install_crawler(monkeypatch, reachable={"ws://remote:9222"})

    async with create_browser_fetch(
        ServerConfig(browser_cdp_url="ws://remote:9222")
    ) as browser:
        assert browser is not None

    assert len(FakeAsyncWebCrawler.started) == 1


async def test_no_browser_when_endpoint_unset(monkeypatch):
    _install_crawler(monkeypatch, reachable=set())

    async with create_browser_fetch(ServerConfig(browser_cdp_url="")) as browser:
        assert browser is None

    assert FakeAsyncWebCrawler.started == []


async def test_no_browser_when_endpoint_unreachable(monkeypatch):
    _install_crawler(monkeypatch, reachable=set())

    async with create_browser_fetch(
        ServerConfig(browser_cdp_url="ws://remote:9222")
    ) as browser:
        assert browser is None


async def _list_tool_names(monkeypatch, *, reachable: bool) -> list[str]:
    monkeypatch.setenv("ROUTER_API_BASE", "http://localhost:4000")
    monkeypatch.setenv("ROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("BROWSER_CDP_URL", "ws://127.0.0.1:9222")
    monkeypatch.delenv("VAULT", raising=False)
    _install_crawler(
        monkeypatch,
        reachable={"ws://127.0.0.1:9222"} if reachable else set(),
    )
    server = create_server(create_config())
    async with Client(server.mcp) as client:
        tools = await client.list_tools()
    return [tool.name for tool in tools]


async def test_startup_failure_hides_only_web_research(monkeypatch):
    names = await _list_tool_names(monkeypatch, reachable=False)

    assert "web_research" not in names


async def test_reachable_browser_keeps_web_research(monkeypatch):
    names = await _list_tool_names(monkeypatch, reachable=True)

    assert "web_research" in names


# ---------------------------------------------------------------------------
# Research lifespan owns one browser connection
# ---------------------------------------------------------------------------


def _set_research_env(monkeypatch) -> None:
    monkeypatch.setenv("ROUTER_API_BASE", "http://localhost:4000")
    monkeypatch.setenv("ROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("BROWSER_CDP_URL", "ws://127.0.0.1:9222")
    monkeypatch.delenv("VAULT", raising=False)


async def test_research_lifespan_reuses_one_browser_connection(monkeypatch):
    _set_research_env(monkeypatch)
    _install_crawler(monkeypatch, reachable={"ws://127.0.0.1:9222"})

    lifespan = build_research_lifespan(create_config())
    async with lifespan(FakeServer()) as ctx:  # type: ignore[arg-type]
        researcher = ctx["researcher"]
        fetch = researcher.config.fetch
        first = await fetch("https://a.example", None)
        second = await fetch("https://b.example", None)
        crawled = _crawled()
        assert len(FakeAsyncWebCrawler.started) == len(crawled) == 1
        crawler = crawled[0]
        assert crawler.is_open
        assert crawler.urls == ["https://a.example", "https://b.example"]

    assert (first.status, second.status) == (FetchStatus.OK, FetchStatus.OK)
    assert "hello world" in first.content
    assert "hello world" in second.content
    assert crawler in FakeAsyncWebCrawler.closed


async def test_research_lifespan_crawler_failure_hides_web_research(monkeypatch):
    _set_research_env(monkeypatch)
    _install_crawler(monkeypatch, reachable=set())

    server = FakeServer()
    lifespan = build_research_lifespan(create_config())
    async with lifespan(server) as ctx:  # type: ignore[arg-type]
        assert ctx["researcher"] is None

    assert server.disabled == {"web_research"}
    assert server.enabled == set()
