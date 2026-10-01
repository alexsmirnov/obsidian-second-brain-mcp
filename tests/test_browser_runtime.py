"""Contract tests for CDP browser startup and web_research availability."""

from __future__ import annotations

import asyncio
import shutil
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from fastmcp import Client

from mcps.config import ServerConfig, create_config
from mcps.research.lifespan import build_research_lifespan
from mcps.research.tools import browser as browser_module
from mcps.research.tools.browser import LOCAL_CDP_URL, create_browser_fetch
from mcps.research.tools.models import FetchStatus
from mcps.server import create_server


class FakeProcess:
    """Stand-in for an ``asyncio.subprocess.Process`` tracking lifecycle calls."""

    def __init__(self, returncode: int | None = None):
        self.returncode = returncode
        self.terminate_calls = 0
        self.kill_calls = 0

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.returncode = 0

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9

    async def wait(self) -> int | None:
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


def _install_spawn(monkeypatch, process: FakeProcess) -> list[tuple[str, ...]]:
    spawns: list[tuple[str, ...]] = []

    async def fake_spawn(*argv: str, **_kwargs: Any) -> FakeProcess:
        spawns.append(argv)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)
    return spawns


class FakeAsyncWebCrawler:
    """Stand-in for crawl4ai.AsyncWebCrawler tracking connection lifetime.

    ``reachable`` is the set of CDP URLs the crawler can enter; the probe and the
    persistent crawler share the same fake, so entry is decided per URL.
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
    """Crawlers that actually fetched a URL (the probe never does)."""
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
    monkeypatch.setattr(shutil, "which", lambda _name: "/test-bin/obscura")
    spawns = _install_spawn(monkeypatch, FakeProcess())
    _install_crawler(monkeypatch, reachable={"ws://remote:9222"})

    async with create_browser_fetch(
        ServerConfig(browser_cdp_url="ws://remote:9222")
    ) as browser:
        assert browser is not None

    assert spawns == []


@pytest.mark.parametrize("cdp_url", ["", "ws://remote:9222"])
async def test_falls_back_to_local_obscura(monkeypatch, cdp_url: str):
    monkeypatch.setattr(shutil, "which", lambda _name: "/test-bin/obscura")
    process = FakeProcess()
    spawns = _install_spawn(monkeypatch, process)
    _install_crawler(monkeypatch, reachable={LOCAL_CDP_URL})

    async with create_browser_fetch(ServerConfig(browser_cdp_url=cdp_url)) as browser:
        assert browser is not None

    assert spawns == [
        ("/test-bin/obscura", "serve", "--stealth", "--allow-private-network")
    ]
    assert process.terminate_calls == 1


async def test_no_browser_when_obscura_absent(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    spawns = _install_spawn(monkeypatch, FakeProcess())
    _install_crawler(monkeypatch, reachable=set())

    async with create_browser_fetch(ServerConfig(browser_cdp_url="")) as browser:
        assert browser is None

    assert spawns == []


async def test_no_browser_when_obscura_exits(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _name: "/test-bin/obscura")
    process = FakeProcess(returncode=1)
    spawns = _install_spawn(monkeypatch, process)
    _install_crawler(monkeypatch, reachable=set())

    async with create_browser_fetch(ServerConfig(browser_cdp_url="")) as browser:
        assert browser is None

    assert len(spawns) == 1
    assert process.returncode == 1


async def test_no_browser_when_local_probe_never_ready(monkeypatch):
    monkeypatch.setattr(browser_module, "LOCAL_STARTUP_SECONDS", 0.05)
    monkeypatch.setattr(browser_module, "_LOCAL_POLL_SECONDS", 0.01)
    monkeypatch.setattr(shutil, "which", lambda _name: "/test-bin/obscura")
    process = FakeProcess()
    _install_spawn(monkeypatch, process)
    _install_crawler(monkeypatch, reachable=set())

    async with create_browser_fetch(ServerConfig(browser_cdp_url="")) as browser:
        assert browser is None

    assert process.terminate_calls == 1


async def _list_tool_names(monkeypatch, *, reachable: bool) -> list[str]:
    monkeypatch.setenv("ROUTER_API_BASE", "http://localhost:4000")
    monkeypatch.setenv("ROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("BROWSER_CDP_URL", "ws://127.0.0.1:9222")
    monkeypatch.delenv("VAULT", raising=False)
    monkeypatch.setattr(shutil, "which", lambda _name: None)
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
    monkeypatch.setattr(shutil, "which", lambda _name: None)


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
        assert len(crawled) == 1
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


async def test_research_lifespan_closes_crawler_before_local_process(monkeypatch):
    events: list[str] = []
    process = FakeProcess()

    class OrderingCrawler(FakeAsyncWebCrawler):
        async def __aexit__(self, *_exc: object) -> None:
            events.append("crawler_close")
            await super().__aexit__(*_exc)

    _set_research_env(monkeypatch)
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    monkeypatch.setattr(shutil, "which", lambda _name: "/test-bin/obscura")
    _install_spawn(monkeypatch, process)
    monkeypatch.setattr(browser_module, "AsyncWebCrawler", OrderingCrawler)
    OrderingCrawler.started = []
    OrderingCrawler.closed = []
    OrderingCrawler.reachable = {LOCAL_CDP_URL}
    original_terminate = process.terminate

    def terminate() -> None:
        events.append("process_terminate")
        original_terminate()

    monkeypatch.setattr(process, "terminate", terminate)

    config = create_config()
    config.browser_cdp_url = ""
    lifespan = build_research_lifespan(config)
    async with lifespan(FakeServer()) as ctx:  # type: ignore[arg-type]
        assert ctx["researcher"] is not None

    # The probe (first) and the persistent crawler (last) both close, and the
    # spawned process is terminated only after them.
    persistent = OrderingCrawler.started[-1]
    assert persistent is not OrderingCrawler.started[0]
    assert persistent in OrderingCrawler.closed
    assert events[-1] == "process_terminate", events
    assert events.count("crawler_close") == len(OrderingCrawler.closed)
