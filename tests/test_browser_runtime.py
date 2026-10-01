"""Contract tests for CDP endpoint startup and web_research availability."""

from __future__ import annotations

import asyncio
import shutil
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from fastmcp import Client

from mcps.config import create_config
from mcps.research.lifespan import build_research_lifespan
from mcps.research.tools import browser as browser_module
from mcps.research.tools.browser import LOCAL_CDP_URL, browser_endpoint
from mcps.research.tools.result import FetchStatus
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


def _fake_probe(probe_results: dict[str, bool]) -> Any:
    calls: list[str] = []

    async def probe(cdp_url: str) -> bool:
        calls.append(cdp_url)
        return probe_results.get(cdp_url, False)

    probe.calls = calls  # type: ignore[attr-defined]
    return probe


def _endpoint_stub(endpoint: str | None) -> Any:
    @asynccontextmanager
    async def stub(cdp_url: str, *, probe: Any = None):
        yield endpoint

    return stub


def _crawler_stub(crawler: Any) -> Any:
    @asynccontextmanager
    async def stub(cdp_url: str | None, *, crawler_factory: Any = None):
        yield crawler

    return stub


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
    process = FakeProcess()
    spawns = _install_spawn(monkeypatch, process)
    probe = _fake_probe({"ws://remote:9222": True})

    async with browser_endpoint("ws://remote:9222", probe=probe) as url:
        assert url == "ws://remote:9222"

    assert spawns == []


@pytest.mark.parametrize("cdp_url", ["", "ws://remote:9222"])
async def test_falls_back_to_local_obscura(monkeypatch, cdp_url: str):
    monkeypatch.setattr(shutil, "which", lambda _name: "/test-bin/obscura")
    process = FakeProcess()
    spawns = _install_spawn(monkeypatch, process)
    probe = _fake_probe({"ws://remote:9222": False, LOCAL_CDP_URL: True})

    async with browser_endpoint(cdp_url, probe=probe) as url:
        assert url == LOCAL_CDP_URL

    assert spawns == [
        ("/test-bin/obscura", "serve", "--stealth", "--allow-private-network")
    ]
    assert process.terminate_calls == 1


async def test_no_browser_when_obscura_absent(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    process = FakeProcess()
    spawns = _install_spawn(monkeypatch, process)
    probe = _fake_probe({})

    async with browser_endpoint("", probe=probe) as url:
        assert url is None

    assert spawns == []


async def test_no_browser_when_obscura_exits(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _name: "/test-bin/obscura")
    process = FakeProcess(returncode=1)
    spawns = _install_spawn(monkeypatch, process)
    probe = _fake_probe({LOCAL_CDP_URL: False})

    async with browser_endpoint("", probe=probe) as url:
        assert url is None

    assert len(spawns) == 1
    assert process.returncode == 1


async def test_no_browser_when_local_probe_never_ready(monkeypatch):
    monkeypatch.setattr(browser_module, "LOCAL_STARTUP_SECONDS", 0.1)
    monkeypatch.setattr(shutil, "which", lambda _name: "/test-bin/obscura")
    process = FakeProcess()
    _install_spawn(monkeypatch, process)
    probe = _fake_probe({LOCAL_CDP_URL: False})

    async with browser_endpoint("", probe=probe) as url:
        assert url is None

    assert process.terminate_calls == 1


async def _list_tool_names(monkeypatch, endpoint: str | None) -> list[str]:
    monkeypatch.setenv("ROUTER_API_BASE", "http://localhost:4000")
    monkeypatch.setenv("ROUTER_API_KEY", "sk-test")
    monkeypatch.delenv("VAULT", raising=False)
    monkeypatch.setattr(
        "mcps.research.lifespan.browser_endpoint", _endpoint_stub(endpoint)
    )
    monkeypatch.setattr(
        "mcps.research.lifespan.browser_crawler",
        _crawler_stub(object() if endpoint else None),
    )
    server = create_server(create_config())
    async with Client(server.mcp) as client:
        tools = await client.list_tools()
    return [tool.name for tool in tools]


async def test_startup_failure_hides_only_web_research(monkeypatch):
    names = await _list_tool_names(monkeypatch, None)

    assert "web_research" not in names


async def test_reachable_browser_keeps_web_research(monkeypatch):
    names = await _list_tool_names(monkeypatch, "ws://127.0.0.1:9222")

    assert "web_research" in names


# ---------------------------------------------------------------------------
# Research lifespan owns one browser connection
# ---------------------------------------------------------------------------


class FakeCrawl4aiCrawler:
    """Stand-in for crawl4ai.AsyncWebCrawler tracking connection lifetime."""

    started: ClassVar[list[FakeCrawl4aiCrawler]] = []
    closed: ClassVar[list[FakeCrawl4aiCrawler]] = []

    def __init__(self, config: Any = None):
        self.config = config
        self.urls: list[str] = []
        self.is_open = False
        FakeCrawl4aiCrawler.started.append(self)

    async def __aenter__(self) -> FakeCrawl4aiCrawler:
        self.is_open = True
        return self

    async def __aexit__(self, *_exc: object) -> None:
        self.is_open = False
        FakeCrawl4aiCrawler.closed.append(self)

    async def arun(self, url: str, config: Any = None) -> Any:
        assert self.is_open, "crawler used while closed"
        self.urls.append(url)
        return SimpleNamespace(
            success=True,
            status_code=200,
            cleaned_html="<html><body><p>hello world</p></body></html>",
            error_message="",
        )


async def test_research_lifespan_reuses_one_browser_connection(monkeypatch):
    monkeypatch.setenv("ROUTER_API_BASE", "http://localhost:4000")
    monkeypatch.setenv("ROUTER_API_KEY", "sk-test")
    monkeypatch.delenv("VAULT", raising=False)
    monkeypatch.setattr(
        "mcps.research.lifespan.browser_endpoint",
        _endpoint_stub("ws://127.0.0.1:9222"),
    )
    monkeypatch.setattr(
        "mcps.research.tools.browser.AsyncWebCrawler", FakeCrawl4aiCrawler
    )
    FakeCrawl4aiCrawler.started = []
    FakeCrawl4aiCrawler.closed = []

    lifespan = build_research_lifespan(create_config())
    async with lifespan(FakeServer()) as ctx:  # type: ignore[arg-type]
        researcher = ctx["researcher"]
        fetch = researcher.config.fetch
        first = await fetch("https://a.example", None)
        second = await fetch("https://b.example", None)
        crawler = FakeCrawl4aiCrawler.started[0]
        assert len(FakeCrawl4aiCrawler.started) == 1
        assert crawler.is_open
        assert crawler.urls == ["https://a.example", "https://b.example"]

    assert (first.status, second.status) == (FetchStatus.OK, FetchStatus.OK)
    assert "hello world" in first.content
    assert "hello world" in second.content
    assert FakeCrawl4aiCrawler.closed == [crawler]


async def test_research_lifespan_crawler_failure_hides_web_research(monkeypatch):
    monkeypatch.setenv("ROUTER_API_BASE", "http://localhost:4000")
    monkeypatch.setenv("ROUTER_API_KEY", "sk-test")
    monkeypatch.delenv("VAULT", raising=False)
    monkeypatch.setattr(
        "mcps.research.lifespan.browser_endpoint",
        _endpoint_stub("ws://127.0.0.1:9222"),
    )
    monkeypatch.setattr(
        "mcps.research.lifespan.browser_crawler", _crawler_stub(None)
    )

    server = FakeServer()
    lifespan = build_research_lifespan(create_config())
    async with lifespan(server) as ctx:  # type: ignore[arg-type]
        assert ctx["researcher"] is None

    assert server.disabled == {"web_research"}
    assert server.enabled == set()


async def test_research_lifespan_closes_crawler_before_local_process(monkeypatch):
    events: list[str] = []
    process = FakeProcess()

    class OrderingCrawler(FakeCrawl4aiCrawler):
        async def __aexit__(self, *_exc: object) -> None:
            events.append("crawler_close")
            await super().__aexit__(*_exc)

    async def fake_wait(proc: FakeProcess, probe: Any) -> bool:
        return True

    monkeypatch.setattr(shutil, "which", lambda _name: "/test-bin/obscura")
    _install_spawn(monkeypatch, process)
    monkeypatch.setattr(browser_module, "_wait_for_local", fake_wait)
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    monkeypatch.setattr(
        "mcps.research.tools.browser.AsyncWebCrawler", OrderingCrawler
    )
    original_terminate = process.terminate

    def terminate() -> None:
        events.append("process_terminate")
        original_terminate()

    monkeypatch.setattr(process, "terminate", terminate)
    OrderingCrawler.started = []
    OrderingCrawler.closed = []

    config = create_config()
    config.browser_cdp_url = ""
    lifespan = build_research_lifespan(config)
    async with lifespan(FakeServer()) as ctx:  # type: ignore[arg-type]
        assert ctx["researcher"] is not None

    assert events[-1] == "process_terminate", events
    assert "crawler_close" in events
    assert events.index("crawler_close") < events.index("process_terminate")
