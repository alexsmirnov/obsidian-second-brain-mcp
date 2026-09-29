"""Contract tests for CDP endpoint startup and web_research availability."""

from __future__ import annotations

import asyncio
import shutil
from contextlib import asynccontextmanager
from typing import Any

import pytest
from fastmcp import Client

from mcps.config import create_config
from mcps.research.tools import browser as browser_module
from mcps.research.tools.browser import LOCAL_CDP_URL, browser_endpoint
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
