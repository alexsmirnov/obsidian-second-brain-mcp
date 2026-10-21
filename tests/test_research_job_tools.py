"""Contract tests for the web_research / web_research_status tools."""

from __future__ import annotations

import asyncio
import re
from typing import Any

import pytest
from fastmcp import Client
from fastmcp.client.client import CallToolResult

from mcps.config import create_config
from mcps.research.jobs import InMemoryJobStore, JobStatus, ResearchJob
from mcps.server import create_server
from tests.research_fakes import FakeResearcher
from tests.test_browser_runtime import _install_crawler, _set_research_env

_JOB_NAME = re.compile(r"[a-z]+-[a-z]+-[a-z]+")


def _data(result: CallToolResult) -> dict[str, Any]:
    assert result.structured_content is not None
    return result.structured_content


@pytest.fixture
def research_server(monkeypatch):
    """Build a server whose researcher is the given fake; short wait budget."""

    def _build(fake: FakeResearcher, *, wait_seconds: str = "0.2"):
        _set_research_env(monkeypatch)
        _install_crawler(monkeypatch, reachable={"ws://127.0.0.1:9222"})
        monkeypatch.setenv("RESEARCH_WAIT_SECONDS", wait_seconds)
        monkeypatch.setattr(
            "mcps.research.lifespan.create_researcher", lambda *a, **k: fake
        )
        return create_server(create_config())

    return _build


async def test_web_research_returns_completed_job(research_server):
    fake = FakeResearcher()
    fake.release.set()
    async with Client(research_server(fake).mcp) as client:
        result = await client.call_tool("web_research", {"query": "q"})

    assert _data(result)["status"] == "completed"
    assert _data(result)["result"]["answer"] == "answer:q"


async def test_web_research_returns_job_then_status_returns_result(research_server):
    fake = FakeResearcher()
    async with Client(research_server(fake).mcp) as client:
        started = await client.call_tool("web_research", {"query": "q"})
        job_id = _data(started)["job_id"]
        fake.release.set()
        finished = await client.call_tool("web_research_status", {"job_id": job_id})

    assert _data(started)["status"] == "running"
    assert _JOB_NAME.fullmatch(job_id)
    assert "web_research_status" in _data(started)["instructions"]
    assert _data(finished)["status"] == "completed"
    assert _data(finished)["result"]["answer"] == "answer:q"


async def test_status_unknown_job_is_tool_error(research_server):
    async with Client(research_server(FakeResearcher()).mcp) as client:
        result = await client.call_tool(
            "web_research_status", {"job_id": "no-such-job"}, raise_on_error=False
        )

    assert result.is_error
    assert "unknown or expired" in str(result.content[0])


async def test_web_research_progress_notifications(research_server):
    fake = FakeResearcher(before=["step a"])
    notifications: list[str | None] = []

    async def on_progress(progress: float, total: float | None, message: str | None):
        notifications.append(message)

    async with Client(research_server(fake).mcp) as client:
        await client.call_tool(
            "web_research", {"query": "q"}, progress_handler=on_progress
        )

    assert "step a" in notifications


async def test_abandoned_web_research_call_cancels_job(research_server, monkeypatch):
    stores: list[InMemoryJobStore] = []
    created_ids: list[str] = []

    class RecordingStore(InMemoryJobStore):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            stores.append(self)

        async def create(self, job: ResearchJob) -> None:
            created_ids.append(job.job_id)
            await super().create(job)

    monkeypatch.setattr("mcps.research.lifespan.InMemoryJobStore", RecordingStore)
    fake = FakeResearcher()
    async with Client(research_server(fake, wait_seconds="5").mcp) as client:
        with pytest.raises(Exception):  # noqa: B017 - client timeout type varies
            await client.call_tool("web_research", {"query": "q"}, timeout=0.1)
        for _ in range(40):
            if fake.cancelled:
                break
            await asyncio.sleep(0.05)
        (store,) = stores
        (job_id,) = created_ids
        job = await store.get(job_id)

    assert fake.cancelled
    assert job is not None
    assert job.status == JobStatus.FAILED
    assert "cancelled" in (job.error or "")


async def test_abandoned_status_call_keeps_job(research_server):
    fake = FakeResearcher()
    async with Client(research_server(fake, wait_seconds="1").mcp) as client:
        started = await client.call_tool("web_research", {"query": "q"}, timeout=5)
        job_id = _data(started)["job_id"]
        with pytest.raises(Exception):  # noqa: B017 - client timeout type varies
            await client.call_tool(
                "web_research_status", {"job_id": job_id}, timeout=0.1
            )
        await asyncio.sleep(0.3)
        still_alive = not fake.cancelled
        fake.release.set()
        finished = await client.call_tool("web_research_status", {"job_id": job_id})

    assert still_alive
    assert _data(finished)["status"] == "completed"


async def test_both_research_tools_listed(research_server):
    async with Client(research_server(FakeResearcher()).mcp) as client:
        names = {tool.name for tool in await client.list_tools()}

    assert {"web_research", "web_research_status"} <= names
