"""Contract tests for deep research agent integration.

All tests mock LLM and HTTP calls — no real network access.
"""

from __future__ import annotations

from typing import Any, cast

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage
from pydantic import Field

from mcps.config import ServerConfig, create_config
from mcps.research.agent import create_researcher
from mcps.research.config import (
    ResearchConfig,
    build_research_config,
)
from mcps.research.deep_research import ResearchAgent
from mcps.research.tools import SearchResult

# ---------------------------------------------------------------------------
# Config contract tests
# ---------------------------------------------------------------------------


class TestServerConfigContract:
    def test_config_has_router_api_base_field(self):
        config = ServerConfig()
        assert hasattr(config, "router_api_base")
        assert config.router_api_base == ""

    def test_config_has_router_api_key_field(self):
        config = ServerConfig()
        assert hasattr(config, "router_api_key")
        assert config.router_api_key == ""

    def test_perplexity_api_key_removed(self):
        config = ServerConfig()
        assert not hasattr(config, "perplexity_api_key")

    def test_create_config_reads_router_env_vars(self, monkeypatch):
        monkeypatch.setenv("ROUTER_API_BASE", "http://localhost:4000")
        monkeypatch.setenv("ROUTER_API_KEY", "sk-test123")
        config = create_config()
        assert config.router_api_base == "http://localhost:4000"
        assert config.router_api_key == "sk-test123"


# ---------------------------------------------------------------------------
# ResearchConfig contract tests
# ---------------------------------------------------------------------------


class TestResearchConfigContract:
    def test_build_research_config_returns_valid_config(self):
        server_config = ServerConfig(
            router_api_base="http://localhost:4000",
            router_api_key="sk-test",
            research_fast_model="gemini-flash-lite",
            research_infer_model="gemini-flash",
        )
        config = build_research_config(
            server_config,
            http_client=httpx.AsyncClient(),
        )
        assert isinstance(config, ResearchConfig)
        assert config.fast is not None
        assert config.small is not None
        assert callable(config.search)
        assert callable(config.fetch)

    def test_research_config_fields_are_callables(self):
        server_config = ServerConfig(
            router_api_base="http://localhost:4000",
            router_api_key="sk-test",
            research_fast_model="gemini-flash-lite",
            research_infer_model="gemini-flash",
        )
        config = build_research_config(
            server_config,
            http_client=httpx.AsyncClient(),
        )
        import asyncio

        assert asyncio.iscoroutinefunction(config.search)
        assert asyncio.iscoroutinefunction(config.fetch)


# ---------------------------------------------------------------------------
# Agent contract tests
# ---------------------------------------------------------------------------


class TestAgentContract:
    @pytest.fixture
    def mock_config(self):
        server_config = ServerConfig(
            router_api_base="http://localhost:4000",
            router_api_key="sk-test",
            research_fast_model="gemini-flash-lite",
            research_infer_model="gemini-flash",
        )
        return build_research_config(
            server_config,
            http_client=httpx.AsyncClient(),
        )

    def test_create_researcher_returns_callable(self, mock_config):
        researcher = create_researcher(mock_config, implementation="deep_research")
        assert callable(researcher)

    def test_create_researcher_rejects_unknown_implementation(self, mock_config):
        with pytest.raises(ValueError, match="Unknown implementation"):
            create_researcher(mock_config, implementation=cast(Any, "bogus"))

    @pytest.mark.asyncio
    async def test_agent_returns_research_response_shape(
        self, mock_config, monkeypatch
    ):
        """
        Full graph execution with mocked LLM responses.
        Verifies the response contract without real network calls.
        """
        from unittest.mock import patch

        researcher = create_researcher(mock_config, implementation="deep_research")
        agent_graph = researcher.graph

        mock_answer = "42"
        mock_explanation = "This was derived from authoritative sources."
        mock_sources = ["https://example.com/source1", "https://example.com/source2"]

        async def mock_invoke(_input, **_kwargs):
            return {
                "answer": mock_answer,
                "explanation": mock_explanation,
                "sources_gathered": mock_sources,
            }

        with patch.object(agent_graph, "ainvoke", side_effect=mock_invoke):
            result = await researcher("What is the answer?")

        assert isinstance(result, dict)
        assert "answer" in result
        assert "explanation" in result
        assert "sources" in result
        assert result["answer"] == mock_answer
        assert result["explanation"] == mock_explanation

    @pytest.mark.asyncio
    async def test_agent_response_matches_research_response_contract(
        self, mock_config
    ):
        """Response follows ResearchResponse TypedDict shape."""
        from unittest.mock import patch

        researcher = create_researcher(mock_config, implementation="deep_research")
        agent_graph = researcher.graph

        async def mock_invoke(_input, **_kwargs):
            return {
                "answer": "Test answer",
                "explanation": "Test explanation",
                "sources_gathered": ["https://test.com"],
            }

        with patch.object(agent_graph, "ainvoke", side_effect=mock_invoke):
            result = await researcher("test query")

        assert isinstance(result["answer"], str)
        assert isinstance(result["explanation"], str)
        assert isinstance(result["sources"], list)


# ---------------------------------------------------------------------------
# Progress reporter contract tests
# ---------------------------------------------------------------------------


class TestProgressReporterContract:
    @pytest.fixture
    def mock_config(self):
        server_config = ServerConfig(
            router_api_base="http://localhost:4000",
            router_api_key="sk-test",
            research_fast_model="gemini-flash-lite",
            research_infer_model="gemini-flash",
        )
        return build_research_config(
            server_config,
            http_client=httpx.AsyncClient(),
        )

    @pytest.mark.asyncio
    async def test_agent_wraps_progress_in_config_for_ainvoke(self, mock_config):
        """__call__ builds config dict from progress callback and forwards it to graph.ainvoke."""
        from unittest.mock import patch

        researcher = create_researcher(mock_config, implementation="deep_research")
        agent_graph = researcher.graph

        captured: dict[str, Any] = {}

        async def mock_invoke(_input, **kwargs):
            captured.update(kwargs)
            return {
                "answer": "42",
                "explanation": "test",
                "sources_gathered": [],
            }

        async def mock_reporter(message: str, progress: float, total: float | None) -> None:
            pass

        with patch.object(agent_graph, "ainvoke", side_effect=mock_invoke):
            result = await researcher("query", progress=mock_reporter)

        assert "config" in captured
        assert captured["config"]["configurable"]["progress_reporter"] is mock_reporter
        assert result["answer"] == "42"

    @pytest.mark.asyncio
    async def test_agent_call_without_config_still_works(self, mock_config):
        """__call__ without config arg preserves original behavior."""
        from unittest.mock import patch

        researcher = create_researcher(mock_config, implementation="deep_research")
        agent_graph = researcher.graph

        async def mock_invoke(_input, **_kwargs):
            return {
                "answer": "ok",
                "explanation": "",
                "sources_gathered": [],
            }

        with patch.object(agent_graph, "ainvoke", side_effect=mock_invoke):
            result = await researcher("bare query")

        assert result["answer"] == "ok"


# ---------------------------------------------------------------------------
# Tool contract tests
# ---------------------------------------------------------------------------


class TestToolContract:
    @pytest.mark.asyncio
    async def test_tool_is_registered(self, monkeypatch):
        """web_research tool appears in the server's registered tools."""
        from contextlib import asynccontextmanager

        from fastmcp import Client

        from mcps.server import create_server

        @asynccontextmanager
        async def reachable_browser(_cdp_url, *, probe=None):
            yield "ws://127.0.0.1:9222"

        monkeypatch.setenv("ROUTER_API_BASE", "http://localhost:4000")
        monkeypatch.setenv("ROUTER_API_KEY", "sk-test")
        monkeypatch.delenv("VAULT", raising=False)
        monkeypatch.setattr(
            "mcps.research.lifespan.browser_endpoint", reachable_browser
        )
        config = create_config()
        server = create_server(config)

        async with Client(server.mcp) as client:
            tools = await client.list_tools()
        assert "web_research" in [t.name for t in tools]


# ---------------------------------------------------------------------------
# Fetch query propagation and evidence filtering
# ---------------------------------------------------------------------------

_QUESTION = "How does quantum routing work?"
_SEARCH_RESULTS = [
    SearchResult(url=f"https://{name}.example", title=name.upper(), snippet="s")
    for name in ("a", "b", "c")
]


class _RecordingModel(FakeMessagesListChatModel):
    """Fake chat model that keeps every message list it was invoked with."""

    calls: list[list[Any]] = Field(default_factory=list)

    def _generate(self, messages, *args, **kwargs):
        self.calls.append(messages)
        return super()._generate(messages, *args, **kwargs)


def _make_agent(fetch_results: dict[str, str] | None = None):
    fetch_calls: list[tuple[str, str | None]] = []
    results = fetch_results or {}

    async def fake_fetch(url: str, query: str | None) -> str:
        fetch_calls.append((url, query))
        return results.get(url, "content")

    async def fake_search(query: str) -> list[SearchResult]:
        return _SEARCH_RESULTS

    fast = _RecordingModel(responses=[AIMessage("CLEANED")] * 4)
    small = FakeMessagesListChatModel(responses=[AIMessage("CLEANED")])
    agent = ResearchAgent(
        ResearchConfig(fast=fast, small=small, search=fake_search, fetch=fake_fetch)
    )
    return agent, fast, fetch_calls


def _state(**overrides: Any) -> Any:
    state = {"original_question": _QUESTION, "search_query": "quantum routing", "id": 0}
    return {**state, **overrides}


class TestWebResearchFetch:
    async def test_initial_branch_uses_original_question(self):
        agent, _, fetch_calls = _make_agent()

        await agent.web_research(_state(), {})

        assert [q for _, q in fetch_calls] == [_QUESTION] * 3

    async def test_follow_up_uses_knowledge_gap(self):
        agent, _, fetch_calls = _make_agent()

        await agent.web_research(
            _state(knowledge_gap="What are the latency limits?"), {}
        )
        await agent.web_research(_state(knowledge_gap="N/A"), {})
        await agent.web_research(_state(knowledge_gap="  "), {})

        queries = [q for _, q in fetch_calls]
        assert queries == ["What are the latency limits?"] * 3 + [_QUESTION] * 6

    async def test_direct_url_passes_query(self):
        agent, _, fetch_calls = _make_agent()

        await agent.web_research(
            _state(search_query="https://source.example/page"), {}
        )

        assert fetch_calls == [("https://source.example/page", _QUESTION)]

    async def test_failed_and_empty_fetches_are_not_evidence(self):
        agent, fast, _ = _make_agent(
            {
                "https://a.example": "content A",
                "https://b.example": "ERROR: http code 403",
                "https://c.example": "",
            }
        )

        result = await agent.web_research(_state(), {})

        assert len(fast.calls) == 1
        human_text = str(fast.calls[0][-1].content)
        assert "https://a.example" in human_text
        assert "b.example" not in human_text
        assert "c.example" not in human_text
        assert "CLEANED" in result["web_results"][0]

    @pytest.mark.parametrize(
        "state_extra", [{}, {"knowledge_gap": "N/A"}, {"knowledge_gap": " "}]
    )
    async def test_absent_knowledge_gap_is_skipped_in_prompt(self, state_extra):
        agent, fast, _ = _make_agent()

        await agent.web_research(_state(**state_extra), {})

        assert "KNOWLEDGE GAP" not in str(fast.calls[0][-1].content)

    async def test_present_knowledge_gap_is_in_prompt(self):
        agent, fast, _ = _make_agent()

        await agent.web_research(_state(knowledge_gap="latency limits"), {})

        assert "KNOWLEDGE GAP: latency limits" in str(fast.calls[0][-1].content)

    async def test_all_failed_fetches_yield_no_evidence_without_llm_call(self):
        agent, fast, _ = _make_agent(
            {
                "https://a.example": "ERROR: http code 403",
                "https://b.example": "",
                "https://c.example": "   ",
            }
        )

        result = await agent.web_research(_state(), {})

        assert fast.calls == []
        assert "NO_RELEVANT_EVIDENCE" in result["web_results"][0]
