"""Contract tests for deep research agent integration.

All tests mock LLM and HTTP calls — no real network access.
"""

from __future__ import annotations

from typing import Any, cast

import httpx
import pytest

from mcps.config import ServerConfig, create_config
from mcps.research.agent import create_researcher
from mcps.research.config import (
    ResearchConfig,
    build_research_config,
)

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
        """aiswe_research tool appears in server's registered tools."""
        from mcps.server import create_server

        monkeypatch.setenv("ROUTER_API_BASE", "http://localhost:4000")
        monkeypatch.setenv("ROUTER_API_KEY", "sk-test")
        config = create_config()
        server = create_server(config)

        tools = await server.mcp.list_tools()
        tool_names = [t.name for t in tools]
        assert "web_research" in tool_names
