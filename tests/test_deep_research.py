"""Contract tests for deep research agent integration.

All tests mock LLM and HTTP calls — no real network access.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from typing import Any, cast

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage

from mcps.config import ServerConfig, create_config
from mcps.research.agent import create_researcher
from mcps.research.config import (
    ResearchConfig,
    build_research_config,
)
from mcps.research.deep_research import (
    _NO_EVIDENCE_ANSWER,
    _UNANSWERED,
    Claim,
    ExtractedRanges,
    FinalAnswer,
    Reflection,
    ResearchAgent,
    ResearchLimits,
)
from mcps.research.tools import SearchResult
from mcps.research.tools.models import FetchResult, FetchStatus

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


class _ReachableCrawler:
    """Minimal crawl4ai crawler stub that always opens successfully."""

    def __init__(self, config: Any = None) -> None:
        self.config = config

    async def __aenter__(self) -> _ReachableCrawler:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    async def arun(self, url: str, config: Any = None, **_kwargs: Any) -> Any:
        raise AssertionError("test crawler must not fetch")


def _fake_research_config() -> ResearchConfig:
    """Build a ResearchConfig backed by fakes for graph-level tests."""

    async def search(_query: str) -> list[SearchResult]:
        return []

    async def fetch(_url: str, _query: str | None = None, /) -> FetchResult:
        return FetchResult("", FetchStatus.OK, "text/markdown", "content")

    return ResearchConfig(
        fast=FakeMessagesListChatModel(responses=[AIMessage("x")]),
        small=FakeMessagesListChatModel(responses=[AIMessage("x")]),
        search=search,
        fetch=fetch,
    )


class TestResearchConfigContract:
    @pytest.fixture
    def browser_config(self, monkeypatch):
        monkeypatch.setattr(
            "mcps.research.tools.browser.AsyncWebCrawler", _ReachableCrawler
        )
        return ServerConfig(
            router_api_base="http://localhost:4000",
            router_api_key="sk-test",
            research_fast_model="fast-model",
            research_infer_model="small-model",
            browser_cdp_url="ws://127.0.0.1:9222",
        )

    @pytest.mark.asyncio
    async def test_build_research_config_returns_valid_config(self, browser_config):
        async with httpx.AsyncClient() as http_client:
            async with build_research_config(
                browser_config, http_client
            ) as config:
                assert isinstance(config, ResearchConfig)
                assert config.fast is not None
                assert config.small is not None
                assert callable(config.search)
                assert callable(config.fetch)

    @pytest.mark.asyncio
    async def test_build_research_config_yields_none_without_browser(
        self, monkeypatch
    ):
        monkeypatch.setattr(shutil, "which", lambda _name: None)
        monkeypatch.setattr(
            "mcps.research.tools.browser.AsyncWebCrawler", _ReachableCrawler
        )
        server_config = ServerConfig(browser_cdp_url="")

        async with httpx.AsyncClient() as http_client:
            async with build_research_config(server_config, http_client) as config:
                assert config is None


# ---------------------------------------------------------------------------
# Agent contract tests
# ---------------------------------------------------------------------------


class TestAgentContract:
    @pytest.fixture
    def mock_config(self):
        return _fake_research_config()

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
        return _fake_research_config()

    @pytest.mark.asyncio
    async def test_agent_wraps_progress_in_config_for_ainvoke(self, mock_config):
        """__call__ forwards the progress callback to graph.ainvoke."""
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

        async def mock_reporter(
            message: str, progress: float, total: float | None
        ) -> None:
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
        from fastmcp import Client

        from mcps.server import create_server

        monkeypatch.setenv("ROUTER_API_BASE", "http://localhost:4000")
        monkeypatch.setenv("ROUTER_API_KEY", "sk-test")
        monkeypatch.setenv("BROWSER_CDP_URL", "ws://127.0.0.1:9222")
        monkeypatch.delenv("VAULT", raising=False)
        monkeypatch.setattr(shutil, "which", lambda _name: None)
        monkeypatch.setattr(
            "mcps.research.tools.browser.AsyncWebCrawler", _ReachableCrawler
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


class _StructuredRunnable:
    """Records one structured-output invocation and delegates to a handler."""

    def __init__(self, model: _StructuredModel, schema: Any) -> None:
        self._model = model
        self._schema = schema

    async def ainvoke(self, messages: list[Any]) -> Any:
        self._model.calls.append((self._schema, messages))
        return self._model.handlers[self._schema](messages)


class _StructuredModel:
    """Fake chat model serving structured-output handlers and recording calls.

    ``calls`` holds ``(schema, messages)`` tuples; unstructured ``ainvoke``
    records ``(None, messages)`` so single-call legacy nodes still run.
    """

    def __init__(
        self,
        handlers: dict[Any, Callable[[list[Any]], Any]],
    ) -> None:
        self.handlers = handlers
        self.calls: list[tuple[Any, list[Any]]] = []

    def with_structured_output(
        self, schema: Any, **_kwargs: Any
    ) -> _StructuredRunnable:
        return _StructuredRunnable(self, schema)

    async def ainvoke(self, messages: list[Any], **_kwargs: Any) -> AIMessage:
        self.calls.append((None, messages))
        return AIMessage("UNSTRUCTURED")


_FAILED = FetchResult("", FetchStatus.HTTP_ERROR, "", http_status=403)


def _fetched(content: str) -> FetchResult:
    return FetchResult("", FetchStatus.OK, "text/markdown", content)


_PAGE = "alpha\nbeta\ngamma\ndelta\nepsilon"
_DEFAULT_LIMITS = ResearchLimits()


def _ranges_by_url(mapping: dict[str, str]) -> Callable[[list[Any]], ExtractedRanges]:
    """Handler returning the range string mapped to whichever URL is in the prompt."""

    def handler(messages: list[Any]) -> ExtractedRanges:
        text = str(messages[-1].content)
        for url, ranges in mapping.items():
            if url in text:
                return ExtractedRanges(ranges=ranges)
        raise AssertionError(f"no URL from {list(mapping)} in extraction prompt")

    return handler


def _make_agent(
    fetch_results: dict[str, FetchResult] | None = None,
    *,
    extract: Callable[[list[Any]], ExtractedRanges] | None = None,
    answer: Callable[[list[Any]], FinalAnswer] | None = None,
    reflect: Callable[[list[Any]], Reflection] | None = None,
    limits: ResearchLimits = _DEFAULT_LIMITS,
):
    fetch_calls: list[tuple[str, str | None]] = []
    results = fetch_results or {}

    async def fake_fetch(url: str, query: str | None = None, /) -> FetchResult:
        fetch_calls.append((url, query))
        return results.get(url, _fetched("content"))

    async def fake_search(query: str) -> list[SearchResult]:
        return _SEARCH_RESULTS

    extract_handler = extract or (lambda _messages: ExtractedRanges(ranges="1"))
    answer_handler = answer or (
        lambda _messages: FinalAnswer(
            answer=[Claim(text="x", evidence=[1])],
            claims=[Claim(text="y", evidence=[2])],
        )
    )
    reflect_handler = reflect or (
        lambda _messages: Reflection(is_sufficient=True, knowledge_gap="")
    )
    fast = _StructuredModel({ExtractedRanges: extract_handler})
    small = _StructuredModel({Reflection: reflect_handler, FinalAnswer: answer_handler})
    agent = ResearchAgent(
        ResearchConfig(
            fast=cast(Any, fast),
            small=cast(Any, small),
            search=fake_search,
            fetch=fake_fetch,
        ),
        limits=limits,
    )
    return agent, fast, small, fetch_calls


def _state(**overrides: Any) -> Any:
    state = {"original_question": _QUESTION, "search_query": "quantum routing", "id": 0}
    return {**state, **overrides}


class TestWebResearchFetch:
    """Per-page verbatim evidence extraction from the web_research node."""

    async def test_initial_branch_uses_original_question(self):
        agent, _, _, fetch_calls = _make_agent()

        await agent.web_research(_state(), {})

        assert [q for _, q in fetch_calls] == [_QUESTION] * 3

    async def test_follow_up_uses_knowledge_gap(self):
        agent, _, _, fetch_calls = _make_agent()

        await agent.web_research(
            _state(knowledge_gap="What are the latency limits?"), {}
        )
        await agent.web_research(_state(knowledge_gap="N/A"), {})
        await agent.web_research(_state(knowledge_gap="  "), {})

        queries = [q for _, q in fetch_calls]
        assert queries == ["What are the latency limits?"] * 3 + [_QUESTION] * 6

    async def test_direct_url_passes_query(self):
        agent, _, _, fetch_calls = _make_agent()

        await agent.web_research(
            _state(search_query="https://source.example/page"), {}
        )

        assert fetch_calls == [("https://source.example/page", _QUESTION)]

    async def test_extracted_ranges_become_verbatim_evidence_per_page(self):
        agent, fast, _, _ = _make_agent(
            {
                "https://a.example": _fetched(_PAGE),
                "https://b.example": _fetched(_PAGE),
                "https://c.example": _fetched(_PAGE),
            },
            extract=_ranges_by_url(
                {
                    "https://a.example": "2-3",
                    "https://b.example": "",
                    "https://c.example": "5",
                }
            ),
        )

        out = await agent.web_research(_state(), {})

        assert out["evidence"] == [
            {"url": "https://a.example", "text": "beta\ngamma"},
            {"url": "https://c.example", "text": "epsilon"},
        ]
        assert len(fast.calls) == 3
        for _schema, messages in fast.calls:
            text = str(messages[-1].content)
            assert "1: alpha" in text
            urls = ("https://a.example", "https://b.example", "https://c.example")
            assert sum(url in text for url in urls) == 1

    async def test_long_lines_are_wrapped_for_ranges_and_sliced_verbatim(self):
        long_line = " ".join(f"word{i:02d}" for i in range(12))
        agent, fast, _, _ = _make_agent(
            {
                "https://a.example": _fetched(long_line),
                "https://b.example": _FAILED,
                "https://c.example": _FAILED,
            },
            extract=_ranges_by_url({"https://a.example": "2-3"}),
            limits=ResearchLimits(line_wrap_chars=20),
        )

        out = await agent.web_research(_state(), {})

        assert "4: " in str(fast.calls[0][1][-1].content)
        evidence_text = out["evidence"][0]["text"]
        assert evidence_text in long_line
        assert 0 < len(evidence_text) < len(long_line)

    async def test_invalid_ranges_are_ignored_clipped_or_merged(self):
        agent, _, _, _ = _make_agent(
            {
                "https://a.example": _fetched(_PAGE),
                "https://b.example": _FAILED,
                "https://c.example": _FAILED,
            },
            extract=_ranges_by_url({"https://a.example": "abc, 2-2, 4-99, 9-12, 3-1"}),
        )

        out = await agent.web_research(_state(), {})

        assert [e["text"] for e in out["evidence"]] == ["beta", "delta\nepsilon"]

        merged_agent, _, _, _ = _make_agent(
            {
                "https://a.example": _fetched(_PAGE),
                "https://b.example": _FAILED,
                "https://c.example": _FAILED,
            },
            extract=_ranges_by_url({"https://a.example": "1-2, 2-3, 4-4"}),
        )

        merged_out = await merged_agent.web_research(_state(), {})

        assert [e["text"] for e in merged_out["evidence"]] == [
            "alpha\nbeta\ngamma\ndelta"
        ]

    async def test_failed_and_blank_fetches_skip_extraction_and_are_recorded(self):
        agent, fast, _, _ = _make_agent(
            {
                "https://a.example": _fetched(_PAGE),
                "https://b.example": _FAILED,
                "https://c.example": _fetched("  "),
            },
            extract=_ranges_by_url({"https://a.example": "1"}),
        )

        out = await agent.web_research(_state(), {})

        assert len(fast.calls) == 1
        assert out["visits"] == [
            {"url": "https://a.example", "ok": True},
            {"url": "https://b.example", "ok": False},
            {"url": "https://c.example", "ok": True},
        ]

    async def test_extraction_error_drops_only_that_page(self):
        def extract(messages: list[Any]) -> ExtractedRanges:
            if "https://b.example" in str(messages[-1].content):
                raise RuntimeError("extraction failed")
            return ExtractedRanges(ranges="1")

        agent, _, _, _ = _make_agent(
            {
                "https://a.example": _fetched(_PAGE),
                "https://b.example": _fetched(_PAGE),
                "https://c.example": _fetched(_PAGE),
            },
            extract=extract,
        )

        out = await agent.web_research(_state(), {})

        assert [e["url"] for e in out["evidence"]] == [
            "https://a.example",
            "https://c.example",
        ]

    @pytest.mark.parametrize(
        ("state_extra", "expected"),
        [
            ({}, "absent"),
            ({"knowledge_gap": "N/A"}, "absent"),
            ({"knowledge_gap": " "}, "absent"),
            ({"knowledge_gap": "latency limits"}, "KNOWLEDGE GAP: latency limits"),
        ],
    )
    async def test_knowledge_gap_header_only_when_present(
        self, state_extra: dict[str, Any], expected: str
    ):
        agent, fast, _, _ = _make_agent()

        await agent.web_research(_state(**state_extra), {})

        assert len(fast.calls) == 3
        for _schema, messages in fast.calls:
            text = str(messages[-1].content)
            if expected == "absent":
                assert "KNOWLEDGE GAP" not in text
            else:
                assert expected in text

    async def test_reflection_and_finalize_see_numbered_evidence_only(self):
        agent, _, small, _ = _make_agent()
        state: Any = {
            "original_question": _QUESTION,
            "evidence": [
                {"url": "https://a.example", "text": "beta"},
                {"url": "https://b.example", "text": "gamma"},
            ],
            "research_loop_count": 0,
        }

        await agent.reflection(state, {})
        out = await agent.finalize_answer(state, {})

        for _schema, messages in small.calls:
            text = str(messages[-1].content)
            assert "[E1] (https://a.example)\nbeta" in text
            assert "[E2] (https://b.example)\ngamma" in text
        assert out["sources_gathered"] == [
            "https://a.example",
            "https://b.example",
        ]


_EVIDENCE = [
    {"url": "https://a.example", "text": "e1"},
    {"url": "https://b.example", "text": "e2"},
    {"url": "https://a.example", "text": "e3"},
]


def _final_state(**overrides: Any) -> Any:
    state = {
        "original_question": _QUESTION,
        "evidence": _EVIDENCE,
        "knowledge_gap": "",
        "searches": [],
    }
    return {**state, **overrides}


class TestFinalizeAnswer:
    """Cited atomic output: only claims backed by existing evidence survive."""

    async def test_only_validly_cited_claims_survive_and_cite_sources(self):
        agent, _, _, _ = _make_agent(
            answer=lambda _messages: FinalAnswer(
                answer=[Claim(text="X.", evidence=[3])],
                claims=[
                    Claim(text="P", evidence=[2]),
                    Claim(text="Q", evidence=[1, 3]),
                    Claim(text="R", evidence=[]),
                    Claim(text="S", evidence=[7]),
                ],
                not_found=["G"],
            )
        )

        out = await agent.finalize_answer(_final_state(), {})

        assert out["sources_gathered"] == ["https://a.example", "https://b.example"]
        assert out["answer"] == "X. [1]"
        assert out["explanation"] == "- P [2]\n- Q [1]\n\nNot found:\n- G"

    async def test_answer_without_valid_citation_is_not_reported(self):
        agent, _, _, _ = _make_agent(
            answer=lambda _messages: FinalAnswer(
                answer=[Claim(text="Invented.", evidence=[9])],
                claims=[Claim(text="P", evidence=[2])],
            )
        )

        out = await agent.finalize_answer(_final_state(), {})

        assert out["answer"] == _UNANSWERED
        assert out["sources_gathered"] == ["https://b.example"]

    async def test_zero_evidence_returns_fixed_reply_without_model_call(self):
        agent, _, small, _ = _make_agent()
        state = _final_state(
            evidence=[], searches=[{"query": "quantum routing", "results": []}]
        )

        out = await agent.finalize_answer(state, {})

        assert small.calls == []
        assert out["sources_gathered"] == []
        assert "quantum routing" in out["explanation"]
        assert out["answer"] == _NO_EVIDENCE_ANSWER
