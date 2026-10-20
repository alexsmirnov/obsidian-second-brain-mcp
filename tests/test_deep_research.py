"""Contract tests for deep research agent integration.

All tests mock LLM and HTTP calls — no real network access.
"""

from __future__ import annotations

import logging
import shutil
import time
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import httpx
import pytest
from fastmcp import Client
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command, Send

from mcps.config import ServerConfig, create_config
from mcps.research.agent import create_researcher
from mcps.research.config import ResearchConfig, build_research_config
from mcps.research.deep_research import (
    _FINALIZE_PROMPT,
    FinalAnswer,
    Reflection,
    ResearchAgent,
    ResearchLimits,
    ResearchPlan,
    SelectedPages,
)
from mcps.research.tools import SearchResult
from mcps.research.tools.models import FetchResult, FetchStatus
from mcps.server import create_server

# ---------------------------------------------------------------------------
# Fakes and helpers
# ---------------------------------------------------------------------------

_AGENT_LOGGER = "mcps.research.deep_research"
_QUESTION = "How does quantum routing work?"
_SEARCH_RESULTS = [
    SearchResult(url=f"https://{name}.example", title=name.upper(), snippet="s")
    for name in ("a", "b", "c")
]
_FAILED = FetchResult("", FetchStatus.HTTP_ERROR, "", http_status=403)
_PAGE = "alpha\nbeta\ngamma\ndelta\nepsilon"
_TRACE_PAGE = "unique-page-line\nother"
_DEFAULT_LIMITS = ResearchLimits()


def _fetched(content: str) -> FetchResult:
    return FetchResult("", FetchStatus.OK, "text/markdown", content)


def _sr(url: str, title: str = "T", snippet: str = "s") -> SearchResult:
    return SearchResult(url=url, title=title, snippet=snippet)


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


def _as_handler(value: Any) -> Callable[[list[Any]], Any]:
    """Normalize a value, iterator, or callable into a message handler."""
    if callable(value):
        return value
    if isinstance(value, Iterator):
        return lambda _messages: next(value)
    return lambda _messages: value


class _StructuredModel:
    """Fake chat model serving structured-output handlers and recording calls."""

    def __init__(self, handlers: dict[Any, Any]) -> None:
        self.handlers = {schema: _as_handler(h) for schema, h in handlers.items()}
        self.calls: list[tuple[Any, list[Any]]] = []

    def with_structured_output(
        self, schema: Any, *, include_raw: bool = False, **_kwargs: Any
    ) -> Any:
        async def ainvoke(messages: list[Any]) -> Any:
            self.calls.append((schema, messages))
            parsed = self.handlers[schema](messages)
            if include_raw:
                return {"raw": AIMessage(parsed.model_dump_json()), "parsed": parsed}
            return parsed

        return SimpleNamespace(ainvoke=ainvoke)


def _sent_urls(command: Command[Any]) -> list[str]:
    """URLs of the web_fetch branches carried by a select_urls command."""
    return [send.arg["url"] for send in cast(list[Send], command.goto)]


def _make_agent(
    fetch_results: dict[str, FetchResult] | None = None,
    *,
    fetch: Callable[..., Any] | None = None,
    answer: Any = None,
    reflect: Any = None,
    plan: Any = None,
    select: Any = None,
    search_results: list[SearchResult] = _SEARCH_RESULTS,
    search_by_query: Callable[[str], list[SearchResult]] | None = None,
    limits: ResearchLimits = _DEFAULT_LIMITS,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[
    ResearchAgent,
    _StructuredModel,
    _StructuredModel,
    list[tuple[str, str | None]],
]:
    fetch_calls: list[tuple[str, str | None]] = []
    results = fetch_results or {}

    async def fake_fetch(url: str, query: str | None = None, /) -> FetchResult:
        fetch_calls.append((url, query))
        if fetch is not None:
            return await fetch(url, query)
        return results.get(url, _fetched("content"))

    async def fake_search(query: str) -> list[SearchResult]:
        return search_by_query(query) if search_by_query is not None else search_results

    fast = _StructuredModel(
        {
            ResearchPlan: plan or ResearchPlan(search_queries=["q"], fetch_query="k"),
            SelectedPages: select or SelectedPages(indices=[]),
        }
    )
    small = _StructuredModel(
        {
            Reflection: reflect or Reflection(is_sufficient=True, knowledge_gap=""),
            FinalAnswer: answer or FinalAnswer(answer="x", explanation="y"),
        }
    )
    agent = ResearchAgent(
        ResearchConfig(
            fast=cast(Any, fast),
            small=cast(Any, small),
            search=fake_search,
            fetch=fake_fetch,
        ),
        limits=limits,
        clock=clock,
    )
    return agent, fast, small, fetch_calls


def _state(**overrides: Any) -> Any:
    return {
        "messages": [],
        "original_question": _QUESTION,
        "fetch_query": _QUESTION,
        **overrides,
    }


def _fetch_state(**overrides: Any) -> Any:
    return {
        "url": "https://a.example",
        "title": "A",
        "fetch_query": _QUESTION,
        **overrides,
    }


def _final_state(**overrides: Any) -> Any:
    return {
        "messages": [HumanMessage("round 1"), AIMessage('{"is_sufficient": true}')],
        **overrides,
    }


# ---------------------------------------------------------------------------
# Config & Server contract tests
# ---------------------------------------------------------------------------


class TestServerConfigContract:
    def test_default_router_fields_and_perplexity_removed(self):
        config = ServerConfig()
        assert config.router_api_base == ""
        assert config.router_api_key == ""
        assert not hasattr(config, "perplexity_api_key")

    def test_create_config_reads_router_env_vars(self, monkeypatch):
        monkeypatch.setenv("ROUTER_API_BASE", "http://localhost:4000")
        monkeypatch.setenv("ROUTER_DOCKER_API_BASE", "http://localhost:4000")
        monkeypatch.setenv("ROUTER_API_KEY", "sk-test123")
        config = create_config()
        assert config.router_api_base == "http://localhost:4000"
        assert config.router_api_key == "sk-test123"


class TestResearchConfigContract:
    def test_create_researcher_validates_implementation(self):
        config = _make_agent()[0].config
        assert callable(create_researcher(config, implementation="deep_research"))
        with pytest.raises(ValueError, match="Unknown implementation"):
            create_researcher(config, implementation=cast(Any, "bogus"))

    async def test_build_research_config_returns_valid_config(self, monkeypatch):
        monkeypatch.setattr(
            "mcps.research.tools.browser.AsyncWebCrawler", _ReachableCrawler
        )
        browser_config = ServerConfig(
            router_api_base="http://localhost:4000",
            router_api_key="sk-test",
            research_fast_model="fast-model",
            research_infer_model="small-model",
            browser_cdp_url="ws://127.0.0.1:9222",
        )
        async with (
            httpx.AsyncClient() as http_client,
            build_research_config(browser_config, http_client) as config,
        ):
            assert isinstance(config, ResearchConfig)
            assert config.fast is not None
            assert config.small is not None
            assert callable(config.search)
            assert callable(config.fetch)

    async def test_build_research_config_yields_none_without_browser(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda _name: None)
        monkeypatch.setattr(
            "mcps.research.tools.browser.AsyncWebCrawler", _ReachableCrawler
        )
        server_config = ServerConfig(browser_cdp_url="")

        async with (
            httpx.AsyncClient() as http_client,
            build_research_config(server_config, http_client) as config,
        ):
            assert config is None


class TestAgentAndProgressContract:
    @pytest.fixture
    def researcher(self):
        config = _make_agent()[0].config
        return create_researcher(config, implementation="deep_research")

    async def test_agent_returns_research_response_contract(self, researcher):
        """Response follows ResearchResponse TypedDict shape and values."""
        mock_sources = ["https://example.com/source1", "https://example.com/source2"]

        async def mock_invoke(_input: Any, **_kwargs: Any) -> dict[str, Any]:
            return {
                "answer": "42",
                "explanation": "Derived from authoritative sources.",
                "sources_gathered": mock_sources,
            }

        with patch.object(researcher.graph, "ainvoke", side_effect=mock_invoke):
            result = await researcher("What is the answer?")

        assert result == {
            "answer": "42",
            "explanation": "Derived from authoritative sources.",
            "sources": mock_sources,
        }

    async def test_agent_forwards_optional_progress_reporter(self, researcher):
        """__call__ forwards progress callback in RunnableConfig when provided."""
        captured: list[dict[str, Any]] = []

        async def mock_invoke(_input: Any, **kwargs: Any) -> dict[str, Any]:
            captured.append(kwargs)
            return {"answer": "42", "explanation": "test", "sources_gathered": []}

        async def mock_reporter(
            message: str, progress: float, total: float | None
        ) -> None:
            pass

        with patch.object(researcher.graph, "ainvoke", side_effect=mock_invoke):
            with_progress = await researcher("query", progress=mock_reporter)
            without_progress = await researcher("bare query")

        reporter = captured[0]["config"]["configurable"]["progress_reporter"]
        assert reporter is mock_reporter
        assert captured[1]["config"] is None
        assert with_progress["answer"] == "42"
        assert without_progress["answer"] == "42"


class TestToolContract:
    async def test_tool_is_registered(self, monkeypatch):
        """web_research tool appears in the server's registered tools."""
        monkeypatch.setenv("ROUTER_API_BASE", "http://localhost:4000")
        monkeypatch.setenv("ROUTER_API_KEY", "sk-test")
        monkeypatch.setenv("BROWSER_CDP_URL", "ws://127.0.0.1:9222")
        monkeypatch.delenv("VAULT", raising=False)
        monkeypatch.setattr(shutil, "which", lambda _name: None)
        monkeypatch.setattr(
            "mcps.research.tools.browser.AsyncWebCrawler", _ReachableCrawler
        )
        server = create_server(create_config())

        async with Client(server.mcp) as client:
            tools = await client.list_tools()
        assert "web_research" in [t.name for t in tools]


# ---------------------------------------------------------------------------
# Page fetching and final answer
# ---------------------------------------------------------------------------


class TestWebFetch:
    """Single-page fetch: the tool's own filter output becomes the round page."""

    async def test_successful_fetch_becomes_page_with_title_and_fetch_query(self):
        agent, _, _, fetch_calls = _make_agent({"https://a.example": _fetched(_PAGE)})

        out = await agent.web_fetch(
            _fetch_state(
                url="https://a.example", title="Title A", fetch_query="qubit routing"
            ),
            {},
        )

        assert out == {
            "visits": [{"url": "https://a.example", "ok": True}],
            "pages": [
                {"url": "https://a.example", "title": "Title A", "content": _PAGE}
            ],
        }
        assert fetch_calls == [("https://a.example", "qubit routing")]

    async def test_failed_and_blank_fetches_are_recorded_without_pages(self):
        agent, _, _, _ = _make_agent(
            {
                "https://b.example": _FAILED,
                "https://c.example": _fetched("  "),
            },
        )

        out_b = await agent.web_fetch(_fetch_state(url="https://b.example"), {})
        out_c = await agent.web_fetch(_fetch_state(url="https://c.example"), {})

        assert out_b == {
            "visits": [{"url": "https://b.example", "ok": False}],
            "pages": [],
        }
        assert out_c == {
            "visits": [{"url": "https://c.example", "ok": True}],
            "pages": [],
        }

    async def test_fetch_uses_no_model_call(self):
        agent, fast, small, _ = _make_agent()

        await agent.web_fetch(_fetch_state(), {})

        assert fast.calls == []
        assert small.calls == []


class TestFinalizeAnswer:
    """finalize_answer asks the model once over the whole conversation."""

    async def test_model_receives_history_then_finalize_prompt_only(self):
        agent, _, small, _ = _make_agent(
            answer=FinalAnswer(answer="X.", explanation="P [https://a.example]")
        )
        state = _final_state()

        await agent.finalize_answer(state, {})

        assert len(small.calls) == 1
        schema, messages = small.calls[0]
        assert schema is FinalAnswer
        assert messages[:-1] == state["messages"]
        assert isinstance(messages[-1], HumanMessage)
        assert messages[-1].content == _FINALIZE_PROMPT

    async def test_output_is_passed_through_with_deduped_http_sources(self):
        agent, _, _, _ = _make_agent(
            answer=FinalAnswer(
                answer="X.",
                explanation="Detail.\n\nNot found:\n- G",
                sources=[
                    "https://a.example",
                    "not a url",
                    "https://b.example",
                    "https://a.example",
                ],
            )
        )

        out = await agent.finalize_answer(_final_state(), {})

        assert out == {
            "answer": "X.",
            "explanation": "Detail.\n\nNot found:\n- G",
            "sources_gathered": ["https://a.example", "https://b.example"],
        }

    def test_finalize_prompt_covers_sufficient_and_limit_stops(self):
        assert "is_sufficient" in _FINALIZE_PROMPT
        assert "Not found" in _FINALIZE_PROMPT


# ---------------------------------------------------------------------------
# Round graph: generate_query, route_round, web_search, select_urls
# ---------------------------------------------------------------------------


class TestGenerateQuery:
    """The start node caps, dedupes and validates the shared round plan."""

    async def test_generate_query_caps_dedupes_and_starts_clock(self):
        agent, _, _, _ = _make_agent(
            limits=ResearchLimits(max_initial_queries=2),
            clock=iter([100.0]).__next__,
            plan=ResearchPlan(
                search_queries=[
                    "  Quantum Routing ",
                    "quantum routing",
                    "NIST PQC",
                    "extra",
                ],
                links=["https://x.example/a", "not a url"],
                fetch_query="qubit routing",
            ),
        )

        out = await agent.generate_query({"question": _QUESTION}, {})

        assert out["search_queries"] == ["Quantum Routing", "NIST PQC"]
        assert out["pending_links"] == ["https://x.example/a"]
        assert out["fetch_query"] == "qubit routing"
        assert out["started_at"] == 100.0

    async def test_blank_fetch_query_falls_back_to_question(self):
        agent, _, _, _ = _make_agent(
            plan=ResearchPlan(search_queries=["q"], fetch_query="  "),
        )

        out = await agent.generate_query({"question": _QUESTION}, {})

        assert out["fetch_query"] == _QUESTION

    async def test_empty_plan_falls_back_to_question(self):
        agent, _, _, _ = _make_agent(plan=ResearchPlan())

        out = await agent.generate_query({"question": _QUESTION}, {})

        assert out["search_queries"] == [_QUESTION]
        assert out["pending_links"] == []


class TestRouteRound:
    """route_round fans out searches or goes straight to link selection."""

    def test_route_round_searches_queries_or_goes_straight_to_selection(self):
        agent, _, _, _ = _make_agent()

        sends = agent.route_round(
            _state(search_queries=["a", "b"], pending_links=["https://x.example"])
        )
        assert [send.arg["search_query"] for send in sends] == ["a", "b"]
        assert all(send.node == "web_search" for send in sends)

        assert (
            agent.route_round(
                _state(search_queries=[], pending_links=["https://x.example"])
            )
            == "select_urls"
        )


class TestWebSearch:
    """web_search records one search run for the current round."""

    async def test_web_search_records_query_and_results(self):
        agent, _, _, _ = _make_agent()

        out = await agent.web_search({"search_query": "quantum routing"}, {})

        assert out == {
            "searches": [{"query": "quantum routing", "results": _SEARCH_RESULTS}]
        }


class TestSelectUrls:
    """select_urls dedupes, keeps links first and sends one branch per page."""

    async def test_select_urls_puts_links_first_dedupes_and_sends_to_web_fetch(self):
        agent, fast, _, _ = _make_agent(
            limits=ResearchLimits(max_pages_per_round=3),
            select=SelectedPages(indices=[2, 0, 0, 99, -1]),
        )
        state = _state(
            fetch_query="qubit routing",
            knowledge_gap="gap1",
            search_queries=["q"],
            pending_links=[
                "https://direct.example/page",
                "https://direct.example/page#again",
            ],
            searches=[
                {
                    "query": "q",
                    "results": [
                        _sr("HTTPS://DIRECT.EXAMPLE/page/#sec"),
                        _sr("https://b.example/one", title="One"),
                        _sr("https://b.example/one#top"),
                        _sr("https://c.example/bad"),
                        _sr("https://d.example/two", title="Two"),
                    ],
                },
            ],
        )

        out = await agent.select_urls(state, {})

        assert isinstance(out, Command)
        assert out.update is None
        assert out.goto == [
            Send(
                "web_fetch",
                {
                    "url": "https://direct.example/page",
                    "title": "T",
                    "fetch_query": "qubit routing",
                },
            ),
            Send(
                "web_fetch",
                {
                    "url": "https://d.example/two",
                    "title": "Two",
                    "fetch_query": "qubit routing",
                },
            ),
            Send(
                "web_fetch",
                {
                    "url": "https://b.example/one",
                    "title": "One",
                    "fetch_query": "qubit routing",
                },
            ),
        ]
        text = str(fast.calls[-1][1][-1].content)
        assert "[0]" in text
        assert "https://c.example/bad" in text

    async def test_all_candidates_reach_triage_before_fetch_budget_applies(self):
        early = [_sr(f"https://early-{i}.example") for i in range(32)]
        agent, fast, _, _ = _make_agent(
            limits=ResearchLimits(max_pages_per_round=3),
            select=SelectedPages(indices=[32, 0, 1]),
        )
        state = _state(
            search_queries=["q1", "q2"],
            pending_links=["https://direct.example/page"],
            searches=[
                {"query": "q1", "results": early},
                {
                    "query": "q2",
                    "results": [
                        _sr("https://early-0.example#section"),
                        _sr("https://late.example"),
                    ],
                },
            ],
        )

        out = await agent.select_urls(state, {})

        text = str(fast.calls[-1][1][-1].content)
        expected_candidates = [f"https://early-{i}.example" for i in range(32)] + [
            "https://late.example"
        ]
        for index, url in enumerate(expected_candidates):
            assert text.count(f"[{index}] {url}\n") == 1
        assert "early-0.example#section" not in text
        assert "direct.example" not in text
        assert _sent_urls(out) == [
            "https://direct.example/page",
            "https://late.example",
            "https://early-0.example",
        ]

    @pytest.mark.parametrize(
        ("knowledge_gap", "expected"),
        [
            ("", None),
            ("  ", None),
            ("  missing csdid vignette  ", "KNOWLEDGE GAP: missing csdid vignette"),
        ],
    )
    async def test_triage_receives_gap_only_when_meaningful(
        self, knowledge_gap: str, expected: str | None
    ):
        agent, fast, _, _ = _make_agent(select=SelectedPages(indices=[0]))
        state = _state(
            knowledge_gap=knowledge_gap,
            search_queries=["q"],
            searches=[{"query": "q", "results": [_sr("https://a.example")]}],
        )

        await agent.select_urls(state, {})

        text = str(fast.calls[-1][1][-1].content)
        if expected is None:
            assert "KNOWLEDGE GAP" not in text
        else:
            assert expected in text.strip()

    async def test_select_urls_skips_model_without_candidates_and_survives_triage_error(
        self,
    ):
        agent, fast, _, _ = _make_agent()

        out = await agent.select_urls(
            _state(pending_links=["https://x.example"]),
            {},
        )

        assert fast.calls == []
        assert isinstance(out, Command)
        assert _sent_urls(out) == ["https://x.example"]

        def boom(_messages: list[Any]) -> SelectedPages:
            raise RuntimeError("triage failed")

        error_agent, _, _, _ = _make_agent(select=boom)
        error_out = await error_agent.select_urls(
            _state(
                pending_links=["https://x.example"],
                search_queries=["q"],
                searches=[
                    {"query": "q", "results": [_sr("https://candidate.example")]}
                ],
            ),
            {},
        )

        assert isinstance(error_out, Command)
        assert _sent_urls(error_out) == ["https://x.example"]

    async def test_select_urls_skips_triage_when_links_fill_budget(self):
        agent, fast, _, _ = _make_agent(
            limits=ResearchLimits(max_pages_per_round=1),
            select=SelectedPages(indices=[0]),
        )
        state = _state(
            pending_links=["https://link.example"],
            search_queries=["q"],
            searches=[{"query": "q", "results": [_sr("https://candidate.example")]}],
        )

        out = await agent.select_urls(state, {})

        assert fast.calls == []
        assert isinstance(out, Command)
        assert _sent_urls(out) == ["https://link.example"]

    async def test_select_urls_routes_to_reflection_when_no_urls_selected(self):
        agent, _, _, _ = _make_agent(select=SelectedPages(indices=[]))
        state = _state(
            pending_links=[],
            search_queries=["q"],
            searches=[{"query": "q", "results": []}],
        )

        out = await agent.select_urls(state, {})

        assert isinstance(out, Command)
        assert out.update is None
        assert out.goto == "reflection"


class TestFetchQueryFallback:
    """select_urls hands every web_fetch branch a non-blank fetch query."""

    async def test_blank_round_fetch_query_falls_back_to_question(self):
        agent, _, _, _ = _make_agent()

        out = await agent.select_urls(
            _state(fetch_query="  ", pending_links=["https://a.example"]), {}
        )

        sends = cast(list[Send], cast(Command[Any], out).goto)
        assert [send.arg["fetch_query"] for send in sends] == [_QUESTION]


class TestReflectionRound:
    """reflection renders this round, extends the history and clears the buffers."""

    async def test_round_message_shows_search_outcomes_and_page_content(self):
        agent, _, small, _ = _make_agent(reflect=Reflection(is_sufficient=False))
        state = _state(
            searches=[
                {
                    "query": "q1",
                    "results": [
                        _sr("https://a.example"),
                        _sr("https://c.example"),
                        _sr("https://d.example"),
                    ],
                },
                {"query": "q2", "results": []},
            ],
            visits=[
                {"url": "https://a.example", "ok": True},
                {"url": "https://c.example", "ok": False},
                {"url": "https://link.example", "ok": True},
            ],
            pages=[{"url": "https://a.example", "title": "A", "content": "page-body"}],
        )

        await agent.reflection(state, {})

        text = str(small.calls[-1][1][-1].content)
        lines = text.splitlines()

        def line_with(fragment: str) -> str:
            return next(line for line in lines if fragment in line)

        assert "VISITED" in line_with("a.example | ")
        assert "FAILED" in line_with("c.example | ")
        assert "not selected" in line_with("d.example | ")
        assert "VISITED" in line_with("link.example")
        assert "(no results)" in lines
        assert "https://a.example" in text
        assert "page-body" in text

    async def test_no_pages_are_stated_explicitly(self):
        agent, _, small, _ = _make_agent(reflect=Reflection(is_sufficient=False))

        await agent.reflection(_state(searches=[], visits=[], pages=[]), {})

        assert "No pages fetched this round." in str(small.calls[-1][1][-1].content)

    async def test_history_grows_and_buffers_are_cleared(self):
        agent, _, small, _ = _make_agent(reflect=Reflection(is_sufficient=False))
        first_state = _state(
            messages=[],
            searches=[{"query": "q1", "results": []}],
            visits=[],
            pages=[{"url": "https://a.example", "title": "A", "content": "round-one"}],
        )

        first = await agent.reflection(first_state, {})

        human, raw = first["messages"]
        assert isinstance(human, HumanMessage)
        assert isinstance(raw, AIMessage)
        for key in ("searches", "visits", "pages"):
            assert first[key].value == []

        second_state = _state(
            messages=first["messages"],
            searches=[{"query": "q2", "results": []}],
            visits=[],
            pages=[{"url": "https://b.example", "title": "B", "content": "round-two"}],
        )
        await agent.reflection(second_state, {})

        sent = small.calls[-1][1]
        assert sent[:2] == [human, raw]
        assert "round-two" in str(sent[-1].content)
        assert "round-one" not in str(sent[-1].content)
        assert "q1" not in str(sent[-1].content)

    async def test_follow_ups_repeating_earlier_queries_and_links_are_kept(self):
        agent, _, _, _ = _make_agent(
            reflect=Reflection(
                is_sufficient=False,
                search_queries=["quantum routing"],
                links=["https://a.example/paper"],
            ),
        )
        state = _state(
            searches=[{"query": "quantum routing", "results": []}],
            visits=[{"url": "https://a.example/paper", "ok": True}],
        )

        out = await agent.reflection(state, {})

        assert out["search_queries"] == ["quantum routing"]
        assert out["pending_links"] == ["https://a.example/paper"]

    async def test_follow_up_queries_and_links_keep_independent_limits(self):
        agent, _, _, _ = _make_agent(
            limits=ResearchLimits(max_follow_ups=2),
            reflect=Reflection(
                is_sufficient=False,
                search_queries=[
                    "  Quantum Routing ",
                    "post quantum crypto",
                    "POST QUANTUM CRYPTO",
                    "third",
                    "fourth",
                ],
                links=[
                    "https://a.example/paper#sec2",
                    "not a url",
                    "https://b.example/next",
                    "https://b.example/next#top",
                    "https://c.example/more",
                ],
                fetch_query="pqc latency",
            ),
        )

        out = await agent.reflection(_state(), {})

        assert out["search_queries"] == ["Quantum Routing", "post quantum crypto"]
        assert out["pending_links"] == [
            "https://a.example/paper#sec2",
            "https://b.example/next",
        ]
        assert out["fetch_query"] == "pqc latency"

    @pytest.mark.parametrize(
        ("knowledge_gap", "expected"),
        [
            ("", _QUESTION),
            ("latency limits", "latency limits"),
        ],
    )
    async def test_fetch_query_falls_back_through_new_gap_to_question(
        self, knowledge_gap: str, expected: str
    ):
        agent, _, _, _ = _make_agent(
            reflect=Reflection(
                is_sufficient=False, fetch_query="  ", knowledge_gap=knowledge_gap
            ),
        )

        out = await agent.reflection(_state(knowledge_gap="previous round gap"), {})

        assert out["fetch_query"] == expected


class TestReflectionParseFailure:
    """An unparsable reflection ends research instead of crashing the graph."""

    async def test_unparsed_reflection_finalizes_and_keeps_raw_out_of_history(self):
        agent, _, small, _ = _make_agent()

        def unparsed(_messages: list[Any]) -> Any:
            return None

        small.handlers[Reflection] = unparsed
        small_chain = small.with_structured_output

        def wrapped(schema: Any, *, include_raw: bool = False, **kwargs: Any) -> Any:
            chain = small_chain(schema, include_raw=False, **kwargs)

            async def ainvoke(messages: list[Any]) -> Any:
                await chain.ainvoke(messages)
                return {"raw": AIMessage("garbage"), "parsed": None}

            return SimpleNamespace(ainvoke=ainvoke)

        small.with_structured_output = wrapped  # type: ignore[method-assign]

        out = await agent.reflection(_state(searches=[], visits=[], pages=[]), {})

        assert out["is_sufficient"] is True
        assert out["search_queries"] == []
        assert out["pending_links"] == []
        assert [type(m) for m in out["messages"]] == [HumanMessage]
        assert agent.evaluate_research({**_state(), **out}) == "finalize_answer"


class TestEvaluateResearch:
    """The sufficiency, time and follow-up gates decide when to finalize."""

    @pytest.mark.parametrize(
        ("clock_value", "is_sufficient", "queries", "links", "expected"),
        [
            (10, True, ["q"], [], "finalize_answer"),
            (30, False, ["q"], [], "finalize_answer"),
            (10, False, [], [], "finalize_answer"),
            (10, False, [], ["https://x.example"], "select_urls"),
            (10, False, ["q"], [], "send"),
        ],
    )
    def test_evaluate_research_routes_by_sufficiency_follow_ups_and_time(
        self, clock_value, is_sufficient, queries, links, expected
    ):
        agent, _, _, _ = _make_agent(
            limits=ResearchLimits(round_start_budget=25),
            clock=iter([clock_value]).__next__,
        )
        state = _state(
            started_at=0,
            is_sufficient=is_sufficient,
            search_queries=queries,
            pending_links=links,
        )

        result = agent.evaluate_research(state)

        if expected == "send":
            assert [send.node for send in result] == ["web_search"]
        else:
            assert result == expected


class TestGraphRound:
    """The wired graph runs rounds end to end and returns the model's answer."""

    async def test_graph_runs_one_round_to_answer(self):
        agent, _, _, _ = _make_agent(
            fetch_results={"https://a.example": _fetched(_PAGE)},
            search_results=[_sr("https://a.example")],
            plan=ResearchPlan(search_queries=["q"], fetch_query="k"),
            select=SelectedPages(indices=[0]),
            reflect=Reflection(is_sufficient=True),
            answer=FinalAnswer(
                answer="A.", explanation="B.", sources=["https://a.example"]
            ),
        )

        result = await agent("question")

        assert result == {
            "answer": "A.",
            "explanation": "B.",
            "sources": ["https://a.example"],
        }

    async def test_graph_second_round_sees_first_round_and_finalize_sees_both(self):
        reflections = iter(
            [
                Reflection(
                    is_sufficient=False, search_queries=["q2"], fetch_query="k2"
                ),
                Reflection(is_sufficient=True),
            ]
        )
        searches = {
            "q1": [_sr("https://a.example")],
            "q2": [_sr("https://c.example")],
        }
        agent, _, small, fetch_calls = _make_agent(
            fetch_results={
                "https://a.example": _fetched("page-one"),
                "https://c.example": _fetched("page-two"),
            },
            plan=ResearchPlan(search_queries=["q1"], fetch_query="k1"),
            select=SelectedPages(indices=[0]),
            reflect=reflections,
            answer=FinalAnswer(answer="A."),
            search_by_query=searches.__getitem__,
        )

        await agent("question")

        assert fetch_calls == [
            ("https://a.example", "k1"),
            ("https://c.example", "k2"),
        ]
        reflection_calls = [m for schema, m in small.calls if schema is Reflection]
        assert len(reflection_calls) == 2
        assert "page-one" in str(reflection_calls[1][-3].content)
        assert "page-two" in str(reflection_calls[1][-1].content)
        finalize_messages = small.calls[-1][1]
        history = "\n".join(str(m.content) for m in finalize_messages)
        assert "page-one" in history
        assert "page-two" in history
        assert finalize_messages[-1].content == _FINALIZE_PROMPT

    async def test_graph_finalizes_after_round_without_pages(self):
        agent, _, small, fetch_calls = _make_agent(
            search_results=[],
            plan=ResearchPlan(search_queries=["q"], fetch_query="k"),
            select=SelectedPages(indices=[]),
            reflect=Reflection(is_sufficient=False),
            answer=FinalAnswer(answer="none", explanation="Not found:\n- q"),
        )

        result = await agent("question")

        assert fetch_calls == []
        assert result["answer"] == "none"
        assert result["sources"] == []
        assert small.calls[-1][0] is FinalAnswer


class TestDebugTrace:
    """DEBUG logs trace each node's inputs and decisions for diagnosis."""

    @staticmethod
    def _one_round_agent() -> ResearchAgent:
        agent, _, _, _ = _make_agent(
            fetch_results={"https://picked.example": _fetched(_TRACE_PAGE)},
            search_results=[
                _sr("https://picked.example"),
                _sr("https://skipped.example"),
            ],
            plan=ResearchPlan(search_queries=["q"], fetch_query="k"),
            select=SelectedPages(indices=[0]),
            reflect=Reflection(is_sufficient=True, knowledge_gap="gap-marker"),
            answer=FinalAnswer(answer="A.", sources=["https://picked.example"]),
        )
        return agent

    async def test_debug_logs_trace_search_selection_pages_and_reflection(
        self, caplog: pytest.LogCaptureFixture
    ):
        agent = self._one_round_agent()

        with caplog.at_level(logging.DEBUG, logger=_AGENT_LOGGER):
            await agent("question")

        debug = "\n".join(
            record.getMessage()
            for record in caplog.records
            if record.levelno == logging.DEBUG
        )
        assert "https://skipped.example" in debug
        assert "https://picked.example" in debug
        assert "unique-page-line" in debug
        assert "is_sufficient" in debug
        assert "gap-marker" in debug

    async def test_info_level_emits_no_debug_records(
        self, caplog: pytest.LogCaptureFixture
    ):
        agent = self._one_round_agent()

        with caplog.at_level(logging.INFO, logger=_AGENT_LOGGER):
            await agent("question")

        levels = {record.levelno for record in caplog.records}
        assert logging.DEBUG not in levels
