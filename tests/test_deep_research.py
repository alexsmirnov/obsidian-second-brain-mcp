"""Contract tests for deep research agent integration.

All tests mock LLM and HTTP calls — no real network access.
"""

from __future__ import annotations

import asyncio
import shutil
import time
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import httpx
import pytest
from fastmcp import Client

from mcps.config import ServerConfig, create_config
from mcps.research.agent import create_researcher
from mcps.research.config import ResearchConfig, build_research_config
from mcps.research.deep_research import (
    _NO_EVIDENCE_ANSWER,
    _UNANSWERED,
    Claim,
    ExtractedRanges,
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

_QUESTION = "How does quantum routing work?"
_SEARCH_RESULTS = [
    SearchResult(url=f"https://{name}.example", title=name.upper(), snippet="s")
    for name in ("a", "b", "c")
]
_FAILED = FetchResult("", FetchStatus.HTTP_ERROR, "", http_status=403)
_PAGE = "alpha\nbeta\ngamma\ndelta\nepsilon"
_DEFAULT_LIMITS = ResearchLimits()
_EVIDENCE = [
    {"url": "https://a.example", "text": "e1"},
    {"url": "https://b.example", "text": "e2"},
    {"url": "https://a.example", "text": "e3"},
]


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

    def with_structured_output(self, schema: Any, **_kwargs: Any) -> Any:
        async def ainvoke(messages: list[Any]) -> Any:
            self.calls.append((schema, messages))
            return self.handlers[schema](messages)

        return SimpleNamespace(ainvoke=ainvoke)


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
    fetch: Callable[..., Any] | None = None,
    extract: Any = None,
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
            ExtractedRanges: extract or ExtractedRanges(ranges="1"),
            ResearchPlan: plan or ResearchPlan(search_queries=["q"], fetch_query="k"),
            SelectedPages: select or SelectedPages(indices=[]),
        }
    )
    small = _StructuredModel(
        {
            Reflection: reflect or Reflection(is_sufficient=True, knowledge_gap=""),
            FinalAnswer: answer
            or FinalAnswer(
                answer=[Claim(text="x", evidence=[1])],
                claims=[Claim(text="y", evidence=[2])],
            ),
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
        "original_question": _QUESTION,
        "fetch_query": _QUESTION,
        "to_visit": [result.url for result in _SEARCH_RESULTS],
        **overrides,
    }


def _final_state(**overrides: Any) -> Any:
    return {
        "original_question": _QUESTION,
        "evidence": _EVIDENCE,
        "knowledge_gap": "",
        "searches": [],
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
# Fetch query propagation and evidence filtering
# ---------------------------------------------------------------------------


class TestWebResearchFetch:
    """Per-page verbatim evidence extraction from the web_research node."""

    async def test_extracted_ranges_become_verbatim_evidence_per_page(self):
        urls = ("https://a.example", "https://b.example", "https://c.example")
        agent, fast, _, _ = _make_agent(
            dict.fromkeys(urls, _fetched(_PAGE)),
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

    @pytest.mark.parametrize(
        ("ranges", "expected"),
        [
            ("abc, 2-2, 4-99, 9-12, 3-1", ["beta", "delta\nepsilon"]),
            ("1-2, 2-3, 4-4", ["alpha\nbeta\ngamma\ndelta"]),
        ],
    )
    async def test_invalid_ranges_are_ignored_clipped_or_merged(
        self, ranges: str, expected: list[str]
    ):
        agent, _, _, _ = _make_agent(
            {
                "https://a.example": _fetched(_PAGE),
                "https://b.example": _FAILED,
                "https://c.example": _FAILED,
            },
            extract=_ranges_by_url({"https://a.example": ranges}),
        )

        out = await agent.web_research(_state(), {})

        assert [e["text"] for e in out["evidence"]] == expected

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

        urls = ("https://a.example", "https://b.example", "https://c.example")
        agent, _, _, _ = _make_agent(
            dict.fromkeys(urls, _fetched(_PAGE)),
            extract=extract,
        )

        out = await agent.web_research(_state(), {})

        assert [e["url"] for e in out["evidence"]] == [
            "https://a.example",
            "https://c.example",
        ]

    async def test_page_extraction_starts_before_slower_fetches_finish(self):
        extraction_started = asyncio.Event()

        async def fetch(url: str, _query: str | None = None) -> FetchResult:
            if url == "https://b.example":
                await asyncio.wait_for(extraction_started.wait(), timeout=1)
            return _fetched(_PAGE)

        def extract(_messages: list[Any]) -> ExtractedRanges:
            extraction_started.set()
            return ExtractedRanges(ranges="1")

        agent, _, _, _ = _make_agent(fetch=fetch, extract=extract)
        out = await agent.web_research(
            _state(to_visit=["https://a.example", "https://b.example"]), {}
        )

        assert out["visits"] == [
            {"url": "https://a.example", "ok": True},
            {"url": "https://b.example", "ok": True},
        ]
        assert [e["url"] for e in out["evidence"]] == [
            "https://a.example",
            "https://b.example",
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


class TestFinalizeAnswer:
    """Cited atomic output: only claims backed by existing evidence survive."""

    async def test_only_validly_cited_claims_survive_and_cite_sources(self):
        agent, _, _, _ = _make_agent(
            answer=FinalAnswer(
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
            answer=FinalAnswer(
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
    """select_urls dedupes, keeps links first and triages search candidates."""

    async def test_select_urls_puts_links_first_dedupes_and_applies_triage(self):
        agent, fast, _, _ = _make_agent(
            limits=ResearchLimits(max_pages_per_round=3),
            select=SelectedPages(indices=[2, 0, 0, 99, -1]),
        )
        state = _state(
            fetch_query="qubit routing",
            search_queries=["q"],
            pending_links=["https://direct.example/page", "https://a.example/doc"],
            visits=[{"url": "https://a.example/doc", "ok": True}],
            searches=[
                {"query": "old", "results": [_sr("https://old.example")]},
                {
                    "query": "q",
                    "results": [
                        _sr("HTTPS://A.EXAMPLE/doc/#sec"),
                        _sr("https://b.example/one"),
                        _sr("https://b.example/one#top"),
                        _sr("https://c.example/bad"),
                        _sr("https://d.example/two"),
                    ],
                },
            ],
        )

        out = await agent.select_urls(state, {})

        assert out["to_visit"] == [
            "https://direct.example/page",
            "https://d.example/two",
            "https://b.example/one",
        ]
        text = str(fast.calls[-1][1][-1].content)
        assert "qubit routing" in text
        assert "[0]" in text
        assert "https://c.example/bad" in text
        assert "old.example" not in text

    async def test_select_urls_skips_model_without_candidates_and_survives_triage_error(
        self,
    ):
        agent, fast, _, _ = _make_agent()

        out = await agent.select_urls(
            _state(pending_links=["https://x.example"]),
            {},
        )

        assert fast.calls == []
        assert out["to_visit"] == ["https://x.example"]

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

        assert error_out["to_visit"] == ["https://x.example"]

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
        assert out["to_visit"] == ["https://link.example"]


class TestWebResearchRound:
    """web_research fetches only the selected pages with the round fetch query."""

    async def test_every_fetch_uses_round_fetch_query(self):
        agent, _, _, fetch_calls = _make_agent()
        state = _state(
            fetch_query="qubit routing",
            to_visit=["https://a.example", "https://b.example"],
            knowledge_gap="latency",
        )

        await agent.web_research(state, {})

        assert fetch_calls == [
            ("https://a.example", "qubit routing"),
            ("https://b.example", "qubit routing"),
        ]


class TestReflectionRound:
    """reflection audits the search view and emits the next round plan."""

    async def test_reflection_sees_search_outcomes(self):
        agent, _, small, _ = _make_agent(reflect=Reflection(is_sufficient=False))
        state = _state(
            searches=[
                {
                    "query": "q1",
                    "results": [
                        _sr("https://a.example"),
                        _sr("https://b.example"),
                        _sr("https://c.example"),
                        _sr("https://d.example"),
                    ],
                },
                {"query": "q2", "results": []},
            ],
            visits=[
                {"url": "https://a.example", "ok": True},
                {"url": "https://b.example", "ok": True},
                {"url": "https://c.example", "ok": False},
            ],
            evidence=[{"url": "https://a.example", "text": "x"}],
        )

        await agent.reflection(state, {})

        lines = str(small.calls[-1][1][-1].content).splitlines()

        def line_with(fragment: str) -> str:
            return next(line for line in lines if fragment in line)

        assert "E1" in line_with("a.example")
        assert "none" in line_with("b.example")
        assert "FAILED" in line_with("c.example")
        assert "not selected" in line_with("d.example")
        assert "(no results)" in lines

    async def test_reflection_drops_repeated_follow_ups_and_caps_total(self):
        agent, _, _, _ = _make_agent(
            limits=ResearchLimits(max_follow_ups=2),
            reflect=Reflection(
                is_sufficient=False,
                search_queries=[
                    "  Quantum Routing ",
                    "post quantum crypto",
                    "third",
                ],
                links=["https://a.example/paper#sec2", "https://b.example/next"],
                fetch_query="pqc latency",
            ),
        )
        state = _state(
            searches=[{"query": "quantum routing", "results": []}],
            visits=[{"url": "https://a.example/paper", "ok": True}],
        )

        out = await agent.reflection(state, {})

        combined = out["search_queries"] + out["pending_links"]
        assert len(combined) == 2
        assert out["search_queries"][0] == "post quantum crypto"
        assert "Quantum Routing" not in combined
        assert "https://a.example/paper#sec2" not in combined
        assert out["fetch_query"] == "pqc latency"

    @pytest.mark.parametrize(
        ("knowledge_gap", "expected"),
        [
            ("", _QUESTION),
            ("N/A", _QUESTION),
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
        state = _state(
            searches=[],
            visits=[],
            knowledge_gap="previous round gap",
        )

        out = await agent.reflection(state, {})

        assert out["fetch_query"] == expected


class TestEvaluateResearch:
    """The time gate and evidence gate decide when to finalize."""

    @pytest.mark.parametrize(
        ("clock_value", "is_sufficient", "evidence", "queries", "links", "expected"),
        [
            (10, True, [{"url": "u", "text": "t"}], ["q"], [], "finalize_answer"),
            (10, True, [], ["q"], [], "send"),
            (30, False, [{"url": "u", "text": "t"}], ["q"], [], "finalize_answer"),
            (10, False, [{"url": "u", "text": "t"}], [], [], "finalize_answer"),
            (
                10,
                False,
                [{"url": "u", "text": "t"}],
                [],
                ["https://x.example"],
                "select_urls",
            ),
        ],
    )
    def test_evaluate_research_routes_by_sufficiency_follow_ups_and_time(
        self, clock_value, is_sufficient, evidence, queries, links, expected
    ):
        agent, _, _, _ = _make_agent(
            limits=ResearchLimits(round_start_budget=25),
            clock=iter([clock_value]).__next__,
        )
        state = _state(
            started_at=0,
            is_sufficient=is_sufficient,
            evidence=evidence,
            search_queries=queries,
            pending_links=links,
        )

        result = agent.evaluate_research(state)

        if expected == "send":
            assert [send.node for send in result] == ["web_search"]
        else:
            assert result == expected


class TestGraphRound:
    """The wired graph runs one round end to end and returns a cited answer."""

    async def test_graph_runs_one_round_to_cited_answer(self):
        agent, _, _, _ = _make_agent(
            fetch_results={"https://a.example": _fetched(_PAGE)},
            search_results=[_sr("https://a.example")],
            plan=ResearchPlan(search_queries=["q"], fetch_query="k"),
            select=SelectedPages(indices=[0]),
            extract=ExtractedRanges(ranges="1"),
            reflect=Reflection(is_sufficient=True),
            answer=FinalAnswer(answer=[Claim(text="A.", evidence=[1])]),
        )

        result = await agent("question")

        assert result == {
            "answer": "A. [1]",
            "explanation": "",
            "sources": ["https://a.example"],
        }

    async def test_graph_runs_two_rounds_and_accumulates_evidence(self):
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
        agent, _, _, fetch_calls = _make_agent(
            fetch_results={
                "https://a.example": _fetched(_PAGE),
                "https://c.example": _fetched(_PAGE),
            },
            plan=ResearchPlan(search_queries=["q1"], fetch_query="k1"),
            select=SelectedPages(indices=[0]),
            extract=ExtractedRanges(ranges="1"),
            reflect=reflections,
            answer=FinalAnswer(answer=[Claim(text="A.", evidence=[1, 2])]),
            search_by_query=searches.__getitem__,
        )

        result = await agent("question")

        assert fetch_calls == [
            ("https://a.example", "k1"),
            ("https://c.example", "k2"),
        ]
        assert result["sources"] == ["https://a.example", "https://c.example"]
