"""Deep research agent using a LangGraph StateGraph with iterative search."""

from __future__ import annotations

# ruff: noqa: E501
import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from operator import add
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypedDict, cast
from urllib.parse import urlparse, urlunparse

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from mcps.research.tools import SearchResult

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from mcps.research.agent import ResearchResponse
    from mcps.research.config import ResearchConfig

__all__ = [
    "FETCH_QUERY_INSTRUCTIONS",
    "FetchQuery",
    "ResearchAgent",
    "create_deep_research_graph",
]

logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

FETCH_QUERY_INSTRUCTIONS = """Fetch query rules:
`fetch_query` is 3-6 space-separated keywords for a BM25 ranker over page passages.
- Use distinctive terms as they appear in the source text: named entities, method or metric names, technical terms, and one common synonym.
- Do not use question words, stopwords, `site:`/quotes/operators, or URLs.
- With a KNOWLEDGE GAP, target the gap's terms plus at most one core entity from the question.

GOOD: staggered adoption parallel trends TWFE bias
BAD: what are the problems with DiD
BAD: site:arxiv.org DiD
"""

_GENERATE_QUERY_PROMPT = f"""You are a web search query generator for a research agent.
Current date context: Today is {datetime.now().strftime("%A, %B %d, %Y")}. Use this to evaluate time-sensitivity, version obsolescence, and real-time data validity.

Produce a research plan:
- `search_queries`: 1-3 orthogonal search queries. Each query is 2-5 keywords (hard maximum 6 words) and contains at least one named entity from the question. Use fewer queries for simple facts. Never produce natural-language questions or compound noun phrases.
- `links`: any concrete HTTP(S) URLs found in the question, to read directly instead of searching.
- `fetch_query`: the shared BM25 query used to select passages inside every fetched page for this round.

{FETCH_QUERY_INSTRUCTIONS}
"""

_SELECT_URLS_PROMPT = """You choose which search results are worth reading.
Judge each candidate URL/domain, title and snippet against the user question and the FETCH QUERY.
Return the 0-based indices of the most promising candidates in priority order, at most MAX PAGES.
Prefer primary sources and official documentation. Return an empty list when nothing is worth reading.
"""

_EXTRACT_PROMPT = """You select the verbatim lines of a fetched page that are relevant to the user question and knowledge gap.

The page is provided one line per numbered line. Every line starts with its 1-based line number.
Return ONLY line ranges in the `ranges` field, for example "1-20, 80-210".

Selection rules:
- Keep lines containing facts, methods, tools, quantitative metrics, software package names, mathematical assumptions, contradictions, or promising links.
- Use broad relevance: keep anything that could support, contextualize, or contradict the question.
- Never restate, summarize, or paraphrase. The ranges alone select the evidence.
- Return an empty string when nothing in the page is relevant.
"""

_REFLECTION_PROMPT = f"""You are the reasoning engine of a web research agent.
Audit the numbered evidence `[E<n>] (url)` gathered so far and the search view, then decide the next step.
- Treat anything not present in the evidence as an unverified hypothesis; never assume it.
- Use the search view to see which queries failed, which results were not selected, and which links appeared in evidence.
- Rephrase failed queries, follow promising `not selected` results, and prefer primary sources when following links.
- Set `is_sufficient` only when the evidence answers the question with authoritative sources.

Current date context: Today is {datetime.now().strftime("%A, %B %d, %Y")}. Use this to evaluate time-sensitivity, version obsolescence, and real-time data validity.

If more work is needed, produce the next round plan:
- `search_queries`: up to 3 orthogonal follow-up queries, each 2-5 keywords (hard maximum 6 words) with at least one named entity from the question. Target one unresolved gap per query.
- `links`: exact HTTP(S) URLs to read directly, when you already know the source.
- `fetch_query`: the shared BM25 query for the next round, applied to the knowledge gap.
- `knowledge_gap`: the concrete missing data or validation step.

{FETCH_QUERY_INSTRUCTIONS}
"""


_NO_EVIDENCE_ANSWER = "No supporting evidence was found for this question."
_UNANSWERED = "The gathered evidence does not support an answer to this question."

_FETCH_QUERY_FIELD = "3-6 BM25 keywords that select relevant passages inside fetched pages"

_FINALIZE_PROMPT = """You write a final answer to the user's question using ONLY the numbered evidence blocks `[E<n>] (url)`.
Return atomic claims, each supported by the `[E#]` ids of the evidence it rests on.

Rules:
- Write the `answer` as 1-3 short sentences that directly answer the question.
- Put the supporting detail into `claims` as short atomic statements.
- Every claim MUST cite at least one existing `[E#]` id in its `evidence` list. Never cite an id that is not shown.
- Never add facts from your own knowledge; if the evidence does not say it, do not claim it.
- If sources contradict each other, state the conflict and cite both sides.
- List every part of the question the evidence does not answer in `not_found`.
- Do not include any source attribution in the claim text; citations are carried by the `evidence` ids.
"""


# ---------------------------------------------------------------------------
# Pydantic models (structured LLM output)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResearchLimits:
    """Tunable bounds for one research run."""

    round_start_budget: float = 25.0  # seconds since start; no new round after it
    max_pages_per_round: int = 8
    max_initial_queries: int = 3
    max_follow_ups: int = 3  # search_queries + links
    max_triage_candidates: int = 30
    line_wrap_chars: int = 300


_DEFAULT_LIMITS = ResearchLimits()


class ExtractedRanges(BaseModel):
    """Line ranges relevant to the question, as delivered by fetch."""

    ranges: str = Field(
        default="",
        description="Comma-separated 1-based line ranges, e.g. '1-20, 80-210'; empty when nothing relevant",
    )


class Evidence(TypedDict):
    """Verbatim page slice; id is its 1-based position in state['evidence']."""

    url: str
    text: str


class PageVisit(TypedDict):
    """Outcome of visiting a page; ok mirrors FetchResult.ok."""

    url: str
    ok: bool


class SearchRun(TypedDict):
    """One executed search query and its results."""

    query: str
    results: list[SearchResult]


class Claim(BaseModel):
    """One atomic statement supported by cited evidence ids."""

    text: str
    evidence: list[int] = Field(default_factory=list, description="[E#] ids supporting the claim")


class FinalAnswer(BaseModel):
    """Cited atomic output of the finalize_answer node."""

    answer: list[Claim] = Field(default_factory=list, description="1-3 sentences answering the question")
    claims: list[Claim] = Field(default_factory=list, description="Supporting cited claims")
    not_found: list[str] = Field(default_factory=list, description="Gaps the evidence does not answer")


class ResearchPlan(BaseModel):
    """Shared round plan produced by generate_query and reflection."""

    search_queries: list[str] = Field(default_factory=list, description="2-5 keyword queries")
    links: list[str] = Field(default_factory=list, description="HTTP(S) URLs to read directly")
    fetch_query: str = Field(default="", description=_FETCH_QUERY_FIELD)


class FetchQuery(BaseModel):
    """Structured output of the fetch-query fragment (evaluation and Phase 3 `ResearchPlan`)."""

    fetch_query: str = Field(default="", description=_FETCH_QUERY_FIELD)


class SelectedPages(BaseModel):
    """Triage output: 0-based candidate indices in priority order."""

    indices: list[int] = Field(default_factory=list, description="0-based candidate indices, priority order")


class Reflection(ResearchPlan):
    """Structured output for the reflection node."""

    is_sufficient: bool = Field(
        description="Whether gathered info is sufficient and authoritative enough to answer the question"
    )
    knowledge_gap: str = Field(default="", description="What information is still missing or unverified community claims")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class OverallState(TypedDict):
    """Shared state for the full research graph."""

    original_question: str
    started_at: float
    search_queries: list[str]  # current round — overwritten by generate_query/reflection
    pending_links: list[str]  # current round — overwritten
    fetch_query: str  # current round — overwritten
    to_visit: list[str]  # written by select_urls
    searches: Annotated[list[SearchRun], add]  # accumulated across rounds
    visits: Annotated[list[PageVisit], add]  # accumulated across rounds
    evidence: Annotated[list[Evidence], add]  # accumulated across rounds
    knowledge_gap: str
    is_sufficient: bool
    research_loop_count: int  # logging/progress only
    sources_gathered: list[str]  # written by finalize_answer


class Input(TypedDict):
    """Input to the overall graph."""

    question: str


class _WebSearchState(TypedDict):
    """Payload sent to each parallel web_search branch."""

    search_query: str


class Result(TypedDict):
    """Final output of the overall graph."""

    answer: Annotated[str, "Only final synthesized answer to the user's question without any explanation or source attribution"]
    explanation: Annotated[str, "Explanation of how the answer was derived"]
    sources_gathered: Annotated[list[str], "List of source URLs that informed the final answer"]


# ---------------------------------------------------------------------------
# URL and text helpers
# ---------------------------------------------------------------------------


def _is_http_url(url: str) -> bool:
    """True when ``url`` has an HTTP(S) scheme and a host."""
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _normalize_url(url: str) -> str:
    """Lowercase scheme/host, drop the fragment and the trailing path slash."""
    stripped = url.strip()
    parsed = urlparse(stripped)
    if not _is_http_url(stripped):
        return stripped
    path = parsed.path.rstrip("/")
    return urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), path, parsed.params, parsed.query, ""))


def _query_key(query: str) -> str:
    """Case-insensitive, whitespace-collapsed key for query deduplication."""
    return " ".join(query.casefold().split())


def _dedupe(
    items: list[str],
    seen_keys: set[str],
    key: Callable[[str], str],
    limit: int,
) -> list[str]:
    """Strip, skip blanks and ``seen_keys``, and keep at most ``limit`` items."""
    if limit <= 0:
        return []
    result: list[str] = []
    for item in items:
        stripped = item.strip()
        if not stripped:
            continue
        item_key = key(stripped)
        if item_key in seen_keys:
            continue
        seen_keys.add(item_key)
        result.append(stripped)
        if len(result) >= limit:
            break
    return result


def _valid_links(links: list[str]) -> list[str]:
    """Keep stripped HTTP(S) URLs with a host."""
    return [stripped for link in links if (stripped := link.strip()) and _is_http_url(stripped)]


def _meaningful_gap(gap: str) -> str | None:
    """Stripped gap text, or None when blank or "N/A"."""
    stripped = gap.strip()
    return stripped if stripped and stripped != "N/A" else None


def _knowledge_gap(state: OverallState) -> str | None:
    """Stripped knowledge gap, or None when absent, blank or "N/A"."""
    return _meaningful_gap(state.get("knowledge_gap", ""))


def _wrap_line(line: str, wrap: int) -> list[str]:
    """Split ``line`` into pieces no longer than ``wrap``, keeping separators.

    A split prefers the last space at or before ``wrap``; when no such space
    exists it cuts hard. Joining the pieces rebuilds the original line.
    """
    if wrap <= 0 or len(line) <= wrap:
        return [line]
    pieces: list[str] = []
    remaining = line
    while len(remaining) > wrap:
        cut = remaining.rfind(" ", 0, wrap + 1)
        if cut <= 0:
            cut = wrap - 1
        pieces.append(remaining[: cut + 1])
        remaining = remaining[cut + 1 :]
    pieces.append(remaining)
    return pieces


def _number_lines(content: str, wrap: int) -> list[tuple[int, str]]:
    """Return ``(source_line_index, piece)`` for every wrapped line of ``content``."""
    return [
        (index, piece)
        for index, line in enumerate(content.split("\n"))
        for piece in _wrap_line(line, wrap)
    ]


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or adjacent 1-based inclusive ranges."""
    merged: list[tuple[int, int]] = []
    for start, end in ranges:
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _parse_ranges(ranges: str, count: int) -> list[tuple[int, int]]:
    """Parse 1-based inclusive ranges, dropping invalid ones and clipping ends."""
    parsed: list[tuple[int, int]] = []
    for token in ranges.split(","):
        token = token.strip()
        if not token:
            continue
        start_text, separator, end_text = token.partition("-")
        try:
            start = int(start_text.strip())
            end = int(end_text.strip()) if separator else start
        except ValueError:
            continue
        if start < 1 or end < start or start > count:
            continue
        parsed.append((start, min(end, count)))
    parsed.sort()
    return _merge_ranges(parsed)


def _slice(lines: list[tuple[int, str]], ranges: list[tuple[int, int]]) -> list[str]:
    """Return the text of each range, joining wrapped pieces verbatim."""
    texts: list[str] = []
    for start, end in ranges:
        text = ""
        previous_index: int | None = None
        for index, piece in lines[start - 1 : end]:
            if previous_index is not None and index != previous_index:
                text += "\n"
            text += piece
            previous_index = index
        stripped = text.strip()
        if stripped:
            texts.append(stripped)
    return texts


def _render_evidence(evidence: list[Evidence]) -> str:
    """Render evidence as ``[E<n>] (url)`` blocks, never raw pages."""
    return "\n".join(
        f"[E{number}] ({item['url']})\n{item['text']}"
        for number, item in enumerate(evidence, start=1)
    )


def _visit_status(
    url: str,
    visited: dict[str, bool],
    evidence_ids: dict[str, list[int]],
) -> str:
    """Render one search result's outcome for the reflection search view."""
    if url not in visited:
        return "not selected"
    if not visited[url]:
        return "FAILED"
    ids = evidence_ids.get(url, [])
    return f"VISITED → {','.join(f'E{number}' for number in ids)}" if ids else "VISITED → none"


def _render_search_view(
    searches: list[SearchRun],
    visits: list[PageVisit],
    evidence: list[Evidence],
) -> str:
    """Render search results grouped by query, tagged with visit outcomes."""
    visited = {_normalize_url(visit["url"]): visit["ok"] for visit in visits}
    evidence_ids: dict[str, list[int]] = {}
    for number, item in enumerate(evidence, start=1):
        evidence_ids.setdefault(_normalize_url(item["url"]), []).append(number)

    lines: list[str] = []
    seen: set[str] = set()
    for run in searches:
        lines.append(f'Query: "{run["query"]}"')
        results = run.get("results", [])
        if not results:
            lines.append("(no results)")
            continue
        for result in results:
            url = _normalize_url(result.url)
            seen.add(url)
            lines.append(
                f"[{_visit_status(url, visited, evidence_ids)}] "
                f"{result.url} | {result.title} | {result.snippet[:200]}"
            )
    followed: dict[str, str] = {}
    for visit in visits:
        url = _normalize_url(visit["url"])
        if url not in seen:
            followed.setdefault(url, visit["url"])
    if followed:
        lines.append("Followed links:")
        for url, original in followed.items():
            lines.append(f"[{_visit_status(url, visited, evidence_ids)}] {original}")
    return "\n".join(lines)


def _render_not_found_queries(state: OverallState) -> str:
    """List every executed query under a ``Not found:`` header (no-model reply)."""
    queries = [item["query"] for item in state.get("searches", [])]
    return "Not found:\n" + "\n".join(f"- {query}" for query in queries)


def _render_cited(final: FinalAnswer, evidence: list[Evidence]) -> tuple[str, str, list[str]]:
    """Drop uncited claims and renumber sources in first-citation order.

    Citation ids are 1-based indexes into ``evidence``. A claim survives only
    when it cites at least one existing id. Source numbers are 1-based indexes
    into the returned URL list, numbered over ``answer`` then ``claims``.
    """
    sources: list[str] = []

    def cited_text(claim: Claim) -> str:
        suffixes: list[str] = []
        for item in claim.evidence:
            if not 1 <= item <= len(evidence):
                continue
            url = evidence[item - 1]["url"]
            if url not in sources:
                sources.append(url)
            suffix = f" [{sources.index(url) + 1}]"
            if suffix not in suffixes:
                suffixes.append(suffix)
        if not suffixes:
            return ""
        return f"{claim.text}{''.join(suffixes)}"

    answer = " ".join(
        text for claim in final.answer if (text := cited_text(claim))
    ) or _UNANSWERED
    explanation = "\n".join(
        f"- {text}" for claim in final.claims if (text := cited_text(claim))
    )
    if final.not_found:
        gaps = "\n".join(f"- {gap}" for gap in final.not_found)
        explanation = f"{explanation}\n\nNot found:\n{gaps}" if explanation else f"Not found:\n{gaps}"
    return answer, explanation, list(sources)


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------


def create_deep_research_graph(config: ResearchConfig) -> CompiledStateGraph:
    """Build and compile the deep research StateGraph.

    Args:
        config: Research configuration providing models and tools.

    Returns:
        A compiled LangGraph runnable ready to invoke.
    """
    return ResearchAgent(config).graph


class ResearchAgent:
    """Deep research agent that owns graph construction and invocation."""

    config: ResearchConfig
    graph: CompiledStateGraph
    limits: ResearchLimits
    clock: Callable[[], float]

    def __init__(
        self,
        config: ResearchConfig,
        *,
        limits: ResearchLimits = _DEFAULT_LIMITS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.limits = limits
        self.clock = clock
        self.graph = self.create_graph()

    def create_graph(self) -> CompiledStateGraph:
        """Wire method references as LangGraph nodes and routes, compile and return."""
        builder = StateGraph(
            OverallState,
            input_schema=Input,
            output_schema=Result,
        )

        builder.add_node(self.generate_query)
        builder.add_node("web_search", self.web_search)
        builder.add_node("select_urls", self.select_urls)
        builder.add_node("web_research", self.web_research)
        builder.add_node(self.reflection)
        builder.add_node("finalize_answer", self.finalize_answer)

        builder.add_edge(START, "generate_query")
        builder.add_conditional_edges(
            "generate_query",
            self.route_round,
            ["web_search", "select_urls"],
        )
        builder.add_edge("web_search", "select_urls")
        builder.add_edge("select_urls", "web_research")
        builder.add_edge("web_research", "reflection")
        builder.add_conditional_edges(
            "reflection",
            self.evaluate_research,
            ["web_search", "select_urls", "finalize_answer"],
        )
        builder.add_edge("finalize_answer", END)

        return builder.compile()

    async def generate_query(self, state: Input, config: RunnableConfig) -> dict[str, Any]:
        """Produce the initial round plan and start the elapsed-time clock."""
        started_at = self.clock()
        reporter = config.get("configurable", {}).get("progress_reporter")
        if reporter:
            await reporter("Generating search queries...", 0.0, None)
        chain = self.config.fast.with_structured_output(ResearchPlan)
        result = cast(
            ResearchPlan,
            await chain.ainvoke(
                [
                    SystemMessage(_GENERATE_QUERY_PROMPT),
                    HumanMessage(state["question"]),
                ]
            ),
        )
        queries = _dedupe(
            result.search_queries,
            set(),
            _query_key,
            self.limits.max_initial_queries,
        )
        links = _valid_links(result.links)
        if not queries and not links:
            queries = [state["question"]]
        logger.info("Initial plan: %d queries, %d links", len(queries), len(links))
        return {
            "original_question": state["question"],
            "started_at": started_at,
            "search_queries": queries,
            "pending_links": links,
            "fetch_query": result.fetch_query.strip() or state["question"],
            "research_loop_count": 0,
            "knowledge_gap": "",
        }

    def route_round(self, state: OverallState) -> list[Send] | Literal["select_urls"]:
        """Fan out one web_search per query, or go straight to link selection."""
        queries = state.get("search_queries", [])
        if queries:
            return [Send("web_search", {"search_query": query}) for query in queries]
        return "select_urls"

    async def web_search(self, state: _WebSearchState, config: RunnableConfig) -> dict[str, Any]:
        """Run one search query and record its results for the round."""
        query = state["search_query"]
        reporter = config.get("configurable", {}).get("progress_reporter")
        if reporter:
            await reporter(f"Searching: {query}", 0.0, None)
        logger.info("Performing web search for query: %s", query)
        results = await self.config.search(query)
        logger.info("Search %r returned %d results", query, len(results))
        return {"searches": [SearchRun(query=query, results=results)]}

    async def select_urls(self, state: OverallState, config: RunnableConfig) -> dict[str, Any]:
        """Choose this round's pages: links first, then LLM-triaged candidates."""
        reporter = config.get("configurable", {}).get("progress_reporter")
        if reporter:
            await reporter("Selecting pages...", 0.0, None)
        visited = {_normalize_url(visit["url"]) for visit in state.get("visits", [])}
        links = _dedupe(
            _valid_links(state.get("pending_links", [])),
            set(visited),
            _normalize_url,
            self.limits.max_pages_per_round,
        )
        current_queries = {_query_key(query) for query in state.get("search_queries", [])}
        exclude = set(visited) | {_normalize_url(link) for link in links}
        candidates = _collect_candidates(
            state.get("searches", []),
            current_queries,
            exclude,
        )[: self.limits.max_triage_candidates]
        budget = self.limits.max_pages_per_round - len(links)
        chosen: list[str] = []
        if candidates and budget > 0:
            chosen = await self._triage(state, candidates, budget)
        logger.info("Selected %d links and %d triaged pages", len(links), len(chosen))
        return {"to_visit": links + chosen}

    async def _triage(
        self,
        state: OverallState,
        candidates: list[SearchResult],
        budget: int,
    ) -> list[str]:
        """Ask the fast model for candidate indices; fall back to no triage."""
        lines = [
            f"Question: {state['original_question']}",
            f"FETCH QUERY: {state.get('fetch_query', '')}",
            f"MAX PAGES: {budget}",
        ]
        for index, candidate in enumerate(candidates):
            lines.append(f"[{index}] {candidate.url}")
            lines.append(candidate.title)
            lines.append(candidate.snippet)
        chain = self.config.fast.with_structured_output(SelectedPages)
        try:
            selected = cast(
                SelectedPages,
                await chain.ainvoke(
                    [SystemMessage(_SELECT_URLS_PROMPT), HumanMessage("\n".join(lines))]
                ),
            )
        except Exception as error:  # triage must never fail a round
            logger.warning("URL triage failed, using links only: %s", error)
            return []
        chosen: list[str] = []
        for index in selected.indices:
            if not 0 <= index < len(candidates):
                continue
            url = candidates[index].url
            if url in chosen:
                continue
            chosen.append(url)
            if len(chosen) >= budget:
                break
        return chosen

    async def web_research(self, state: OverallState, config: RunnableConfig) -> dict[str, Any]:
        """Fetch every selected page and extract verbatim evidence per page."""
        reporter = config.get("configurable", {}).get("progress_reporter")
        if reporter:
            await reporter("Reading pages...", 0.0, None)
        urls = state.get("to_visit", [])
        if not urls:
            logger.warning("No pages selected for this round")
            return {"evidence": [], "visits": []}
        titles = _search_titles(state.get("searches", []))
        pages = await asyncio.gather(
            *(self._visit_page(url, titles.get(_normalize_url(url), ""), state) for url in urls)
        )
        visits = [visit for visit, _ in pages]
        evidence = [item for _, page_evidence in pages for item in page_evidence]
        logger.info("Visited %d pages, kept %d evidence items", len(visits), len(evidence))
        return {"evidence": evidence, "visits": visits}

    async def _visit_page(
        self,
        url: str,
        title: str,
        state: OverallState,
    ) -> tuple[PageVisit, list[Evidence]]:
        """Fetch one page and extract its evidence as soon as it arrives."""
        fetch_query = state.get("fetch_query") or state["original_question"]
        fetched = await self.config.fetch(url, fetch_query)
        visit = PageVisit(url=url, ok=fetched.ok)
        if not fetched.ok or not fetched.content.strip():
            return visit, []
        try:
            evidence = await self._extract(
                url=url,
                title=title,
                content=fetched.content,
                question=state["original_question"],
                gap=_knowledge_gap(state),
            )
        except Exception as error:  # one page's extraction must never fail the round
            logger.warning("Extraction failed for %s: %s", url, error)
            return visit, []
        return visit, evidence

    async def _extract(
        self,
        url: str,
        title: str,
        content: str,
        question: str,
        gap: str | None,
    ) -> list[Evidence]:
        """Ask the fast model for relevant line ranges and slice them verbatim."""
        numbered = _number_lines(content, self.limits.line_wrap_chars)
        numbered_text = "\n".join(
            f"{number}: {piece}"
            for number, (_, piece) in enumerate(numbered, start=1)
        )
        gap_section = f"KNOWLEDGE GAP: {gap}\n" if gap else ""
        prompt = (
            f"QUESTION: {question}\n"
            f"{gap_section}"
            f"URL: {url}\n"
            f"TITLE: {title}\n"
            f"{numbered_text}"
        )
        chain = self.config.fast.with_structured_output(ExtractedRanges)
        predicted = cast(
            ExtractedRanges,
            await chain.ainvoke([SystemMessage(_EXTRACT_PROMPT), HumanMessage(prompt)]),
        )
        ranges = _parse_ranges(predicted.ranges, len(numbered))
        return [Evidence(url=url, text=text) for text in _slice(numbered, ranges)]

    async def reflection(self, state: OverallState, config: RunnableConfig) -> dict[str, Any]:
        """Audit evidence and the search view, then plan the next round."""
        loop_count = state.get("research_loop_count", 0) + 1
        reporter = config.get("configurable", {}).get("progress_reporter")
        if reporter:
            await reporter(f"Evaluating research (round {loop_count})...", 0.0, None)
        gap = state.get("knowledge_gap", "").strip()
        gap_section = f"\n\nKnowledge gap: {gap}" if gap else ""
        human = (
            f"Question: {state['original_question']}{gap_section}\n\n"
            f"Search results:\n{_render_search_view(state.get('searches', []), state.get('visits', []), state.get('evidence', []))}\n\n"
            f"Evidence:\n{_render_evidence(state.get('evidence', []))}"
        )
        chain = self.config.small.with_structured_output(Reflection)
        parsed = cast(
            Reflection,
            await chain.ainvoke([SystemMessage(_REFLECTION_PROMPT), HumanMessage(human)]),
        )
        executed = {_query_key(run["query"]) for run in state.get("searches", [])}
        visited = {_normalize_url(visit["url"]) for visit in state.get("visits", [])}
        queries = _dedupe(
            parsed.search_queries,
            executed,
            _query_key,
            self.limits.max_follow_ups,
        )
        remaining = self.limits.max_follow_ups - len(queries)
        links = _dedupe(_valid_links(parsed.links), visited, _normalize_url, remaining)
        logger.info(
            "Reflection round %d: is_sufficient=%s, queries=%d, links=%d, knowledge_gap=%s",
            loop_count,
            parsed.is_sufficient,
            len(queries),
            len(links),
            parsed.knowledge_gap[:1200],
        )
        return {
            "research_loop_count": loop_count,
            "is_sufficient": parsed.is_sufficient,
            "knowledge_gap": parsed.knowledge_gap,
            "search_queries": queries,
            "pending_links": links,
            "fetch_query": parsed.fetch_query.strip()
            or _meaningful_gap(parsed.knowledge_gap)
            or state["original_question"],
        }

    def evaluate_research(
        self,
        state: OverallState,
    ) -> Literal["finalize_answer"] | list[Send] | Literal["select_urls"]:
        """Finalize when sufficient, out of time, or out of follow-ups; else continue."""
        if state.get("is_sufficient") and state.get("evidence"):
            return "finalize_answer"
        elapsed = self.clock() - state.get("started_at", 0)
        if elapsed > self.limits.round_start_budget:
            logger.info("Round budget exceeded (%.1fs), finalizing", elapsed)
            return "finalize_answer"
        if not state.get("search_queries") and not state.get("pending_links"):
            return "finalize_answer"
        return self.route_round(state)

    async def finalize_answer(self, state: OverallState, config: RunnableConfig) -> dict[str, Any]:
        """Render a cited atomic answer, or a fixed reply when no evidence exists."""
        reporter = config.get("configurable", {}).get("progress_reporter")
        if reporter:
            await reporter("Synthesizing final answer...", 0.0, None)
        evidence = state.get("evidence", [])
        if not evidence:
            return {
                "answer": _NO_EVIDENCE_ANSWER,
                "explanation": _render_not_found_queries(state),
                "sources_gathered": [],
            }
        gap = state.get("knowledge_gap", "").strip()
        gap_section = f"\n\nKnowledge gap: {gap}" if gap else ""
        human = (
            f"Question: {state['original_question']}\n\n"
            f"Evidence:\n{_render_evidence(evidence)}{gap_section}"
        )
        response = cast(
            FinalAnswer,
            await self.config.small.with_structured_output(FinalAnswer).ainvoke(
                [SystemMessage(_FINALIZE_PROMPT), HumanMessage(human)]
            ),
        )
        answer, explanation, sources_gathered = _render_cited(response, evidence)
        logger.info("Final answer generated: %s. Explanation: %s", answer, explanation[:1000])
        return {
            "answer": answer,
            "explanation": explanation,
            "sources_gathered": sources_gathered,
        }

    async def __call__(
        self, input: str, progress: Any | None = None
    ) -> ResearchResponse:
        """Invoke the compiled graph; forward optional progress reporter to all nodes."""
        config: RunnableConfig | None = (
            {"configurable": {"progress_reporter": progress}} if progress is not None else None
        )
        result = await self.graph.ainvoke({"question": input}, config=config)
        return {
            "answer": result["answer"],
            "explanation": result.get("explanation", ""),
            "sources": result.get("sources_gathered", []),
        }


def _collect_candidates(
    searches: list[SearchRun],
    current_queries: set[str],
    exclude: set[str],
) -> list[SearchResult]:
    """Deduplicated search results of the current round, excluding seen URLs."""
    candidates: list[SearchResult] = []
    for run in searches:
        if _query_key(run["query"]) not in current_queries:
            continue
        for result in run.get("results", []):
            url = _normalize_url(result.url)
            if url in exclude:
                continue
            exclude.add(url)
            candidates.append(SearchResult(url=url, title=result.title, snippet=result.snippet))
    return candidates


def _search_titles(searches: list[SearchRun]) -> dict[str, str]:
    """Map normalized result URLs to their titles for extraction prompts."""
    titles: dict[str, str] = {}
    for run in searches:
        for result in run.get("results", []):
            titles.setdefault(_normalize_url(result.url), result.title)
    return titles
