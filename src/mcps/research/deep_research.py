"""Deep research agent using a LangGraph StateGraph with iterative search."""

from __future__ import annotations

# ruff: noqa: E501
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from operator import add
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypedDict, cast
from urllib.parse import urlparse, urlunparse

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, Overwrite, Send
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
`fetch_query` is space-separated keywords for a BM25 filter over page passages.
- Use distinctive terms as they appear in the source text: named entities, method or metric names, technical terms, and common synonyms.
- Do not use question words, stopwords, `site:`/quotes/operators, or URLs.
- Words are normalized and stemmed during search, use base forms only
- With a KNOWLEDGE GAP, target the gap's terms plus core entities from the question.

GOOD: staggered adoption parallel trends TWFE bias
BAD: what are the problems with DiD
BAD: site:arxiv.org DiD
"""

_GENERATE_QUERY_PROMPT = f"""You are a web search query generator for a research agent.
Current date context: Today is {datetime.now().strftime("%A, %B %d, %Y")}. Use this to evaluate time-sensitivity, version obsolescence, and real-time data validity.

Produce a research plan:
- split the question into 1-3 distinct subquestions, using fewer queries for simple questions. If you think you know an answer, the subquestion is how to confirm it.
- `search_queries`: concise keyword queries for each subquestion. Each query is 2-5 keywords (hard maximum 6 words) and uses relevant entities, including ones known from domain knowledge and not only names present in the question. Never produce natural-language questions. Keep the wording concise, as keyword queries.
- `links`: any concrete HTTP(S) URLs observed in the question, to read directly instead of searching.
- `fetch_query`: the query used to select passages inside every fetched page for this round.

{FETCH_QUERY_INSTRUCTIONS}
"""

_SELECT_URLS_PROMPT = """You choose which search results are worth reading.
Consider the whole candidate pool against the user question, and the KNOWLEDGE GAP when present.
Select complementary sources that cover distinct subquestions, in priority order by relevance and authority.
Prefer primary sources and official documentation. Prefer HTML over raw PDFs or binary files when the evidence is comparable.
Return the 0-based indices of at most MAX PAGES candidates without duplicates of the same article, or fewer when fewer are useful, or an empty list when nothing is worth reading.
"""

_REFLECTION_PROMPT = f"""You are the reasoning engine of a web research agent.
Audit the fetched pages and search results gathered so far in the conversation, then decide the next step.
Your output is evidence for an upstream agent that will draft the answer: collect the most relevant support as fast as possible.
- Treat anything not present in the fetched pages as an unverified hypothesis; never assume it.
- Use domain knowledge to propose search hypotheses, naming relevant entities, methods, packages, standards, or authors. Keep them unverified until fetched pages support them.
- Use the search view to see which queries failed, which results were not selected, and which links appeared in fetched pages.
- Rephrase failed queries, follow promising `not selected` results, and prefer primary sources when following links.
- Set `is_sufficient` once the fetched pages support the core factual answer.
- Do not extend research for peripheral details or unavailable private measurements. Disclose them as unresolved instead.
- Follow-up queries may name hypothesized entities. Follow only exact HTTP(S) URLs observed in the question, search results, or fetched pages; never invent a URL.

Current date context: Today is {datetime.now().strftime("%A, %B %d, %Y")}. Use this to evaluate time-sensitivity, version obsolescence, and real-time data validity.

If more work is needed, produce the next round plan:
- `search_queries`: up to 3 distinct follow-up queries, each 2-5 keywords (hard maximum 6 words). Target one unresolved gap per query.
- `links`: exact HTTP(S) URLs to read directly, when you already know the source.
- `fetch_query`: the shared BM25 query for the next round, applied to the knowledge gap.
- `knowledge_gap`: the concrete missing data or validation step.

{FETCH_QUERY_INSTRUCTIONS}
"""


_FETCH_QUERY_FIELD = "keywords that select relevant passages inside fetched pages"

_FINALIZE_PROMPT = """Write the final answer to the user's question using ONLY the fetched pages in the conversation above.

- `answer`: 1-3 short sentences that directly answer the question.
- `explanation`: supporting detail. Preserve relevant metrics, identifiers, package names, and author names. Cite the source URL inline after each statement it supports. If sources contradict each other, state the conflict and cite both sides. End with a `Not found:` list of every part of the question the pages do not answer, including unavailable private measurements.
- `sources`: URLs of the fetched pages the answer rests on.
- Use domain knowledge only to organize the evidence, never to add unsupported facts.
- If the last reflection has `is_sufficient` false, research stopped at a time or round limit. Answer only what the fetched pages support, and list the open `knowledge_gap` and other uncertain points under `Not found:`.
- If no fetched page supports an answer, say so in `answer` and list the questions under `Not found:`.
"""


# ---------------------------------------------------------------------------
# Pydantic models (structured LLM output)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResearchLimits:
    """Tunable bounds for one research run."""

    round_start_budget: float = 180.0  # seconds since start; no new round after it
    max_pages_per_round: int = 10
    max_initial_queries: int = 3
    max_follow_ups: int = 5  # independent limit for search_queries and for links


_DEFAULT_LIMITS = ResearchLimits()


class Page(TypedDict):
    """Filtered content of one fetched page."""

    url: str
    title: str
    content: str


class PageVisit(TypedDict):
    """Outcome of visiting a page; ok mirrors FetchResult.ok."""

    url: str
    ok: bool


class SearchRun(TypedDict):
    """One executed search query and its results."""

    query: str
    results: list[SearchResult]


class FinalAnswer(BaseModel):
    """Output of the finalize_answer node."""

    answer: str = Field(default="", description="1-3 sentences answering the question")
    explanation: str = Field(default="", description="Supporting detail with inline source URLs, ending with 'Not found:' gaps")
    sources: list[str] = Field(default_factory=list, description="URLs of fetched pages the answer rests on")


class FetchQuery(BaseModel):
    """Structured output of the fetch-query fragment (also the base of `ResearchPlan`)."""

    fetch_query: str = Field(default="", description=_FETCH_QUERY_FIELD)


class ResearchPlan(FetchQuery):
    """Shared round plan produced by generate_query and reflection."""

    search_queries: list[str] = Field(default_factory=list, description="2-5 keyword queries")
    links: list[str] = Field(default_factory=list, description="HTTP(S) URLs to read directly")


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


class OverallState(MessagesState):
    """Shared state for the full research graph."""

    original_question: str
    started_at: float
    search_queries: list[str]  # current round — overwritten by generate_query/reflection
    pending_links: list[str]  # current round — overwritten
    fetch_query: str  # current round — overwritten
    # round buffers: filled by the round's nodes, rendered and cleared by reflection
    searches: Annotated[list[SearchRun], add]
    visits: Annotated[list[PageVisit], add]
    pages: Annotated[list[Page], add]
    knowledge_gap: str
    is_sufficient: bool
    research_loop_count: int  # logging/progress only


class Input(TypedDict):
    """Input to the overall graph."""

    question: str


class _WebSearchState(TypedDict):
    """Payload sent to each parallel web_search branch."""

    search_query: str


class _WebFetchState(TypedDict):
    """Payload sent to each parallel web_fetch branch."""

    url: str
    title: str
    fetch_query: str


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


def _visit_status(url: str, visited: dict[str, bool]) -> str:
    """Render one search result's outcome for the reflection search view."""
    if url not in visited:
        return "not selected"
    return "VISITED" if visited[url] else "FAILED"


def _render_search_view(searches: list[SearchRun], visits: list[PageVisit]) -> str:
    """Render the round's search results grouped by query, tagged with visit outcomes."""
    visited = {_normalize_url(visit["url"]): visit["ok"] for visit in visits}
    lines: list[str] = []
    seen: set[str] = set()
    for run in searches:
        lines.append(f'Query: "{run["query"]}"')
        results = run["results"]
        if not results:
            lines.append("(no results)")
            continue
        for result in results:
            url = _normalize_url(result.url)
            seen.add(url)
            lines.append(
                f"[{_visit_status(url, visited)}] "
                f"{result.url} | {result.title} | {result.snippet[:200]}"
            )
    followed = {
        url: visit["url"]
        for visit in visits
        if (url := _normalize_url(visit["url"])) not in seen
    }
    if followed:
        lines.append("Followed links:")
        lines.extend(f"[{_visit_status(url, visited)}] {original}" for url, original in followed.items())
    return "\n".join(lines)


def _render_pages(pages: list[Page]) -> str:
    """Render fetched pages as ``Source URL / Title / CONTENT`` blocks."""
    if not pages:
        return "No pages fetched this round."
    return "\n\n".join(
        f"Source URL: {page['url']}\nTitle: {page['title']}\nCONTENT:\n{page['content']}"
        for page in pages
    )


def _collect_candidates(searches: list[SearchRun], exclude: set[str]) -> list[SearchResult]:
    """Deduplicated search results of the round, excluding ``exclude`` URLs."""
    candidates: list[SearchResult] = []
    for run in searches:
        for result in run["results"]:
            url = _normalize_url(result.url)
            if url in exclude:
                continue
            exclude.add(url)
            candidates.append(SearchResult(url=url, title=result.title, snippet=result.snippet))
    return candidates


def _search_titles(searches: list[SearchRun]) -> dict[str, str]:
    """Map normalized result URLs to their titles."""
    titles: dict[str, str] = {}
    for run in searches:
        for result in run["results"]:
            titles.setdefault(_normalize_url(result.url), result.title)
    return titles


async def _report(config: RunnableConfig, message: str) -> None:
    """Forward a progress update when a reporter is attached to config."""
    reporter = config.get("configurable", {}).get("progress_reporter")
    if reporter:
        await reporter(message, 0.0, None)


async def _invoke_structured_history[T: BaseModel](
    model: Any,
    schema: type[T],
    messages: list[BaseMessage],
) -> T:
    """Invoke a model with structured output over an existing message list."""
    chain = model.with_structured_output(schema)
    return cast(T, await chain.ainvoke(messages))


async def _invoke_structured[T: BaseModel](
    model: Any,
    schema: type[T],
    system: str,
    human: str,
) -> T:
    """Invoke a model with structured output for a system + human message pair."""
    return await _invoke_structured_history(
        model, schema, [SystemMessage(system), HumanMessage(human)]
    )


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
        builder.add_node("web_fetch", self.web_fetch)
        builder.add_node(self.reflection)
        builder.add_node("finalize_answer", self.finalize_answer)

        builder.add_edge(START, "generate_query")
        builder.add_conditional_edges(
            "generate_query",
            self.route_round,
            ["web_search", "select_urls"],
        )
        builder.add_edge("web_search", "select_urls")
        builder.add_edge("web_fetch", "reflection")
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
        await _report(config, "Generating search queries...")
        result = await _invoke_structured(
            self.config.fast,
            ResearchPlan,
            _GENERATE_QUERY_PROMPT,
            state["question"],
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
        fetch_query = result.fetch_query.strip() or state["question"]
        logger.info("Initial plan: %d queries, %d links", len(queries), len(links))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[generate_query] queries=%s links=%s fetch_query=%r",
                queries,
                links,
                fetch_query,
            )
        return {
            "messages": [SystemMessage(_REFLECTION_PROMPT)],
            "original_question": state["question"],
            "started_at": started_at,
            "search_queries": queries,
            "pending_links": links,
            "fetch_query": fetch_query,
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
        await _report(config, f"Searching: {query}")
        logger.info("Performing web search for query: %s", query)
        results = await self.config.search(query)
        logger.info("Search %r returned %d results", query, len(results))
        return {"searches": [SearchRun(query=query, results=results)]}

    async def select_urls(
        self, state: OverallState, config: RunnableConfig
    ) -> Command[Literal["web_fetch", "reflection"]]:
        """Choose this round's pages and fan out one web_fetch branch per page."""
        await _report(config, "Selecting pages...")
        links = _dedupe(
            _valid_links(state.get("pending_links", [])),
            set(),
            _normalize_url,
            self.limits.max_pages_per_round,
        )
        candidates = _collect_candidates(
            state.get("searches", []),
            {_normalize_url(link) for link in links},
        )
        budget = self.limits.max_pages_per_round - len(links)
        chosen: list[str] = []
        if candidates and budget > 0:
            chosen = await self._triage(state, candidates, budget)
        logger.info("Selected %d links and %d triaged pages", len(links), len(chosen))
        urls = links + chosen
        fetch_query = (
            (state.get("fetch_query") or "").strip() or state["original_question"]
        )
        logger.debug(
            "[select_urls] candidates=%d budget=%d links=%s triaged=%s fetch_query=%r",
            len(candidates),
            budget,
            links,
            chosen,
            fetch_query,
        )
        if not urls:
            logger.warning("No pages selected for this round")
            return Command(goto="reflection")
        titles = _search_titles(state.get("searches", []))
        return Command(
            goto=[
                Send(
                    "web_fetch",
                    _WebFetchState(
                        url=url,
                        title=titles.get(_normalize_url(url), ""),
                        fetch_query=fetch_query,
                    ),
                )
                for url in urls
            ]
        )

    async def _triage(
        self,
        state: OverallState,
        candidates: list[SearchResult],
        budget: int,
    ) -> list[str]:
        """Ask the fast model for candidate indices; fall back to no triage."""
        gap = state.get("knowledge_gap", "").strip()
        lines = [
            f"Question: {state['original_question']}",
            *([f"KNOWLEDGE GAP: {gap}"] if gap else []),
            f"MAX PAGES: {budget}",
        ]
        for index, candidate in enumerate(candidates):
            lines.append(f"[{index}] {candidate.url}")
            lines.append(candidate.title)
            lines.append(candidate.snippet)
        prompt = "\n".join(lines)
        logger.debug("[triage] prompt:\n%s", prompt)
        try:
            selected = await _invoke_structured(
                self.config.fast,
                SelectedPages,
                _SELECT_URLS_PROMPT,
                prompt,
            )
        except Exception as error:  # triage must never fail a round
            logger.warning("URL triage failed, using links only: %s", error)
            return []
        valid_urls = [
            candidates[index].url
            for index in selected.indices
            if 0 <= index < len(candidates)
        ]
        chosen = _dedupe(valid_urls, set(), str, budget)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[triage] indices=%s chosen=%s not selected=%s",
                selected.indices,
                chosen,
                [candidate.url for candidate in candidates if candidate.url not in chosen],
            )
        return chosen

    async def web_fetch(self, state: _WebFetchState, config: RunnableConfig) -> dict[str, Any]:
        """Fetch one selected page; the fetch tool's own filter picks its passages."""
        url = state["url"]
        await _report(config, f"Reading: {url}")
        fetch_query = state["fetch_query"].strip()
        fetched = await self.config.fetch(url, fetch_query)
        visit = PageVisit(url=url, ok=fetched.ok)
        logger.debug(
            "[web_fetch] %s fetch_query=%r status=%s http_status=%s mime=%s chars=%d",
            url,
            fetch_query,
            fetched.status,
            fetched.http_status,
            fetched.mime,
            len(fetched.content),
        )
        if not fetched.ok or not fetched.content.strip():
            return {"visits": [visit], "pages": []}
        # logger.debug("[web_fetch] %s content:\n%s", url, fetched.content)
        page = Page(url=url, title=state.get("title", ""), content=fetched.content)
        return {"visits": [visit], "pages": [page]}

    async def reflection(self, state: OverallState, config: RunnableConfig) -> dict[str, Any]:
        """Add this round to the conversation, then plan the next round."""
        loop_count = state.get("research_loop_count", 0) + 1
        await _report(config, f"Evaluating research (round {loop_count})...")
        gap = state.get("knowledge_gap", "").strip()
        gap_section = f"\n\nKNOWLEDGE GAP: {gap}" if gap else ""
        round_message = HumanMessage(
            f"QUESTION: {state['original_question']}{gap_section}\n\n"
            f"Search results:\n{_render_search_view(state.get('searches', []), state.get('visits', []))}\n\n"
            f"Fetched pages:\n{_render_pages(state.get('pages', []))}"
        )
        logger.debug("[reflection] round %d prompt:\n%s", loop_count, round_message.content)
        chain = self.config.small.with_structured_output(
            Reflection, include_raw=True, method="json_schema"
        )
        result = cast(
            dict[str, Any],
            await chain.ainvoke([*state["messages"], round_message]),
        )
        parsed = cast(Reflection | None, result["parsed"])
        if parsed is None:
            logger.warning("Reflection round %d returned unparsable output, finalizing", loop_count)
            return {
                "messages": [round_message],
                "searches": Overwrite(value=[]),
                "visits": Overwrite(value=[]),
                "pages": Overwrite(value=[]),
                "research_loop_count": loop_count,
                "is_sufficient": True,
                "search_queries": [],
                "pending_links": [],
            }
        queries = _dedupe(
            parsed.search_queries,
            set(),
            _query_key,
            self.limits.max_follow_ups,
        )
        links = _dedupe(
            _valid_links(parsed.links),
            set(),
            _normalize_url,
            self.limits.max_follow_ups,
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("[reflection] round %d raw: %s", loop_count, parsed.model_dump_json())
        logger.info(
            "Reflection round %d: is_sufficient=%s, queries=%d, links=%d, knowledge_gap=%s",
            loop_count,
            parsed.is_sufficient,
            len(queries),
            len(links),
            parsed.knowledge_gap[:1200],
        )
        return {
            "messages": [round_message, result["raw"]],
            "searches": Overwrite(value=[]),
            "visits": Overwrite(value=[]),
            "pages": Overwrite(value=[]),
            "research_loop_count": loop_count,
            "is_sufficient": parsed.is_sufficient,
            "knowledge_gap": parsed.knowledge_gap,
            "search_queries": queries,
            "pending_links": links,
            "fetch_query": parsed.fetch_query.strip()
            or parsed.knowledge_gap
            or state["original_question"],
        }

    def evaluate_research(
        self,
        state: OverallState,
    ) -> Literal["finalize_answer"] | list[Send] | Literal["select_urls"]:
        """Finalize when sufficient, out of time, or out of follow-ups; else continue."""
        if state.get("is_sufficient"):
            logger.debug("[evaluate_research] sufficient, finalizing")
            return "finalize_answer"
        elapsed = self.clock() - state.get("started_at", 0)
        if elapsed > self.limits.round_start_budget:
            logger.info("Round budget exceeded (%.1fs), finalizing", elapsed)
            return "finalize_answer"
        if not state.get("search_queries") and not state.get("pending_links"):
            logger.debug("[evaluate_research] no follow-up queries or links, finalizing")
            return "finalize_answer"
        logger.debug("[evaluate_research] starting next round after %.1fs", elapsed)
        return self.route_round(state)

    async def finalize_answer(self, state: OverallState, config: RunnableConfig) -> dict[str, Any]:
        """Answer from the whole conversation, whether research ended sufficient or at a limit."""
        await _report(config, "Synthesizing final answer...")
        response = await _invoke_structured_history(
            self.config.small,
            FinalAnswer,
            [*state["messages"], HumanMessage(_FINALIZE_PROMPT)],
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("[finalize] raw: %s", response.model_dump_json())
        sources = list(dict.fromkeys(url for url in response.sources if _is_http_url(url)))
        logger.info("Final answer generated: %s. Explanation: %s", response.answer, response.explanation[:1000])
        return {
            "answer": response.answer,
            "explanation": response.explanation,
            "sources_gathered": sources,
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
