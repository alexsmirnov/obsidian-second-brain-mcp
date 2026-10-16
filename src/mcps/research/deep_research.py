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
from typing import TYPE_CHECKING, Annotated, Any, Literal, NotRequired, TypedDict, cast
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from mcps.research.tools import FetchResult, SearchResult

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from mcps.research.agent import ResearchResponse
    from mcps.research.config import ResearchConfig

__all__ = [
    "ResearchAgent",
    "create_deep_research_graph",
]

logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_GENERATE_QUERY_PROMPT = f"""You are a web search query generator.
Generate exactly up to 2 short web search queries for the user question.
Current date context: Today is {datetime.now().strftime("%A, %B %d, %Y")}. Use this to evaluate time-sensitivity, version obsolescence, and real-time data validity.

Hard rules:
1) Each query MUST be 2-5 keywords. Never exceed 6 words.
2) Each query must include at least one named entity from the question.
3) Queries must be orthogonal — each targets a DIFFERENT facet:
   - Facet 1: core entity
   - Facet 2: related entity or alternative name
4) Preserve source/time constraints when present:
   - Use site: filters for specified sources
   - Include year when version matters
5) Never produce natural language questions or compound noun phrases.
6) If user question contains a concrete URL, return it as-is.

BAD (too long, compound):
  "ChatGPT Enterprise native Microsoft Purview DLP integration support"
  "compare Copilot Microsoft 365 vs ChatGPT Enterprise security governance"

GOOD (short, orthogonal):
  "ChatGPT Enterprise DLP integration"
  "Microsoft Purview ChatGPT"

GOOD: ( user asks about concrete page )
  "https://example.com/openai/enterprise"
Start with a broad area search, and adapt query strategy based on search results in next iterations.
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

_REFLECTION_PROMPT = f"""You are the core reasoning engine of an advanced web research agent. 
Your task is to audit the numbered evidence gathered so far for correctness, sufficiency, and source integrity, and then determine the next strategic step.

### 1. SOURCE QUALITY MATRIX
- Tier 1: Official documentation, primary records, peer-reviewed papers, official archives.
- Tier 2: Expert/community technical sources (e.g., GitHub issues, developer forums, vendor blogs).
- Tier 3: General summaries, news articles, and tertiary sources.

### 2. DETERMINISTIC SUFFICIENCY GATE
Evaluate your current evidence against a strict binary gate (`is_sufficient`). 
Set `is_sufficient = False` if ANY of the following conditions are met:
- Specific empirical metrics, numerical benchmarks, software library names, or mathematical assumptions requested in the user prompt are missing.
- Explicit user constraints (e.g., specific dates, version numbers, or site-specific restrictions) remain unverified in the gathered evidence.
- The gathered evidence contains unresolved, active contradictions between Tier 1 sources.

*Exception*: For purely self-contained logic or mathematical questions, you may set `is_sufficient = True` if you provide a mathematically rigorous derivation, even if external web sources are weak.
*Best-Effort Fallback*: If you hit your maximum reflection loops or constraints cannot be fully met, set `is_sufficient = True` under a "best-effort" policy, but you MUST explicitly declare your uncertainty metrics in the `knowledge_gap`.

Current date context: Today is {datetime.now().strftime("%A, %B %d, %Y")}. Use this to evaluate time-sensitivity, version obsolescence, and real-time data validity.

### 3. OUTPUT SPECIFICATION (JSON Format)
Your response must strictly conform to this schema:
{{
  "is_sufficient": boolean,
  "knowledge_gap": "String. Describe the exact, concrete missing piece of data or validation step required.",
  "follow_ups": ["Array of strings. Next-step target queries conforming to the Search Query Rules below."]
}}
Audit only the numbered evidence blocks `[E<n>] (url)`; do not invent findings beyond them.

### 4. WEB SEARCH QUERY RULES (HARD MANIFESTO)
If `is_sufficient = False`, generate up to 3 follow-up queries. Each query must target ONE unresolved gap. Never combine multiple concepts.
If you know exact URL that may contain some answers, put it directly instead of search query
Concentrate of the most important knowledge gap, refine search strategy on the each iteration.

1) Length: 2-5 keywords. Hard maximum of 6 words per query.
2) Named Entities: Every query MUST contain at least one primary named entity or technical keyword from the original user question.
3) Absolute Orthogonality: Each query must isolate and target a completely different dimension of the problem:
   - Query 1 (Core Axis): Core entity + primary missing attribute/metric.
   - Query 2 (Comparative Axis): Core entity + secondary alternative name, framework, or version modifier.
   - Query 3 (Validation Axis): Specific technical constraint, data point, or official documentation source filter (e.g., site:domain.com).
4) Syntax Rules: 
   - Never generate natural language questions or conversational phrases.
   - Never use compound noun phrases.
   - Preserved dates/versions must be represented as single tokens (e.g., "2026" or "v4.8").

GOOD (Short, Orthogonal, Machine-Ready):
- "ChatGPT Enterprise DLP integration"
- "Microsoft Purview ChatGPT"
- "ChatGPT Enterprise security features"
- "https://example.com/page_with_related_content.html"

BAD (Conversational, Long, Compound):
- "How do I integrate ChatGPT Enterprise natively with Microsoft Purview DLP?"
- "compare Copilot Microsoft 365 vs ChatGPT Enterprise security governance"
"""


_FINALIZE_PROMPT = """Synthesize a final answer to the user's original question based on the research findings and sources gathered.
Provide concise answer and detailed explanation.
The answer itself should be short exact response to the user's question without any explanation or source attribution.
The explanation MUST comprehensively cover EVERY facet of the user's question. Do not summarize away complex methodological, technical, or quantitative details; include them explicitly. If the user asks about specific frame
works, mechanisms, limitations, or theories, ensure your explanation provides the exact detailed rationale and nuanced context found in the research.
If there are multiply possible solution or conroversial information, also include it into explanation
Guidelines:
 Accuracy: Ensure the answer is technically correct and directly addresses ALL parts of the user's question.
 Clarity: Write in a clear, concise manner. Directly answer the question without unnecessary preamble.
 Traceability: In explanation, describe how the answer was derived, with verifiable evidence and logic
Source Attribution: Explicitly reference the sources that support each part of the answer. Mark sources that may contradict provided answer
If the answer includes code snippets, ensure they are well-formatted and tested against the gathered research
 Finalization checklist (mandatory before writing answer):
 - Verify requested answer format exactly (units, scale, rounding, casing, phrasing constraints).
 - Verify arithmetic and unit conversions explicitly.
 - Example: if question asks for answer in "thousands of hours", convert 17000 hours to 17.
 - Address every single sub-question and implicit requirement in the prompt comprehensively.
 - Resolve source-constraint requirements where possible; if unresolved, provide best-effort answer with explicit uncertainty in explanation.
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


class SearchQueryList(BaseModel):
    """Structured output for the generate_query node."""

    query: list[str] = Field(description="up to 2 search queries, or exact URLs for direct fetch.")


class Reflection(BaseModel):
    """Structured output for the reflection node."""

    is_sufficient: bool = Field(
        description="Whether gathered info is sufficient and authoritative enough to answer the question"
    )
    knowledge_gap: str = Field(description="What information is still missing or unverified community claims")
    follow_ups: list[str] = Field(
        default_factory=list,
        description="Short (2-5 keyword) follow-up search queries to fill gaps, or exact URLs for direct fetch.",
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class OverallState(TypedDict):
    """Shared state for the full research graph."""

    original_question: str
    queries: list[str]  # current queries — overwrite each generate_query
    follow_ups: list[str]  # reflection output — overwrite each loop
    evidence: Annotated[list[Evidence], add]  # accumulated across loops
    visits: Annotated[list[PageVisit], add]  # accumulated across loops
    knowledge_gap: str
    sources_gathered: list[str]  # written by finalize_answer
    research_loop_count: int
    max_research_loops: int
    is_sufficient: bool


def _knowledge_gap(state: WebSearchState) -> str | None:
    """Stripped knowledge gap, or None when absent, blank or "N/A"."""
    gap = state.get("knowledge_gap", "").strip()
    return gap if gap and gap != "N/A" else None


def _fetch_query(state: WebSearchState) -> str:
    """Knowledge gap when meaningful, else the original question."""
    return _knowledge_gap(state) or state["original_question"]


class WebSearchState(TypedDict):
    """Minimal payload sent to each parallel web_research branch."""
    original_question: str
    knowledge_gap: NotRequired[str]

    search_query: str
    id: int


class Input(TypedDict):
    """Input to the overall graph."""
    question: str
    max_research_loops: int

class Result(TypedDict):
    """Final output of the overall graph."""
    answer: Annotated[str, "Only final synthesized answer to the user's question without any explanation or source attribution"]
    explanation: Annotated[str, "Explanation of how the answer was derived"]
    sources_gathered: Annotated[list[str],"List of source URLs that informed the final answer"]


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------
def _is_valid_url(url):
    try:
        result = urlparse(url)
        # A valid URL must have a scheme (http/https) and a network location (domain)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


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
    max_research_loops: int

    def __init__(
        self,
        config: ResearchConfig,
        *,
        limits: ResearchLimits = _DEFAULT_LIMITS,
        clock: Callable[[], float] = time.monotonic,
        max_research_loops: int = 3,
    ) -> None:
        self.config = config
        self.limits = limits
        self.clock = clock
        self.max_research_loops = max_research_loops
        self.graph = self.create_graph()

    def create_graph(self) -> CompiledStateGraph:
        """Wire method references as LangGraph nodes and routes, compile and return."""
        builder = StateGraph(
            OverallState,
            input_schema=Input,
            output_schema=Result,
        )

        builder.add_node(self.generate_query)
        builder.add_node("web_research", self.web_research)
        builder.add_node(self.reflection)
        builder.add_node("finalize_answer", self.finalize_answer)

        builder.add_edge(START, "generate_query")
        builder.add_conditional_edges(
            "generate_query",
            self.continue_to_web_research,
            ["web_research"],
        )
        builder.add_edge("web_research", "reflection")
        builder.add_conditional_edges(
            "reflection",
            self.evaluate_research,
            ["web_research", "finalize_answer"],
        )
        builder.add_edge("finalize_answer", END)

        return builder.compile()

    async def generate_query(self, state: Input, config: RunnableConfig) -> dict[str, Any]:
        """Generate initial search queries from the user's question."""
        reporter = config.get("configurable", {}).get("progress_reporter")
        if reporter:
            await reporter("Generating search queries...", 0.0, None)
        chain = self.config.fast.with_structured_output(SearchQueryList)
        result: SearchQueryList = await chain.ainvoke(
            [
                SystemMessage(_GENERATE_QUERY_PROMPT),
                HumanMessage(state["question"]),
            ]
        )  # type: ignore
        logger.info("Generated queries: %d", len(result.query))
        return {
            "queries": result.query,
            "max_research_loops": state.get("max_research_loops", 3),
            "original_question": state["question"],
        }

    async def web_research(self, state: WebSearchState, config: RunnableConfig) -> dict[str, Any]:
        """Fetch every page of one query and extract verbatim evidence per page."""
        query = state["search_query"]
        reporter = config.get("configurable", {}).get("progress_reporter")
        if reporter:
            await reporter("Searching", 0.0, None)
        # If query is a direct URL, fetch it directly without searching
        if _is_valid_url(query):
            logger.info("Direct URL fetch for query: %s", query)
            search_results = [SearchResult(url=query, title="", snippet="")]
        else:
            logger.info("Performing web search for query: %s", query)
            search_results = await self.config.search(query)
        if not search_results:
            logger.warning("No web search results found for query: %s", query)
            return {"evidence": [], "visits": []}
        fetch_query = _fetch_query(state)
        fetch_results = await asyncio.gather(
            *(self.config.fetch(result.url, fetch_query) for result in search_results)
        )
        logger.info("Web search result: %d", len(fetch_results))
        return await self._extract_pages(search_results, fetch_results, state)

    async def _extract_pages(
        self,
        search_results: list[SearchResult],
        fetch_results: list[FetchResult],
        state: WebSearchState,
    ) -> dict[str, Any]:
        """Extract one evidence list per fetchable page, dropping failed pages."""
        pages = list(zip(search_results, fetch_results, strict=False))
        visits = [PageVisit(url=result.url, ok=fetched.ok) for result, fetched in pages]
        extracted = await asyncio.gather(
            *(self._extract_page(result, fetched, state) for result, fetched in pages),
            return_exceptions=True,
        )
        evidence: list[Evidence] = []
        for (result, _), item in zip(pages, extracted, strict=False):
            if isinstance(item, asyncio.CancelledError):
                raise item
            if isinstance(item, BaseException):
                logger.warning("Extraction failed for %s: %s", result.url, item)
                continue
            evidence.extend(item)
        return {"evidence": evidence, "visits": visits}

    async def _extract_page(
        self,
        result: SearchResult,
        fetched: FetchResult,
        state: WebSearchState,
    ) -> list[Evidence]:
        """Extract verbatim evidence from one successfully fetched page."""
        if not fetched.ok or not fetched.content.strip():
            return []
        return await self._extract(
            url=result.url,
            title=result.title,
            content=fetched.content,
            question=state["original_question"],
            gap=_knowledge_gap(state),
        )

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
        """Evaluate research completeness and identify knowledge gaps."""
        loop_count = state.get("research_loop_count", 0) + 1
        max_loops = float(state.get("max_research_loops", self.max_research_loops))
        reporter = config.get("configurable", {}).get("progress_reporter")
        if reporter:
            await reporter(
                f"Evaluating research (loop {loop_count}/{int(max_loops)})...",
                float(loop_count),
                max_loops,
            )
        gap = state.get("knowledge_gap", "").strip()
        gap_section = f"\nKnowledge gap: {gap}" if gap else ""
        human = (
            f"Question: {state['original_question']}{gap_section}\n\n"
            f"Evidence:\n{_render_evidence(state.get('evidence', []))}"
        )
        chain = self.config.small.with_structured_output(Reflection)
        parsed = cast(
            Reflection,
            await chain.ainvoke([SystemMessage(_REFLECTION_PROMPT), HumanMessage(human)]),
        )
        logger.info(
            "Reflection loop %d: is_sufficient=%s, follow_ups=%d, knowledge_gap=%s",
            loop_count,
            parsed.is_sufficient,
            len(parsed.follow_ups),
            parsed.knowledge_gap[:1200],
        )
        return {
            "research_loop_count": loop_count,
            "is_sufficient": parsed.is_sufficient,
            "follow_ups": parsed.follow_ups,
            "knowledge_gap": parsed.knowledge_gap,
        }

    async def finalize_answer(self, state: OverallState, config: RunnableConfig) -> dict[str, Any]:
        """Synthesise all gathered research into a final report."""
        reporter = config.get("configurable", {}).get("progress_reporter")
        if reporter:
            await reporter("Synthesizing final answer...", 0.0, None)
        evidence = state.get("evidence", [])
        gap = state.get("knowledge_gap", "").strip()
        gap_section = f"\n\nKnowledge gap: {gap}" if gap else ""
        human = (
            f"Question: {state['original_question']}\n\n"
            f"Evidence:\n{_render_evidence(evidence)}{gap_section}"
        )
        response = cast(
            Result,
            await self.config.small.with_structured_output(Result).ainvoke(
                [SystemMessage(_FINALIZE_PROMPT), HumanMessage(human)]
            ),
        )
        sources_gathered = list(dict.fromkeys(item["url"] for item in evidence))
        logger.info(
            "Final answer generated: %s. Explanation: %s",
            response["answer"],
            response.get("explanation", "")[:1000],
        )
        return {
            "answer": response["answer"],
            "explanation": response.get("explanation", ""),
            "sources_gathered": sources_gathered,
        }

    def continue_to_web_research(self, state: OverallState) -> list[Send]:
        """Fan-out: one parallel web_research branch per query."""
        return [
            Send("web_research", {
                "original_question": state["original_question"],
                "search_query": q,
                "id": i})
            for i, q in enumerate(state["queries"])
        ]

    def evaluate_research(
        self,
        state: OverallState,
    ) -> Literal["web_research", "finalize_answer"] | list[Send]:
        """Route to finalize or fan-out more searches based on reflection."""
        max_loops = state.get("max_research_loops", 3)
        follow_ups = state.get("follow_ups", [])

        if (
            state.get("is_sufficient")
            or state.get("research_loop_count", 0) >= max_loops
            or not follow_ups
        ):
            return "finalize_answer"

        return [
            Send("web_research", {
                "original_question": state["original_question"],
                "knowledge_gap": state.get("knowledge_gap", ""),
                "search_query": q,
                "id": i
                })
            for i, q in enumerate(follow_ups)
        ]

    async def __call__(
        self, input: str, progress: Any | None = None
    ) -> ResearchResponse:
        """Invoke the compiled graph; forward optional progress reporter to all nodes."""
        config: RunnableConfig | None = (
            {"configurable": {"progress_reporter": progress}} if progress is not None else None
        )
        result = await self.graph.ainvoke(
            {"question": input, "max_research_loops": self.max_research_loops},
            config=config,
        )
        return {
            "answer": result["answer"],
            "explanation": result.get("explanation", ""),
            "sources": result.get("sources_gathered", []),
        }
