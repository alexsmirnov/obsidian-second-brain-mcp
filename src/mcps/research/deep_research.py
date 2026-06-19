"""Deep research agent using a LangGraph StateGraph with iterative search."""

from __future__ import annotations

# ruff: noqa: E501, W291
import asyncio
import logging
from operator import add
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypedDict, cast
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages as _add_messages
from langgraph.types import Overwrite, Send
from pydantic import BaseModel, Field
from mcps.research.tools import SearchResult

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from mcps.research.agent import ResearchResponse
    from mcps.research.config import ResearchConfig

__all__ = [
    "create_deep_research_graph",
    "ResearchAgent",
    "RawWebResult",
    "SummarizeSourceState",
]

logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_GENERATE_QUERY_PROMPT = """You are a search query optimizer.
Generate exactly 5 short web search queries for the user question.

Hard rules:
1) Each query MUST be 2-5 keywords. Never exceed 6 words.
2) Each query must include at least one named entity from the question.
3) Queries must be orthogonal — each targets a DIFFERENT facet:
   - Facet 1: core entity + primary attribute
   - Facet 2: core entity + secondary attribute or comparison
   - Facet 3: related entity or alternative name
   - Facet 4: official source or documentation
   - Facet 5: specific data point or constraint from the question
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
  "ChatGPT Enterprise security features"
  "Copilot 365 data governance"
  "ChatGPT Enterprise admin controls"

Output: Return only 5 query strings.
"""
_CLEAN_RESULT_PROMPT = """You extract evidence relevant to the user question and knowledge gap.
Use broad relevance criteria, keep text even slyghtly related to the question.
Hard constraints:
- Keep verbatim excerpts only. No summarization, no conclusions.
- Remove boilerplate, ads, nav, comments, and completely unrelated text.
- Never output conversational text (e.g., "Based on...", "Please provide...", "Here is...").
- Ignore tool/error artifacts and placeholders.

If relevant evidence exists, output one or more blocks in this exact format:
- Source URL: [url]
- Content: [verbatim relevant excerpt]

If no relevant evidence exists in provided content, output exactly:
NO_RELEVANT_EVIDENCE
"""

_REFLECTION_PROMPT = """Analyze gathered research for correctness, sufficiency, and source quality.

Source tiers:
- Tier 1: official documentation, primary records, peer-reviewed papers, official archives.
- Tier 2: expert/community technical sources.
- Tier 3: general summaries and tertiary sources.

Sufficiency gate (strict):
- Do not set is_sufficient=True unless the core claim is directly supported.
- If the user requested required source/time/version constraints, do not set is_sufficient=True until those constraints are verified in evidence.
- If evidence is conflicting, unresolved, or mostly NO_RELEVANT_EVIDENCE, set is_sufficient=False.
- For self-contained logic/math questions, it is acceptable to set is_sufficient=True with rigorous derivation even when web sources are weak.

Best-effort policy:
- When constraints cannot be fully met, continue with follow-up queries and produce best-effort findings, explicitly stating uncertainty.

Output requirements:
- findings_summary must include all supporting evidence snippets grouped by tier and explicit conflict notes.
- relevant_links should include only URLs actually used as evidence.
- knowledge_gap must be concrete and actionable.
- follow_ups must be SHORT keyword queries (2-5 words each).
  Target ONE unresolved gap per query. Never combine multiple concepts.
  BAD: "official documentation for X Y Z integration compatibility"
  GOOD: "X Y compatibility", "Z integration docs"
  Or contain exact URLs for direct fetch.
"""

_FINALIZE_PROMPT = """Synthesize a final answer to the user's original question based on the research findings and sources gathered.
The answer itself should be short exact response to the user's question without any explanation or source attribution. The explanation should provide a brief rationale of how the answer was derived from the research findings, including information that supported the answer and highlighting any key sources that informed the conclusion.
For answer explanation, include a brief rationale of how the answer was derived from the research findings, include information that supported answer, highlighting any key sources that informed the conclusion.
Guidelines:
 Accuracy: Ensure the answer is technically correct and directly addresses the user's question.
 Clarity: Write in a clear, concise manner. Directly answer the question without unnecessary preamble.
Source Attribution: Explicitly reference the sources that support each part of the answer.
If the answer includes code snippets, ensure they are well-formatted and tested against the gathered research
 Finalization checklist (mandatory before writing answer):
 - Verify requested answer format exactly (units, scale, rounding, casing, phrasing constraints).
 - Verify arithmetic and unit conversions explicitly.
 - Example: if question asks for answer in "thousands of hours", convert 17000 hours to 17.
 - Resolve source-constraint requirements where possible; if unresolved, provide best-effort answer with explicit uncertainty in explanation.
"""

# ---------------------------------------------------------------------------
# Pydantic models (structured LLM output)
# ---------------------------------------------------------------------------


class SearchQueryList(BaseModel):
    """Structured output for the generate_query node."""

    query: list[str] = Field(description="Exactly 5 short (2-5 keyword) search queries, or exact URLs for direct fetch.")


class Reflection(BaseModel):
    """Structured output for the reflection node."""

    is_sufficient: bool = Field(
        description="Whether gathered info is sufficient and authoritative enough to answer the question"
    )
    findings_summary: str = Field(
        default="",
        description=(
            "Summary of research findings categorized by source authority, "
            "with text fragments that support the answer"
        ),
    )
    relevant_links: list[str] = Field(
        default_factory=list,
        description="URL links to relevant sources used to generate findings",
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

    messages: Annotated[list, _add_messages]
    original_question: str
    queries: list[str]  # current queries — overwrite each generate_query
    follow_ups: list[str]  # reflection output — overwrite each loop
    web_results: Annotated[list[str], add]
    summarized_findings: Annotated[list[str], add]
    findings: Annotated[list[str], add]  # accumulated across loops
    knowledge_gap: str
    sources_gathered: Annotated[list[str], add]  # accumulated across loops
    research_loop_count: int
    max_research_loops: int
    is_sufficient: bool


class WebSearchState(TypedDict):
    """Minimal payload sent to each parallel web_research branch."""
    original_question: str
    knowledge_gap: str

    search_query: str
    id: int


class RawWebResult(TypedDict):
    text: str
    search_query: str
    id: int
    url: str


class SummarizeSourceState(TypedDict):
    raw_text: str
    search_query: str
    id: int
    url: str


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

def create_deep_research_graph(config: ResearchConfig) -> CompiledStateGraph:
    """Build and compile the deep research StateGraph.

    Args:
        config: Research configuration providing models and tools.

    Returns:
        A compiled LangGraph runnable ready to invoke.
    """
    return ResearchAgent(config).create_graph()


class ResearchAgent:
    """Deep research agent that owns graph construction and invocation."""

    config: ResearchConfig
    graph: CompiledStateGraph
    max_research_loops: int

    def __init__(
        self,
        config: ResearchConfig,
        *,
        max_research_loops: int = 3,
    ) -> None:
        self.config = config
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

    async def generate_query(self, state: Input) -> dict[str, Any]:
        """Generate initial search queries from the user's question."""
        chain = self.config.fast.with_structured_output(SearchQueryList)
        result: SearchQueryList = await chain.ainvoke(
            [
                SystemMessage(_GENERATE_QUERY_PROMPT),
                HumanMessage(state["question"]),
            ]
        )  # type: ignore
        logger.info("Generated queries: %d", len(result.query))
        return {
            "messages": [SystemMessage(_REFLECTION_PROMPT)],
            "queries": result.query,
            "max_research_loops": state.get("max_research_loops", 3),
            "original_question": state["question"],
        }

    async def web_research(self, state: WebSearchState) -> dict[str, Any]:
        """Execute a single web search and return normalized raw results."""
        query = state["search_query"]
        # If query is a direct URL, fetch it directly and clean up without searching
        if _is_valid_url(query):
            logger.info("Direct URL fetch for query: %s", query)
            fetch_result = await self.config.fetch(query)
            cleaned = await self.clean_result(
                results=[SearchResult(url=query, title="", snippet="")],
                fetch_results=[fetch_result],
                question=state['original_question'],
                knowledge_gap=state.get('knowledge_gap', 'N/A'),
            )
            return {
                "web_results": [f"Direct URL fetch: {query}\nResult:\n{cleaned}"],
            }
        logger.info("Performing web search for query: %s", query)
        results = await self.config.search(query)
        if not results:
            logger.warning("No web search results found for query: %s", query)
            return {
                "web_results": [f"Web search query: {query}\nNo results found."],
            }
        fetch_tasks = [self.config.fetch(result.url) for result in results]
        fetch_results = await asyncio.gather(*fetch_tasks)
        logger.info("Web search result: %d", len(fetch_results))
        question = state['original_question']
        knowledge_gap = state.get('knowledge_gap', 'N/A')
        clean_result = await self.clean_result(results, fetch_results, question, knowledge_gap)
        logger.info("Web search raw results: %s", clean_result[:50])
        return {
            "web_results": [f"Web search query: {query}\nResult:\n{clean_result}"],
        }

    async def clean_result(self, results: list[SearchResult], fetch_results: list[str], question: str, knowledge_gap: str) -> str:
        formatted_result = (
            f"Search Query: {question}\n"
            f"Knowledge Gaps: {knowledge_gap}\n\n"
            "Raw Web Search Results:\n"
            + "\n\n".join((f"Source: {sr.url}\nSnippet: {sr.snippet}\nContent: {fr}" for sr, fr in zip(results, fetch_results)))
        )
        clean_result = (await self.config.fast.ainvoke(
            [
                SystemMessage(_CLEAN_RESULT_PROMPT),
                HumanMessage(formatted_result),
            ]
        )).content

        return clean_result # type: ignore


    async def reflection(self, state: OverallState) -> dict[str, Any]:
        """Evaluate research completeness and identify knowledge gaps."""
        loop_count = state.get("research_loop_count", 0) + 1
        research_content = "\n\n".join(state.get("web_results", []))
        research_msg = HumanMessage(
            content=(
                f"Question: {state['original_question']}\n\n"
                f"Research:\n{research_content}"
            ),
            name="research_aggregator",
        )
        chain = self.config.small.with_structured_output(
            Reflection,
            include_raw=True,
            method="json_schema"
        )
        result = cast(
            dict[str, Any],
            await chain.ainvoke(state.get("messages", []) + [research_msg]),
        )
        raw_message = result["raw"]
        parsed = cast(Reflection, result["parsed"])
        logger.info(
            (
                "Reflection loop %d: is_sufficient=%s, findings: %s, "
                "follow_ups=%d, knowledge_gap=%s"
            ),
            loop_count,
            parsed.is_sufficient,
            parsed.findings_summary[:1200],
            len(parsed.follow_ups),
            parsed.knowledge_gap[:1200],
        )
        return {
            "messages": [research_msg, raw_message],
            "research_loop_count": loop_count,
            "is_sufficient": parsed.is_sufficient,
            "follow_ups": parsed.follow_ups,
            "findings": [parsed.findings_summary],
            "knowledge_gap": parsed.knowledge_gap,
            "sources_gathered": parsed.relevant_links,
            "web_results": Overwrite(value=[]),
            "summarized_findings": Overwrite(value=[]),
        }

    async def finalize_answer(self, state: OverallState) -> dict[str, Any]:
        """Synthesise all gathered research into a final report."""
        response = cast(
            Result,
            await self.config.small.with_structured_output(
            Result
            ).ainvoke(state["messages"] + [HumanMessage(_FINALIZE_PROMPT)]),
        )
        sources_gathered = list(dict.fromkeys(state.get("sources_gathered", [])))
        logger.info("Final answer generated: %s. Explanation: %s", response["answer"], response.get("explanation", "")[:1000]  )
        return {
            "answer": response["answer"],
            "explanation": response.get("explanation", ""),
            "sources_gathered": Overwrite(value=sources_gathered),
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

    async def __call__(self, input: str) -> ResearchResponse:
        """Invoke the compiled graph and return the public response payload."""
        result = await self.graph.ainvoke(
            {"question": input, "max_research_loops": self.max_research_loops}
        )
        return {
            "answer": result["answer"],
            "explanation": result.get("explanation", ""),
            "sources": result.get("sources_gathered", []),
        }
