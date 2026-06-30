"""Deep research agent using a LangGraph StateGraph with iterative search."""

from __future__ import annotations

# ruff: noqa: E501
import asyncio
import logging
from datetime import datetime
from operator import add
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypedDict, cast
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
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
    "RawWebResult",
    "ResearchAgent",
    "SummarizeSourceState",
    "create_deep_research_graph",
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
Use broad relevance criteria, keep text even slyghtly related to the question. Contradiction is also important information
Include links that may contain related information
Include facts, methods, and tools that does not explicitly related to question but support, or even contradict extracted facts
Hard constraints:
- Keep verbatim excerpts only. No summarization, no conclusions.
- Keep verbatim excerpts. Do not summarize, but specifically extract sentences that contain exact quantitative metrics, software package names, mathematical assumptions, and specific methodological frameworks.
- Remove boilerplate, ads, nav, comments, and completely unrelated text.
- Never output conversational text (e.g., "Based on...", "Please provide...", "Here is...").

If relevant evidence exists, output one or more blocks in this exact format:
- Source URL: [url]
- Content: [verbatim relevant excerpt]

If no relevant evidence exists in provided content, output exactly:
NO_RELEVANT_EVIDENCE
"""

_FAILURE_RECOVERY_PROMPT = """You are an autonomous web research agent tasked to fill a specific information gap.

You will be given the User's core Question and Knowledge Gap, along with a list of target URLs that may contain required information.

CRITICAL INSTRUCTIONS:
1. For each URL, read the web page and extract raw, verbatim evidence that answers or relates to the User's question and knowledge gap. Keep verbatim excerpts.
2. If the URL is behind a hard login wall, completely blank, or returns an explicit error page even through your tool, omit it from the final result. Do not invent contents for a page you cannot see.

OUTPUT FORMAT REQUIREMENTS:
If relevant evidence present on web page, format your entire response into one or more blocks matching this exact schema:
- Source URL: [insert exact fetched url]
- Content: [verbatim relevant excerpt extracted from the tool's page payload]

If no relevant evidence can be extracted from any of the recovered URLs, your entire final response must be exactly:
NO_RELEVANT_EVIDENCE

Important: Never output any conversational filler (e.g., "I have fetched the pages for you...", "According to the tool..."). Output only the matching evidence blocks or the exact fallback phrase.
"""

_REFLECTION_PROMPT = f"""You are the core reasoning engine of an advanced web research agent. 
Your task is to audit the gathered research for correctness, sufficiency, and source integrity, and then determine the next strategic step.

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
- The current evidence consists primarily of 'NO_RELEVANT_EVIDENCE'.

*Exception*: For purely self-contained logic or mathematical questions, you may set `is_sufficient = True` if you provide a mathematically rigorous derivation, even if external web sources are weak.
*Best-Effort Fallback*: If you hit your maximum reflection loops or constraints cannot be fully met, set `is_sufficient = True` under a "best-effort" policy, but you MUST explicitly declare your uncertainty metrics in the `knowledge_gap`.

Current date context: Today is {datetime.now().strftime("%A, %B %d, %Y")}. Use this to evaluate time-sensitivity, version obsolescence, and real-time data validity.

### 3. OUTPUT SPECIFICATION (JSON Format)
Your response must strictly conform to this schema:
{{
  "is_sufficient": boolean,
  "findings_summary": "String. Group all supporting evidence snippets explicitly by Tier (1, 2, or 3). Include verbatim quotes. Dedicate a specific paragraph to analyzing conflicting data if present.",
  "relevant_links": ["Array of strings. Include ONLY the valid, un-truncated URLs directly mapped to the evidence snippets above."],
  "knowledge_gap": "String. Describe the exact, concrete missing piece of data or validation step required.",
  "follow_ups": ["Array of strings. Next-step target queries conforming to the Search Query Rules below."]
}}

### 4. WEB SEARCH QUERY RULES (HARD MANIFESTO)
If `is_sufficient = False`, generate exactly 3 follow-up queries. Each query must target ONE unresolved gap. Never combine multiple concepts.

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

BAD (Conversational, Long, Compound):
- "How do I integrate ChatGPT Enterprise natively with Microsoft Purview DLP?"
- "compare Copilot Microsoft 365 vs ChatGPT Enterprise security governance"
"""


_FINALIZE_PROMPT = """You are the synthesis engine of a frontier research agent. 
Review the entire corpus of gathered research, reflection loops, and source materials to construct your final output. 

Your response must strictly follow a dual-component format, separated by a structural delimiter.

---

### COMPONENT 1: THE DIRECT RESOLUTION
[Provide ONLY the exact, naked, short response to the user's question. This component must contain zero preamble, zero introductory text, zero explanation, and zero source citations. It must strictly match the user's requested units, scale, rounding, casing, or phrasing format. Example: If the user asks for hours in thousands and the answer is 17,000, write exactly: 17]

--- STRUCTURAL BREAK: BEGIN DETAILED TECHNICAL APPENDIX ---

### COMPONENT 2: THE TECHNICAL APPENDIX & METHODOLOGICAL EVIDENCE
[Provide an exhaustive, unrestricted, hyper-detailed technical breakdown. This section must systematically address every explicit and implicit sub-question within the prompt. Do not summarize away complex technical, quantitative, or methodological details. You will be penalized for brevity.]

Execute your analysis according to these four mandatory pillars:

1. COMPREHENSIVE FACET ANALYSIS
Deconstruct and explain every architectural layer, mechanism, limitation, framework, or theory mentioned in the research. If multiple technical solutions exist, map them out in full detail alongside their trade-offs.

2. LOGICAL TRACEABILITY & SYSTEMATIC DERIVATION
Provide a step-by-step narrative showing exactly how Component 1 was derived from the data. Explicitly include any mathematical formulas, unit conversions, or arithmetic calculations used during evaluation.

3. RESOLUTION OF CONTRADICTIONS
If the search history revealed conflicting data or divergent view points between sources, do not choose one arbitrarily. Dedicate a section to detailing the friction, identifying the source tiers involved, and stating why a particular source was favored or outlining the remaining uncertainty.

4. INLINE EXPLICIT SOURCE ATTRIBUTION
Every factual claim, metric, or code sample in this Appendix must be immediately followed by an inline citation to its source URL from your gathered research (e.g., [Source URL]). Mark potential source reliability issues transparently.

### FINALIZATION SANITY CHECKLIST
Before generating your output, execute this verification step internally:
- Did I fulfill the precise format constraints of Component 1?
- Did I double-check all arithmetic operations using raw numbers before applying unit transformations?
- Did I include the granular, low-level technical specifics instead of high-level generalizations?
- If data was missing or uncertain, did I explicitly quantify that uncertainty in the appendix rather than omitting the facet entirely?
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


def extract_text(response_content) -> str:
    """Extract text from model response, handling Gemini's list-of-dicts format."""
    if isinstance(response_content, str):
        return response_content
    if isinstance(response_content, list):
        return "".join(
            [
                chunk.get("text", "")
                for chunk in response_content
                if isinstance(chunk, dict)
            ]
        )
    return str(response_content)


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
            "messages": [SystemMessage(_REFLECTION_PROMPT)],
            "queries": result.query,
            "max_research_loops": state.get("max_research_loops", 3),
            "original_question": state["question"],
        }

    async def web_research(self, state: WebSearchState, config: RunnableConfig) -> dict[str, Any]:
        """Execute a single web search and return normalized raw results."""
        query = state["search_query"]
        reporter = config.get("configurable", {}).get("progress_reporter")
        if reporter:
            await reporter("Searching", 0.0, None)
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

    async def clean_result(
        self,
        results: list[SearchResult],
        fetch_results: list[str],
        question: str,
        knowledge_gap: str,
    ) -> str:
        success_results = [
            f"Source URL: {sr.url}\nTitle:{sr.title}\nPage Sippet: {sr.snippet}\nCONTENT: {fr}"
            for sr, fr in zip(results, fetch_results, strict=False)
            if not fr.startswith("ERROR")
        ]
        failed_fetches = [
            sr for sr, fr in zip(results, fetch_results, strict=False) if fr.startswith("ERROR")
        ]
        logger.info(
            "Extract information from %d success and %d failed results. User question %.10s, knowledge gap: %.20s",
            len(success_results),
            len(failed_fetches),
            question,
            knowledge_gap,
        )
        knowledge_gap_section = (
            f"KNOWLEDGE GAP: {knowledge_gap}\n" if knowledge_gap else ""
        )
        formatted_message = (
            f"SEARCH QUERY: {question}\n"
            f"{knowledge_gap_section}"
            "Web Search results:\n"
        )
        if success_results:
            response = await self.config.fast.ainvoke(
                [
                    SystemMessage(_CLEAN_RESULT_PROMPT),
                    HumanMessage(formatted_message + "\n\n".join(success_results)),
                ]
            )
            clean_fetch_result = extract_text(response.content)
        else:
            clean_fetch_result = "NO_RELEVANT_EVIDENCE"
        if failed_fetches:
            logger.info("Try to recover %d failed fetches", len(failed_fetches))
            try:
                grounded_model = self.config.fast.bind_tools(
                    [{"url_context": {}}],
                    tool_choice="any",
                )
                # Do not recover more than 4 failures, gemini model got confused
                formatted_urls_block = "\n".join(
                    [
                        f"<target_url id='{i+1}'>{sr.url}</target_url>"
                        for i, sr in enumerate(failed_fetches[:4])
                    ]
                )
                user_payload = (
                    f"USER QUESTION: {question}\n"
                    f"{knowledge_gap_section}"
                    f"<web_sources>\n{formatted_urls_block}\n</web_sources>"
                )
                response = await grounded_model.ainvoke(
                    [
                        SystemMessage(_FAILURE_RECOVERY_PROMPT),
                        HumanMessage(user_payload),
                    ],
                    thinking_config= { "thinking_budget": 0 }
                )
                metadata = response.response_metadata.get("grounding_metadata", {})
                requested_urls = [
                    chunk["web"]["uri"]
                    for chunk in metadata.get("grounding_chunks", [])
                    if chunk.get("web") and chunk["web"].get("uri")
                ]
                logger.info("Fall back recovery requested urls: %s", requested_urls)
                fallback_result = extract_text(response.content)
                logger.info("Fallback result: %.100s", fallback_result)
                if clean_fetch_result.startswith("NO_RELEVANT_EVIDENCE"):
                    clean_fetch_result = fallback_result
                elif not fallback_result.startswith("NO_RELEVANT_EVIDENCE"):
                    clean_fetch_result += "\n\n" + fallback_result
            except Exception:
                # Model does not support web content grounded - throws exception on bind tools
                pass

        return clean_fetch_result


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
            await chain.ainvoke([*state.get("messages", []), research_msg]),
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

    async def finalize_answer(self, state: OverallState, config: RunnableConfig) -> dict[str, Any]:
        """Synthesise all gathered research into a final report."""
        reporter = config.get("configurable", {}).get("progress_reporter")
        if reporter:
            await reporter("Synthesizing final answer...", 0.0, None)
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
