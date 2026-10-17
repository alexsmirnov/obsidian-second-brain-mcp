"""Relevance filtering of real fetched pages, compared across ranker implementations.

Pages and queries come from the fetch smoke dataset. ``expectations.json`` lists,
per case, passages the filter must keep and boilerplate it must drop.
"""

from __future__ import annotations

import itertools
import json
import logging
from collections.abc import (
    AsyncGenerator,
    Callable,
    Generator,
    Iterable,
    Mapping,
)
from dataclasses import dataclass, field
from typing import cast

import httpx
import pytest
from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from web_fetch_evaluation import CASES_DIR, FetchCase, load_cases

from mcps.common.decisions import DecisionClient
from mcps.config import ServerConfig, create_config
from mcps.research.config import _create_chat_model
from mcps.research.deep_research import FETCH_QUERY_INSTRUCTIONS, FetchQuery
from mcps.research.tools.bm25_ranker import Bm25Ranker
from mcps.research.tools.decision_ranker import DecisionRanker
from mcps.research.tools.filtering import RelevanceFilter
from mcps.research.tools.filtering_models import RouterPassageModels, valid_router_url
from mcps.research.tools.llm_ranker import LlmRanker
from mcps.research.tools.markdown_chunker import MarkdownChunker
from mcps.research.tools.models import FetchResult, FetchStatus

logger = logging.getLogger(__name__)

PAGES_DIR = CASES_DIR / "filtering"
EXPECTATIONS: dict[str, dict] = json.loads(
    (PAGES_DIR / "expectations.json").read_text(encoding="utf-8")
)
CASES = {
    case.case_id: case for case in load_cases(CASES_DIR / "fetch-cases-smoke.jsonl")
}

QUERY_SOURCES = ("dataset", "generated")
GENERATED_QUERIES: dict[str, str] = {}


@dataclass(frozen=True)
class FetchQueryInput:
    """Question and optional knowledge gap sent to the fetch-query fragment."""

    question: str
    gap: str = ""


def fetch_query_input(case: FetchCase, cases: Iterable[FetchCase]) -> FetchQueryInput:
    """Build the fragment input for a case: own query at round 0, root+gap later."""
    question = case.query or ""
    if case.research_round in (None, 0) or case.task_id is None:
        return FetchQueryInput(question=question)
    root = next(
        (
            candidate
            for candidate in cases
            if candidate.task_id == case.task_id
            and candidate.research_round == 0
        ),
        None,
    )
    if root is None:
        return FetchQueryInput(question=question)
    return FetchQueryInput(question=root.query or "", gap=question)


def fetch_query_message(inp: FetchQueryInput) -> str:
    """Render the fragment human message with an optional gap header."""
    if inp.gap:
        return f"QUESTION: {inp.question}\nKNOWLEDGE GAP: {inp.gap}"
    return f"QUESTION: {inp.question}"


FilterBuilder = Callable[[ServerConfig, httpx.AsyncClient], RelevanceFilter]


def bm25_filter(
    _config: ServerConfig, _http_client: httpx.AsyncClient
) -> RelevanceFilter:
    return RelevanceFilter(MarkdownChunker(), Bm25Ranker())


def llm_filter(config: ServerConfig, http_client: httpx.AsyncClient) -> RelevanceFilter:
    if not (
        config.fetch_model
        and config.rag_embedding_model
        and config.router_api_key
        and valid_router_url(config.router_api_base)
    ):
        pytest.skip("FETCH_MODEL, RAG_EMBEDDING_MODEL or router settings not set")
    models = RouterPassageModels(
        http_client,
        router_url=config.router_api_base,
        router_key=config.router_api_key,
        fetch_model=config.fetch_model,
        embedding_model=config.rag_embedding_model,
        embedding_dimensions=config.rag_embedding_dimensions,
    )
    return RelevanceFilter(MarkdownChunker(), LlmRanker(models))


def decision_filter(
    config: ServerConfig, http_client: httpx.AsyncClient
) -> RelevanceFilter:
    if not (config.decision_model and valid_router_url(config.router_api_base)):
        pytest.skip("DECISION_MODEL or router settings not set")
    client = DecisionClient(
        http_client,
        router_url=config.router_api_base,
        router_key=config.router_api_key,
        model=config.decision_model,
    )
    return RelevanceFilter(MarkdownChunker(), DecisionRanker(client))


# Append an entry to compare another filter implementation on the same pages.
FILTERS: dict[str, FilterBuilder] = {
    "bm25": bm25_filter,
    "llm": llm_filter,
    "decision": decision_filter,
}


def calculate_precision(tp: int, fp: int) -> float:
    """Calculate precision: proportion of kept passages that are relevant."""
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def calculate_recall(tp: int, fn: int) -> float:
    """Calculate recall: proportion of relevant passages that were kept."""
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def calculate_f1(precision: float, recall: float) -> float:
    """Calculate harmonic mean of precision and recall."""
    if precision + recall == 0.0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def evaluate_retrieval(
    kept_content: str, relevant: list[str], irrelevant: list[str]
) -> tuple[int, int, int, int]:
    """Evaluate kept content against expected relevant and irrelevant passages.

    Returns:
        tuple of (tp, fp, fn, tn) counts.
    """
    tp = sum(1 for text in relevant if text in kept_content)
    fn = len(relevant) - tp
    fp = sum(1 for text in irrelevant if text in kept_content)
    tn = len(irrelevant) - fp
    return tp, fp, fn, tn


@dataclass(frozen=True)
class CaseMetrics:
    case_id: str
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def total_relevant(self) -> int:
        return self.tp + self.fn

    @property
    def total_irrelevant(self) -> int:
        return self.fp + self.tn

    @property
    def precision(self) -> float:
        return calculate_precision(self.tp, self.fp)

    @property
    def recall(self) -> float:
        return calculate_recall(self.tp, self.fn)

    @property
    def f1(self) -> float:
        return calculate_f1(self.precision, self.recall)


@dataclass
class FilterEvaluation:
    filter_name: str
    cases: dict[str, CaseMetrics] = field(default_factory=dict)

    def add_case(self, case: CaseMetrics) -> None:
        self.cases[case.case_id] = case

    @property
    def total_tp(self) -> int:
        return sum(c.tp for c in self.cases.values())

    @property
    def total_fp(self) -> int:
        return sum(c.fp for c in self.cases.values())

    @property
    def total_fn(self) -> int:
        return sum(c.fn for c in self.cases.values())

    @property
    def total_tn(self) -> int:
        return sum(c.tn for c in self.cases.values())

    @property
    def micro_precision(self) -> float:
        return calculate_precision(self.total_tp, self.total_fp)

    @property
    def micro_recall(self) -> float:
        return calculate_recall(self.total_tp, self.total_fn)

    @property
    def micro_f1(self) -> float:
        return calculate_f1(self.micro_precision, self.micro_recall)

    @property
    def macro_precision(self) -> float:
        if not self.cases:
            return 0.0
        return sum(c.precision for c in self.cases.values()) / len(self.cases)

    @property
    def macro_recall(self) -> float:
        if not self.cases:
            return 0.0
        return sum(c.recall for c in self.cases.values()) / len(self.cases)

    @property
    def macro_f1(self) -> float:
        """Arithmetic mean of per-case F1 scores (macro-averaged F1)."""
        if not self.cases:
            return 0.0
        return sum(c.f1 for c in self.cases.values()) / len(self.cases)


EVALUATIONS: dict[str, FilterEvaluation] = {}


def format_evaluation_report(
    evaluations: dict[str, FilterEvaluation],
    queries: Mapping[str, str] | None = None,
) -> str:
    """Format evaluation metrics as an aligned summary table."""
    active_evals = {name: ev for name, ev in evaluations.items() if ev.cases}
    if not active_evals:
        return "No relevance filter evaluations recorded."

    hdr = (
        f"{'Filter':<18} {'Cases':<6} {'Precision':<10} {'Recall':<10} {'F1':<10} "
        f"{'Macro-P':<10} {'Macro-R':<10} {'Macro-F1':<10}"
    )
    sep = "=" * len(hdr)
    dash_sep = "-" * len(hdr)

    lines = [
        "",
        sep,
        "RELEVANCE FILTER EVALUATION REPORT",
        sep,
        hdr,
        dash_sep,
    ]
    for name, ev in active_evals.items():
        lines.append(
            f"{name:<18} {len(ev.cases):<6} "
            f"{ev.micro_precision:<10.4f} {ev.micro_recall:<10.4f} "
            f"{ev.micro_f1:<10.4f} {ev.macro_precision:<10.4f} "
            f"{ev.macro_recall:<10.4f} {ev.macro_f1:<10.4f}"
        )
    lines.append(dash_sep)

    for name, ev in active_evals.items():
        total_eval = ev.total_tp + ev.total_fp + ev.total_fn + ev.total_tn
        lines.append(
            f"  [{name}] Counts: TP={ev.total_tp}, FP={ev.total_fp}, "
            f"FN={ev.total_fn}, TN={ev.total_tn} (Total: {total_eval})"
        )

    lines.append("")
    lines.append("Per-case breakdown:")
    case_hdr = (
        f"  {'Filter':<18} {'Case ID':<14} {'Precision':<10} {'Recall':<10} "
        f"{'F1':<10} {'TP/Rel':<8} {'FP/Irrel':<8}"
    )
    lines.append(case_hdr)
    lines.append(f"  {'-' * (len(case_hdr) - 2)}")
    for name, ev in active_evals.items():
        for c in ev.cases.values():
            rel_ratio = f"{c.tp}/{c.total_relevant}"
            irrel_ratio = f"{c.fp}/{c.total_irrelevant}"
            lines.append(
                f"  {name:<18} {c.case_id:<14} {c.precision:<10.4f} "
                f"{c.recall:<10.4f} {c.f1:<10.4f} {rel_ratio:<8} {irrel_ratio:<8}"
            )
    if queries:
        lines.append("")
        lines.append("Generated fetch queries:")
        lines.extend(
            f"  {case_id}: {query}" for case_id, query in sorted(queries.items())
        )
    lines.append(sep)
    return "\n".join(lines)


def page_content(case_id: str) -> str:
    return (PAGES_DIR / EXPECTATIONS[case_id]["file"]).read_text(encoding="utf-8")


@pytest.fixture
async def http_client() -> AsyncGenerator[httpx.AsyncClient]:
    async with httpx.AsyncClient(timeout=120.0) as client:
        yield client


@pytest.fixture(scope="module")
def config() -> ServerConfig:
    load_dotenv(find_dotenv(usecwd=True))
    return create_config()


@pytest.fixture(scope="module", autouse=True)
def report_filter_metrics(
    request: pytest.FixtureRequest,
) -> Generator[None]:
    EVALUATIONS.clear()
    GENERATED_QUERIES.clear()
    for name, source in itertools.product(FILTERS, QUERY_SOURCES):
        key = f"{name}/{source}"
        EVALUATIONS[key] = FilterEvaluation(filter_name=key)
    yield
    report = format_evaluation_report(EVALUATIONS, GENERATED_QUERIES)
    logger.info("%s", report)
    reporter = request.config.pluginmanager.get_plugin("terminalreporter")
    if reporter:
        reporter.write_line(report)
    else:
        print(report)


@pytest.fixture(params=list(FILTERS))
def filter_name(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(params=QUERY_SOURCES)
def query_source(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def page_filter(
    filter_name: str,
    config: ServerConfig,
    http_client: httpx.AsyncClient,
) -> RelevanceFilter:
    return FILTERS[filter_name](config, http_client)


def _generation_model_name(case: FetchCase, config: ServerConfig) -> str:
    """Round 0/unknown uses the fast model; later rounds use the infer model."""
    if case.research_round in (None, 0):
        return config.research_fast_model
    return config.research_infer_model


async def generate_fetch_query(
    case: FetchCase,
    config: ServerConfig,
    http_client: httpx.AsyncClient,
) -> str:
    """Generate (and cache) one fetch query per case via the shared fragment."""
    if case.case_id in GENERATED_QUERIES:
        return GENERATED_QUERIES[case.case_id]
    inp = fetch_query_input(case, CASES.values())
    model_name = _generation_model_name(case, config)
    if not (
        config.research_fast_model
        and config.research_infer_model
        and config.router_api_key
        and valid_router_url(config.router_api_base)
    ):
        pytest.skip(
            "RESEARCH_FAST_MODEL, RESEARCH_INFER_MODEL or router settings not set"
        )
    model = _create_chat_model(
        model_name=model_name,
        router_url=config.router_api_base,
        router_key=config.router_api_key,
        http_client=http_client,
    )
    parsed = cast(
        FetchQuery,
        await model.with_structured_output(FetchQuery).ainvoke(
            [
                SystemMessage(FETCH_QUERY_INSTRUCTIONS),
                HumanMessage(fetch_query_message(inp)),
            ]
        ),
    )
    query = parsed.fetch_query.strip() or inp.question
    GENERATED_QUERIES[case.case_id] = query
    logger.info("Generated fetch query for %s: %s", case.case_id, query)
    return query


def test_calculate_precision_recall_f1() -> None:
    assert calculate_precision(3, 1) == 0.75
    assert calculate_precision(0, 0) == 0.0
    assert calculate_recall(3, 1) == 0.75
    assert calculate_recall(0, 0) == 0.0
    assert calculate_f1(0.75, 0.75) == 0.75
    assert calculate_f1(0.0, 0.0) == 0.0


def test_evaluate_retrieval() -> None:
    content = "the quick brown fox jumps over the lazy dog"
    tp, fp, fn, tn = evaluate_retrieval(
        content,
        relevant=["quick brown", "silent cat"],
        irrelevant=["lazy dog", "loud bird"],
    )
    assert tp == 1
    assert fn == 1
    assert fp == 1
    assert tn == 1


def test_filter_evaluation_aggregates() -> None:
    eval_filter = FilterEvaluation(filter_name="test")
    eval_filter.add_case(CaseMetrics(case_id="c1", tp=2, fp=1, fn=0, tn=2))
    eval_filter.add_case(CaseMetrics(case_id="c2", tp=1, fp=0, fn=1, tn=2))
    assert eval_filter.total_tp == 3
    assert eval_filter.total_fp == 1
    assert eval_filter.total_fn == 1
    assert eval_filter.total_tn == 4
    assert eval_filter.micro_precision == 3 / 4
    assert eval_filter.micro_recall == 3 / 4
    assert eval_filter.micro_f1 == 0.75
    assert eval_filter.macro_precision == (2 / 3 + 1.0) / 2
    assert eval_filter.macro_recall == (1.0 + 0.5) / 2
    # c1: P=2/3, R=1.0, F1=0.8; c2: P=1.0, R=0.5, F1=2/3
    assert eval_filter.macro_f1 == pytest.approx((0.8 + 2 / 3) / 2)

    # Test report formatting
    report = format_evaluation_report({"test": eval_filter})
    assert "RELEVANCE FILTER EVALUATION REPORT" in report
    assert "test" in report


def test_fetch_query_input_round_zero_uses_case_query_as_question() -> None:
    case = CASES["baseline-079"]

    assert fetch_query_input(case, CASES.values()) == FetchQueryInput(
        question=case.query or ""
    )


@pytest.mark.parametrize(
    ("case_id", "root_id"),
    [("baseline-263", "baseline-172"), ("baseline-124", "baseline-079")],
)
def test_fetch_query_input_later_round_uses_task_question_and_gap(
    case_id: str, root_id: str
) -> None:
    case = CASES[case_id]

    assert fetch_query_input(case, CASES.values()) == FetchQueryInput(
        question=CASES[root_id].query or "", gap=case.query or ""
    )


def test_fetch_query_input_without_root_round_falls_back_to_own_query() -> None:
    orphan = FetchCase(
        case_id="x",
        url="https://x.example",
        query="gap text",
        baseline_error=None,
        baseline_chars=None,
        task_id="none",
        research_round=1,
    )

    assert fetch_query_input(orphan, [orphan, *CASES.values()]) == FetchQueryInput(
        question="gap text"
    )


def test_fetch_query_input_without_task_id_does_not_match_root() -> None:
    orphan = FetchCase(
        case_id="y",
        url="https://y.example",
        query="gap text",
        baseline_error=None,
        baseline_chars=None,
        task_id=None,
        research_round=1,
    )

    assert fetch_query_input(orphan, [orphan, *CASES.values()]) == FetchQueryInput(
        question="gap text"
    )


def test_fetch_query_message_adds_gap_header_only_when_present() -> None:
    assert fetch_query_message(FetchQueryInput("Q")) == "QUESTION: Q"
    assert (
        fetch_query_message(FetchQueryInput("Q", "G"))
        == "QUESTION: Q\nKNOWLEDGE GAP: G"
    )


def test_report_lists_source_rows_and_generated_queries() -> None:
    evaluation = FilterEvaluation("bm25/generated")
    evaluation.add_case(CaseMetrics(case_id="c1", tp=1, fp=0, fn=0, tn=1))

    report = format_evaluation_report(
        {"bm25/generated": evaluation}, {"c1": "parallel trends staggered"}
    )

    assert any(line.startswith("bm25/generated ") for line in report.splitlines())
    assert "c1: parallel trends staggered" in report


@pytest.mark.parametrize("case_id", sorted(EXPECTATIONS))
def test_expected_substrings_occur_once_in_page(case_id: str) -> None:
    content = page_content(case_id)
    expected = EXPECTATIONS[case_id]

    counts = {
        text: content.count(text)
        for text in expected["relevant"] + expected["irrelevant"]
    }

    assert expected["relevant"]
    assert expected["irrelevant"]
    assert {text: count for text, count in counts.items() if count != 1} == {}


@pytest.mark.parametrize("case_id", sorted(EXPECTATIONS))
async def test_filter_keeps_relevant_and_drops_irrelevant_passages(
    filter_name: str,
    query_source: str,
    page_filter: RelevanceFilter,
    case_id: str,
    config: ServerConfig,
    http_client: httpx.AsyncClient,
) -> None:
    case = CASES[case_id]
    expected = EXPECTATIONS[case_id]
    content = page_content(case_id)
    page = FetchResult(
        url=case.url, status=FetchStatus.OK, mime=expected["mime"], content=content
    )

    query = (
        case.query
        if query_source == "dataset"
        else await generate_fetch_query(case, config, http_client)
    )
    assert query and query.strip()

    result = await page_filter(page, query)

    kept = result.content
    assert result.status is FetchStatus.OK

    tp, fp, fn, tn = evaluate_retrieval(
        kept, expected["relevant"], expected["irrelevant"]
    )
    case_metric = CaseMetrics(case_id=case_id, tp=tp, fp=fp, fn=fn, tn=tn)
    evaluation_key = f"{filter_name}/{query_source}"
    if evaluation_key not in EVALUATIONS:
        EVALUATIONS[evaluation_key] = FilterEvaluation(filter_name=evaluation_key)
    EVALUATIONS[evaluation_key].add_case(case_metric)

    logger.info(
        "%s/%s [%s] kept %d/%d chars | P=%.3f R=%.3f F1=%.3f "
        "(TP=%d, FP=%d, FN=%d, TN=%d)",
        filter_name,
        query_source,
        case_id,
        len(kept),
        len(content),
        case_metric.precision,
        case_metric.recall,
        case_metric.f1,
        tp,
        fp,
        fn,
        tn,
    )
    if (fn > 0 or fp > 0) and logger.isEnabledFor(logging.DEBUG):
        missing = [text for text in expected["relevant"] if text not in kept]
        leaked = [text for text in expected["irrelevant"] if text in kept]
        logger.debug(
            "%s/%s [%s] missing=%d leaked=%d",
            filter_name,
            query_source,
            case_id,
            len(missing),
            len(leaked),
        )
