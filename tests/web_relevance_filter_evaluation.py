#!/usr/bin/env python3
"""Evaluate relevance filtering of real fetched pages against labelled passages.

Run from the repository root:
``uv run tests/web_relevance_filter_evaluation.py --filter bm25 --limit 5``.

Filters: ``bm25`` and ``decision`` chunk the page and rank the fragments;
``llm`` sends the whole page to ``RESEARCH_FAST_MODEL`` and keeps the
extracted text.

Tuning arguments override the chunker size and overlap, the ranker threshold,
the fetch-query prompt, the BM25 relative cutoff, smoothing and link penalty, and
the decision ranker question, for example
``--filter bm25 --max-chars 3000 --overlap-chars 500 --threshold 2.0``. Every
report row records the effective values under ``settings``.

Pages and queries come from the fetch baseline and smoke datasets; smoke rows
win on a shared case id. ``expectations.json`` lists,
per case, passages the filter must keep and boilerplate it must drop.

Outputs go to ``tmp/``: ``relevance-<filter>-<timestamp>.log`` and a
``.jsonl`` report with one row per page. Each row lists the selected fragments
(line range in the page and their first 30 characters; chunked filters only),
the relevant passages that were missed and the irrelevant ones that leaked into
the result.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import httpx
import nltk
from dotenv import find_dotenv, load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from web_fetch_evaluation import CASES_DIR, FetchCase, load_cases

from mcps.common.decisions import DecisionClient, Noul, NoulCriteria
from mcps.config import ServerConfig, create_config
from mcps.research.deep_research import FETCH_QUERY_INSTRUCTIONS, FetchQuery
from mcps.research.research_config import create_chat_model
from mcps.research.tools.bm25_ranker import Bm25Ranker
from mcps.research.tools.decision_ranker import RELEVANCE_QUESTION, DecisionRanker
from mcps.research.tools.fetch import (
    BM25_RELATIVE_CUTOFF,
    BM25_SMOOTHING,
    BM25_THRESHOLD,
    BM25_WINDOW,
    DECISION_WINDOW,
)
from mcps.research.tools.filtering import (
    ChunkedRelevanceFilter,
    LlmRelevanceFilter,
    Ranker,
    RelevanceFilter,
)
from mcps.research.tools.markdown_chunker import MarkdownChunker
from mcps.research.tools.models import FetchResult, FetchStatus

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / "tmp"
PAGES_DIR = CASES_DIR / "filtering"
PREVIEW_CHARS = 30
QUERY_SOURCES = ("dataset", "generated")

CHUNKER_OPTIONS = ("max_chars", "overlap_chars", "threshold")
BM25_OPTIONS = ("relative_cutoff", "smoothing", "link_penalty")
DECISION_OPTIONS = ("decision_instructions", "decision_true", "decision_false")
# Tuning options each filter accepts; the rest must stay unset.
FILTER_OPTIONS: dict[str, tuple[str, ...]] = {
    "bm25": CHUNKER_OPTIONS + BM25_OPTIONS,
    "decision": CHUNKER_OPTIONS + DECISION_OPTIONS,
    "llm": (),
}
TUNING_OPTIONS = CHUNKER_OPTIONS + BM25_OPTIONS + DECISION_OPTIONS


def _configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        encoding="utf-8",
        level=logging.INFO,
        format="%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def _override[T](value: T | None, default: T) -> T:
    """Explicit CLI value, empty strings included, else the default."""
    return default if value is None else value


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
            if candidate.task_id == case.task_id and candidate.research_round == 0
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


def router_chat_model(
    config: ServerConfig, model_name: str | None, http_client: httpx.AsyncClient
) -> BaseChatModel:
    """Router-backed chat model; fail fast when the model or router is unset."""
    if not (model_name and config.router.api_key and config.router.api_base):
        raise RuntimeError("research model or router settings not set")
    return create_chat_model(
        model_name=model_name,
        router_url=config.router.api_base,
        router_key=config.router.api_key,
        http_client=http_client,
    )


class RecordingRanker:
    """Ranker wrapper that remembers the last fragments and the indices chosen."""

    def __init__(self, ranker: Ranker) -> None:
        self._ranker = ranker
        self.fragments: list[str] = []
        self.selected: list[int] = []

    async def __call__(self, fragments: list[str], query: str) -> list[int]:
        self.fragments = fragments
        self.selected = await self._ranker(fragments, query)
        return self.selected


@dataclass(frozen=True)
class FilterSetup:
    """Filter under evaluation, its effective settings and optional recorder."""

    page_filter: RelevanceFilter
    settings: dict[str, object]
    recorder: RecordingRanker | None = None


def chunked_setup(
    args: argparse.Namespace,
    window: tuple[int, int],
    ranker: Ranker,
    ranker_settings: dict[str, object],
) -> FilterSetup:
    """Wrap a ranker into a recorded chunked filter with CLI window overrides."""
    max_chars = _override(args.max_chars, window[0])
    overlap_chars = _override(args.overlap_chars, window[1])
    recorder = RecordingRanker(ranker)
    return FilterSetup(
        page_filter=ChunkedRelevanceFilter(
            MarkdownChunker(max_chars=max_chars, overlap_chars=overlap_chars),
            recorder,
        ),
        settings={
            "max_chars": max_chars,
            "overlap_chars": overlap_chars,
            **ranker_settings,
        },
        recorder=recorder,
    )


def bm25_filter(
    args: argparse.Namespace, _config: ServerConfig, _http_client: httpx.AsyncClient
) -> FilterSetup:
    ranker = Bm25Ranker(
        threshold=_override(args.threshold, BM25_THRESHOLD),
        relative_cutoff=_override(args.relative_cutoff, BM25_RELATIVE_CUTOFF),
        smoothing=_override(args.smoothing, BM25_SMOOTHING),
        link_penalty=_override(args.link_penalty, 0.0),
    )
    return chunked_setup(
        args,
        BM25_WINDOW,
        ranker,
        {
            "threshold": ranker.threshold,
            "relative_cutoff": ranker.relative_cutoff,
            "smoothing": ranker.smoothing,
            "link_penalty": ranker.link_penalty,
        },
    )


def decision_question(args: argparse.Namespace) -> Noul:
    """Relevance question with CLI overrides applied field by field."""
    criteria = RELEVANCE_QUESTION.criteria or NoulCriteria()
    return Noul(
        instructions=_override(
            args.decision_instructions, RELEVANCE_QUESTION.instructions
        ),
        criteria=NoulCriteria(
            true=_override(args.decision_true, criteria.true),
            false=_override(args.decision_false, criteria.false),
        ),
    )


def decision_filter(
    args: argparse.Namespace, config: ServerConfig, http_client: httpx.AsyncClient
) -> FilterSetup:
    if not (config.fetch.decision_model and config.router.api_base):
        raise RuntimeError("DECISION_MODEL or router settings not set")
    threshold = {} if args.threshold is None else {"threshold": args.threshold}
    question = decision_question(args)
    ranker = DecisionRanker(
        DecisionClient(
            http_client,
            router_url=config.router.api_base,
            router_key=config.router.api_key,
            model=config.fetch.decision_model,
        ),
        question=question,
        **threshold,
    )
    criteria = question.criteria or NoulCriteria()
    return chunked_setup(
        args,
        DECISION_WINDOW,
        ranker,
        {
            "threshold": ranker.threshold,
            "decision_question": {
                "instructions": question.instructions,
                "true": criteria.true,
                "false": criteria.false,
            },
        },
    )


def llm_filter(
    _args: argparse.Namespace, config: ServerConfig, http_client: httpx.AsyncClient
) -> FilterSetup:
    model_name = config.research.fast_model
    return FilterSetup(
        page_filter=LlmRelevanceFilter(
            router_chat_model(config, model_name, http_client)
        ),
        settings={"model": model_name},
    )


# Append an entry to compare another filter implementation on the same pages.
FILTERS: dict[
    str,
    Callable[[argparse.Namespace, ServerConfig, httpx.AsyncClient], FilterSetup],
] = {
    "bm25": bm25_filter,
    "decision": decision_filter,
    "llm": llm_filter,
}


def calculate_precision(tp: int, fp: int) -> float:
    """Proportion of kept passages that are relevant."""
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def calculate_recall(tp: int, fn: int) -> float:
    """Proportion of relevant passages that were kept."""
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def calculate_f1(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall."""
    if precision + recall == 0.0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


@dataclass(frozen=True)
class CaseMetrics:
    case_id: str
    tp: int
    fp: int
    fn: int
    tn: int

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
    cases: list[CaseMetrics] = field(default_factory=list)

    @property
    def micro_precision(self) -> float:
        return calculate_precision(
            sum(c.tp for c in self.cases), sum(c.fp for c in self.cases)
        )

    @property
    def micro_recall(self) -> float:
        return calculate_recall(
            sum(c.tp for c in self.cases), sum(c.fn for c in self.cases)
        )

    @property
    def micro_f1(self) -> float:
        return calculate_f1(self.micro_precision, self.micro_recall)

    @property
    def macro_f1(self) -> float:
        """Arithmetic mean of per-case F1 scores."""
        return sum(c.f1 for c in self.cases) / len(self.cases) if self.cases else 0.0


def format_summary(evaluation: FilterEvaluation, report_path: Path) -> str:
    """Format the aggregate metrics and per-case breakdown as an aligned table."""
    header = (
        f"{'Case ID':<14} {'Precision':<10} {'Recall':<10} {'F1':<10} "
        f"{'TP/Rel':<8} {'FP/Irrel':<8}"
    )
    rows = [
        f"{c.case_id:<14} {c.precision:<10.4f} {c.recall:<10.4f} {c.f1:<10.4f} "
        f"{f'{c.tp}/{c.tp + c.fn}':<8} {f'{c.fp}/{c.fp + c.tn}':<8}"
        for c in evaluation.cases
    ]
    return "\n".join(
        [
            f"RELEVANCE FILTER EVALUATION: {evaluation.filter_name}",
            f"cases={len(evaluation.cases)} "
            f"micro-P={evaluation.micro_precision:.4f} "
            f"micro-R={evaluation.micro_recall:.4f} "
            f"micro-F1={evaluation.micro_f1:.4f} "
            f"macro-F1={evaluation.macro_f1:.4f}",
            "",
            header,
            "-" * len(header),
            *rows,
            "",
            f"Report: {report_path}",
        ]
    )


def evaluate_retrieval(
    kept_content: str, relevant: list[str], irrelevant: list[str]
) -> tuple[int, int, int, int]:
    """Count (tp, fp, fn, tn) of expected passages found in the kept content."""
    tp = sum(1 for text in relevant if text in kept_content)
    fp = sum(1 for text in irrelevant if text in kept_content)
    return tp, fp, len(relevant) - tp, len(irrelevant) - fp


def line_span(content: str, start: int, end: int) -> tuple[int, int]:
    """1-based first and last line of ``content[start:end]``."""
    last = max(end, start + 1) - 1
    return content.count("\n", 0, start) + 1, content.count("\n", 0, last) + 1


def _locate(content: str, fragment: str, cursor: int) -> tuple[int, int] | None:
    """Find a fragment in the page at or after ``cursor``.

    A fragment carries its section heading as a prefix and may be a hard-split
    piece of a line, so fall back to the body and then to its first line.
    """
    body = fragment.split("\n\n", 1)[-1]
    first_line = body.splitlines()[0] if body else ""
    for needle in (fragment, body, first_line):
        start = content.find(needle, cursor) if needle else -1
        if start >= 0:
            return start, start + len(needle)
    return None


def fragment_rows(
    content: str, fragments: list[str], selected: list[int]
) -> list[dict[str, object]]:
    """Describe selected fragments in document order with page line ranges."""
    kept = {index for index in selected if 0 <= index < len(fragments)}
    rows: list[dict[str, object]] = []
    cursor = 0
    for index, fragment in enumerate(fragments):
        found = _locate(content, fragment, cursor)
        if found:
            cursor = found[0] + 1
        if index not in kept:
            continue
        start_line, end_line = line_span(content, *found) if found else (None, None)
        rows.append(
            {
                "index": index,
                "start_line": start_line,
                "end_line": end_line,
                "preview": fragment[:PREVIEW_CHARS],
            }
        )
    return rows


def passage_row(content: str, text: str) -> dict[str, object]:
    """Describe an expected passage by page line range and first characters."""
    start = content.find(text)
    start_line, end_line = (
        line_span(content, start, start + len(text)) if start >= 0 else (None, None)
    )
    return {
        "start_line": start_line,
        "end_line": end_line,
        "preview": text[:PREVIEW_CHARS],
    }


async def generate_fetch_query(
    case: FetchCase,
    cases: Iterable[FetchCase],
    config: ServerConfig,
    http_client: httpx.AsyncClient,
    instructions: str = FETCH_QUERY_INSTRUCTIONS,
) -> str:
    """Generate one fetch query for a case via the shared production fragment."""
    inp = fetch_query_input(case, cases)
    # Round 0/unknown uses the fast model; later rounds use the infer model.
    model_name = (
        config.research.fast_model
        if case.research_round in (None, 0)
        else config.research.infer_model
    )
    model = router_chat_model(config, model_name, http_client)
    parsed = cast(
        FetchQuery,
        await model.with_structured_output(FetchQuery).ainvoke(
            [
                SystemMessage(instructions),
                HumanMessage(fetch_query_message(inp)),
            ]
        ),
    )
    return parsed.fetch_query.strip() or inp.question


async def evaluate_case(
    case: FetchCase,
    expected: dict,
    query: str,
    setup: FilterSetup,
) -> tuple[CaseMetrics, dict[str, object]]:
    """Filter one page and build its metrics and JSONL report row."""
    content = (PAGES_DIR / expected["file"]).read_text(encoding="utf-8")
    page = FetchResult(
        url=case.url, status=FetchStatus.OK, mime=expected["mime"], content=content
    )
    result = await setup.page_filter(page, query)
    if result.status is not FetchStatus.OK:
        raise RuntimeError(f"{case.case_id}: filter returned {result.status}")
    kept = result.content
    tp, fp, fn, tn = evaluate_retrieval(
        kept, expected["relevant"], expected["irrelevant"]
    )
    metrics = CaseMetrics(case_id=case.case_id, tp=tp, fp=fp, fn=fn, tn=tn)
    recorder = setup.recorder
    row = {
        "case_id": case.case_id,
        "url": case.url,
        "query": query,
        "page_chars": len(content),
        "kept_chars": len(kept),
        "precision": round(metrics.precision, 4),
        "recall": round(metrics.recall, 4),
        "f1": round(metrics.f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "selected": (
            fragment_rows(content, recorder.fragments, recorder.selected)
            if recorder
            else None
        ),
        "missed": [
            passage_row(content, t) for t in expected["relevant"] if t not in kept
        ],
        "false_positives": [
            passage_row(content, t) for t in expected["irrelevant"] if t in kept
        ],
    }
    return metrics, row


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filter", choices=sorted(FILTERS), default="bm25")
    parser.add_argument(
        "--limit", type=int, help="number of dataset records (default: all)"
    )
    parser.add_argument(
        "--query-source",
        choices=QUERY_SOURCES,
        default="dataset",
        help="use the recorded query or generate one with the fetch-query fragment",
    )
    parser.add_argument(
        "--fetch-query-prompt",
        type=Path,
        help="file replacing FETCH_QUERY_INSTRUCTIONS; "
        "requires --query-source generated",
    )
    chunked = parser.add_argument_group(
        "chunked filter tuning", "--filter bm25 or decision"
    )
    chunked.add_argument(
        "--max-chars",
        type=int,
        help=f"chunker fragment size (default bm25 {BM25_WINDOW[0]}, "
        f"decision {DECISION_WINDOW[0]})",
    )
    chunked.add_argument(
        "--overlap-chars",
        type=int,
        help=f"chunker fragment overlap (default bm25 {BM25_WINDOW[1]}, "
        f"decision {DECISION_WINDOW[1]})",
    )
    chunked.add_argument(
        "--threshold",
        type=float,
        help=f"BM25 score cutoff (default {BM25_THRESHOLD}) "
        "or decision probability (default 0.6)",
    )
    bm25 = parser.add_argument_group("bm25 tuning", "--filter bm25")
    bm25.add_argument(
        "--relative-cutoff",
        type=float,
        help="keep fragments scoring at least this fraction of the page best "
        f"(0-1, default {BM25_RELATIVE_CUTOFF})",
    )
    bm25.add_argument(
        "--smoothing",
        type=float,
        help="weight of each neighbour score blended in "
        f"(0-0.5, default {BM25_SMOOTHING})",
    )
    bm25.add_argument(
        "--link-penalty",
        type=float,
        help="score reduction per link/DOI/year density (default 0)",
    )
    decision = parser.add_argument_group(
        "decision prompts", "override RELEVANCE_QUESTION fields; --filter decision"
    )
    decision.add_argument("--decision-instructions", help="question instructions")
    decision.add_argument("--decision-true", help="criteria for a yes answer")
    decision.add_argument("--decision-false", help="criteria for a no answer")
    args = parser.parse_args(argv)

    prompt: Path | None = args.fetch_query_prompt
    if prompt is not None and args.query_source != "generated":
        parser.error("--fetch-query-prompt requires --query-source generated")
    if prompt is not None and not prompt.is_file():
        parser.error(f"fetch query prompt file not found: {prompt}")
    unsupported = [
        f"--{name.replace('_', '-')}"
        for name in TUNING_OPTIONS
        if name not in FILTER_OPTIONS[args.filter] and getattr(args, name) is not None
    ]
    if unsupported:
        parser.error(
            f"{', '.join(unsupported)} not supported by --filter {args.filter}"
        )
    return args


def ensure_nltk_data() -> None:
    for resource in ("corpora/stopwords", "tokenizers/punkt_tab"):
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource.rsplit("/", 1)[-1], raise_on_error=True)


def load_evaluation_cases() -> dict[str, FetchCase]:
    """Dataset cases by id; smoke rows override baseline rows sharing an id."""
    return {
        case.case_id: case
        for dataset in ("fetch-cases-baseline.jsonl", "fetch-cases-smoke.jsonl")
        for case in load_cases(CASES_DIR / dataset)
    }


async def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    instructions = (
        args.fetch_query_prompt.read_text(encoding="utf-8")
        if args.fetch_query_prompt is not None
        else FETCH_QUERY_INSTRUCTIONS
    )

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"relevance-{args.filter}-{args.query_source}-{stamp}"
    report_path = OUTPUT_DIR / f"{stem}.jsonl"
    _configure_logging(OUTPUT_DIR / f"{stem}.log")
    ensure_nltk_data()

    load_dotenv(find_dotenv(usecwd=True))
    config = create_config()
    expectations: dict[str, dict] = json.loads(
        (PAGES_DIR / "expectations.json").read_text(encoding="utf-8")
    )
    cases = load_evaluation_cases()
    case_ids = sorted(expectations)[: args.limit]
    logger.info(
        "Evaluating filter=%s query_source=%s on %d cases",
        args.filter,
        args.query_source,
        len(case_ids),
    )

    evaluation = FilterEvaluation(filter_name=f"{args.filter}/{args.query_source}")
    async with httpx.AsyncClient(timeout=120.0) as http_client:
        setup = FILTERS[args.filter](args, config, http_client)
        settings = {
            **setup.settings,
            "fetch_query_prompt": (
                str(args.fetch_query_prompt) if args.fetch_query_prompt else None
            ),
        }
        logger.info("Settings: %s", json.dumps(settings))
        with report_path.open("w", encoding="utf-8") as report:
            for number, case_id in enumerate(case_ids, start=1):
                case = cases[case_id]
                query = (
                    case.query or ""
                    if args.query_source == "dataset"
                    else await generate_fetch_query(
                        case, cases.values(), config, http_client, instructions
                    )
                )
                metrics, row = await evaluate_case(
                    case, expectations[case_id], query, setup
                )
                evaluation.cases.append(metrics)
                row |= {
                    "filter": args.filter,
                    "query_source": args.query_source,
                    "settings": settings,
                }
                report.write(json.dumps(row) + "\n")
                report.flush()
                logger.info(
                    "[%d/%d] %s kept %d/%d chars | P=%.3f R=%.3f F1=%.3f "
                    "(TP=%d, FP=%d, FN=%d, TN=%d)",
                    number,
                    len(case_ids),
                    case_id,
                    row["kept_chars"],
                    row["page_chars"],
                    metrics.precision,
                    metrics.recall,
                    metrics.f1,
                    metrics.tp,
                    metrics.fp,
                    metrics.fn,
                    metrics.tn,
                )

    summary = format_summary(evaluation, report_path)
    logger.info("\n%s", summary)
    print(summary)


if __name__ == "__main__":
    asyncio.run(main())
