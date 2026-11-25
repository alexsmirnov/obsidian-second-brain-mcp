#!/usr/bin/env python3
"""Evaluate the deep-research ``select_urls`` node against labelled results.

Run from the repository root: ``uv run tests/web_select_urls_evaluation.py --limit 1``.

The dataset ``tests/evaluation/data/select-urls-cases.json`` stores, per case, the
original question, the optional knowledge gap, the web queries and their results
(url, title, snippet). Results that a good triage should pick carry
``"expected": true``. The script calls the real ``ResearchAgent.select_urls`` with
the real fast model (``RESEARCH_FAST_MODEL``, or ``--model``) and compares the
URLs it sends to ``web_fetch`` with the expected ones.

Metrics per run: precision, recall against the page budget (the number of
expected URLs is capped by the budget), F1, plus flags for an empty selection
and for exceeding the budget. ``--repeat`` runs every case several times to show
model variance.

Outputs go to ``tmp/``: ``select-urls-<timestamp>.log`` and a ``.jsonl`` report
with one row per run.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import httpx
from dotenv import find_dotenv, load_dotenv
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, Send
from web_fetch_evaluation import CASES_DIR

from mcps.common.ai_models import create_chat_model
from mcps.config import ResearchLimits, ServerConfig, create_config
from mcps.research.deep_research import (
    OverallState,
    ResearchAgent,
    SearchRun,
    _normalize_url,
)
from mcps.research.research_config import ResearchConfig
from mcps.research.tools.models import SearchResult

DATASET_PATH = CASES_DIR / "select-urls-cases.json"
OUTPUT_DIR = Path(__file__).parent.parent / "tmp"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LabelledResult:
    """One search result with the expectation that triage selects it."""

    url: str
    title: str
    snippet: str
    expected: bool
    reason: str = ""


@dataclass(frozen=True)
class LabelledRun:
    """One web query and its labelled results."""

    query: str
    results: list[LabelledResult]


@dataclass(frozen=True)
class SelectCase:
    """One ``select_urls`` input with its expected selection (normalized URLs)."""

    case_id: str
    question: str
    knowledge_gap: str
    pending_links: list[str]
    searches: list[LabelledRun]
    expected: set[str]
    candidates: set[str]


@dataclass(frozen=True)
class CaseMetrics:
    """Score of one selection."""

    tp: int
    fp: int
    fn: int
    recall_base: int
    precision: float
    recall: float
    f1: float
    over_budget: bool
    empty: bool


@dataclass
class Evaluation:
    """All scored runs of one invocation."""

    runs: list[CaseMetrics] = field(default_factory=list)


def _parse_case(raw: dict[str, Any]) -> SelectCase:
    searches = [
        LabelledRun(
            query=run["query"],
            results=[
                LabelledResult(
                    url=item["url"],
                    title=item["title"],
                    snippet=item["snippet"],
                    expected=bool(item.get("expected", False)),
                    reason=item.get("reason", ""),
                )
                for item in run["results"]
            ],
        )
        for run in raw["searches"]
    ]
    results = [result for run in searches for result in run.results]
    return SelectCase(
        case_id=raw["case_id"],
        question=raw["question"],
        knowledge_gap=raw.get("knowledge_gap", ""),
        pending_links=list(raw.get("pending_links", [])),
        searches=searches,
        expected={_normalize_url(r.url) for r in results if r.expected},
        candidates={_normalize_url(r.url) for r in results},
    )


def load_cases(path: Path) -> list[SelectCase]:
    """Load the labelled dataset."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [_parse_case(raw) for raw in data["cases"]]


def build_state(case: SelectCase) -> dict[str, Any]:
    """The graph state ``select_urls`` sees after the case's search round."""
    searches = [
        SearchRun(
            query=run.query,
            results=[
                SearchResult(url=r.url, title=r.title, snippet=r.snippet)
                for r in run.results
            ],
        )
        for run in case.searches
    ]
    return {
        "original_question": case.question,
        "knowledge_gap": case.knowledge_gap,
        "pending_links": case.pending_links,
        "fetch_query": "",
        "searches": searches,
    }


def selected_urls(command: Command) -> list[str]:
    """URLs ``select_urls`` sent to ``web_fetch``; empty when it went to reflection."""
    goto = command.goto
    if not isinstance(goto, list):
        return []
    return [
        send.arg["url"]
        for send in goto
        if isinstance(send, Send) and send.node == "web_fetch"
    ]


def score_selection(
    case: SelectCase, selected: list[str], budget: int
) -> CaseMetrics:
    """Precision, recall against the budget and F1 of one selection."""
    chosen = set(dict.fromkeys(_normalize_url(url) for url in selected))
    tp = len(chosen & case.expected)
    fp = len(chosen - case.expected)
    fn = len(case.expected - chosen)
    recall_base = min(len(case.expected), budget)
    precision = tp / len(chosen) if chosen else 0.0
    recall = tp / recall_base if recall_base else 1.0
    f1 = (
        2 * precision * recall / (precision + recall) if precision + recall else 0.0
    )
    return CaseMetrics(
        tp=tp,
        fp=fp,
        fn=fn,
        recall_base=recall_base,
        precision=precision,
        recall=recall,
        f1=f1,
        over_budget=len(chosen) > budget,
        empty=not chosen,
    )


async def _unused(*_args: object, **_kwargs: object) -> Any:
    raise AssertionError("select_urls must not search or fetch")


def create_agent(
    server_config: ServerConfig,
    model_name: str,
    max_pages: int,
    http_client: httpx.AsyncClient,
) -> ResearchAgent:
    """Agent whose fast model is real; search and fetch must never be called."""
    if not (
        model_name
        and server_config.router.api_key
        and bool(server_config.router.api_base)
    ):
        raise RuntimeError("RESEARCH_FAST_MODEL or router settings not set")
    model = create_chat_model(
        model_name,
        server_config.router,
        http_client,
    )
    research_config = ResearchConfig(
        fast=model,
        small=model,  # not used by select_urls
        search=cast(Any, _unused),
        fetch=cast(Any, _unused),
    )
    return ResearchAgent(
        research_config, limits=ResearchLimits(max_pages_per_round=max_pages)
    )


def _row(
    case: SelectCase,
    run: int,
    selected: list[str],
    metrics: CaseMetrics,
    settings: dict[str, object],
) -> dict[str, object]:
    chosen = {_normalize_url(url) for url in selected}
    titles = {
        _normalize_url(r.url): r.title for run_ in case.searches for r in run_.results
    }
    return {
        "case_id": case.case_id,
        "run": run,
        "selected": [
            {"url": url, "expected": url in case.expected} for url in sorted(chosen)
        ],
        "missed": sorted(case.expected - chosen),
        "false_positives": [
            {"url": url, "title": titles.get(url, "")}
            for url in sorted(chosen - case.expected)
        ],
        "metrics": {
            "tp": metrics.tp,
            "fp": metrics.fp,
            "fn": metrics.fn,
            "precision": round(metrics.precision, 3),
            "recall": round(metrics.recall, 3),
            "f1": round(metrics.f1, 3),
            "over_budget": metrics.over_budget,
            "empty": metrics.empty,
        },
        "settings": settings,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def format_summary(evaluation: Evaluation, report_path: Path) -> str:
    """Macro and micro scores over all runs."""
    runs = evaluation.runs
    tp = sum(m.tp for m in runs)
    fp = sum(m.fp for m in runs)
    base = sum(m.recall_base for m in runs)
    micro_p = tp / (tp + fp) if tp + fp else 0.0
    micro_r = tp / base if base else 1.0
    micro_f1 = (
        2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0
    )
    f1s = [m.f1 for m in runs]
    return "\n".join(
        [
            f"select_urls evaluation: {len(runs)} runs",
            f"macro  P={_mean([m.precision for m in runs]):.3f} "
            f"R@budget={_mean([m.recall for m in runs]):.3f} F1={_mean(f1s):.3f} "
            f"(F1 min={min(f1s, default=0.0):.3f} max={max(f1s, default=0.0):.3f})",
            f"micro  P={micro_p:.3f} R@budget={micro_r:.3f} F1={micro_f1:.3f}",
            f"empty selections: {sum(m.empty for m in runs)}, "
            f"over budget: {sum(m.over_budget for m in runs)}",
            f"report: {report_path}",
        ]
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--limit", type=int, help="number of cases (default: all)")
    parser.add_argument(
        "--repeat", type=int, default=1, help="runs per case to show model variance"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=ResearchLimits().max_pages_per_round,
        help="page budget of the round",
    )
    parser.add_argument(
        "--model", help="fast model name (default: RESEARCH_FAST_MODEL)"
    )
    return parser.parse_args(argv)


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


async def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = OUTPUT_DIR / f"select-urls-{stamp}.jsonl"
    _configure_logging(OUTPUT_DIR / f"select-urls-{stamp}.log")

    load_dotenv(find_dotenv(usecwd=True))
    server_config = create_config()
    model_name = args.model or server_config.research.fast_model
    cases = load_cases(args.dataset)[: args.limit]
    settings: dict[str, object] = {
        "model": model_name,
        "max_pages": args.max_pages,
        "repeat": args.repeat,
    }
    logger.info("Evaluating %d cases, settings: %s", len(cases), json.dumps(settings))

    evaluation = Evaluation()
    async with httpx.AsyncClient(timeout=120.0) as http_client:
        agent = create_agent(server_config, model_name, args.max_pages, http_client)
        with report_path.open("w", encoding="utf-8") as report:
            for number, case in enumerate(cases, start=1):
                budget = args.max_pages - len(case.pending_links)
                for run in range(1, args.repeat + 1):
                    command = await agent.select_urls(
                        cast(OverallState, build_state(case)), RunnableConfig()
                    )
                    selected = selected_urls(command)
                    metrics = score_selection(case, selected, budget)
                    evaluation.runs.append(metrics)
                    report.write(
                        json.dumps(_row(case, run, selected, metrics, settings)) + "\n"
                    )
                    report.flush()
                    logger.info(
                        "[%d/%d] %s run %d: selected %d | P=%.3f R=%.3f F1=%.3f "
                        "(TP=%d, FP=%d, FN=%d)",
                        number,
                        len(cases),
                        case.case_id,
                        run,
                        len(selected),
                        metrics.precision,
                        metrics.recall,
                        metrics.f1,
                        metrics.tp,
                        metrics.fp,
                        metrics.fn,
                    )

    summary = format_summary(evaluation, report_path)
    logger.info("\n%s", summary)
    print(summary)


if __name__ == "__main__":
    asyncio.run(main())
