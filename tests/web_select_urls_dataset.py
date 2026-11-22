#!/usr/bin/env python3
"""Build the unlabelled dataset for ``web_select_urls_evaluation.py``.

Run from the repository root: ``uv run tests/web_select_urls_dataset.py``.

For every case in ``CASE_SPECS`` (taken from DRACO evaluation logs) the script
loads the DRACO question and runs the logged web queries through the Yahoo
search tool. Results are stored with ``"expected": false``; label them by hand
(set ``expected`` and a short ``reason``) before running the evaluation.

An existing dataset file holds labels, so it is only replaced with ``--force``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import httpx
from evaluation.dataset import load_draco_questions
from web_select_urls_evaluation import DATASET_PATH

from mcps.research.tools.yahoo import create_yahoo_search

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CaseSpec:
    """A research round recorded in the evaluation logs."""

    case_id: str
    task_id: str
    research_round: int
    queries: tuple[str, ...]
    knowledge_gap: str = ""


DIDS_TASK = "0c2c668a-c3bf-41af-93c9-b5614ff63508"
JETSON_TASK = "e1f2c310-d311-49da-b0e4-ee855603469d"
QUANTUM_TASK = "79224558-75f7-4e51-bc24-51f1e3ede921"

CASE_SPECS = (
    CaseSpec(
        case_id="did-r1",
        task_id=DIDS_TASK,
        research_round=1,
        queries=(
            "Callaway Sant'Anna interaction-weighted estimator",
            "Sun Abraham staggered adoption",
            "Borusyak staggered adoption imputation",
        ),
    ),
    CaseSpec(
        case_id="did-r2",
        task_id=DIDS_TASK,
        research_round=2,
        queries=(
            "Callaway Sant'Anna vs Sun Abraham vs Borusyak comparison",
            "AER QJE JPE staggered DiD labor health",
            "Roth 2023 What's Trending in Difference-in-Differences pre-trend",
        ),
        knowledge_gap=(
            "Identify the dominance of Callaway-Sant'Anna, Sun-Abraham, and "
            "Borusyak-Jaravel-Spiess estimators in AER, QJE, and JPE (2020-2024) "
            "specifically for labor/health applications, including adoption rates "
            "and common justification methods (Monte Carlo vs. sensitivity "
            "analysis). Also, clarify Roth's (2023) specific concerns regarding "
            "pre-trend testing and how these estimators address them."
        ),
    ),
    CaseSpec(
        case_id="jetson-r1",
        task_id=JETSON_TASK,
        research_round=1,
        queries=(
            "YOLOv8 EfficientDet-D4 DetectNet_v2 Jetson Orin",
            "INT8 quantization mAP Jetson Orin",
            "TensorRT optimization Jetson AGX Orin",
        ),
    ),
    CaseSpec(
        case_id="quantum-r1",
        task_id=QUANTUM_TASK,
        research_round=1,
        queries=(
            "quantum computing drug discovery progress",
            "quantum computing cryptography impact",
            "quantum computing optimization applications",
        ),
    ),
)


async def build_cases(questions: dict[str, str]) -> list[dict[str, object]]:
    """Run every spec's queries through Yahoo and return unlabelled case rows."""
    async with httpx.AsyncClient(timeout=60.0) as http_client:
        search = create_yahoo_search(http_client=http_client)
        cases: list[dict[str, object]] = []
        for spec in CASE_SPECS:
            batches = await asyncio.gather(*(search(query) for query in spec.queries))
            searches = [
                {
                    "query": query,
                    "results": [
                        {**result.model_dump(), "expected": False, "reason": ""}
                        for result in results
                    ],
                }
                for query, results in zip(spec.queries, batches, strict=True)
            ]
            logger.info(
                "%s: %s", spec.case_id, [len(item["results"]) for item in searches]
            )
            cases.append(
                {
                    "case_id": spec.case_id,
                    "task_id": spec.task_id,
                    "research_round": spec.research_round,
                    "question": questions[spec.task_id],
                    "knowledge_gap": spec.knowledge_gap,
                    "pending_links": [],
                    "searches": searches,
                }
            )
        return cases


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DATASET_PATH)
    parser.add_argument(
        "--force", action="store_true", help="replace an existing (labelled) dataset"
    )
    args = parser.parse_args(argv)
    if args.output.exists() and not args.force:
        parser.error(f"{args.output} exists and may hold labels; use --force")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    task_ids = {spec.task_id for spec in CASE_SPECS}
    questions = {
        question.id: question.problem
        for question in load_draco_questions()
        if question.id in task_ids
    }
    missing = task_ids - questions.keys()
    if missing:
        parser.error(f"DRACO questions not found: {sorted(missing)}")

    cases = asyncio.run(build_cases(questions))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"cases": cases}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote %d cases to %s", len(cases), args.output)


if __name__ == "__main__":
    main()
