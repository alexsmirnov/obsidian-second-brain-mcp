"""Evaluation pipeline orchestrator for DRACO benchmark."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from langchain_core.language_models import BaseChatModel

from mcps.research.agent import ResearchResponse

from .dataset import DracoQuestion
from .judge import judge_criterion
from .runner import (
    EvaluationResult,
    EvaluationSummary,
    calculate_summary,
    evaluate_single,
)
from .scoring import CriterionVerdict, compute_draco_score

logger = logging.getLogger(__name__)

JUDGE_CONCURRENCY = 10


async def _judge_all_criteria(
    problem: str,
    response_text: str,
    question: DracoQuestion,
    judge_model: BaseChatModel,
) -> list[CriterionVerdict]:
    """Judge all criteria for a task concurrently with semaphore."""
    semaphore = asyncio.Semaphore(JUDGE_CONCURRENCY)

    all_criteria = [
        criterion
        for section in question.rubric
        for criterion in section.criteria
    ]

    async def _judge_one(criterion):
        async with semaphore:
            return await judge_criterion(
                problem, response_text, criterion, judge_model
            )

    verdicts = await asyncio.gather(
        *(_judge_one(c) for c in all_criteria)
    )
    return list(verdicts)


async def run_evaluation(
    agent: Callable[[str], Awaitable[ResearchResponse]],
    judge_model: BaseChatModel,
    questions: list[DracoQuestion],
) -> tuple[EvaluationSummary, list[EvaluationResult]]:
    """Run full DRACO evaluation pipeline."""
    results: list[EvaluationResult] = []

    for i, q in enumerate(questions, 1):
        logger.info(
            "[%d/%d] Evaluating: %s (%s)",
            i,
            len(questions),
            q.id,
            q.domain,
        )

        result = await evaluate_single(q, agent)

        if not result.error:
            verdicts = await _judge_all_criteria(
                q.problem, f"{result.prediction}\nExplanation:{result.explanation}", q, judge_model
            )
            result.verdicts = verdicts
            result.score = compute_draco_score(verdicts, q.rubric)

        logger.info(
            "  Score: %.1f%% (%d criteria judged)",
            result.score * 100,
            len(result.verdicts),
        )
        results.append(result)

    summary = calculate_summary(results)
    return summary, results