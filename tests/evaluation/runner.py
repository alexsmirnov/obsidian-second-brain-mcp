"""Evaluation runner for DRACO benchmark tasks."""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from mcps.research.agent import ResearchResponse

from .dataset import DracoQuestion
from .scoring import CriterionVerdict

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Single task evaluation result."""

    question_id: str
    problem: str
    domain: str
    prediction: str
    score: float
    latency: float
    verdicts: list[CriterionVerdict] = field(default_factory=list)
    explanation: str = ""
    error: str | None = None
    token_usage: int = 0


@dataclass
class EvaluationSummary:
    """Aggregated evaluation metrics."""

    total: int
    avg_score: float
    execution_errors: int
    avg_latency: float
    total_tokens: int = 0
    domain_scores: dict[str, float] = field(default_factory=dict)


async def evaluate_single(
    question: DracoQuestion,
    agent: Callable[[str], Awaitable[ResearchResponse]],
) -> EvaluationResult:
    """Run agent on one DRACO task (scoring done separately in pipeline)."""
    start = time.monotonic()
    prediction = ""
    explanation = ""
    error = None

    try:
        response = await agent(question.problem)
        prediction = response["answer"]
        explanation = response["explanation"]
    except Exception as exc:
        error = str(exc)
        logger.warning("Agent error on %r: %s", question.id, error)

    latency = time.monotonic() - start

    return EvaluationResult(
        question_id=question.id,
        problem=question.problem,
        domain=question.domain,
        prediction=prediction,
        explanation=explanation,
        score=0.0,
        latency=latency,
        error=error,
    )


def calculate_summary(
    results: list[EvaluationResult],
) -> EvaluationSummary:
    """Compute aggregate metrics from evaluation results."""
    if not results:
        return EvaluationSummary(
            total=0,
            avg_score=0.0,
            execution_errors=0,
            avg_latency=0.0,
        )

    execution_errors = sum(1 for r in results if r.error)
    total_latency = sum(r.latency for r in results)
    total_tokens = sum(r.token_usage for r in results)
    avg_score = sum(r.score for r in results) / len(results)

    # Per-domain breakdown
    domain_scores: dict[str, list[float]] = {}
    for r in results:
        domain_scores.setdefault(r.domain, []).append(r.score)

    return EvaluationSummary(
        total=len(results),
        avg_score=avg_score,
        execution_errors=execution_errors,
        avg_latency=total_latency / len(results),
        total_tokens=total_tokens,
        domain_scores={
            domain: sum(scores) / len(scores)
            for domain, scores in domain_scores.items()
        },
    )
