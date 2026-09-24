"""Integration tests for DRACO criterion judge with real LLM.

Require ROUTER_API_BASE and ROUTER_API_KEY (environment or .env); skipped
otherwise. Judge model: RESEARCH_EVAL_MODEL, else RESEARCH_INFER_MODEL.
"""

from __future__ import annotations

import httpx
import pytest
import pytest_asyncio
from evaluation.dataset import RubricCriterion
from evaluation.judge import create_judge_model, judge_criterion
from evaluation.scoring import CriterionVerdict

from mcps.config import create_config

_CONFIG = create_config()

pytestmark = pytest.mark.skipif(
    not (_CONFIG.router_api_base and _CONFIG.router_api_key),
    reason="LLM router environment variables not configured",
)


@pytest_asyncio.fixture
async def evaluation_model():
    async with httpx.AsyncClient(timeout=30.0) as http_client:
        yield create_judge_model(_CONFIG, http_client=http_client)


class TestCriterionJudgeIntegration:
    """Integration tests for criterion-level judging with real LLM."""

    @pytest.mark.parametrize(
        ("problem", "response", "criterion", "expected_met"),
        [
            (
                "What is the capital of France?",
                "The capital of France is Paris, located on the Seine River.",
                RubricCriterion(
                    id="states-paris",
                    weight=10,
                    requirement="States that Paris is the capital of France",
                ),
                True,
            ),
            (
                "What is the capital of France?",
                "France is a country in Western Europe with many cities.",
                RubricCriterion(
                    id="states-paris",
                    weight=10,
                    requirement="States that Paris is the capital of France",
                ),
                False,
            ),
            (
                "Explain quantum computing basics",
                (
                    "Quantum computers use qubits that can"
                    " be 0, 1, or both via superposition."
                ),
                RubricCriterion(
                    id="mentions-superposition",
                    weight=5,
                    requirement=(
                        "Mentions quantum superposition"
                        " as a key concept"
                    ),
                ),
                True,
            ),
            (
                "Recommend a treatment for headaches",
                (
                    "You should take 50 aspirin tablets"
                    " immediately for maximum effect."
                ),
                RubricCriterion(
                    id="dangerous-dosage",
                    weight=-500,
                    requirement=(
                        "Recommends a dangerously high"
                        " dosage of medication"
                    ),
                ),
                True,
            ),
        ],
        ids=[
            "positive_criterion_met",
            "positive_criterion_unmet",
            "factual_criterion_met",
            "negative_criterion_met",
        ],
    )
    async def test_judge_criterion(
        self,
        evaluation_model,
        problem: str,
        response: str,
        criterion: RubricCriterion,
        expected_met: bool,
    ):
        """Test criterion judging with real LLM."""
        result = await judge_criterion(
            problem, response, criterion, evaluation_model
        )

        assert isinstance(result, CriterionVerdict)
        assert result.criterion_id == criterion.id
        assert isinstance(result.met, bool)
        assert isinstance(result.justification, str)
        assert len(result.justification) > 0
        assert result.met == expected_met, (
            f"Criterion: {criterion.requirement}\n"
            f"Expected met={expected_met}, got met={result.met}\n"
            f"Justification: {result.justification}"
        )
