"""DRACO dataset loader for evaluation."""

from __future__ import annotations

import itertools
import json
import logging
from dataclasses import dataclass

from datasets import load_dataset

logger = logging.getLogger(__name__)


@dataclass
class RubricCriterion:
    """Single evaluation criterion with weight."""

    id: str
    weight: int
    requirement: str


@dataclass
class RubricSection:
    """Evaluation axis containing multiple criteria."""

    id: str
    title: str
    criteria: list[RubricCriterion]


@dataclass
class DracoQuestion:
    """DRACO benchmark task with rubric."""

    id: str
    problem: str
    rubric: list[RubricSection]
    domain: str


def _parse_rubric(answer_json: str) -> list[RubricSection]:
    """Parse rubric JSON string into structured sections."""
    data = json.loads(answer_json)
    sections: list[RubricSection] = []
    for section in data["sections"]:
        criteria = [
            RubricCriterion(
                id=c["id"],
                weight=c["weight"],
                requirement=c["requirement"],
            )
            for c in section["criteria"]
        ]
        sections.append(
            RubricSection(
                id=section["id"],
                title=section["title"],
                criteria=criteria,
            )
        )
    return sections


def load_draco_questions(
    max_questions: int | None = None,
    domains: tuple[str, ...] = ("Technology", "Academic"),
) -> list[DracoQuestion]:
    """Load DRACO evaluation tasks filtered by domain.

    Args:
        max_questions: Maximum number of questions to load (None = all).
        domains: Domain filter tuple.

    Returns:
        List of DracoQuestion objects.
    """
    dataset = load_dataset("perplexity-ai/draco", split="test")

    filtered = (
        example
        for example in dataset
        if example["domain"] in domains
    )

    if max_questions is not None:
        filtered = itertools.islice(filtered, max_questions)

    questions = [
        DracoQuestion(
            id=example["id"],
            problem=example["problem"],
            rubric=_parse_rubric(example["answer"]),
            domain=example["domain"],
        )
        for example in filtered
    ]

    logger.info(
        "Loaded %d DRACO questions (domains: %s)",
        len(questions),
        ", ".join(domains),
    )
    return questions
