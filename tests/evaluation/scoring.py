"""DRACO rubric-based scoring."""

from __future__ import annotations

from dataclasses import dataclass

from .dataset import RubricSection


@dataclass
class CriterionVerdict:
    """Judge verdict for a single criterion."""

    criterion_id: str
    met: bool
    justification: str


def compute_draco_score(
    verdicts: list[CriterionVerdict],
    rubric: list[RubricSection],
) -> float:
    """Compute DRACO normalized score (0.0-1.0).

    Score formula per DRACO spec:
        raw_score = sum(v_i * w_i for all criteria)
        normalized = clamp(raw_score / sum(w_i for w_i > 0), 0, 1)
    """
    weight_lookup: dict[str, int] = {
        c.id: c.weight
        for section in rubric
        for c in section.criteria
    }

    raw_score = sum(
        weight_lookup[v.criterion_id]
        for v in verdicts
        if v.met and v.criterion_id in weight_lookup
    )

    positive_sum = sum(w for w in weight_lookup.values() if w > 0)
    if positive_sum == 0:
        return 0.0

    return max(0.0, min(1.0, raw_score / positive_sum))
