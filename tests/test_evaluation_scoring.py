"""Tests for DRACO rubric scoring logic."""

from __future__ import annotations

from evaluation.dataset import RubricCriterion, RubricSection
from evaluation.scoring import CriterionVerdict, compute_draco_score


def _section(criteria: list[RubricCriterion]) -> RubricSection:
    return RubricSection(id="test-section", title="Test", criteria=criteria)


def test_all_positive_criteria_met() -> None:
    rubric = [_section([
        RubricCriterion(id="a", weight=10, requirement="r1"),
        RubricCriterion(id="b", weight=20, requirement="r2"),
    ])]
    verdicts = [
        CriterionVerdict(criterion_id="a", met=True, justification="ok"),
        CriterionVerdict(criterion_id="b", met=True, justification="ok"),
    ]

    score = compute_draco_score(verdicts, rubric)

    assert score == 1.0


def test_partial_positive_criteria_met() -> None:
    rubric = [_section([
        RubricCriterion(id="a", weight=10, requirement="r1"),
        RubricCriterion(id="b", weight=10, requirement="r2"),
    ])]
    verdicts = [
        CriterionVerdict(criterion_id="a", met=True, justification="ok"),
        CriterionVerdict(criterion_id="b", met=False, justification="no"),
    ]

    score = compute_draco_score(verdicts, rubric)

    assert score == 0.5


def test_negative_weight_criteria_reduce_score() -> None:
    """MET on negative criterion reduces raw_score."""
    rubric = [_section([
        RubricCriterion(id="a", weight=10, requirement="r1"),
        RubricCriterion(
            id="bad", weight=-5, requirement="error present"
        ),
    ])]
    verdicts = [
        CriterionVerdict(
            criterion_id="a", met=True, justification="ok"
        ),
        CriterionVerdict(
            criterion_id="bad", met=True,
            justification="error found",
        ),
    ]

    score = compute_draco_score(verdicts, rubric)

    # raw_score = 10 + (-5) = 5, positive_sum = 10, normalized = 0.5
    assert score == 0.5


def test_negative_weight_unmet_does_not_penalize() -> None:
    """UNMET on negative criterion means no penalty."""
    rubric = [_section([
        RubricCriterion(id="a", weight=10, requirement="r1"),
        RubricCriterion(
            id="bad", weight=-5, requirement="error present"
        ),
    ])]
    verdicts = [
        CriterionVerdict(
            criterion_id="a", met=True, justification="ok"
        ),
        CriterionVerdict(
            criterion_id="bad", met=False,
            justification="no error",
        ),
    ]

    score = compute_draco_score(verdicts, rubric)

    # raw_score = 10, positive_sum = 10, normalized = 1.0
    assert score == 1.0


def test_score_clamped_to_zero_on_extreme_negative() -> None:
    """Score cannot go below 0 with large negative weights."""
    rubric = [_section([
        RubricCriterion(id="a", weight=10, requirement="r1"),
        RubricCriterion(
            id="bad", weight=-500, requirement="dangerous"
        ),
    ])]
    verdicts = [
        CriterionVerdict(
            criterion_id="a", met=True, justification="ok"
        ),
        CriterionVerdict(
            criterion_id="bad", met=True,
            justification="dangerous content",
        ),
    ]

    score = compute_draco_score(verdicts, rubric)

    # raw_score = 10 + (-500) = -490, clamped to 0
    assert score == 0.0


def test_empty_verdicts_returns_zero() -> None:
    rubric = [_section([
        RubricCriterion(id="a", weight=10, requirement="r1"),
    ])]

    score = compute_draco_score([], rubric)

    assert score == 0.0


def test_no_positive_weights_returns_zero() -> None:
    rubric = [_section([
        RubricCriterion(id="bad", weight=-10, requirement="error"),
    ])]
    verdicts = [
        CriterionVerdict(criterion_id="bad", met=True, justification="yes"),
    ]

    score = compute_draco_score(verdicts, rubric)

    assert score == 0.0


def test_multiple_sections() -> None:
    rubric = [
        RubricSection(
            id="factual",
            title="Factual Accuracy",
            criteria=[RubricCriterion(id="f1", weight=10, requirement="r1")],
        ),
        RubricSection(
            id="depth",
            title="Breadth and Depth",
            criteria=[RubricCriterion(id="d1", weight=5, requirement="r2")],
        ),
    ]
    verdicts = [
        CriterionVerdict(criterion_id="f1", met=True, justification="ok"),
        CriterionVerdict(criterion_id="d1", met=False, justification="no"),
    ]

    score = compute_draco_score(verdicts, rubric)

    # raw = 10, positive_sum = 15, normalized = 10/15 ≈ 0.6667
    assert abs(score - 10 / 15) < 1e-9
