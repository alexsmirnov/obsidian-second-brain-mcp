"""Tests for DRACO evaluation summary aggregation."""

from __future__ import annotations

from evaluation.runner import EvaluationResult, calculate_summary


def _result(
    *,
    score: float,
    domain: str = "Technology",
    error: str | None = None,
) -> EvaluationResult:
    return EvaluationResult(
        question_id="test-id",
        problem="Test problem",
        domain=domain,
        prediction="Test prediction",
        score=score,
        latency=1.0,
        verdicts=[],
        error=error,
    )


def test_calculate_summary_computes_avg_score() -> None:
    results = [
        _result(score=0.8),
        _result(score=0.6),
        _result(score=0.4),
    ]

    summary = calculate_summary(results)

    assert summary.total == 3
    assert abs(summary.avg_score - 0.6) < 1e-9
    assert summary.execution_errors == 0


def test_calculate_summary_tracks_execution_errors() -> None:
    results = [
        _result(score=0.9),
        _result(score=0.0, error="timeout"),
        _result(score=0.5),
    ]

    summary = calculate_summary(results)

    assert summary.total == 3
    assert summary.execution_errors == 1
    assert abs(summary.avg_score - (0.9 + 0.0 + 0.5) / 3) < 1e-9


def test_calculate_summary_domain_breakdown() -> None:
    results = [
        _result(score=0.8, domain="Technology"),
        _result(score=0.6, domain="Technology"),
        _result(score=0.9, domain="Academic"),
    ]

    summary = calculate_summary(results)

    assert abs(summary.domain_scores["Technology"] - 0.7) < 1e-9
    assert abs(summary.domain_scores["Academic"] - 0.9) < 1e-9


def test_calculate_summary_empty_results() -> None:
    summary = calculate_summary([])

    assert summary.total == 0
    assert summary.avg_score == 0.0
    assert summary.execution_errors == 0
