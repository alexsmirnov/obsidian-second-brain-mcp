"""Contract tests for the pure helpers of the select_urls evaluation script."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langgraph.types import Command, Send
from web_select_urls_evaluation import (
    SelectCase,
    build_state,
    load_cases,
    score_selection,
    selected_urls,
)


def _result(url: str, *, expected: bool = False) -> dict[str, object]:
    return {"url": url, "title": f"t {url}", "snippet": "s", "expected": expected}


def _case_json(**overrides: object) -> dict[str, object]:
    raw: dict[str, object] = {
        "case_id": "c1",
        "question": "What is DiD?",
        "knowledge_gap": "",
        "pending_links": [],
        "searches": [
            {
                "query": "q1",
                "results": [
                    _result("https://a.com/x", expected=True),
                    _result("https://b.com/y"),
                ],
            },
            {
                "query": "q2",
                "results": [
                    _result("https://a.com/x/", expected=True),
                    _result("https://c.com/z", expected=True),
                ],
            },
        ],
    }
    return raw | overrides


def _write(tmp_path: Path, *cases: dict[str, object]) -> Path:
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"cases": list(cases)}), encoding="utf-8")
    return path


def test_load_cases_parses_questions_gap_and_results(tmp_path: Path) -> None:
    path = _write(tmp_path, _case_json(knowledge_gap="gap"))

    [case] = load_cases(path)

    assert case.case_id == "c1"
    assert case.question == "What is DiD?"
    assert case.knowledge_gap == "gap"
    assert [run.query for run in case.searches] == ["q1", "q2"]


def test_expected_urls_are_normalized_and_merged_across_queries(tmp_path: Path) -> None:
    [case] = load_cases(_write(tmp_path, _case_json()))

    assert case.expected == {"https://a.com/x", "https://c.com/z"}


def test_candidate_urls_are_deduplicated_across_queries(tmp_path: Path) -> None:
    [case] = load_cases(_write(tmp_path, _case_json()))

    assert case.candidates == {"https://a.com/x", "https://b.com/y", "https://c.com/z"}


def test_build_state_matches_select_urls_input(tmp_path: Path) -> None:
    raw = _case_json(knowledge_gap="gap", pending_links=["https://p.com"])
    [case] = load_cases(_write(tmp_path, raw))

    state = build_state(case)

    assert state["original_question"] == "What is DiD?"
    assert state["knowledge_gap"] == "gap"
    assert state["pending_links"] == ["https://p.com"]
    assert [run["query"] for run in state["searches"]] == ["q1", "q2"]
    assert state["searches"][0]["results"][0].url == "https://a.com/x"


def _fetch(url: str) -> dict[str, str]:
    return {"url": url, "title": "", "fetch_query": "q"}


def test_selected_urls_reads_web_fetch_sends() -> None:
    command = Command(
        goto=[
            Send("web_fetch", _fetch("https://a.com/x")),
            Send("web_fetch", _fetch("https://b.com/y")),
        ]
    )

    assert selected_urls(command) == ["https://a.com/x", "https://b.com/y"]


def test_selected_urls_is_empty_when_round_goes_to_reflection() -> None:
    assert selected_urls(Command(goto="reflection")) == []


def _case(expected: set[str], candidates: set[str] | None = None) -> SelectCase:
    return SelectCase(
        case_id="c",
        question="q",
        knowledge_gap="",
        pending_links=[],
        searches=[],
        expected=expected,
        candidates=candidates or expected,
    )


def test_perfect_selection_scores_one() -> None:
    metrics = score_selection(_case({"u1", "u2"}), ["u1", "u2"], budget=10)

    assert (metrics.tp, metrics.fp, metrics.fn) == (2, 0, 0)
    assert metrics.precision == metrics.recall == metrics.f1 == 1.0


def test_false_positive_and_miss_are_counted() -> None:
    case = _case({"u1", "u2"}, {"u1", "u2", "u3"})

    metrics = score_selection(case, ["u1", "u3"], budget=10)

    assert (metrics.tp, metrics.fp, metrics.fn) == (1, 1, 1)
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5


def test_selection_is_compared_by_normalized_url() -> None:
    case = _case({"https://a.com/x"})

    metrics = score_selection(case, ["HTTPS://A.com/x/#frag"], budget=10)

    assert metrics.tp == 1


def test_recall_is_capped_by_budget() -> None:
    expected = {f"u{i}" for i in range(6)}

    metrics = score_selection(_case(expected), ["u0", "u1", "u2"], budget=3)

    assert metrics.recall == 1.0
    assert metrics.fn == 3


def test_empty_selection_is_flagged_with_zero_scores() -> None:
    metrics = score_selection(_case({"u1"}), [], budget=10)

    assert metrics.empty
    assert metrics.precision == metrics.recall == metrics.f1 == 0.0


def test_selection_over_budget_is_flagged() -> None:
    case = _case({"u1"}, {"u1", "u2", "u3"})

    metrics = score_selection(case, ["u1", "u2", "u3"], budget=2)

    assert metrics.over_budget


@pytest.mark.parametrize("selected", [["x"], []])
def test_case_without_expected_urls_has_full_recall(selected: list[str]) -> None:
    metrics = score_selection(_case(set(), {"x"}), selected, budget=10)

    assert metrics.recall == 1.0
