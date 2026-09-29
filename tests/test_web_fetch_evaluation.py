"""Contract tests for the fetch evaluation replay helpers."""

import json
from pathlib import Path

import pytest
from web_fetch_evaluation import CASES_DIR, FetchCase, load_cases, summarize

BASELINE_OK = {
    "observations": [
        {
            "status": "success",
            "pre_truncation_chars": 20000,
            "inferred_returned_chars": 15021,
        }
    ]
}
BASELINE_BAD = {
    "observations": [
        {
            "status": "retrieval_error",
            "pre_truncation_chars": None,
            "inferred_returned_chars": None,
        }
    ]
}


@pytest.fixture
def cases_file(tmp_path: Path) -> Path:
    rows = [
        {
            "case_id": "ok",
            "url": "https://a.example",
            "query": "q",
            "baseline": BASELINE_OK,
        },
        {
            "case_id": "bad",
            "url": "https://b.example",
            "query": "q",
            "baseline": BASELINE_BAD,
        },
        {"url": "https://c.example", "query": None},
    ]
    path = tmp_path / "cases.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n\n")
    return path


def test_load_cases_reads_minimal_and_baseline_rows(cases_file: Path):
    cases = load_cases(cases_file)

    assert len(cases) == 3
    ok, bad, minimal = cases
    assert minimal.case_id == "line-3"
    assert minimal.query is None
    assert minimal.baseline_error is None
    assert ok.baseline_error is False
    assert ok.baseline_chars == 15021
    assert bad.baseline_error is True


def test_summarize_compares_matched_baseline(cases_file: Path):
    cases = load_cases(cases_file)
    rows: list[dict[str, object]] = [
        {"case_id": "ok", "error": None, "response_size": 1000},
        {"case_id": "bad", "error": None, "response_size": 500},
        {"case_id": "line-3", "error": "ERROR: x", "response_size": 0},
    ]

    assert summarize(cases, rows) == {
        "cases": 3,
        "errors": 1,
        "empty": 0,
        "baseline_errors": 1,
        "recovered": 1,
        "regressed": 0,
        "mean_size": 750.0,
        "baseline_mean_size": 15021.0,
    }


def test_shipped_case_files_load():
    for name in ("fetch-cases-smoke.jsonl", "fetch-cases-baseline.jsonl"):
        cases = load_cases(CASES_DIR / name)
        assert cases
        assert all(isinstance(c, FetchCase) and c.url for c in cases)
