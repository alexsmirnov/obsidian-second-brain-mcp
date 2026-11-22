"""Contract tests for the fetch evaluation replay helpers."""

import argparse
import asyncio
import json
from pathlib import Path

import pytest
from web_fetch_evaluation import (
    CASES_DIR,
    FetchCase,
    _positive_int,
    load_cases,
    run_cases,
    save_baseline,
    summarize,
)

from mcps.research.tools.common import failure
from mcps.research.tools.models import FetchResult, FetchStatus

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


def test_load_cases_reads_task_and_round(tmp_path: Path):
    path = tmp_path / "c.jsonl"
    rows = [
        {
            "case_id": "a",
            "url": "https://a.example",
            "query": "q",
            "task_id": "t1",
            "research_round": 2,
        },
        {"url": "https://b.example"},
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    a, b = load_cases(path)

    assert (a.task_id, a.research_round) == ("t1", 2)
    assert (b.task_id, b.research_round) == (None, None)


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


@pytest.mark.parametrize(
    ("mime", "name"),
    [
        ("text/html", "baseline-3.html"),
        ("text/markdown", "baseline-3.md"),
        ("text/plain", "baseline-3.txt"),
    ],
)
def test_save_baseline_writes_raw_content_by_mime(
    tmp_path: Path, mime: str, name: str
):
    result = FetchResult(
        url="https://a.example", status=FetchStatus.OK, mime=mime, content="<p>x</p>"
    )

    path = save_baseline(tmp_path, "baseline-3", result)

    assert path is not None
    assert path == tmp_path / name
    assert path.read_text(encoding="utf-8") == "<p>x</p>"


@pytest.mark.parametrize(
    "result",
    [
        failure("https://a.example", FetchStatus.HTTP_ERROR, 404),
        FetchResult(
            url="https://a.example",
            status=FetchStatus.OK,
            mime="text/html",
            content="  ",
        ),
    ],
)
def test_save_baseline_skips_errors_and_blank_content(
    tmp_path: Path, result: FetchResult
):
    assert save_baseline(tmp_path, "baseline-1", result) is None
    assert not any(tmp_path.iterdir())


# ---------------------------------------------------------------------------
# Parallel batches
# ---------------------------------------------------------------------------


class RecordingFetch:
    """Fetch that yields to the event loop so overlapping calls are observable."""

    def __init__(self) -> None:
        self.in_flight = 0
        self.peak = 0
        self.trace: list[str] = []

    async def __call__(self, url: str, query: str | None = None, /) -> FetchResult:
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        self.trace.append(f"start:{url}")
        try:
            for _ in range(3):
                await asyncio.sleep(0)
            return FetchResult(
                url=url, status=FetchStatus.OK, mime="text/plain", content=f"of {url}"
            )
        finally:
            self.in_flight -= 1
            self.trace.append(f"end:{url}")


def _cases(count: int) -> list[FetchCase]:
    return [
        FetchCase(
            case_id=f"c{number}",
            url=f"https://{number}.example",
            query=None,
            baseline_error=None,
            baseline_chars=None,
        )
        for number in range(count)
    ]


async def test_run_cases_fetches_a_batch_in_parallel():
    batch_size, count = 3, 7
    fetch = RecordingFetch()

    _, peak = await run_cases(fetch, _cases(count), batch_size=batch_size)

    assert peak == fetch.peak == batch_size


async def test_run_cases_batch_size_one_is_sequential():
    fetch = RecordingFetch()

    _, peak = await run_cases(fetch, _cases(4), batch_size=1)

    assert peak == fetch.peak == 1


async def test_run_cases_peak_is_capped_by_the_number_of_cases():
    count = 2
    fetch = RecordingFetch()

    _, peak = await run_cases(fetch, _cases(count), batch_size=count + 5)

    assert peak == count


async def test_run_cases_starts_a_batch_after_the_previous_one_ended():
    cases = _cases(4)
    fetch = RecordingFetch()

    await run_cases(fetch, cases, batch_size=2)

    first_batch_end = max(fetch.trace.index(f"end:{case.url}") for case in cases[:2])
    second_batch_start = min(
        fetch.trace.index(f"start:{case.url}") for case in cases[2:]
    )
    assert first_batch_end < second_batch_start


async def test_run_cases_returns_rows_in_case_order():
    cases = _cases(5)

    rows, _ = await run_cases(RecordingFetch(), cases, batch_size=2)

    assert [row["case_id"] for row in rows] == [case.case_id for case in cases]
    assert [row["content"] for row in rows] == [f"of {case.url}" for case in cases]


async def test_run_cases_reports_each_batch_when_it_ends():
    batch_size, count = 2, 5
    reported: list[list[object]] = []

    rows, _ = await run_cases(
        RecordingFetch(),
        _cases(count),
        batch_size=batch_size,
        on_rows=lambda batch: reported.append([row["case_id"] for row in batch]),
    )

    assert [len(batch) for batch in reported] == [2, 2, 1]
    assert [case_id for batch in reported for case_id in batch] == [
        row["case_id"] for row in rows
    ]


async def test_run_cases_rejects_non_positive_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        await run_cases(RecordingFetch(), _cases(1), batch_size=0)


def test_positive_int_parses_counts_and_rejects_the_rest():
    assert _positive_int("3") == 3
    for bad in ("0", "-1", "many"):
        with pytest.raises(argparse.ArgumentTypeError):
            _positive_int(bad)
