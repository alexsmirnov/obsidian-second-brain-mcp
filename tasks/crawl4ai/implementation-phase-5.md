<!-- Context: read Goal, Specification, Out of Scope, Research Findings, Implementation Research Findings and Conventions in implementation-plan.md before starting this phase. -->

## Phase 5: Fetch evaluation script [NEW_FEATURE]

### RED — `tests/test_web_fetch_evaluation.py:1` (NEW)
**Source under test:** `tests/web_fetch_evaluation.py:1` (NEW)
**Functions under test:** `load_cases()`, `summarize()`
**Fixtures:** `tmp_path/cases.jsonl` rows: `{"case_id":"ok","url":"https://a.example","query":"q","baseline":{"observations":[{"status":"success","pre_truncation_chars":20000,"inferred_returned_chars":15021}]}}`; `{"case_id":"bad","url":"https://b.example","query":"q","baseline":{"observations":[{"status":"retrieval_error","pre_truncation_chars":null,"inferred_returned_chars":null}]}}`; `{"url":"https://c.example","query":null}`.

#### `test_load_cases_reads_minimal_and_baseline_rows`
- **When:** `load_cases(path)`
- **Then:** 3 cases; third `case_id == "line-3"`, `query is None`, `baseline_error is None`; first `baseline_error is False`, `baseline_chars == 15021`; second `baseline_error is True`

#### `test_summarize_compares_matched_baseline`
- **Given:** rows `{"case_id":"ok","error":None,"response_size":1000}`, `{"case_id":"bad","error":None,"response_size":500}`, `{"case_id":"line-3","error":"ERROR: x","response_size":0}`
- **When:** `summarize(cases, rows)`
- **Then:** `== {"cases": 3, "errors": 1, "empty": 0, "baseline_errors": 1, "recovered": 1, "regressed": 0, "mean_size": 750.0, "baseline_mean_size": 15021.0}`. Definitions: `errors` = rows with non-null `error`; `empty` = rows with no error and `response_size == 0`; `baseline_errors`/`recovered`/`regressed` count only cases whose `baseline_error is not None`; `mean_size` averages non-error, non-empty rows; `baseline_mean_size` averages `baseline_chars` of baseline-successful cases.

→ **EXPECTED: FAIL** — module missing.

### CONFIRM_RED
Run `test.sh "$(pwd)/tests/test_web_fetch_evaluation.py"`. Get approval.

### GREEN — `tests/web_fetch_evaluation.py:1` (NEW)
- Move (not copy) `tasks/crawl4ai/fetch-cases-baseline.jsonl` and `tasks/crawl4ai/fetch-cases-smoke.jsonl` to `tests/evaluation/data/` (NEW directory, no `__init__.py`); contents unchanged; files are tracked by git.
- `CASES_DIR = Path(__file__).parent / "evaluation" / "data"` module constant.
- `@dataclass(frozen=True) FetchCase(case_id: str, url: str, query: str | None, baseline_error: bool | None, baseline_chars: int | None)` — baseline from first observation.
- `load_cases(path: Path) -> list[FetchCase]` — skip blank lines; missing `case_id` → `f"line-{n}"`.
- `summarize(cases: list[FetchCase], rows: list[dict[str, object]]) -> dict[str, object]` — keys as in the test.
- `async main(argv: list[str] | None = None) -> None` — argparse `--cases` (default `CASES_DIR / "fetch-cases-smoke.jsonl"`), `--output` required; `load_dotenv()`, `create_config()`, httpx client, `browser_endpoint`, `create_fetch_tool`; sequentially fetch each case, measure `time.perf_counter`, write JSONL row `{case_id, url, query, response_size: len(content), response_time_ms, error: content if content.startswith("ERROR") else None, content}`; print `json.dumps(summarize(...), indent=2)`. `if __name__ == "__main__": asyncio.run(main())`.
→ **EXPECTED: PASS**.

### VERIFY_GREEN
Run test; lint/compile. `git status --short tests/evaluation/data` lists both JSONL files as untracked-but-not-ignored; `tasks/crawl4ai/*.jsonl` no longer exist. **Manual check (user permission):** `uv run tests/web_fetch_evaluation.py --output tmp/fetch-smoke.jsonl`; full run adds `--cases tests/evaluation/data/fetch-cases-baseline.jsonl`.
