# Fetch integration-test datasets

Amendment to the crawl4ai implementation plan (simplified revision). The evaluation script is `tests/web_fetch_evaluation.py`, specified in Phase 5 of `implementation-plan.md`; that phase is authoritative.

## Saved inputs

- `tasks/crawl4ai/fetch-cases-baseline.jsonl:1`: 279 distinct task/round/URL cases, preserving all 288 original observations (197 successful extraction records and 91 errors). Distinct cases comprise 192 successes and 87 errors; there are 267 unique URLs. The same URL in different rounds remains a separate case because its relevance query changes.
- `tasks/crawl4ai/fetch-cases-smoke.jsonl:1`: eight cases: generic HTML, blocked HTML, non-arXiv PDF, arXiv, GitHub repository, empty extraction, HTTP 404, Wikipedia. Operational check, not a benchmark.
- Source: `tmp/evaluation-20260928_120233.log:1`; SHA-256 `e33769ccd6a34ec78b4808f1a0198a10a291e9bf35a73d26f5a3681cbcba0784`.
- Queries: original DRACO questions (round 0) or reflection knowledge gaps (later rounds), embedded per row.

## JSONL input contract

One JSON object per nonblank line. Required: `url: str`, `query: str | null`. Optional: `case_id`, `task_id`, `research_round`, `baseline{source_log, source_sha256, observations[{line, status, error, pre_truncation_chars, inferred_returned_chars}]}`.

`inferred_returned_chars` is computed from the old logger (≤15,000 chars returned as-is, else 15,000 + 21-char marker), not measured. Failed observations have null sizes.

## Script interface

- `--cases PATH` and `--output PATH` required; no default dataset; nothing runs at import.
- Runs cases sequentially through the production fetch (browser + configured `SCRAPER_PROVIDER`); prints a summary compared with baseline observations.
- Routine: `uv run tests/web_fetch_evaluation.py --cases tasks/crawl4ai/fetch-cases-smoke.jsonl --output tmp/fetch-smoke.jsonl`. Full corpus only by explicit choice of `fetch-cases-baseline.jsonl`.

## Dropped from the previous revision (user decision: personal tool, failures fixed by redeploy)

`--live` opt-in, output-overwrite refusal, per-provider independent runs, malformed-row/line-number validation tests, attempt-count preview. Malformed input fails with the Python exception.
