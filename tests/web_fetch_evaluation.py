#!/usr/bin/env python3
"""Replay JSONL fetch cases through the production fetch and compare to baseline.

Run from the repository root:
``uv run tests/web_fetch_evaluation.py --output tmp/fetch-smoke.jsonl``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import httpx
from dotenv import load_dotenv

from mcps.config import create_config
from mcps.research.config import create_fetch_tool
from mcps.research.tools.browser import browser_crawler, browser_endpoint
from mcps.research.tools.result import FetchResult, FetchStatus

CASES_DIR = Path(__file__).parent / "evaluation" / "data"

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


@dataclass(frozen=True)
class FetchCase:
    case_id: str
    url: str
    query: str | None
    baseline_error: bool | None
    baseline_chars: int | None


def _parse_case(number: int, raw: dict[str, object]) -> FetchCase:
    baseline = raw.get("baseline")
    observations = baseline.get("observations") if isinstance(baseline, dict) else None
    first = observations[0] if observations else None
    query = raw.get("query")
    return FetchCase(
        case_id=str(raw.get("case_id") or f"line-{number}"),
        url=str(raw["url"]),
        query=query if isinstance(query, str) else None,
        baseline_error=first["status"] != "success" if first else None,
        baseline_chars=first.get("inferred_returned_chars") if first else None,
    )


def _error_label(result: FetchResult) -> str | None:
    """Failure label for the report; restricted domains are not failures."""
    if result.ok or result.status is FetchStatus.RESTRICTED:
        return None
    if result.http_status is not None:
        return f"http {result.http_status}"
    return str(result.status)


def load_cases(path: Path) -> list[FetchCase]:
    with path.open(encoding="utf-8") as lines:
        return [
            _parse_case(number, json.loads(line))
            for number, line in enumerate(lines, start=1)
            if line.strip()
        ]


def _mean_or_zero(values: list[int]) -> float:
    return float(mean(values)) if values else 0.0


def summarize(
    cases: list[FetchCase], rows: list[dict[str, object]]
) -> dict[str, object]:
    by_id = {str(row["case_id"]): row for row in rows}
    errors = empty = baseline_errors = recovered = regressed = 0
    sizes: list[int] = []
    for case in cases:
        row = by_id.get(case.case_id)
        if row is None:
            continue
        raw_size = row.get("response_size")
        size = raw_size if isinstance(raw_size, int) else 0
        failed = row.get("error") is not None
        if failed:
            errors += 1
        elif size == 0:
            empty += 1
        else:
            sizes.append(size)
        if case.baseline_error is None:
            continue
        if case.baseline_error:
            baseline_errors += 1
            recovered += not failed
        else:
            regressed += failed
    baseline_sizes = [
        case.baseline_chars
        for case in cases
        if case.baseline_error is False and case.baseline_chars is not None
    ]
    return {
        "cases": len(cases),
        "errors": errors,
        "empty": empty,
        "baseline_errors": baseline_errors,
        "recovered": recovered,
        "regressed": regressed,
        "mean_size": _mean_or_zero(sizes),
        "baseline_mean_size": _mean_or_zero(baseline_sizes),
    }


async def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", type=Path, default=CASES_DIR / "fetch-cases-smoke.jsonl"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    _configure_logging()
    load_dotenv()
    server_config = create_config()
    cases = load_cases(args.cases)
    logger.info("Replaying %d cases from %s", len(cases), args.cases)
    rows: list[dict[str, object]] = []
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as http_client:
        async with browser_endpoint(server_config.browser_cdp_url) as cdp_url:
            async with browser_crawler(cdp_url) as crawler:
                fetch = create_fetch_tool(
                    config=server_config,
                    http_client=http_client,
                    browser_crawler=crawler,
                )
                for number, case in enumerate(cases, start=1):
                    started = time.perf_counter()
                    result = await fetch(case.url, case.query)
                    content = result.content
                    elapsed_ms = (time.perf_counter() - started) * 1000
                    logger.info(
                        "[%d/%d] %s %s: %d chars in %.0f ms",
                        number,
                        len(cases),
                        case.case_id,
                        case.url,
                        len(content),
                        elapsed_ms,
                    )
                    rows.append(
                        {
                            "case_id": case.case_id,
                            "url": case.url,
                            "query": case.query,
                            "response_size": len(content),
                            "response_time_ms": round(elapsed_ms),
                            "error": _error_label(result),
                            "content": content,
                        }
                    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    print(json.dumps(summarize(cases, rows), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
