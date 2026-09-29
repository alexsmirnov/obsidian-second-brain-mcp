#!/usr/bin/env python3
"""DRACO evaluation CLI for the web_research deep-research agent.

Run from the repository root: ``uv run python tests/web_research_evaluation.py``.
Judge model: ``RESEARCH_EVAL_MODEL`` (falls back to ``RESEARCH_INFER_MODEL``).
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime
import logging

import httpx
from dotenv import load_dotenv
from evaluation.dataset import load_draco_questions
from evaluation.judge import create_judge_model
from evaluation.pipeline import run_evaluation
from evaluation.report import REPORT_DIR, generate_html_report, save_report

from mcps.config import create_config
from mcps.research.agent import create_researcher
from mcps.research.config import build_research_config
from mcps.research.tools.browser import browser_endpoint

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
REPORT_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=REPORT_DIR / f"evaluation-{timestamp}.log",
    encoding="utf-8",
    level=logging.INFO,
    format="%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


async def main():
    """Run DRACO evaluation on research agent."""
    load_dotenv()
    server_config = create_config()
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as http_client:
        async with browser_endpoint(server_config.browser_cdp_url) as cdp_url:
            if cdp_url is None:
                raise RuntimeError(
                    "No browser available: set BROWSER_CDP_URL or install Obscura "
                    "on PATH before running the evaluation."
                )
            logger.info("Creating research configuration...")
            config = build_research_config(
                dataclasses.replace(server_config, browser_cdp_url=cdp_url),
                http_client=http_client,
            )
            judge = create_judge_model(server_config, http_client=http_client)

            logger.info("Creating research agent...")
            agent = create_researcher(config, implementation="deep_research")

            logger.info("Loading DRACO questions (Technology + Academic)...")
            questions = load_draco_questions()[:3]

            logger.info("Running evaluation on %d questions...", len(questions))
            summary, results = await run_evaluation(agent, judge, questions)

    print("\n" + "=" * 60)
    print("DRACO EVALUATION RESULTS")
    print(f"Total tasks: {summary.total}")
    print(f"Average score: {summary.avg_score * 100:.1f}%")
    print(f"Execution errors: {summary.execution_errors}")
    print(f"Avg latency: {summary.avg_latency:.1f}s")
    for domain, score in sorted(summary.domain_scores.items()):
        print(f"  {domain}: {score * 100:.1f}%")
    print("=" * 60)

    logger.info("Saving HTML report...")
    report = generate_html_report(summary, results, questions)
    output_path = save_report(report, "web_research_evaluation_" + timestamp)
    logger.info("Report saved to: %s", output_path)


if __name__ == "__main__":
    asyncio.run(main())
