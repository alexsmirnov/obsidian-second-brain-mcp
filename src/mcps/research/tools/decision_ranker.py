"""Ranker: per-fragment relevance probability from the router's decision model."""

from __future__ import annotations

import asyncio
import logging

from mcps.common.decisions import DecisionClient, DecisionError, Noul, NoulCriteria
from mcps.research.tools.filtering_models import FilterModelError

__all__ = ["DecisionRanker"]

logger = logging.getLogger(__name__)

_QUESTION_ID = "relevant"
_RELEVANT = {
    _QUESTION_ID: Noul(
        instructions="Does the passage help answer the query?",
        criteria=NoulCriteria(
            true="The passage contains information that answers or directly "
            "supports an answer to the query.",
            false="The passage is off topic or only shares vocabulary with the query.",
        ),
    )
}


class DecisionRanker:
    """Ranker keeping fragments whose relevance probability meets a threshold.

    Any decision failure raises :class:`FilterModelError`.
    """

    def __init__(
        self,
        client: DecisionClient,
        *,
        threshold: float = 0.6,
        concurrency: int = 4,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in the range [0.0, 1.0]")
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        self._client = client
        self._threshold = threshold
        self._semaphore = asyncio.Semaphore(concurrency)

    @property
    def threshold(self) -> float:
        return self._threshold

    async def __call__(self, fragments: list[str], query: str) -> list[int]:
        if not fragments:
            return []
        try:
            async with asyncio.TaskGroup() as group:
                tasks = [
                    group.create_task(self._relevance(fragment, query))
                    for fragment in fragments
                ]
        except* DecisionError as failures:
            raise FilterModelError(
                "decision request failed"
            ) from failures.exceptions[0]
        kept = [
            index
            for index, task in enumerate(tasks)
            if task.result() >= self._threshold
        ]
        logger.info(
            "Decision model %s kept %d of %d fragments",
            self._client.model,
            len(kept),
            len(fragments),
        )
        return kept

    async def _relevance(self, fragment: str, query: str) -> float:
        async with self._semaphore:
            response = await self._client.decide(
                {"query": query, "passage": fragment}, _RELEVANT
            )
        return response.noul(_QUESTION_ID)
