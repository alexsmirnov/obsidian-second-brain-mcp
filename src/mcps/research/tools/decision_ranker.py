"""Ranker: per-fragment relevance probability from the router's decision model."""

from __future__ import annotations

import asyncio
import logging

from mcps.common.decisions import DecisionClient, DecisionError, Noul, NoulCriteria
from mcps.research.tools.filtering import FilterModelError

__all__ = ["RELEVANCE_QUESTION", "DecisionRanker"]

logger = logging.getLogger(__name__)

_QUESTION_ID = "relevant"
RELEVANCE_QUESTION = Noul(
    instructions="Does the passage relevant to the query?",
    criteria=NoulCriteria(
        true="The passage contains text from the area related to the query, "
        "or at least from similar domain.",
        false="The passage is completely off topic.",
    ),
)


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
        question: Noul = RELEVANCE_QUESTION,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in the range [0.0, 1.0]")
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        self._client = client
        self._threshold = threshold
        self._semaphore = asyncio.Semaphore(concurrency)
        self._questions = {_QUESTION_ID: question}

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def question(self) -> Noul:
        return self._questions[_QUESTION_ID]

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
                {"query": query, "passage": fragment}, self._questions
            )
        return response.noul(_QUESTION_ID)
