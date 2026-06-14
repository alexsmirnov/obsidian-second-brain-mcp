"""Provider-neutral reranking for RAG search results."""

import logging
from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from .interfaces import Chunk

logger = logging.getLogger("mcps.rag.reranking")


class IRerankingService(ABC):
    """Interface for asynchronous search-result reranking."""

    @abstractmethod
    async def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        """Return chunks ordered and annotated by relevance."""
        pass


class LangChainReranker(IRerankingService):
    """Rerank chunks by asking an injected LangChain chat model for scores."""

    def __init__(self, model: BaseChatModel) -> None:
        self.model = model

    async def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return []

        scored_chunks = [
            self._with_relevance_score(
                chunk,
                await self._score_document(query, chunk.content),
            )
            for chunk in chunks
        ]
        return sorted(
            scored_chunks,
            key=lambda chunk: getattr(chunk, "_relevance_score", 0.0),
            reverse=True,
        )

    async def _score_document(self, query: str, document: str) -> float:
        prompt = self._create_relevance_prompt(query, document)
        try:
            response = await self.model.ainvoke([HumanMessage(content=prompt)])
        except Exception:
            logger.exception("Failed to score document with LangChain chat model")
            return 0.0

        content = response.content
        if isinstance(content, str):
            return self._parse_score(content)
        return self._parse_score(str(content))

    @staticmethod
    def _with_relevance_score(chunk: Chunk, score: float) -> Chunk:
        scored = chunk.model_copy()
        object.__setattr__(scored, "_relevance_score", score)
        return scored

    @staticmethod
    def _create_relevance_prompt(query: str, document: str) -> str:
        return f"""
Given the query and document below, rate how relevant the document is to answering
the query. Output a single word:
PERFECT if they are relevant
GOOD if they are close but not exact, like both about programming but different
    languages or libraries
SOME if they are relevant in broad sense, like both about programming but one about
    coding practices and another about algorithms
BAD if there is only little similarity, like one about programming and another about
    job interviews
NONE if document is unrelated to question, like one about astronomy and another about
    cooking recipes.

Query: {query}

Document: {document}

Relevance:"""

    @staticmethod
    def _parse_score(score_text: str) -> float:
        score_text = score_text.lower()
        if "perfect" in score_text:
            return 1.0
        if "good" in score_text:
            return 0.75
        if "some" in score_text:
            return 0.5
        if "bad" in score_text:
            return 0.25
        return 0.0
