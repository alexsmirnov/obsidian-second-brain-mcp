"""Provider-neutral reranking for RAG search results."""

import logging
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from .interfaces import Chunk

logger = logging.getLogger("mcps.rag.reranking")

RERANK_SYSTEM_PROMPT = """
You are a search result reranker. Return IDs of chunks that are relevant to
answering the query. Consider relevance in broad mean, 
include chunk if it even slightly related.
Order IDs from most relevant to least relevant. Return only
IDs from the provided candidates.
""".strip()


class RelevantChunks(BaseModel):
    """Structured reranker response containing relevant chunk IDs in order."""

    relevant_chunk_ids: list[str] = Field(default_factory=list)


class IRerankingService(ABC):
    """Interface for asynchronous search-result reranking."""

    @abstractmethod
    async def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        """Return chunks ordered and annotated by relevance."""
        pass


class LangChainReranker(IRerankingService):
    """Rerank chunks with one structured-output chat model call."""

    def __init__(self, model: BaseChatModel) -> None:
        self.model = model
        self.structured_model = model.with_structured_output(RelevantChunks)

    async def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return []

        try:
            response = await self.structured_model.ainvoke(
                [
                    SystemMessage(content=RERANK_SYSTEM_PROMPT),
                    HumanMessage(content=self._create_user_prompt(query, chunks)),
                ]
            )
        except Exception:
            logger.exception("Failed to rerank documents with LangChain chat model")
            raise

        relevant_chunk_ids = self._extract_relevant_chunk_ids(response)
        chunks_by_id = {chunk.id: chunk for chunk in chunks}
        selected_chunks = []
        selected_ids = set()
        for chunk_id in relevant_chunk_ids:
            if chunk_id in selected_ids or chunk_id not in chunks_by_id:
                continue
            selected_chunks.append(chunks_by_id[chunk_id])
            selected_ids.add(chunk_id)
        logger.info("Rerank filtered %s chunks from %s",len(selected_chunks),len(chunks))
        return selected_chunks

    @staticmethod
    def _extract_relevant_chunk_ids(response: Any) -> list[str]:
        if isinstance(response, RelevantChunks):
            return response.relevant_chunk_ids
        if isinstance(response, BaseModel):
            response = response.model_dump()
        if not isinstance(response, dict):
            return []
        relevant_chunk_ids = response.get("relevant_chunk_ids", [])
        return [str(chunk_id) for chunk_id in relevant_chunk_ids]

    @staticmethod
    def _create_user_prompt(query: str, chunks: list[Chunk]) -> str:
        documents = "\n\n".join(
            f"ID: {chunk.id}\nContent:\n{chunk.content}"
            for chunk in chunks
        )
        return f"""
Query: {query}

Candidate chunks:
{documents}"""
