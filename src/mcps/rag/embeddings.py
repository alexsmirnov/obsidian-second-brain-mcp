"""Embedding service implementations for the RAG search system."""

from langchain_core.embeddings import Embeddings

from .interfaces import IEmbeddingService


class LangChainEmbeddingService(IEmbeddingService):
    """Embedding service backed by a LangChain Embeddings implementation."""

    def __init__(
        self,
        embeddings: Embeddings,
        dimensions: int,
    ) -> None:
        self.embeddings = embeddings
        self.dimensions = dimensions

    async def documents_embeddings(
        self, texts: list[str]
    ) -> list[list[float]]:
        if not texts:
            return []
        return await self.embeddings.aembed_documents(texts)

    async def query_embeddings(
        self, query: str
    ) -> list[float]:
        if not query:
            return []
        return await self.embeddings.aembed_query(query)

    def ndims(self) -> int:
        return self.dimensions
