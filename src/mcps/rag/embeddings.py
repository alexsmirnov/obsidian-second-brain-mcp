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

    async def generate_embeddings(
        self, texts: list[str], query: bool = False
    ) -> list[list[float]]:
        """Generate embeddings for texts using the injected LangChain adapter."""
        if not texts:
            return []

        if query:
            return [await self.embeddings.aembed_query(text) for text in texts]

        return await self.embeddings.aembed_documents(texts)

    def ndims(self) -> int:
        """Return the number of dimensions of the embeddings."""
        return self.dimensions
