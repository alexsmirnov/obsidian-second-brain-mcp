"""
Embedding service implementations for the RAG search system.
"""

import logging
from operator import le

import openai
from .interfaces import IEmbeddingService

logger = logging.getLogger("mcps")


class OpenAIEmbedding(IEmbeddingService):
    """OpenAI api compatible embedding service."""

    def __init__(
        self,
        model_name: str,
        dimensions: int,
        api_key: str | None = None,
        api_base: str | None = None,
        provider: str = "openai",
    ):
        self.model_name = model_name
        self.dimensions = dimensions
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=api_base)
        self.provider = provider

    async def generate_embeddings(
        self, texts: list[str], query: bool = False
    ) -> list[list[float]]:
        """Generate embedding for a texts using OpenAI API."""
        if len(texts) == 0:
            return []
        try:
            extra_body = (
                {
                    "output_dimension": self.dimensions,
                    "input_type": "query" if query else "document",
                }
                if self.provider == "voyage"
                else None
            )
            response = await self.client.embeddings.create(
                model=self.model_name,
                dimensions=(
                    self.dimensions if self.provider == "openai" else openai.NotGiven()
                ),
                extra_body=extra_body,
                input=texts,
            )
            return [d.embedding for d in response.data]
        except Exception as e:
            logger.error(f"Failed to generate OpenAI embedding: {e}")
            # Return zero vector as fallback
            return [[0.0] * self.dimensions] * len(texts)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.client.close()
