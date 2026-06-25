"""Integration tests for model-backed RAG embeddings."""

from dataclasses import replace

from typing import AsyncIterator

import httpx
import pytest

from mcps.config import ServerConfig, create_config
from mcps.rag.interfaces import IEmbeddingService
from mcps.rag.vault import create_embeddings

MODEL_CASES: list[tuple[str, int]] = [
    ("text-embedding-3-small", 1536),
    ("text-embedding-3-small", 512),
    ("gemini-embed", 512),
    ("bge-embed", 1024),
    ("gemma-embed", 768),
    ("nomic-embed", 768),
]


@pytest.fixture
async def async_client() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient() as client:
        yield client


@pytest.fixture
def server_config() -> ServerConfig:
    server_config = create_config()
    if not server_config.router_api_base or not server_config.router_api_key:
        pytest.skip("ROUTER_API_BASE / ROUTER_API_KEY are not configured")
    return server_config


@pytest.fixture( params = MODEL_CASES)
def embedding_service(
    request,
    server_config: ServerConfig,
    async_client: httpx.AsyncClient,
) -> IEmbeddingService:
    (model, dimensions) = request.param
    config = replace(server_config,
    rag_embedding_model = model,
    rag_embedding_dimensions = dimensions
    )
    return create_embeddings(config, async_client)


@pytest.mark.asyncio
async def test_generate_embeddings_documents_returns_expected_shape(
    embedding_service: IEmbeddingService,
) -> None:

    result = await embedding_service.documents_embeddings(["first document", "second document"])
    dimensions = embedding_service.ndims()
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(vector, list) for vector in result)
    assert all(len(vector) == dimensions for vector in result)
    assert all(all(isinstance(value, float) for value in vector) for vector in result)


@pytest.mark.asyncio
async def test_generate_embeddings_query_returns_expected_shape(
    embedding_service: IEmbeddingService,
) -> None:
    result = await embedding_service.query_embeddings("query text")

    dimensions = embedding_service.ndims()
    assert isinstance(result, list)
    assert isinstance(result, list)
    assert len(result) == dimensions
    assert all(isinstance(value, float) for value in result)
