"""Integration tests for `LlmReranker` (hybrid rerank) with different model pairs."""

import logging
from collections.abc import AsyncIterator
from dataclasses import replace

import httpx
import pyarrow as pa
import pytest
from lancedb.rerankers import Reranker, RRFReranker
from langchain_core.embeddings import Embeddings

from mcps.config import ServerConfig, create_config
from mcps.rag.llm_reranker import LlmReranker
from mcps.rag.vault import create_reranker

logger = logging.getLogger(__name__)

# (inference_model, embedding_model, embedding_dimensions)
MODEL_CASES: list[tuple[str, str, int]] = [
    ("gemini-flash-lite", "nomic-embed", 768),
    ("gemini-flash-lite", "gemma-embed", 768),
    ("gpt-5-nano", "nomic-embed", 768),
    ("local-gemma", "gemma-embed", 768),
    ("", "gemma-embed", 768),
]



def _reranker_label(reranker: Reranker) -> str:
    """Return a human-readable label for the active model pair (logging only)."""
    chat_model = getattr(reranker, "chat_model", None)
    chat = chat_model.model_name if chat_model is not None else "none"
    embed = getattr(getattr(reranker, "embeddings", None), "model", "?")
    return f"{chat}+{embed}"


class FakeEmbeddings(Embeddings):
    """Deterministic in-memory embeddings implementation for unit tests."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def embed_query(self, text: str) -> list[float]:
        return self.vectors.get(text, [1.0, 0.0])

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.vectors.get(text, [1.0, 0.0]) for text in texts]


@pytest.fixture
async def async_client() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient() as client:
        yield client


@pytest.fixture
def server_config() -> ServerConfig:
    config = create_config()
    if not config.router_api_base or not config.router_api_key:
        pytest.skip(
            "ROUTER_API_BASE / ROUTER_API_KEY are not set; "
            "reranker tests need them to build a chat model + embeddings."
        )
    return config


@pytest.fixture(params=MODEL_CASES)
def reranker(
    request,
    server_config: ServerConfig,
    async_client: httpx.AsyncClient,
) -> Reranker:
    infer_model, embed_model, dimensions = request.param
    config = replace(
        server_config,
        rag_reranker_infer_model=infer_model,
        rag_embedding_model=embed_model,
        rag_embedding_dimensions=dimensions,
    )
    return create_reranker(config, async_client)


class TestLlmRerankerHybrid:
    """Test `LlmReranker.rerank_hybrid` across model pairs."""

    @pytest.fixture
    def sample_vector_results(self) -> pa.Table:
        return pa.Table.from_pydict(
            {
                "_rowid": [1, 2, 3],
                "content": [
                    "Cream of wild mushroom recipe",
                    "Ollama is the service to run large language models",
                    "RED6k is a comprehensive dataset containing **~6,000 samples** "
                    "across **10 domains** created by **Aizip** for evaluating "
                    "language models as summarizers in retrieval-augmented "
                    "generation (RAG) systems.",
                ],
                "id": [1, 2, 3],
                "_distance": [0.1, 0.2, 0.3],
            }
        )

    @pytest.fixture
    def sample_fts_results(self) -> pa.Table:
        return pa.Table.from_pydict(
            {
                "_rowid": [4, 5],
                "content": [
                    "Common Mistakes in Vector Search (and How to Avoid Them) "
                    "Neglecting Evaluations from the Get-Go",
                    "Reduce heat to medium and simmer for 20 minutes.",
                ],
                "id": [4, 5],
                "_score": [0.9, 0.8],
            }
        )

    def test_rerank_hybrid_empty_results(self, reranker: Reranker) -> None:
        empty_vector = pa.table(
            {
                "content": pa.array([], type=pa.string()),
                "_rowid": pa.array([], type=pa.int64()),
            }
        )
        empty_fts = pa.table(
            {
                "content": pa.array([], type=pa.string()),
                "_rowid": pa.array([], type=pa.int64()),
            }
        )

        result = reranker.rerank_hybrid("test query", empty_vector, empty_fts)

        assert isinstance(result, pa.Table)
        assert "_relevance_score" in result.column_names
        assert result.num_rows == 0

class TestLlmRerankerEmbeddingOnly:
    """Unit tests for `LlmReranker` when no chat model is configured."""

    @pytest.fixture
    def reranker(self) -> LlmReranker:
        vectors = {
            "query": [1.0, 0.0],
            "direct answer": [1.0, 0.0],
            "somewhat related": [0.7, 0.7],
            "unrelated": [0.0, 1.0],
        }
        return LlmReranker(chat_model=None, embeddings=FakeEmbeddings(vectors))

    def test_score_with_llm_returns_zeros_without_chat_model(
        self, reranker: LlmReranker
    ) -> None:
        scores = reranker._score_with_llm("query", ["direct answer", "unrelated"])

        assert scores == [0.0, 0.0]

    def test_rerank_vector_uses_only_embedding_similarity(
        self, reranker: LlmReranker
    ) -> None:
        results = pa.Table.from_pydict(
            {
                "_rowid": [1, 2, 3],
                "content": ["direct answer", "somewhat related", "unrelated"],
            }
        )

        result = reranker.rerank_vector("query", results)

        scores = result["_relevance_score"].to_pylist()
        assert scores[0] > scores[1] > scores[2]


class TestCreateRerankerEmbeddingOnly:
    """Integration tests for factory-created embedding-only `LlmReranker`."""

    @pytest.fixture
    def embedding_only_config(self, server_config: ServerConfig) -> ServerConfig:
        return replace(server_config, rag_reranker_infer_model="")

    def test_factory_creates_llm_reranker_without_chat_model(
        self,
        embedding_only_config: ServerConfig,
        async_client: httpx.AsyncClient,
    ) -> None:
        reranker = create_reranker(embedding_only_config, async_client)

        assert isinstance(reranker, RRFReranker)

    def test_factory_rerank_hybrid_uses_embeddings_only(
        self,
        embedding_only_config: ServerConfig,
        async_client: httpx.AsyncClient,
    ) -> None:
        reranker = create_reranker(embedding_only_config, async_client)
        vector_results = pa.Table.from_pydict(
            {
                "_rowid": [1, 2],
                "content": ["direct answer", "unrelated topic"],
                "id": [1, 2],
                "_distance": [0.1, 0.9],
            }
        )
        fts_results = pa.Table.from_pydict(
            {
                "_rowid": [3, 4],
                "content": ["another direct answer", "cooking recipe"],
                "id": [3, 4],
                "_score": [0.9, 0.1],
            }
        )

        result = reranker.rerank_hybrid("query", vector_results, fts_results)

        assert isinstance(result, pa.Table)
        assert "_relevance_score" in result.column_names
        assert result.num_rows == 4
        assert all(0 <= score <= 1 for score in result["_relevance_score"].to_pylist())
