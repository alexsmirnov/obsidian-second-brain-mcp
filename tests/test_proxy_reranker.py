"""Integration tests for `LiteLLMProxyReranker` with LiteLLM proxy models."""

import logging
from collections.abc import AsyncIterator
from dataclasses import replace

import httpx
import pyarrow as pa
import pytest
from lancedb.rerankers import Reranker

from mcps.config import ServerConfig, create_config
from mcps.rag.proxy_reranker import ProxyReranker
from mcps.rag.vault import create_reranker

logger = logging.getLogger(__name__)

MODEL_CASES: list[str] = ["rerank", "rerank-lite"]


def _reranker_label(reranker: Reranker) -> str:
    """Return a human-readable label for the active proxy model."""
    return str(getattr(reranker, "model_name", "?"))


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
            "proxy reranker tests need them to call the model router."
        )
    return config


@pytest.fixture(params=MODEL_CASES)
def reranker(
    request: pytest.FixtureRequest,
    server_config: ServerConfig,
    async_client: httpx.AsyncClient,
) -> Reranker:
    config = replace(
        server_config,
        rag_reranker_model=str(request.param),
        rag_reranker_infer_model="",
        rag_embedding_model="",
        rag_embedding_dimensions=0,
    )
    return create_reranker(config, async_client)


class TestProxyRerankerHybrid:
    """Test `LiteLLMProxyReranker.rerank_hybrid` across proxy models."""

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

    def test_rerank_hybrid_successful(
        self,
        reranker: Reranker,
        sample_vector_results: pa.Table,
        sample_fts_results: pa.Table,
    ) -> None:
        result = reranker.rerank_hybrid(
            "how to perform evaluation for RAG",
            sample_vector_results,
            sample_fts_results,
        )

        assert isinstance(result, pa.Table)
        assert "_relevance_score" in result.column_names
        assert result.num_rows == 5

        scores = result["_relevance_score"].to_pylist()
        assert all(0 <= score <= 1 for score in scores)

        rowids = result["_rowid"].to_pylist()
        rowid_to_score = dict(zip(rowids, scores, strict=True))
        logger.info(
            "Proxy model %s | RowID to Score: %s",
            _reranker_label(reranker),
            rowid_to_score,
        )
        assert rowid_to_score[3] > rowid_to_score[1]
        assert rowid_to_score[4] > rowid_to_score[5]

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


class TestProxyRerankerPayload:
    """Unit tests for request payload construction (no live router)."""

    @staticmethod
    def _capture_post(monkeypatch: pytest.MonkeyPatch) -> dict:
        captured: dict = {}

        class _Response:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return {"results": []}

        def fake_post(_self: httpx.Client, url: str, *, json: dict) -> _Response:
            captured["url"] = url
            captured["payload"] = json
            return _Response()

        monkeypatch.setattr("httpx.Client.post", fake_post)
        return captured

    def test_documents_are_plain_strings_when_content_has_null(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = self._capture_post(monkeypatch)
        reranker = ProxyReranker(
            model_name="rerank", api_key="key", proxy_url="http://router"
        )
        vector = pa.table(
            {
                "content": pa.array(["alpha", None], type=pa.string()),
                "_rowid": pa.array([1, 2], type=pa.int64()),
            }
        )
        fts = pa.table(
            {
                "content": pa.array(["beta"], type=pa.string()),
                "_rowid": pa.array([3], type=pa.int64()),
            }
        )

        reranker.rerank_hybrid("query", vector, fts)

        documents = captured["payload"]["documents"]
        assert documents
        assert all(isinstance(document, str) for document in documents)
