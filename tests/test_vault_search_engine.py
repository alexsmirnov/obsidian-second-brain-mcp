from __future__ import annotations

from unittest.mock import MagicMock, patch

from mcps.config import RagSettings, RouterSettings, SearchSettings, ServerConfig
from mcps.rag.llm_reranker import LlmReranker
from mcps.rag.reranking import LangChainReranker
from mcps.rag.search import HypotheticalDocumentGenerator, SemanticSearchEngine
from mcps.rag.vault import _create_search_engine, create_reranker


def test_create_search_engine_without_rag_infer_model_uses_vector_only() -> None:
    config = ServerConfig(
        rag=RagSettings(infer_model=""),
        search=SearchSettings(limit=7, MIN_SCORE=0.25),
    )
    vector_store = MagicMock()
    http_client = MagicMock()

    with patch("mcps.rag.vault.create_chat_model") as chat_model:
        search_engine = _create_search_engine(vector_store, config, http_client)

    assert isinstance(search_engine, SemanticSearchEngine)
    assert search_engine.vector_store is vector_store
    assert search_engine.limit == 7
    assert search_engine.min_score == 0.25
    assert search_engine.hypothetical_document_generator is None
    assert search_engine.reranker is None
    chat_model.assert_not_called()


def test_create_search_engine_with_rag_infer_model_wires_hyde_and_reranker() -> None:
    config = ServerConfig(
        rag=RagSettings(infer_model="search-model"),
        router=RouterSettings(api_base="http://router", api_key="token"),
        search=SearchSettings(limit=3, MIN_SCORE=0.4),
    )
    vector_store = MagicMock()
    http_client = MagicMock()
    chat_instance = MagicMock()

    with patch(
        "mcps.rag.vault.create_chat_model", return_value=chat_instance
    ) as chat_model:
        search_engine = _create_search_engine(vector_store, config, http_client)

    assert isinstance(search_engine, SemanticSearchEngine)
    assert search_engine.limit == 3
    assert search_engine.min_score == 0.4
    assert isinstance(
        search_engine.hypothetical_document_generator,
        HypotheticalDocumentGenerator,
    )
    assert isinstance(search_engine.reranker, LangChainReranker)
    assert search_engine.hypothetical_document_generator.model is chat_instance
    assert search_engine.reranker.model is chat_instance
    chat_model.assert_called_once_with("search-model", config.router, http_client)


def test_create_reranker_with_infer_model_wires_factories() -> None:
    config = ServerConfig(
        rag=RagSettings(
            reranker_embedding_model="emb-r",
            reranker_embedding_dimensions=256,
            reranker_infer_model="rerank-m",
        ),
        router=RouterSettings(api_base="http://router", api_key="token"),
    )
    http_client = MagicMock()

    with (
        patch("mcps.rag.vault.create_embeddings_model") as embeddings,
        patch("mcps.rag.vault.create_chat_model") as chat_model,
    ):
        reranker = create_reranker(config, http_client)

    assert isinstance(reranker, LlmReranker)
    embeddings.assert_called_once_with("emb-r", 256, config.router, http_client)
    chat_model.assert_called_once_with("rerank-m", config.router, http_client)
