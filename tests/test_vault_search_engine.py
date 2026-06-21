from __future__ import annotations

from unittest.mock import MagicMock, patch

from mcps.config import ServerConfig
from mcps.rag.reranking import LangChainReranker
from mcps.rag.search import HypotheticalDocumentGenerator, SemanticSearchEngine
from mcps.rag.vault import _create_search_engine


def test_create_search_engine_without_rag_infer_model_uses_vector_only() -> None:
    config = ServerConfig(rag_infer_model="", search_limit=7)
    vector_store = MagicMock()
    http_client = MagicMock()

    with patch("mcps.rag.vault.ChatOpenAI") as chat_model:
        search_engine = _create_search_engine(vector_store, config, http_client)

    assert isinstance(search_engine, SemanticSearchEngine)
    assert search_engine.vector_store is vector_store
    assert search_engine.limit == 7
    assert search_engine.hypothetical_document_generator is None
    assert search_engine.reranker is None
    chat_model.assert_not_called()


def test_create_search_engine_with_rag_infer_model_wires_hyde_and_reranker() -> None:
    config = ServerConfig(
        rag_infer_model="search-model",
        litellm_router="http://router",
        litellm_router_key="token",
        search_limit=3,
    )
    vector_store = MagicMock()
    http_client = MagicMock()
    chat_instance = MagicMock()
    chat_instance.with_structured_output.return_value = MagicMock()

    with patch("mcps.rag.vault.ChatOpenAI", return_value=chat_instance) as chat_model:
        search_engine = _create_search_engine(vector_store, config, http_client)

    assert isinstance(search_engine, SemanticSearchEngine)
    assert search_engine.limit == 3
    assert isinstance(
        search_engine.hypothetical_document_generator,
        HypotheticalDocumentGenerator,
    )
    assert isinstance(search_engine.reranker, LangChainReranker)
    assert search_engine.hypothetical_document_generator.model is chat_instance
    assert search_engine.reranker.model is chat_instance
    chat_model.assert_called_once()
