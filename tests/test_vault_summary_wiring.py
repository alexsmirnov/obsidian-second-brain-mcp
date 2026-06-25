from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcps.config import ServerConfig
from mcps.rag.summarization import LangChainDocumentSummaryGenerator
from mcps.rag.vault import _create_document_summary_generator, create_vault


def test_create_summary_generator_without_rag_infer_model_returns_none() -> None:
    config = ServerConfig(rag_infer_model="")
    http_client = MagicMock()

    with patch("mcps.rag.vault.ChatOpenAI") as chat_model:
        generator = _create_document_summary_generator(config, http_client)

    assert generator is None
    chat_model.assert_not_called()


def test_create_summary_generator_with_rag_infer_model_wires_chat_model() -> None:
    config = ServerConfig(
        rag_infer_model="summary-model",
        router_api_base="http://router",
        router_api_key="token",
    )
    http_client = MagicMock()
    chat_instance = MagicMock()

    with patch("mcps.rag.vault.ChatOpenAI", return_value=chat_instance) as chat_model:
        generator = _create_document_summary_generator(config, http_client)

    assert isinstance(generator, LangChainDocumentSummaryGenerator)
    assert generator.model is chat_instance
    chat_model.assert_called_once()
    assert chat_model.call_args.kwargs["model"] == "summary-model"
    assert chat_model.call_args.kwargs["base_url"] == "http://router"
    assert chat_model.call_args.kwargs["http_async_client"] is http_client


@pytest.mark.parametrize("summary_generator", [None, MagicMock()])
async def test_create_vault_passes_document_summary_generator_to_vault(
    summary_generator: object | None,
) -> None:
    config = ServerConfig(vault_dir=MagicMock())
    http_client = MagicMock()
    vector_store = AsyncMock()

    with (
        patch("mcps.rag.vault._create_file_traversal", return_value=MagicMock()),
        patch("mcps.rag.vault._create_document_processor", return_value=MagicMock()),
        patch("mcps.rag.vault._create_chunker", return_value=MagicMock()),
        patch("mcps.rag.vault.create_embeddings", return_value=MagicMock()),
        patch("mcps.rag.vault.create_reranker", return_value=MagicMock()),
        patch("mcps.rag.vault._create_vector_store") as create_vector_store,
        patch("mcps.rag.vault._create_search_engine", return_value=MagicMock()),
        patch(
            "mcps.rag.vault._create_document_summary_generator",
            return_value=summary_generator,
        ),
        patch("mcps.rag.vault.Vault") as vault_class,
    ):
        create_vector_store.return_value.__aenter__.return_value = vector_store
        create_vector_store.return_value.__aexit__.return_value = False
        vault = AsyncMock()
        vault_class.return_value = vault

        async with create_vault(config, http_client):
            pass

    created_summary_generator = vault_class.call_args.kwargs[
        "document_summary_generator"
    ]
    assert created_summary_generator is summary_generator
    vault.initialize.assert_awaited_once()
    vault.cleanup.assert_awaited_once()
