"""Contract tests for provider-neutral LangChain reranking."""

from datetime import datetime
from typing import Any

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from mcps.rag.interfaces import Chunk
from mcps.rag.reranking import LangChainReranker, RelevantChunks


def make_chunk(id_: str, content: str) -> Chunk:
    return Chunk(
        id=id_,
        content=content,
        title=None,
        description=None,
        source_path=f"{id_}.md",
        wikilink_name=id_,
        modified_at=1234.345,
        position=0,
        offset=0,
        file_size=len(content),
    )


class StructuredModel:
    def __init__(self, response: RelevantChunks | Exception):
        self.response = response
        self.calls: list[Any] = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class ChatModel:
    def __init__(self, response: RelevantChunks | Exception):
        self.structured_model = StructuredModel(response)
        self.schema = None

    def with_structured_output(self, schema):
        self.schema = schema
        return self.structured_model


async def test_rerank_empty_chunks_returns_empty_without_model_calls() -> None:
    model = ChatModel(RelevantChunks(relevant_chunk_ids=[]))
    reranker = LangChainReranker(model)

    result = await reranker.rerank("query", [])

    assert result == []
    assert model.structured_model.calls == []


async def test_rerank_structured_output_returns_selected_chunks_in_id_order() -> None:
    model = ChatModel(RelevantChunks(relevant_chunk_ids=["high", "low"]))
    reranker = LangChainReranker(model)
    low = make_chunk("low", "related")
    high = make_chunk("high", "directly answers")
    ignored = make_chunk("ignored", "not needed")

    result = await reranker.rerank("query", [low, high, ignored])

    assert [chunk.id for chunk in result] == ["high", "low"]
    assert len(model.structured_model.calls) == 1


async def test_rerank_splits_system_prompt_from_query_and_chunks() -> None:
    model = ChatModel(RelevantChunks(relevant_chunk_ids=["doc"]))
    reranker = LangChainReranker(model)

    await reranker.rerank("query", [make_chunk("doc", "content")])

    messages = model.structured_model.calls[0]
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert "Order IDs from most relevant to least relevant" in messages[0].content
    assert "Query: query" in messages[1].content
    assert "ID: doc" in messages[1].content
    assert "Order IDs from most relevant to least relevant" not in messages[1].content


async def test_rerank_structured_output_ignores_unknown_ids() -> None:
    reranker = LangChainReranker(
        ChatModel(RelevantChunks(relevant_chunk_ids=["missing", "doc"]))
    )

    result = await reranker.rerank("query", [make_chunk("doc", "content")])

    assert [chunk.id for chunk in result] == ["doc"]


async def test_rerank_structured_output_ignores_duplicate_ids() -> None:
    reranker = LangChainReranker(
        ChatModel(RelevantChunks(relevant_chunk_ids=["doc", "doc"]))
    )

    result = await reranker.rerank("query", [make_chunk("doc", "content")])

    assert [chunk.id for chunk in result] == ["doc"]


async def test_rerank_model_error_raises_after_logging() -> None:
    reranker = LangChainReranker(ChatModel(RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="boom"):
        await reranker.rerank("query", [make_chunk("doc", "content")])
