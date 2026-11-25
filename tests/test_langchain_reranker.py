"""Contract tests for provider-neutral LangChain reranking."""

from typing import Any

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage

from mcps.common.ai_models import DEFAULT_PARSE_ATTEMPTS, StructuredOutputError
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


type Response = RelevantChunks | Exception


class StructuredModel:
    """Replies with one response, or with a list of responses in order."""

    def __init__(self, response: Response | list[Response]):
        self.responses = response if isinstance(response, list) else None
        self.response = response
        self.calls: list[Any] = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        if self.responses is None:
            response = self.response
        else:
            response = self.responses.pop(0)  # IndexError on an unexpected extra call
        if isinstance(response, Exception):
            raise response
        return response


class ChatModel:
    def __init__(self, response: Response | list[Response]):
        self.structured_model = StructuredModel(response)
        self.schema = None

    def with_structured_output(self, schema, strict: bool = False):
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


def parse_error() -> OutputParserException:
    return OutputParserException("bad json", llm_output="not json")


async def test_rerank_reprompts_after_parse_error() -> None:
    model = ChatModel([parse_error(), RelevantChunks(relevant_chunk_ids=["doc"])])
    reranker = LangChainReranker(model)

    result = await reranker.rerank("query", [make_chunk("doc", "content")])

    assert [chunk.id for chunk in result] == ["doc"]
    assert len(model.structured_model.calls) == 2


async def test_rerank_raises_structured_output_error_after_all_attempts() -> None:
    model = ChatModel(parse_error())
    reranker = LangChainReranker(model)

    with pytest.raises(StructuredOutputError):
        await reranker.rerank("query", [make_chunk("doc", "content")])

    assert len(model.structured_model.calls) == DEFAULT_PARSE_ATTEMPTS
