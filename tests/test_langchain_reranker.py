"""Contract tests for provider-neutral LangChain reranking."""

from datetime import datetime

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from mcps.rag.interfaces import Chunk
from mcps.rag.reranking import LangChainReranker


def make_chunk(id_: str, content: str) -> Chunk:
    return Chunk(
        id=id_,
        content=content,
        title=None,
        description=None,
        source_path=f"{id_}.md",
        modified_at=datetime.now(),
        position=0,
    )


async def test_rerank_empty_chunks_returns_empty_without_model_calls() -> None:
    model = FakeListChatModel(responses=[])
    reranker = LangChainReranker(model)

    result = await reranker.rerank("query", [])

    assert result == []


async def test_rerank_scores_and_orders_chunks() -> None:
    reranker = LangChainReranker(FakeListChatModel(responses=["BAD", "PERFECT"]))
    low = make_chunk("low", "unrelated")
    high = make_chunk("high", "directly answers")

    result = await reranker.rerank("query", [low, high])

    assert [chunk.id for chunk in result] == ["high", "low"]
    assert result[0]._relevance_score == 1.0
    assert result[1]._relevance_score == 0.25


async def test_rerank_malformed_score_maps_to_zero() -> None:
    reranker = LangChainReranker(FakeListChatModel(responses=["unknown"]))

    result = await reranker.rerank("query", [make_chunk("doc", "content")])

    assert result[0]._relevance_score == 0.0
