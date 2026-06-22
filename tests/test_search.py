from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

from mcps.rag.interfaces import Chunk, SearchQuery, SearchScope
from mcps.rag.search import SemanticSearchEngine


def make_chunk(
    id_: str,
    content: str = "content",
    relevance_score: float | None = None,
) -> Chunk:
    chunk = Chunk(
        id=id_,
        content=content,
        title=None,
        description=None,
        source_path=f"{id_}.md",
        wikilink_name=id_,
        modified_at=datetime.now(UTC),
        position=0,
        offset=0,
        file_size=len(content),
    )
    if relevance_score is not None:
        object.__setattr__(chunk, "_relevance_score", relevance_score)
    return chunk


def make_vector_store(chunks: list[Chunk]):
    vector_store = AsyncMock()
    vector_store.search = AsyncMock(return_value=chunks)
    return vector_store


async def test_search_hyde_success_passes_hypothetical_document() -> None:
    vector_store = make_vector_store([make_chunk("match")])
    generator = AsyncMock()
    generator.generate = AsyncMock(return_value="hypothetical answer document")
    engine = SemanticSearchEngine(
        vector_store,
        hypothetical_document_generator=generator,
    )

    await engine.search(SearchQuery(text="query", tags=["tag"], path="notes/"))

    vector_store.search.assert_awaited_once_with(
        query="query",
        hypotetical_document="hypothetical answer document",
        tags=["tag"],
        file_path="notes/",
        scope=SearchScope.ALL,
        limit=25,
    )


async def test_search_hyde_failure_falls_back_to_original_vector_results() -> None:
    chunk = make_chunk("match", relevance_score=0.8)
    vector_store = make_vector_store([chunk])
    generator = AsyncMock()
    generator.generate = AsyncMock(side_effect=RuntimeError("boom"))
    engine = SemanticSearchEngine(
        vector_store,
        hypothetical_document_generator=generator,
    )

    result = await engine.search(SearchQuery(text="query", tags=[]))

    assert result == [chunk]
    assert vector_store.search.await_args.kwargs["hypotetical_document"] is None


async def test_search_blank_hyde_result_uses_original_query_only() -> None:
    vector_store = make_vector_store([make_chunk("match")])
    generator = AsyncMock()
    generator.generate = AsyncMock(return_value="   ")
    engine = SemanticSearchEngine(
        vector_store,
        hypothetical_document_generator=generator,
    )

    await engine.search(SearchQuery(text="query", tags=[]))

    assert vector_store.search.await_args.kwargs["hypotetical_document"] is None


async def test_search_filters_vector_results_by_min_score() -> None:
    high = make_chunk("high", relevance_score=0.6)
    low = make_chunk("low", relevance_score=0.4)
    engine = SemanticSearchEngine(make_vector_store([high, low]), min_score=0.5)

    result = await engine.search(SearchQuery(text="query", tags=[]))

    assert result == [high]


async def test_search_chunks_without_relevance_score_pass_min_score_filter() -> None:
    chunk = make_chunk("unscored")
    engine = SemanticSearchEngine(make_vector_store([chunk]), min_score=0.5)

    result = await engine.search(SearchQuery(text="query", tags=[]))

    assert result == [chunk]


async def test_search_successful_rerank_returns_reranked_chunks() -> None:
    first = make_chunk("first", relevance_score=0.8)
    second = make_chunk("second", relevance_score=0.9)
    reranker = AsyncMock()
    reranker.rerank = AsyncMock(return_value=[second, first])
    engine = SemanticSearchEngine(
        make_vector_store([first, second]),
        reranker=reranker,
    )

    result = await engine.search(SearchQuery(text="query", tags=[]))

    assert result == [second, first]


async def test_search_rerank_failure_returns_filtered_vector_results() -> None:
    high = make_chunk("high", relevance_score=0.8)
    low = make_chunk("low", relevance_score=0.2)
    reranker = AsyncMock()
    reranker.rerank = AsyncMock(side_effect=RuntimeError("boom"))
    engine = SemanticSearchEngine(
        make_vector_store([high, low]),
        reranker=reranker,
        min_score=0.5,
    )

    result = await engine.search(SearchQuery(text="query", tags=[]))

    assert result == [high]


async def test_search_reranker_receives_only_min_score_filtered_candidates() -> None:
    high = make_chunk("high", relevance_score=0.8)
    low = make_chunk("low", relevance_score=0.2)
    reranker = AsyncMock()
    reranker.rerank = AsyncMock(return_value=[high])
    engine = SemanticSearchEngine(
        make_vector_store([high, low]),
        reranker=reranker,
        min_score=0.5,
    )

    await engine.search(SearchQuery(text="query", tags=[]))

    reranker.rerank.assert_awaited_once_with("query", [high])
