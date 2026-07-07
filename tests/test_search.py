from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

from mcps.rag.interfaces import Chunk, SearchQuery, SearchScope
from mcps.rag.search import SemanticSearchEngine


def make_chunk(
    doc_id: str,
    content: str = "content",
    relevance_score: float | None = None,
    position: int = 0,
    source_path: str | None = None,
    tags: list[str] | None = None,
    outgoing_links: list[str] | None = None,
    offset: int = 0,
) -> Chunk:
    id_ = f"{doc_id}_{position}"
    chunk = Chunk(
        id=id_,
        content=content,
        title=None,
        description=None,
        source_path=source_path or f"{doc_id}.md",
        wikilink_name=doc_id,
        modified_at=1234.567,
        position=position,
        offset=offset,
        file_size=len(content),
        tags=tags or [],
        outgoing_links=outgoing_links or [],
    )
    if relevance_score is not None:
        object.__setattr__(chunk, "_relevance_score", relevance_score)
    return chunk


def make_vector_store(
    search_results: list[Chunk],
    all_chunks: list[Chunk] | None = None,
):
    vector_store = AsyncMock()
    vector_store.search = AsyncMock(return_value=search_results)
    fetchable = all_chunks if all_chunks is not None else search_results
    vector_store.get_chunks_by_ids = AsyncMock(
        side_effect=lambda ids: [c for c in fetchable if c.id in ids]
    )
    return vector_store


async def test_search_hyde_success_passes_hypothetical_document() -> None:
    vector_store = make_vector_store([make_chunk("match")])
    generator = AsyncMock()
    generator.generate = AsyncMock(return_value="hypothetical answer document")
    engine = SemanticSearchEngine(
        vector_store,
        hypothetical_document_generator=generator,
        neighbor_offset=0,
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
        neighbor_offset=0,
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
        neighbor_offset=0,
    )

    await engine.search(SearchQuery(text="query", tags=[]))

    assert vector_store.search.await_args.kwargs["hypotetical_document"] is None


async def test_search_filters_vector_results_by_min_score() -> None:
    high = make_chunk("high", relevance_score=0.6)
    low = make_chunk("low", relevance_score=0.4)
    engine = SemanticSearchEngine(
        make_vector_store([high, low]),
        min_score=0.5,
        neighbor_offset=0,
    )

    result = await engine.search(SearchQuery(text="query", tags=[]))

    assert result == [high]


async def test_search_chunks_without_relevance_score_pass_min_score_filter() -> None:
    chunk = make_chunk("unscored")
    engine = SemanticSearchEngine(
        make_vector_store([chunk]),
        min_score=0.5,
        neighbor_offset=0,
    )

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
        neighbor_offset=0,
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
        neighbor_offset=0,
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
        neighbor_offset=0,
    )

    await engine.search(SearchQuery(text="query", tags=[]))

    reranker.rerank.assert_awaited_once_with("query", [high])


async def test_search_neighbor_offset_zero_disables_neighbor_fetch() -> None:
    chunk = make_chunk("doc_1", relevance_score=0.8, position=1)
    vector_store = make_vector_store([chunk])
    engine = SemanticSearchEngine(vector_store, neighbor_offset=0)

    result = await engine.search(SearchQuery(text="query", tags=[]))

    assert result == [chunk]
    vector_store.get_chunks_by_ids.assert_not_awaited()


async def test_search_neighbor_offset_one_fetches_adjacent_chunks() -> None:
    center = make_chunk("doc", content="center", relevance_score=0.8, position=1)
    prev_chunk = make_chunk("doc", content="previous", position=0)
    next_chunk = make_chunk("doc", content="next", position=2)
    vector_store = make_vector_store(
        search_results=[center],
        all_chunks=[prev_chunk, center, next_chunk],
    )
    engine = SemanticSearchEngine(vector_store, neighbor_offset=1)

    result = await engine.search(SearchQuery(text="query", tags=[]))

    assert len(result) == 1
    assert result[0].id == "doc_0_2"
    assert "previous" in result[0].content
    assert "center" in result[0].content
    assert "next" in result[0].content
    assert result[0].position == 0
    vector_store.get_chunks_by_ids.assert_awaited_once_with(
        ["doc_0", "doc_1", "doc_2"]
    )


async def test_search_overlapping_neighbors_merge_into_single_window() -> None:
    chunk_2 = make_chunk("doc", content="two", relevance_score=0.8, position=2)
    chunk_3 = make_chunk("doc", content="three", relevance_score=0.7, position=3)
    chunk_4 = make_chunk("doc", content="four", relevance_score=0.6, position=4)
    chunk_5 = make_chunk("doc", content="five", relevance_score=0.5, position=5)
    all_chunks = [chunk_2, chunk_3, chunk_4, chunk_5]
    vector_store = make_vector_store(
        search_results=[chunk_2, chunk_5],
        all_chunks=all_chunks,
    )
    engine = SemanticSearchEngine(vector_store, neighbor_offset=1)

    result = await engine.search(SearchQuery(text="query", tags=[]))

    assert len(result) == 1
    assert result[0].id == "doc_2_5"
    assert result[0].content == "two\n\nthree\n\nfour\n\nfive"
    assert result[0].position == 2
    assert getattr(result[0], "_relevance_score") == 0.8


async def test_search_non_overlapping_neighbors_remain_separate() -> None:
    chunk_2 = make_chunk("doc", content="two", relevance_score=0.8, position=2)
    chunk_6 = make_chunk("doc", content="six", relevance_score=0.7, position=6)
    neighbors = [
        make_chunk("doc", content="zero", position=0),
        make_chunk("doc", content="one", position=1),
        make_chunk("doc", content="three", position=3),
        make_chunk("doc", content="five", position=5),
        make_chunk("doc", content="seven", position=7),
    ]
    all_chunks = [chunk_2, chunk_6] + neighbors
    vector_store = make_vector_store(
        search_results=[chunk_2, chunk_6],
        all_chunks=all_chunks,
    )
    engine = SemanticSearchEngine(vector_store, neighbor_offset=1)

    result = await engine.search(SearchQuery(text="query", tags=[]))

    assert len(result) == 2
    ids = {chunk.id for chunk in result}
    assert ids == {"doc_1_3", "doc_5_7"}


async def test_search_neighbor_boundary_clamps_to_zero() -> None:
    chunk_0 = make_chunk("doc", content="zero", relevance_score=0.8, position=0)
    chunk_1 = make_chunk("doc", content="one", position=1)
    vector_store = make_vector_store(
        search_results=[chunk_0],
        all_chunks=[chunk_0, chunk_1],
    )
    engine = SemanticSearchEngine(vector_store, neighbor_offset=1)

    result = await engine.search(SearchQuery(text="query", tags=[]))

    assert len(result) == 1
    assert result[0].id == "doc_0_1"
    assert result[0].position == 0
    vector_store.get_chunks_by_ids.assert_awaited_once_with(
        ["doc_0", "doc_1"]
    )


async def test_search_neighbor_merging_unions_tags_and_links() -> None:
    center = make_chunk(
        "doc",
        content="center",
        relevance_score=0.8,
        position=1,
        tags=["center"],
        outgoing_links=["center_link"],
    )
    neighbor = make_chunk(
        "doc",
        content="neighbor",
        position=2,
        tags=["neighbor", "center"],
        outgoing_links=["neighbor_link"],
    )
    vector_store = make_vector_store(
        search_results=[center],
        all_chunks=[center, neighbor],
    )
    engine = SemanticSearchEngine(vector_store, neighbor_offset=1)

    result = await engine.search(SearchQuery(text="query", tags=[]))

    assert result[0].tags == ["center", "neighbor"]
    assert result[0].outgoing_links == ["center_link", "neighbor_link"]


async def test_search_neighbor_merged_chunk_drops_embeddings() -> None:
    center = make_chunk("doc", content="center", relevance_score=0.8, position=1)
    neighbor = make_chunk("doc", content="neighbor", position=2)
    center.embeddings = [0.1, 0.2]
    neighbor.embeddings = [0.3, 0.4]
    vector_store = make_vector_store(
        search_results=[center],
        all_chunks=[center, neighbor],
    )
    engine = SemanticSearchEngine(vector_store, neighbor_offset=1)

    result = await engine.search(SearchQuery(text="query", tags=[]))

    assert result[0].embeddings is None
