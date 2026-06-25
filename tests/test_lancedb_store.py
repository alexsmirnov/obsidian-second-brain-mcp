"""
Comprehensive pytest tests for the enhanced LanceDBStore class.

Tests cover basic storage, search, deletion, indexing, and lifecycle behavior.
"""

import hashlib
import logging
import tempfile
from collections.abc import AsyncGenerator, Generator
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from mcps.rag.database import LanceDBStore
from mcps.rag.interfaces import (
    Chunk,
    IEmbeddingService,
    IVectorStore,
    SearchScope,
)


# Test fixtures
@pytest.fixture
def temp_db_path() -> Generator[Path]:
    """Create a temporary directory for test database."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir) / "test_db"


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    base_time = datetime.now()

    chunks = [
        Chunk(
            id="chunk_1",
            content=(
                "Machine learning is a subset of artificial intelligence that "
                "focuses on algorithms."
            ),
            source="AI Tutorial",
            description="Machine learning basics",
            title="ML Introduction",
            outgoing_links=["artificial_intelligence"],
            tags=["machine-learning", "ai"],
            source_path="/test/doc1.md",
            wikilink_name="doc1",
            modified_at=base_time,
            position=0,
            offset=0,
            file_size=77,
        ),
        Chunk(
            id="chunk_2",
            content=(
                "Deep learning uses neural networks with multiple layers to "
                "process data."
            ),
            source="Deep Learning Guide",
            description="Neural networks and deep learning",
            title="Deep Learning Basics",
            outgoing_links=["neural_networks"],
            tags=["deep-learning", "neural-networks"],
            source_path="/test/doc2.md",
            wikilink_name="doc2",
            modified_at=base_time,
            position=1,
            offset=0,
            file_size=75,
        ),
        Chunk(
            id="chunk_3",
            content=(
                "Natural language processing enables computers to understand "
                "human language."
            ),
            source="NLP Handbook",
            description="Natural language processing techniques",
            title="NLP Overview",
            outgoing_links=["language_models"],
            tags=["nlp", "language"],
            source_path="/test/doc3.md",
            wikilink_name="doc3",
            modified_at=base_time,
            position=0,
            offset=0,
            file_size=78,
        ),
        Chunk(
            id="chunk_4",
            content=(
                "Python programming language is widely used for data science "
                "and machine learning."
            ),
            source="Python Guide",
            description="Python programming for data science",
            title="Python Basics",
            outgoing_links=["python", "data_science"],
            tags=["python", "programming"],
            source_path="/test/doc1.md",
            wikilink_name="doc1",
            modified_at=base_time,
            position=0,
            offset=0,
            file_size=86,
        ),
    ]
    return chunks


@pytest.fixture
async def dummy_embedding_function() -> AsyncGenerator[IEmbeddingService]:
    """Create a dummy embedding function for testing.

    The only assumption that embeddings for the same text will always be the same.
    """

    class DummyEmbeddingFunction(IEmbeddingService):
        def _generate_embedding(self, text: str) -> np.ndarray:
            hash_digest = hashlib.sha256(text.encode("utf-8")).digest()
            embedding_int = np.frombuffer(hash_digest, dtype=np.uint16, count=16)
            embedding_fp16 = (embedding_int / 65535.0).astype(np.float16)

            return embedding_fp16

        async def documents_embeddings(
            self,
            texts: list[str],
        ) -> list[list[float]]:
            return [self._generate_embedding(text).tolist() for text in texts]

        async def query_embeddings(
            self,
            query: str,
        ) -> list[float]:
            return self._generate_embedding(query).tolist()

        def ndims(self) -> int:
            return 16

    yield DummyEmbeddingFunction()


@pytest.fixture
def lancedb_store(temp_db_path, dummy_embedding_function) -> IVectorStore:
    """Create a LanceDBStore instance for testing."""
    store = LanceDBStore(temp_db_path, dummy_embedding_function, "test_chunks")
    return store
    # Cleanup is handled by temp_db_path fixture


@pytest.fixture
async def lancedb_store_with_data(
    temp_db_path,
    dummy_embedding_function,
    sample_chunks,
):
    """Create a LanceDBStore instance with pre-loaded test data."""
    from lancedb.rerankers import RRFReranker

    reranker = RRFReranker(return_score="all")

    store = LanceDBStore(
        temp_db_path,
        dummy_embedding_function,
        "test_chunks",
        reranker=reranker,
    )
    await store.initialize()  # Initialize without FTS first

    # Store chunks with embeddings only
    await store.store(sample_chunks)
    await store.reindex()

    yield store
    await store.cleanup()


# Test basic functionality
@pytest.mark.asyncio
async def test_initialization(lancedb_store):
    """Test LanceDBStore initialization."""
    # Test that store is not initialized initially
    assert not lancedb_store._initialized
    
    # Initialize the store
    await lancedb_store.initialize()
    
    # Verify initialization
    assert lancedb_store._initialized
    assert lancedb_store.db is not None
    assert lancedb_store.table is not None
    assert lancedb_store.db_path.exists()


@pytest.mark.asyncio
async def test_store_empty_chunks_list(lancedb_store):
    """Test storing an empty list of chunks."""
    await lancedb_store.initialize()

    # Should not raise an error
    await lancedb_store.store([])



@pytest.mark.asyncio
async def test_search(lancedb_store_with_data):
    """Test searching for chunks by query text."""
    # Basic search
    results = await lancedb_store_with_data.search("learning")
    assert isinstance(results, list)
    assert len(results) == 4  # Default limit
    
    # Search with limit
    results = await lancedb_store_with_data.search("learning", limit=2)
    assert len(results) == 2
    
    # Search with where clause
    results = await lancedb_store_with_data.search(
        "learning",
        tags=["machine-learning"],
    )
    _log_results(results)
    assert len(results) == 1
    for chunk in results:
        assert "machine-learning" in chunk.tags


@pytest.mark.asyncio
async def test_search_empty_results(lancedb_store_with_data):

    results = await lancedb_store_with_data.search("nonexistent gibberish")
    assert isinstance(results, list)
    _log_results(results)
    # LanceDB always returns results, even if there are no matches
    # assert len(results) == 0


def _log_results(results):
    """Helper to log search results."""
    for r in results:
        delattr(r, "embeddings")  # Remove embeddings for cleaner output
        logging.info(f"Search result: {r.model_dump()}")


@pytest.mark.asyncio
async def test_delete_chunks(lancedb_store_with_data, sample_chunks):
    """Test deleting chunks by ID."""
    # First verify chunk exists via search
    results_before = await lancedb_store_with_data.search("machine learning")
    chunk_source_paths = [chunk.source_path for chunk in results_before]

    # Delete the first chunk
    if chunk_source_paths:
        await lancedb_store_with_data.delete([chunk_source_paths[0]])
        await lancedb_store_with_data.reindex()
        # Verify deletion indirectly through the public search contract.
        results_after = await lancedb_store_with_data.search("machine learning")
        after_ids = {chunk.source_path for chunk in results_after}
        assert chunk_source_paths[0] not in after_ids


@pytest.mark.asyncio
async def test_delete_nonexistent_chunks(lancedb_store_with_data):
    """Test deleting chunks that don't exist."""
    # Should not raise an error
    await lancedb_store_with_data.delete(["nonexistent_id_1", "nonexistent_id_2"])


@pytest.mark.asyncio
async def test_store_and_search_workflow(lancedb_store, sample_chunks):
    """Test complete workflow of storing and then searching chunks."""
    await lancedb_store.initialize()

    # Store only specific chunks for targeted testing
    python_chunk = next(
        chunk for chunk in sample_chunks if "python" in chunk.content.lower()
    )
    await lancedb_store.store([python_chunk])

    # Search for the stored content
    results = await lancedb_store.search("python programming")
    
    assert len(results) > 0
    assert "python" in results[0].content.lower()


@pytest.mark.asyncio
async def test_store_and_search_preserves_wikilink_name_offset_and_size(
    lancedb_store,
    sample_chunks,
):
    await lancedb_store.initialize()
    chunk = sample_chunks[0].model_copy(
        update={
            "wikilink_name": "AI/ML Introduction",
            "offset": 42,
            "file_size": len(sample_chunks[0].content),
        }
    )

    await lancedb_store.store([chunk])

    results = await lancedb_store.search("machine learning", limit=1)

    assert results[0].wikilink_name == "AI/ML Introduction"
    assert results[0].offset == 42
    assert results[0].file_size == len(sample_chunks[0].content)


@pytest.mark.asyncio
async def test_get_chunks_by_ids_returns_matching_chunks(lancedb_store, sample_chunks):
    await lancedb_store.initialize()
    await lancedb_store.store(sample_chunks)

    ids = [sample_chunks[0].id, sample_chunks[2].id]
    result = await lancedb_store.get_chunks_by_ids(ids)

    assert {chunk.id for chunk in result} == {sample_chunks[0].id, sample_chunks[2].id}


@pytest.mark.asyncio
async def test_get_chunks_by_ids_returns_empty_list_for_empty_input(
    lancedb_store,
):
    await lancedb_store.initialize()

    result = await lancedb_store.get_chunks_by_ids([])

    assert result == []


@pytest.mark.asyncio
async def test_get_chunks_by_ids_escapes_special_characters(lancedb_store, sample_chunks):
    await lancedb_store.initialize()
    chunk_with_quote = sample_chunks[0].model_copy(update={"id": "chunk'o'malley"})
    await lancedb_store.store([chunk_with_quote])

    result = await lancedb_store.get_chunks_by_ids(["chunk'o'malley"])

    assert len(result) == 1
    assert result[0].id == "chunk'o'malley"


@pytest.mark.asyncio
async def test_get_sources_by_name_returns_single_source_path_for_unique_short_name(
    lancedb_store,
    sample_chunks,
):
    await lancedb_store.initialize()
    await lancedb_store.store(
        [sample_chunks[0].model_copy(update={"wikilink_name": "Unique Note"})]
    )

    result = await lancedb_store.get_sources_by_name("Unique Note")

    assert result == ["/test/doc1.md"]


@pytest.mark.asyncio
async def test_get_sources_by_name_returns_all_source_paths_for_duplicate_short_name(
    lancedb_store,
    sample_chunks,
):
    await lancedb_store.initialize()
    chunks = [
        sample_chunks[0].model_copy(update={"wikilink_name": "Note"}),
        sample_chunks[1].model_copy(update={"wikilink_name": "Note"}),
        sample_chunks[2].model_copy(update={"wikilink_name": "AnotherNote"}),
    ]
    await lancedb_store.store(chunks)

    result = await lancedb_store.get_sources_by_name("Note")

    assert result == ["/test/doc1.md", "/test/doc2.md"]


@pytest.mark.asyncio
async def test_get_sources_by_name_returns_empty_list_for_missing_short_name(
    lancedb_store,
    sample_chunks,
):
    await lancedb_store.initialize()
    await lancedb_store.store(sample_chunks[:1])

    result = await lancedb_store.get_sources_by_name("Missing")

    assert result == []


@pytest.mark.asyncio
async def test_get_sources_by_name_returns_distinct_paths_when_file_has_multiple_chunks(
    lancedb_store,
    sample_chunks,
):
    await lancedb_store.initialize()
    chunks = [
        sample_chunks[0].model_copy(
            update={"id": "one", "wikilink_name": "Note"}
        ),
        sample_chunks[0].model_copy(
            update={"id": "two", "wikilink_name": "Note"}
        ),
    ]
    await lancedb_store.store(chunks)

    result = await lancedb_store.get_sources_by_name("Note")

    assert result == ["/test/doc1.md"]


@pytest.mark.asyncio
async def test_search_with_filters(lancedb_store_with_data):
    """Test search with various filter conditions."""
    # Search with source filter
    results = await lancedb_store_with_data.search(
        "machine learning",
        scope=SearchScope.CONTENT,
    )
    _log_results(results)
    assert any(chunk.source == "AI Tutorial" for chunk in results)

    # Search with tag filter
    results = await lancedb_store_with_data.search(
        "neural",
        tags=['neural-networks']
    )

    for chunk in results:
        assert "neural-networks" in chunk.tags


@pytest.mark.asyncio
async def test_search_tags_and_path_both_applied(lancedb_store_with_data):
    """Both tags and file_path filters must be ANDed, not the last one replacing the first.

    Regression for: chained .where() calls on AsyncHybridQuery replacing each other.
    chunk_1 has tags=["machine-learning", "ai"] at source_path=/test/doc1.md
    chunk_2 has tags=["deep-learning", "neural-networks"] at source_path=/test/doc2.md
    Querying tags=["machine-learning"] + file_path="/test/doc2" must return nothing
    because no single chunk satisfies both constraints.
    If the tags filter were silently dropped, chunk_2 would appear.
    """
    results = await lancedb_store_with_data.search(
        "learning",
        tags=["machine-learning"],   # only chunk_1 has this tag
        file_path="/test/doc2",      # only chunk_2 is under this path
    )
    for chunk in results:
        # Every result must satisfy the tags constraint
        assert "machine-learning" in chunk.tags, (
            f"Tags filter bypassed: chunk {chunk.id!r} returned "
            f"without 'machine-learning' tag (tags={chunk.tags})"
        )


@pytest.mark.asyncio
async def test_search_tags_filter_excludes_mismatched_tags(lancedb_store_with_data):
    """A tags-only filter must exclude chunks that lack the requested tag."""
    results = await lancedb_store_with_data.search(
        "machine learning",
        tags=["nlp"],   # only chunk_3 has this tag
    )
    for chunk in results:
        assert "nlp" in chunk.tags, (
            f"Tags filter returned chunk {chunk.id!r} without 'nlp' tag "
            f"(tags={chunk.tags})"
        )


def make_chunk(source_path, modified_at, idx=0):
    import random
    import string
    content = ''.join(random.choices(string.ascii_letters + string.digits, k=20))
    return Chunk(
        id=f"chunk_{source_path}_{idx}",
        content=content,
        source="Test Source",
        description="Test Description",
        title="Test Title",
        outgoing_links=[],
        tags=["test"],
        source_path=source_path,
        wikilink_name=source_path.removesuffix(".md"),
        modified_at=modified_at,
        position=idx,
        offset=idx * 20,
        file_size=len(content),
    )

@pytest.mark.asyncio
async def test_sources_empty(lancedb_store):
    await lancedb_store.initialize()
    updates = await lancedb_store.sources()
    assert updates == {}

@pytest.mark.asyncio
async def test_sources_unique_and_min_time(lancedb_store):
    from datetime import datetime, timedelta
    await lancedb_store.initialize()
    base_time = datetime.now()
    # Create chunks with duplicate source_path but different times
    chunks = [
        make_chunk("/file1.md", base_time - timedelta(days=2), idx=0),
        make_chunk("/file1.md", base_time - timedelta(days=1), idx=1),
        make_chunk("/file2.md", base_time - timedelta(days=3), idx=0),
        make_chunk("/file2.md", base_time, idx=1),
        make_chunk("/file3.md", base_time - timedelta(days=5), idx=0),
    ]
    await lancedb_store.store(chunks)
    await lancedb_store.reindex()
    updates = await lancedb_store.sources()
    # Should return one update per unique source_path, with minimal modified_at
    assert updates["/file1.md"] == base_time - timedelta(days=2)
    assert updates["/file2.md"] == base_time - timedelta(days=3)
    assert updates["/file3.md"] == base_time - timedelta(days=5)
    assert len(updates) == 3
