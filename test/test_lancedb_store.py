"""
Comprehensive pytest tests for the enhanced LanceDBStore class.

Tests cover:
- Basic functionality (initialization, storage, search, deletion)
- Pydantic model integration
- OpenAI embedding functionality
- Full text search (FTS) with Tantivy-based indexing
- Hybrid search capabilities
- Error handling and edge cases
"""

import asyncio
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

from numpy._core.multiarray import array
import pytest

from src.mcps.rag.database import LanceDBStore
from src.mcps.rag.interfaces import Chunk, Metadata
from lancedb.embeddings import EmbeddingFunction


# Test fixtures
@pytest.fixture
def temp_db_path():
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
            content="Machine learning is a subset of artificial intelligence that focuses on algorithms.",
            metadata=Metadata(
                source="AI Tutorial",
                description="Machine learning basics",
                title="ML Introduction",
                created_at=base_time,
                modified_at=base_time
            ),
            outgoing_links=["artificial_intelligence"],
            tags=["machine-learning", "ai"],
            source_path="/test/doc1.md",
            created_at=base_time,
            modified_at=base_time,
            position=0,
        ),
        Chunk(
            id="chunk_2", 
            content="Deep learning uses neural networks with multiple layers to process data.",
            metadata=Metadata(
                source="Deep Learning Guide",
                description="Neural networks and deep learning",
                title="Deep Learning Basics",
                created_at=base_time,
                modified_at=base_time
            ),
            outgoing_links=["neural_networks"],
            tags=["deep-learning", "neural-networks"],
            source_path="/test/doc2.md",
            created_at=base_time,
            modified_at=base_time,
            position=1,
        ),
        Chunk(
            id="chunk_3",
            content="Natural language processing enables computers to understand human language.",
            metadata=Metadata(
                source="NLP Handbook",
                description="Natural language processing techniques",
                title="NLP Overview",
                created_at=base_time,
                modified_at=base_time
            ),
            outgoing_links=["language_models"],
            tags=["nlp", "language"],
            source_path="/test/doc3.md",
            created_at=base_time,
            modified_at=base_time,
            position=0,
        ),
        Chunk(
            id="chunk_4",
            content="Python programming language is widely used for data science and machine learning.",
            metadata=Metadata(
                source="Python Guide",
                description="Python programming for data science",
                title="Python Basics",
                created_at=base_time,
                modified_at=base_time
            ),
            outgoing_links=["python", "data_science"],
            tags=["python", "programming"],
            source_path="/test/doc4.md",
            created_at=base_time,
            modified_at=base_time,
            position=0,
        )
    ]
    return chunks


@pytest.fixture
def sample_chunks_no_embeddings():
    """Create sample chunks without embeddings for testing embedding generation."""
    base_time = datetime.now()
    
    chunks = [
        Chunk(
            id="chunk_no_emb_1",
            content="Artificial intelligence is transforming various industries.",
            metadata=Metadata(
                source="AI Overview",
                description="Artificial intelligence applications",
                title="AI in Industry",
                created_at=base_time,
                modified_at=base_time
            ),
            outgoing_links=[],
            tags=["ai", "industry"],
            source_path="/test/doc_no_emb1.md",
            created_at=base_time,
            modified_at=base_time,
            position=0,
        ),
        Chunk(
            id="chunk_no_emb_2",
            content="Computer vision enables machines to interpret visual information.",
            metadata=Metadata(
                source="Computer Vision Guide",
                description="Machine vision and image processing",
                title="Computer Vision Basics",
                created_at=base_time,
                modified_at=base_time
            ),
            outgoing_links=[],
            tags=["computer-vision", "ai"],
            source_path="/test/doc_no_emb2.md",
            created_at=base_time,
            modified_at=base_time,
            position=0,
        )
    ]
    return chunks

@pytest.fixture
def dummy_embedding_function():
    """Create a dummy embedding function for testing."""
    class DummyEmbeddingFunction(EmbeddingFunction):
        def compute_query_embeddings(self, *args, **kwargs):
            # Return a fixed embedding for testing
            return [[0.1, 0.2, 0.3, 0.4, 0.5]]
        def ndims(self):
            return 5

        def compute_source_embeddings(self, *args, **kwargs) -> list[Any | None]:
            raise NotImplementedError

    
    return DummyEmbeddingFunction()

@pytest.fixture
async def lancedb_store(temp_db_path,dummy_embedding_function):
    """Create a LanceDBStore instance for testing."""
    store = LanceDBStore(temp_db_path, dummy_embedding_function, "test_chunks")
    yield store
    # Cleanup is handled by temp_db_path fixture


@pytest.fixture
async def lancedb_store_with_data(temp_db_path, dummy_embedding_function, sample_chunks):
    """Create a LanceDBStore instance with pre-loaded test data."""
    store = LanceDBStore(temp_db_path, dummy_embedding_function, "test_chunks")
    await store.initialize(create_fts_index=False)  # Initialize without FTS first
    
    # Store chunks with embeddings only
    chunks_with_embeddings = [chunk for chunk in sample_chunks if chunk.embeddings is not None]
    await store.store(chunks_with_embeddings)
    
    yield store


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
async def test_store_chunks_with_embeddings(lancedb_store, sample_chunks):
    """Test storing chunks that already have embeddings."""
    await lancedb_store.initialize(create_fts_index=False)
    
    # Store chunks with embeddings
    chunks_with_embeddings = [chunk for chunk in sample_chunks if chunk.embeddings is not None]
    await lancedb_store.store(chunks_with_embeddings)
    
    # Verify chunks were stored by searching
    query_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    results = await lancedb_store.search(query_embedding, top_k=5)
    
    assert len(results) == len(chunks_with_embeddings)
    assert all(isinstance(result, Chunk) for result in results)


@pytest.mark.asyncio
async def test_store_empty_chunks_list(lancedb_store):
    """Test storing an empty list of chunks."""
    await lancedb_store.initialize()
    
    # Should not raise an error
    await lancedb_store.store([])


@pytest.mark.asyncio
async def test_search_vector(lancedb_store_with_data):
    """Test vector similarity search."""
    query_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    results = await lancedb_store_with_data.search(query_embedding, top_k=3)
    
    assert len(results) <= 3
    assert all(isinstance(result, Chunk) for result in results)
    assert all(result.embeddings is not None for result in results)


@pytest.mark.asyncio
async def test_search_empty_database(lancedb_store):
    """Test searching in an empty database."""
    await lancedb_store.initialize()
    
    query_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    results = await lancedb_store.search(query_embedding, top_k=5)
    
    assert results == []


@pytest.mark.asyncio
async def test_delete_chunks(lancedb_store_with_data, sample_chunks):
    """Test chunk deletion functionality."""
    # Delete specific chunks
    chunk_ids_to_delete = ["chunk_1", "chunk_2"]
    await lancedb_store_with_data.delete(chunk_ids_to_delete)
    
    # Verify chunks were deleted by searching
    query_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    results = await lancedb_store_with_data.search(query_embedding, top_k=10)
    
    # Should have fewer results now
    result_ids = [result.id for result in results]
    for deleted_id in chunk_ids_to_delete:
        assert deleted_id not in result_ids


@pytest.mark.asyncio
async def test_delete_nonexistent_chunks(lancedb_store_with_data):
    """Test deleting chunks that don't exist."""
    # Should not raise an error
    await lancedb_store_with_data.delete(["nonexistent_1", "nonexistent_2"])


# Test Pydantic model integration
@pytest.mark.asyncio
async def test_pydantic_validation(lancedb_store):
    """Test that Pydantic model validation works correctly."""
    await lancedb_store.initialize()
    
    # Create a chunk with valid data
    valid_chunk = Chunk(
        id="valid_chunk",
        content="Test content",
        metadata=Metadata(
            source="test_source",
            description="test description",
            title="test title",
            created_at=datetime.now(),
            modified_at=datetime.now()
        ),
        outgoing_links=[],
        tags=["test"],
        source_path="/test/valid.md",
        created_at=datetime.now(),
        modified_at=datetime.now(),
        position=0,
    )
    
    # Should store successfully
    await lancedb_store.store([valid_chunk])
    
    # Verify it was stored correctly
    results = await lancedb_store.search([0.1, 0.2, 0.3], top_k=1)
    assert len(results) == 1
    assert results[0].id == "valid_chunk"


@pytest.mark.asyncio
async def test_chunk_serialization(lancedb_store_with_data):
    """Test proper serialization/deserialization of chunks."""
    query_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    results = await lancedb_store_with_data.search(query_embedding, top_k=1)
    
    assert len(results) >= 1
    chunk = results[0]
    
    # Verify all fields are properly deserialized
    assert isinstance(chunk.id, str)
    assert isinstance(chunk.content, str)
    assert isinstance(chunk.metadata, dict)
    assert isinstance(chunk.outgoing_links, list)
    assert isinstance(chunk.tags, list)
    assert isinstance(chunk.source_path, str)
    assert isinstance(chunk.created_at, datetime)
    assert isinstance(chunk.modified_at, datetime)
    assert isinstance(chunk.position, int)
    assert isinstance(chunk.embeddings, list)


@pytest.mark.asyncio
async def test_invalid_data_handling(lancedb_store):
    """Test handling of invalid data during search result parsing."""
    await lancedb_store.initialize()
    
    # Store a valid chunk first
    valid_chunk = Chunk(
        id="valid_chunk",
        content="Test content",
        metadata=Metadata(
            created_at=datetime.now(),
            modified_at=datetime.now()
        ),
        outgoing_links=[],
        tags=[],
        source_path="/test/valid.md",
        created_at=datetime.now(),
        modified_at=datetime.now(),
        position=0,
    )
    await lancedb_store.store([valid_chunk])
    
    # Mock the table search to return invalid data
    with patch.object(lancedb_store.table, 'search') as mock_search:
        mock_search.return_value.limit.return_value.to_list.return_value = [
            {"id": "invalid", "content": "test"},  # Missing required fields
            valid_chunk.model_dump()  # Valid chunk
        ]
        
        results = await lancedb_store.search([0.1, 0.2, 0.3], top_k=2)
        
        # Should only return the valid chunk, invalid one should be filtered out
        assert len(results) == 1
        assert results[0].id == "valid_chunk"


# Test OpenAI embedding integration
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OpenAI API key not available"
)
@pytest.mark.asyncio
async def test_embedding_generation(temp_db_path, sample_chunks_no_embeddings):
    """Test automatic embedding generation with real OpenAI API."""
    store = LanceDBStore(temp_db_path, "test_chunks", embedding_model="text-embedding-ada-002")
    await store.initialize(create_fts_index=False)
    
    # Store chunks without embeddings - should generate them automatically
    await store.store(sample_chunks_no_embeddings)
    
    # Verify embeddings were generated by searching
    # Use a simple query embedding (will be generated by the embedding function)
    if store.embedding_function:
        query_embedding = store.embedding_function.compute_query_embeddings("artificial intelligence")[0]
        results = await store.search(query_embedding, top_k=5)
        
        assert len(results) == len(sample_chunks_no_embeddings)
        assert all(result.embeddings is not None for result in results)
        assert all(result.embeddings is not None and len(result.embeddings) > 0 for result in results)


@pytest.mark.asyncio
async def test_embedding_error_handling(temp_db_path, sample_chunks_no_embeddings):
    """Test graceful handling of embedding generation failures."""
    # Create store with invalid embedding model to trigger errors
    store = LanceDBStore(temp_db_path, "test_chunks", embedding_model="invalid-model")
    await store.initialize(create_fts_index=False)
    
    # Mock the embedding function to raise an error
    if store.embedding_function:
        with patch.object(store.embedding_function, 'compute_query_embeddings', side_effect=Exception("API Error")):
            # Should not raise an error, but should skip chunks
            await store.store(sample_chunks_no_embeddings)
            
            # Verify no chunks were stored due to embedding failures
            results = await store.search([0.1, 0.2, 0.3], top_k=5)
            assert len(results) == 0


@pytest.mark.asyncio
async def test_chunks_with_existing_embeddings(lancedb_store, sample_chunks):
    """Test that existing embeddings are preserved and not regenerated."""
    await lancedb_store.initialize(create_fts_index=False)
    
    # Store chunks that already have embeddings
    chunks_with_embeddings = [chunk for chunk in sample_chunks if chunk.embeddings is not None]
    original_embeddings = {chunk.id: chunk.embeddings[:] for chunk in chunks_with_embeddings}
    
    await lancedb_store.store(chunks_with_embeddings)
    
    # Retrieve and verify embeddings weren't changed
    query_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    results = await lancedb_store.search(query_embedding, top_k=10)
    
    for result in results:
        if result.id in original_embeddings:
            assert result.embeddings == original_embeddings[result.id]


@pytest.mark.asyncio
async def test_no_embedding_function_handling(temp_db_path, sample_chunks_no_embeddings):
    """Test behavior when no embedding function is available."""
    # Create store and manually disable embedding function
    store = LanceDBStore(temp_db_path, "test_chunks")
    store.embedding_function = None
    await store.initialize(create_fts_index=False)
    
    # Try to store chunks without embeddings
    await store.store(sample_chunks_no_embeddings)
    
    # Should not store any chunks since they have no embeddings and no function to generate them
    results = await store.search([0.1, 0.2, 0.3], top_k=5)
    assert len(results) == 0


# Test FTS functionality
@pytest.mark.asyncio
async def test_create_fts_index(lancedb_store_with_data):
    """Test FTS index creation."""
    # Create FTS index
    await lancedb_store_with_data.create_fts_index()
    
    # Index creation should not raise an error
    # Actual functionality will be tested in text search tests


@pytest.mark.asyncio
async def test_create_fts_index_custom_columns(lancedb_store_with_data):
    """Test FTS index creation on custom columns."""
    # Create FTS index on multiple columns
    await lancedb_store_with_data.create_fts_index(
        columns=["content", "tags"],
        tokenizer_name="en_stem"
    )
    
    # Should not raise an error


@pytest.mark.asyncio
async def test_text_search(lancedb_store_with_data):
    """Test full text search functionality."""
    # Create FTS index first
    await lancedb_store_with_data.create_fts_index()
    
    # Perform text search
    results = await lancedb_store_with_data.search_text("machine learning", limit=5)
    
    # Should find relevant chunks
    assert isinstance(results, list)
    assert all(isinstance(result, Chunk) for result in results)
    
    # Check that results contain relevant content
    if results:
        content_texts = [result.content.lower() for result in results]
        # At least one result should contain "machine" or "learning"
        assert any("machine" in text or "learning" in text for text in content_texts)


@pytest.mark.asyncio
async def test_text_search_with_filtering(lancedb_store_with_data):
    """Test FTS with where clauses."""
    # Create FTS index first
    await lancedb_store_with_data.create_fts_index()
    
    # Perform text search with filtering
    results = await lancedb_store_with_data.search_text(
        "learning",
        limit=10,
        where="tags LIKE '%ai%'"
    )
    
    # Should return results that match both text and filter
    assert isinstance(results, list)
    assert all(isinstance(result, Chunk) for result in results)


@pytest.mark.asyncio
async def test_text_search_no_index(lancedb_store_with_data):
    """Test text search behavior when no FTS index exists."""
    # Don't create FTS index, try to search
    results = await lancedb_store_with_data.search_text("machine learning", limit=5)
    
    # Should handle gracefully (might return empty results or work with basic search)
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_text_search_empty_query(lancedb_store_with_data):
    """Test text search with empty query."""
    await lancedb_store_with_data.create_fts_index()
    
    results = await lancedb_store_with_data.search_text("", limit=5)
    
    # Should handle empty query gracefully
    assert isinstance(results, list)


# Test hybrid search functionality
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OpenAI API key not available for embedding generation"
)
@pytest.mark.asyncio
async def test_hybrid_search_with_embeddings(temp_db_path, sample_chunks):
    """Test hybrid search combining vector and text search."""
    store = LanceDBStore(temp_db_path, "test_chunks", embedding_model="text-embedding-ada-002")
    await store.initialize()
    
    # Store chunks with embeddings
    chunks_with_embeddings = [chunk for chunk in sample_chunks if chunk.embeddings is not None]
    await store.store(chunks_with_embeddings)
    
    # Perform hybrid search
    results = await store.hybrid_search("machine learning", top_k=3, text_limit=5)
    
    assert isinstance(results, list)
    assert all(isinstance(result, Chunk) for result in results)
    
    # Results should be unique (no duplicates)
    result_ids = [result.id for result in results]
    assert len(result_ids) == len(set(result_ids))


@pytest.mark.asyncio
async def test_hybrid_search_no_embedding_function(lancedb_store_with_data):
    """Test hybrid search when embedding function is not available."""
    # Disable embedding function
    lancedb_store_with_data.embedding_function = None
    
    # Create FTS index for text search
    await lancedb_store_with_data.create_fts_index()
    
    # Perform hybrid search (should fall back to text search only)
    results = await lancedb_store_with_data.hybrid_search("learning", top_k=3, text_limit=5)
    
    assert isinstance(results, list)
    # Should still return results from text search


@pytest.mark.asyncio
async def test_hybrid_search_empty_database(lancedb_store):
    """Test hybrid search on empty database."""
    await lancedb_store.initialize()
    
    results = await lancedb_store.hybrid_search("test query", top_k=5, text_limit=10)
    
    assert results == []


# Test error handling and edge cases
@pytest.mark.asyncio
async def test_operations_before_initialization():
    """Test that operations work correctly when called before explicit initialization."""
    temp_dir = tempfile.mkdtemp()
    try:
        store = LanceDBStore(Path(temp_dir) / "test_db", "test_chunks")
        
        # Operations should auto-initialize
        await store.store([])  # Empty list should work
        
        assert store._initialized
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


@pytest.mark.asyncio
async def test_invalid_search_queries(lancedb_store_with_data):
    """Test handling of invalid search inputs."""
    # Test with invalid embedding dimensions
    invalid_embedding = [0.1, 0.2]  # Too short
    results = await lancedb_store_with_data.search(invalid_embedding, top_k=5)
    
    # Should handle gracefully and return empty results
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_database_persistence(temp_db_path, sample_chunks):
    """Test that data persists across sessions."""
    # Create first store instance and add data
    store1 = LanceDBStore(temp_db_path, "test_chunks")
    await store1.initialize(create_fts_index=False)
    
    chunks_with_embeddings = [chunk for chunk in sample_chunks if chunk.embeddings is not None]
    await store1.store(chunks_with_embeddings)
    
    # Create second store instance (simulating new session)
    store2 = LanceDBStore(temp_db_path, "test_chunks")
    await store2.initialize(create_fts_index=False)
    
    # Data should still be accessible
    query_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    results = await store2.search(query_embedding, top_k=5)
    
    assert len(results) == len(chunks_with_embeddings)


# Test configuration options
@pytest.mark.asyncio
async def test_custom_embedding_model(temp_db_path):
    """Test initialization with different embedding models."""
    # Test with different model name
    store = LanceDBStore(temp_db_path, "test_chunks", embedding_model="text-embedding-3-small")
    
    # Should initialize without error (even if API key is not available)
    await store.initialize(create_fts_index=False)
    
    assert store.embedding_model == "text-embedding-3-small"


@pytest.mark.asyncio
async def test_fts_configuration_options(lancedb_store_with_data):
    """Test FTS configuration with different options."""
    # Test different tokenizer
    await lancedb_store_with_data.create_fts_index(
        columns=["content"],
        use_tantivy=True,
        tokenizer_name="default",
        replace=True
    )
    
    # Should not raise an error


@pytest.mark.asyncio
async def test_initialization_options(temp_db_path):
    """Test various initialization parameters."""
    store = LanceDBStore(temp_db_path, "custom_table", embedding_model="text-embedding-ada-002")
    
    # Test initialization without FTS index
    await store.initialize(create_fts_index=False)
    
    assert store.table_name == "custom_table"
    assert store._initialized


@pytest.mark.asyncio
async def test_table_creation_and_reopening(temp_db_path):
    """Test that tables are created correctly and can be reopened."""
    # Create first store and initialize
    store1 = LanceDBStore(temp_db_path, "test_table")
    await store1.initialize(create_fts_index=False)
    
    # Create second store with same table name - should open existing table
    store2 = LanceDBStore(temp_db_path, "test_table")
    await store2.initialize(create_fts_index=False)
    
    # Both should be initialized successfully
    assert store1._initialized
    assert store2._initialized


# Test concurrent operations
@pytest.mark.asyncio
async def test_concurrent_operations(lancedb_store_with_data):
    """Test concurrent search operations."""
    query_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    # Run multiple searches concurrently
    tasks = [
        lancedb_store_with_data.search(query_embedding, top_k=3)
        for _ in range(5)
    ]
    
    results = await asyncio.gather(*tasks)
    
    # All searches should complete successfully
    assert len(results) == 5
    assert all(isinstance(result, list) for result in results)


# Test large batch operations
@pytest.mark.asyncio
async def test_large_batch_storage(lancedb_store):
    """Test storing a large batch of chunks."""
    await lancedb_store.initialize(create_fts_index=False)
    
    # Create a large batch of chunks
    base_time = datetime.now()
    large_batch = []
    
    for i in range(100):
        chunk = Chunk(
            id=f"batch_chunk_{i}",
            content=f"Test content for chunk {i} with various keywords like machine learning and AI.",
            metadata=Metadata(
                source=f"batch_source_{i}",
                description=f"Batch chunk {i}",
                title=f"Batch Item {i}",
                created_at=base_time,
                modified_at=base_time
            ),
            outgoing_links=[],
            tags=[f"tag_{i % 10}"],
            source_path=f"/test/batch_{i}.md",
            created_at=base_time,
            modified_at=base_time,
            position=i,
        )
        large_batch.append(chunk)
    
    # Store large batch
    await lancedb_store.store(large_batch)
    
    # Verify storage
    query_embedding = [0.1, 0.1, 0.1, 0.1, 0.1]
    results = await lancedb_store.search(query_embedding, top_k=50)
    
    assert len(results) <= 50  # Should return up to top_k results
    assert len(results) > 0  # Should find some results


if __name__ == "__main__":
    pytest.main([__file__])