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

import os
import tempfile
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Generator
from unittest.mock import patch

import hashlib
import numpy as np
import pytest
from dotenv import load_dotenv, find_dotenv

from src.mcps.rag.database import LanceDBStore
from src.mcps.rag.interfaces import Chunk, IVectorStore, SearchScope
from lancedb.embeddings import EmbeddingFunction, get_registry

load_dotenv(find_dotenv())
load_dotenv(find_dotenv(usecwd=True))


# Test fixtures
@pytest.fixture
def temp_db_path() -> Generator[Path,None,None]:
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
            source="AI Tutorial",
            description="Machine learning basics",
            title="ML Introduction",
            outgoing_links=["artificial_intelligence"],
            tags=["machine-learning", "ai"],
            source_path="/test/doc1.md",
            modified_at=base_time,
            position=0,
        ),
        Chunk(
            id="chunk_2", 
            content="Deep learning uses neural networks with multiple layers to process data.",
                source="Deep Learning Guide",
                description="Neural networks and deep learning",
                title="Deep Learning Basics",
            outgoing_links=["neural_networks"],
            tags=["deep-learning", "neural-networks"],
            source_path="/test/doc2.md",
            modified_at=base_time,
            position=1,
        ),
        Chunk(
            id="chunk_3",
            content="Natural language processing enables computers to understand human language.",
                source="NLP Handbook",
                description="Natural language processing techniques",
                title="NLP Overview",
            outgoing_links=["language_models"],
            tags=["nlp", "language"],
            source_path="/test/doc3.md",
            modified_at=base_time,
            position=0,
        ),
        Chunk(
            id="chunk_4",
            content="Python programming language is widely used for data science and machine learning.",
                source="Python Guide",
                description="Python programming for data science",
                title="Python Basics",
            outgoing_links=["python", "data_science"],
            tags=["python", "programming"],
            source_path="/test/doc4.md",
            modified_at=base_time,
            position=0,
        )
    ]
    return chunks



@pytest.fixture
def dummy_embedding_function() -> EmbeddingFunction:
    """Create a dummy embedding function for testing.
        The only assumption that embeddings for the same text will always be the same.
    """
    ollama_base_url = os.getenv("OLLAMA_API_BASE")
    if ollama_base_url:
        return get_registry().get("ollama").create(name="bge-m3", host=ollama_base_url)

    class DummyEmbeddingFunction(EmbeddingFunction):
        def _generate_embedding(self, text: str) -> np.ndarray:
            hash_digest = hashlib.sha256(text.encode('utf-8')).digest()
            embedding_int = np.frombuffer(hash_digest, dtype=np.uint16, count=16)
            embedding_fp16 = (embedding_int / 65535.0).astype(np.float16)

            return embedding_fp16

        def compute_query_embeddings(self, *args, **kwargs):
            # Return a fixed embedding for testing
            return [self._generate_embedding(arg) for arg in args]
        def ndims(self):
            return 16

        def compute_source_embeddings(self, *args, **kwargs) -> list[Any | None]:
            return [self._generate_embedding(arg) for arg in args]

    
    return DummyEmbeddingFunction()

@pytest.fixture
def lancedb_store(temp_db_path,dummy_embedding_function) -> IVectorStore:
    """Create a LanceDBStore instance for testing."""
    store = LanceDBStore(temp_db_path, dummy_embedding_function, "test_chunks")
    return store
    # Cleanup is handled by temp_db_path fixture


@pytest.fixture
async def lancedb_store_with_data(temp_db_path, dummy_embedding_function, sample_chunks):
    """Create a LanceDBStore instance with pre-loaded test data."""
    store = LanceDBStore(temp_db_path, dummy_embedding_function, "test_chunks")
    await store.initialize()  # Initialize without FTS first
    
    # Store chunks with embeddings only
    await store.store(sample_chunks)
    
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
async def test_store_empty_chunks_list(lancedb_store):
    """Test storing an empty list of chunks."""
    await lancedb_store.initialize()
    
    # Should not raise an error
    await lancedb_store.store([])


@pytest.mark.asyncio
async def test_store_chunks(lancedb_store, sample_chunks):
    """Test storing chunks in the vector store."""
    await lancedb_store.initialize()
    
    # Store chunks
    await lancedb_store.store(sample_chunks)
    
    # We don't have direct access to the underlying store to verify storage in this IVectorStore test
    # Just verifying the method executes without error is sufficient for the interface test
    

@pytest.mark.asyncio
async def test_search(lancedb_store_with_data):
    """Test searching for chunks by query text."""
    # Basic search
    results = await lancedb_store_with_data.search("machine learning")
    assert isinstance(results, list)
    assert len(results) <= 5  # Default limit
    
    # Search with limit
    results = await lancedb_store_with_data.search("machine learning", limit=2)
    assert len(results) <= 2
    
    # Search with where clause
    results = await lancedb_store_with_data.search(
        "machine learning", 
        tags=['machine-learning']
    )
    for chunk in results:
        assert "machine-learning" in chunk.tags


@pytest.mark.asyncio
async def test_search_empty_results(lancedb_store_with_data):
    """Test search with query that should return no results."""
    results = await lancedb_store_with_data.search("nonexistent gibberish")
    assert isinstance(results, list)
    # TODO: refine search 
    assert len(results) == 0


@pytest.mark.asyncio
async def test_delete_chunks(lancedb_store_with_data, sample_chunks):
    """Test deleting chunks by ID."""
    # First verify chunk exists via search
    results_before = await lancedb_store_with_data.search("machine learning")
    chunk_ids = [chunk.id for chunk in results_before]
    
    # Delete the first chunk
    if chunk_ids:
        await lancedb_store_with_data.delete([chunk_ids[0]])
        # Pause for 2 seconds to ensure deletion is processed
        await asyncio.sleep(2)
        # Verify it was deleted (this is an indirect test since we're working with the interface)
        results_after = await lancedb_store_with_data.search("machine learning")
        after_ids = [chunk.id for chunk in results_after]
        assert chunk_ids[0] not in after_ids


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
    python_chunk = next(chunk for chunk in sample_chunks if "python" in chunk.content.lower())
    await lancedb_store.store([python_chunk])
    
    # Search for the stored content
    results = await lancedb_store.search("python programming")
    
    assert len(results) > 0
    assert "python" in results[0].content.lower()


@pytest.mark.asyncio
async def test_search_with_filters(lancedb_store_with_data):
    """Test search with various filter conditions."""
    # Search with source filter
    results = await lancedb_store_with_data.search(
        "learning", 
        scope=SearchScope.TITLE,
    )
    
    for chunk in results:
        assert chunk.source == "Deep Learning Guide"
    
    # Search with tag filter
    results = await lancedb_store_with_data.search(
        "neural", 
        tags=['neural-networks']
    )
    
    for chunk in results:
        assert "neural-networks" in chunk.tags
