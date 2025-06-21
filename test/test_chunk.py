"""
Tests for lazy chunking functionality with generators.
"""

import tempfile
from pathlib import Path
from datetime import datetime
from typing import Generator

import pytest

from src.mcps.rag.document_processing import FixedSizeChunker, SemanticChunker
from src.mcps.rag.interfaces import Document, Chunk


class TestLazyChunking:
    """Test cases for lazy chunking with generators."""

    @pytest.fixture
    def sample_document(self):
        """Create a sample document for testing."""
        return Document(
            id="test_doc_123",
            content="# Test Document\n\nThis is a test document with multiple paragraphs.\n\n## Section 1\n\nFirst section content here.\n\n## Section 2\n\nSecond section with more content to test chunking behavior.",
            metadata={"title": "Test Document", "author": "Test Author"},
            outgoing_links=["link1", "link2"],
            tags=["test", "document"],
            source_path=Path("/tmp/test.md"),
            created_at=datetime.now(),
            modified_at=datetime.now()
        )

    @pytest.fixture
    def large_document(self):
        """Create a large document for testing."""
        content = "# Large Document\n\n"
        content += "\n\n".join([f"## Section {i}\n\nContent for section {i}. " + "This is additional content to make the section longer. " * 10 for i in range(20)])
        
        return Document(
            id="large_doc_456",
            content=content,
            metadata={"title": "Large Document"},
            outgoing_links=[],
            tags=["large", "test"],
            source_path=Path("/tmp/large.md"),
            created_at=datetime.now(),
            modified_at=datetime.now()
        )

    @pytest.fixture
    def empty_document(self):
        """Create an empty document for testing."""
        return Document(
            id="empty_doc_789",
            content="",
            metadata={},
            outgoing_links=[],
            tags=[],
            source_path=Path("/tmp/empty.md"),
            created_at=datetime.now(),
            modified_at=datetime.now()
        )

    def test_fixed_size_chunker_returns_generator(self, sample_document):
        """Test that FixedSizeChunker.chunk returns a generator."""
        chunker = FixedSizeChunker(chunk_size=100, overlap=20)
        result = chunker.chunk(sample_document)
        
        # Check that it's a generator
        assert isinstance(result, Generator)
        assert hasattr(result, '__iter__')
        assert hasattr(result, '__next__')

    def test_semantic_chunker_returns_generator(self, sample_document):
        """Test that SemanticChunker.chunk returns a generator."""
        chunker = SemanticChunker(max_chunk_size=200, min_chunk_size=50)
        result = chunker.chunk(sample_document)
        
        # Check that it's a generator
        assert isinstance(result, Generator)
        assert hasattr(result, '__iter__')
        assert hasattr(result, '__next__')

    def test_fixed_size_chunker_lazy_evaluation(self, sample_document):
        """Test that FixedSizeChunker creates chunks lazily."""
        chunker = FixedSizeChunker(chunk_size=50, overlap=10)
        generator = chunker.chunk(sample_document)
        
        # Generator should not have processed anything yet
        # We can't directly test this, but we can test that we can get chunks one by one
        first_chunk = next(generator)
        assert isinstance(first_chunk, Chunk)
        assert first_chunk.content
        assert first_chunk.position == 0
        
        second_chunk = next(generator)
        assert isinstance(second_chunk, Chunk)
        assert second_chunk.position == 1

    def test_semantic_chunker_lazy_evaluation(self, sample_document):
        """Test that SemanticChunker creates chunks lazily."""
        chunker = SemanticChunker(max_chunk_size=200, min_chunk_size=20)
        generator = chunker.chunk(sample_document)
        
        # Get chunks one by one
        first_chunk = next(generator)
        assert isinstance(first_chunk, Chunk)
        assert first_chunk.content
        assert first_chunk.position == 0

    def test_generator_exhaustion(self, sample_document):
        """Test that generators can only be consumed once."""
        chunker = FixedSizeChunker(chunk_size=100, overlap=20)
        generator = chunker.chunk(sample_document)
        
        # Convert to list (exhausts generator)
        chunks_first = list(generator)
        assert len(chunks_first) > 0
        
        # Generator should be exhausted now
        chunks_second = list(generator)
        assert len(chunks_second) == 0
        
        # Need to create a new generator for more chunks
        new_generator = chunker.chunk(sample_document)
        chunks_new = list(new_generator)
        assert len(chunks_new) == len(chunks_first)

    def test_fixed_size_chunker_content_unchanged(self, sample_document):
        """Test that FixedSizeChunker produces the same content as before."""
        chunker = FixedSizeChunker(chunk_size=100, overlap=20)
        chunks = list(chunker.chunk(sample_document))
        
        # Verify basic properties
        assert len(chunks) > 0
        for i, chunk in enumerate(chunks):
            assert isinstance(chunk, Chunk)
            assert chunk.id == f"{sample_document.id}_{i}"
            assert chunk.content.strip()  # Non-empty content
            assert chunk.position == i
            assert chunk.source_path == sample_document.source_path
            assert chunk.metadata == sample_document.metadata
            assert chunk.outgoing_links == sample_document.outgoing_links
            assert chunk.tags == sample_document.tags

    def test_semantic_chunker_content_unchanged(self, sample_document):
        """Test that SemanticChunker produces the same content as before."""
        chunker = SemanticChunker(max_chunk_size=200, min_chunk_size=20)
        chunks = list(chunker.chunk(sample_document))
        
        # Verify basic properties
        assert len(chunks) > 0
        for i, chunk in enumerate(chunks):
            assert isinstance(chunk, Chunk)
            assert chunk.id == f"{sample_document.id}_{i}"
            assert chunk.content.strip()  # Non-empty content
            assert chunk.position == i
            assert chunk.source_path == sample_document.source_path
            assert chunk.metadata == sample_document.metadata
            assert chunk.outgoing_links == sample_document.outgoing_links
            assert chunk.tags == sample_document.tags

    def test_fixed_size_chunker_with_empty_document(self, empty_document):
        """Test FixedSizeChunker with empty document."""
        chunker = FixedSizeChunker(chunk_size=100, overlap=20)
        chunks = list(chunker.chunk(empty_document))
        
        # Should produce no chunks for empty document
        assert len(chunks) == 0

    def test_semantic_chunker_with_empty_document(self, empty_document):
        """Test SemanticChunker with empty document."""
        chunker = SemanticChunker(max_chunk_size=200, min_chunk_size=20)
        chunks = list(chunker.chunk(empty_document))
        
        # Should produce no chunks for empty document
        assert len(chunks) == 0

    def test_fixed_size_chunker_with_large_document(self, large_document):
        """Test FixedSizeChunker with large document."""
        chunker = FixedSizeChunker(chunk_size=500, overlap=50)
        generator = chunker.chunk(large_document)
        
        # Test that we can iterate through chunks
        chunk_count = 0
        for chunk in generator:
            assert isinstance(chunk, Chunk)
            assert len(chunk.content) <= 500 + 100  # Allow some flexibility for word boundaries
            chunk_count += 1
            
            # Test early termination (lazy processing benefit)
            if chunk_count >= 5:
                break
        
        assert chunk_count == 5

    def test_semantic_chunker_with_large_document(self, large_document):
        """Test SemanticChunker with large document."""
        chunker = SemanticChunker(max_chunk_size=1000, min_chunk_size=50)
        generator = chunker.chunk(large_document)
        
        # Test that we can iterate through chunks
        chunk_count = 0
        for chunk in generator:
            assert isinstance(chunk, Chunk)
            chunk_count += 1
            
            # Test early termination (lazy processing benefit)
            if chunk_count >= 3:
                break
        
        assert chunk_count == 3

    def test_chunk_ordering_consistency(self, sample_document):
        """Test that chunk ordering is consistent across multiple generator creations."""
        chunker = FixedSizeChunker(chunk_size=80, overlap=15)
        
        # Create multiple generators and compare first few chunks
        gen1 = chunker.chunk(sample_document)
        gen2 = chunker.chunk(sample_document)
        
        chunk1_1 = next(gen1)
        chunk1_2 = next(gen1)
        
        chunk2_1 = next(gen2)
        chunk2_2 = next(gen2)
        
        # Content should be identical
        assert chunk1_1.content == chunk2_1.content
        assert chunk1_2.content == chunk2_2.content
        assert chunk1_1.position == chunk2_1.position
        assert chunk1_2.position == chunk2_2.position

    def test_memory_efficiency_simulation(self, large_document):
        """Test that generators don't pre-compute all chunks."""
        chunker = FixedSizeChunker(chunk_size=200, overlap=30)
        generator = chunker.chunk(large_document)
        
        # This simulates processing chunks as they come without storing all
        processed_count = 0
        for chunk in generator:
            # Simulate processing each chunk individually
            assert len(chunk.content) > 0
            processed_count += 1
            
            # Early exit to demonstrate we don't need all chunks
            if processed_count >= 10:
                break
        
        assert processed_count == 10

    def test_generator_with_iteration_patterns(self, sample_document):
        """Test different ways of consuming the generator."""
        chunker = FixedSizeChunker(chunk_size=60, overlap=10)
        
        # Test list conversion
        chunks_list = list(chunker.chunk(sample_document))
        assert len(chunks_list) > 0
        
        # Test manual iteration
        generator = chunker.chunk(sample_document)
        manual_chunks = []
        try:
            while True:
                chunk = next(generator)
                manual_chunks.append(chunk)
        except StopIteration:
            pass
        
        # Should have same number of chunks
        assert len(manual_chunks) == len(chunks_list)
        
        # Content should be identical
        for i, (chunk1, chunk2) in enumerate(zip(chunks_list, manual_chunks)):
            assert chunk1.content == chunk2.content
            assert chunk1.position == chunk2.position

    def test_chunker_parameters_affect_generation(self, sample_document):
        """Test that different chunker parameters produce different results."""
        chunker_small = FixedSizeChunker(chunk_size=50, overlap=10)
        chunker_large = FixedSizeChunker(chunk_size=200, overlap=20)
        
        chunks_small = list(chunker_small.chunk(sample_document))
        chunks_large = list(chunker_large.chunk(sample_document))
        
        # Small chunks should produce more chunks
        assert len(chunks_small) > len(chunks_large)
        
        # Verify all chunks have content
        for chunk in chunks_small:
            assert chunk.content.strip()
        for chunk in chunks_large:
            assert chunk.content.strip()

    def test_semantic_chunker_section_splitting(self, sample_document):
        """Test that SemanticChunker properly splits on sections."""
        chunker = SemanticChunker(max_chunk_size=1000, min_chunk_size=10)
        chunks = list(chunker.chunk(sample_document))
        
        # Should have multiple chunks for the document with sections
        assert len(chunks) > 1
        
        # First chunk should contain the main title
        assert "# Test Document" in chunks[0].content
        
        # Subsequent chunks should contain section headers
        section_found = False
        for chunk in chunks[1:]:
            if "## Section" in chunk.content:
                section_found = True
                break
        assert section_found

    @pytest.mark.parametrize("chunk_size,overlap", [
        (50, 10),
        (100, 20),
        (200, 30),
    ])
    def test_fixed_size_chunker_parametrized(self, sample_document, chunk_size, overlap):
        """Parametrized test for different FixedSizeChunker configurations."""
        chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)
        generator = chunker.chunk(sample_document)
        
        # Should return a generator
        assert isinstance(generator, Generator)
        
        # Should produce at least one chunk
        chunks = list(generator)
        assert len(chunks) > 0
        
        # All chunks should have valid properties
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.content.strip()
            assert chunk.id.startswith(sample_document.id)

    @pytest.mark.parametrize("max_size,min_size", [
        (100, 20),
        (500, 50),
        (1000, 100),
    ])
    def test_semantic_chunker_parametrized(self, sample_document, max_size, min_size):
        """Parametrized test for different SemanticChunker configurations."""
        chunker = SemanticChunker(max_chunk_size=max_size, min_chunk_size=min_size)
        generator = chunker.chunk(sample_document)
        
        # Should return a generator
        assert isinstance(generator, Generator)
        
        # Should produce at least one chunk
        chunks = list(generator)
        assert len(chunks) > 0
        
        # All chunks should meet size requirements
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert len(chunk.content.strip()) >= min_size or len(chunks) == 1  # Allow single chunk exception
            assert chunk.id.startswith(sample_document.id)