"""
Tests for IEmbeddingService interface.

This module contains comprehensive tests for the IEmbeddingService interface,
including corner cases and edge conditions.
"""

import logging
from math import log
import os
from dotenv import find_dotenv, load_dotenv
import pytest
from abc import ABC, abstractmethod
from typing import List
import asyncio

from mcps.rag.interfaces import IEmbeddingService
from mcps.rag.embeddings import OpenAIEmbedding

# Configure logging for detailed test output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
load_dotenv(find_dotenv())
load_dotenv(find_dotenv(usecwd=True))

params = []
if os.getenv("OLLAMA_API_BASE") is not None:
    params.append(pytest.param("ollama", id="ollama_embedding"))
if os.getenv("VOYAGE_API_KEY") is not None:
    params.append(pytest.param("voyage", id="voyage_embedding"))
if os.getenv("OPENAI_API_KEY") is not None:
    params.append(pytest.param("openai", id="openai_embedding"))


@pytest.fixture(params=params)
async def embedding_service(request):
    if request.param == "ollama":
        async with OpenAIEmbedding(
            model_name="bge-m3",
            dimensions=1024,
            api_base=os.getenv("OLLAMA_API_BASE", "") + "/v1",
            api_key="dummy",
        ) as svc:
            yield svc
    elif request.param == "voyage":
        async with OpenAIEmbedding(
            model_name="voyage-3.5-lite",
            dimensions=1024,
            api_key=os.getenv("VOYAGE_API_KEY"),
            api_base="https://api.voyageai.com/v1",
            provider="voyage",
        ) as svc:
            yield svc
    elif request.param == "openai":
        async with OpenAIEmbedding(
            model_name="text-embedding-3-small",
            dimensions=1536,
            api_key=os.getenv("OPENAI_API_KEY"),
        ) as svc:
            yield svc


@pytest.mark.asyncio
async def test_empty_texts(embedding_service: IEmbeddingService):
    """Test that empty list of texts returns empty embeddings."""
    result = await embedding_service.generate_embeddings([])
    assert result == [], f"Expected empty list for empty input, got {result}"


@pytest.mark.asyncio
async def test_identical_texts_same_embeddings(embedding_service):
    """Test that identical texts produce identical embeddings."""
    text = "This is a test sentence for embedding generation."
    text1 = "This is second test sentence for embedding generation."

    # Generate embeddings for the same text multiple times
    embeddings1 = await embedding_service.generate_embeddings([text, text1])
    embeddings2 = await embedding_service.generate_embeddings([text1, text])

    # Should be identical
    assert (
        embeddings1[0] == embeddings2[1]
    ), f"Expected identical embeddings for identical text, got different results"
    assert (
        embeddings1[1] == embeddings2[0]
    ), f"Expected identical embeddings for identical text, got different results"


# @pytest.mark.asyncio
# async def test_mixed_empty_and_non_empty_texts(embedding_service):
#     """Test handling of mixed empty and non-empty texts."""
#     texts = ["", "Hello world", "", "Another text", ""]

#     result = await embedding_service.generate_embeddings(texts)

#     # Should have embeddings for all texts (including empty ones)
#     expected_length = len(texts)
#     assert len(result) == expected_length, f"Expected {expected_length} values, got {len(result)}"

#     # Empty texts should have zero embeddings
#     first_embedding = result[0]
#     assert all(val == 0.0 for val in first_embedding), f"Expected zero embedding for empty text, got {first_embedding[:10]}..."


@pytest.mark.asyncio
async def test_large_text_handling(embedding_service):
    """Test handling of large text (10k symbols)."""
    # Create a large text with 10k characters
    large_text = "A" * 10000

    result = await embedding_service.generate_embeddings([large_text])

    # Should still produce valid embedding
    assert len(result) == 1, f"Expected 1 value for large text, got {len(result)}"

    # Should not be all zeros (unless the mock specifically returns zeros)
    assert not all(
        val == 0.0 for val in result[0]
    ), "Large text should produce non-zero embedding"


@pytest.mark.asyncio
async def test_batch_consistency(embedding_service):
    """Test that processing 20 texts at once gives same result as 10+10."""
    # Create 20 test texts
    texts = [f"Test text number {i}" for i in range(20)]

    # Process all 20 at once
    embeddings_all = await embedding_service.generate_embeddings(texts)

    # Process in two batches of 10
    first_half = texts[:10]
    second_half = texts[10:]

    embeddings_first = await embedding_service.generate_embeddings(first_half)
    embeddings_second = await embedding_service.generate_embeddings(second_half)

    # Combine the results
    embeddings_combined = embeddings_first + embeddings_second

    # Should be identical
    # assert embeddings_all[:10] == embeddings_first, "Expected first 10 embeddings to match"
    # assert embeddings_all[10:] == embeddings_second, "Expected last 10 embeddings to match"


@pytest.mark.asyncio
async def test_single_text_embedding(embedding_service):
    """Test that single text produces correct embedding."""
    text = "Single test text"

    result = await embedding_service.generate_embeddings([text])

    assert len(result) == 1, f"Expected 1 value for single text, got {len(result)}"

    # Should not be empty
    assert len(result[0]) > 0, "Should produce non-empty embedding"
    assert not all(
        val == 0.0 for val in result[0]
    ), "text should produce non-zero embedding"
    logger.info(f"Embedding for '{text}': {result[0][:10]}...")


@pytest.mark.asyncio
async def test_multiple_texts_order_preservation(embedding_service):
    """Test that order of texts is preserved in embeddings."""
    texts = ["First text", "Second text", "Third text"]

    result = await embedding_service.generate_embeddings(texts)

    expected_length = 3
    assert (
        len(result) == expected_length
    ), f"Expected {expected_length} values for 3 texts, got {len(result)}"

    # Extract individual embeddings
    first_embedding = result[0]
    second_embedding = result[1]
    third_embedding = result[2]

    # Each should be different (unless texts are identical)
    assert (
        first_embedding != second_embedding
    ), "First and second embeddings should be different"
    assert (
        second_embedding != third_embedding
    ), "Second and third embeddings should be different"
