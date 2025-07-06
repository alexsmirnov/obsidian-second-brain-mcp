import os
from dotenv import load_dotenv, find_dotenv
import logging

import numpy as np
import pyarrow as pa
import pytest

from mcps import main
from mcps.rag.ollama_reranker import OllamaReranker

# Configure logging for detailed test output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv(find_dotenv())
load_dotenv(find_dotenv(usecwd=True))

# Skip all tests if OLLAMA_API_BASE is not set
pytestmark = pytest.mark.skipif(
    os.getenv("OLLAMA_API_BASE") is None,
    reason="OLLAMA_API_BASE environment variable not set. Set it to run Ollama reranker tests.",
)


class TestOllamaRerankerHybrid:
    """Test class for OllamaReranker.rerank_hybrid method"""

    @pytest.fixture
    def ollama_base_url(self):
        """Get Ollama base URL from environment"""
        return os.getenv("OLLAMA_API_BASE", "http://localhost:11434")

    @pytest.fixture
    def reranker(self, ollama_base_url):
        """Create OllamaReranker instance with mocked client"""
        return OllamaReranker(
            model_name="phi4-mini:latest",
            ollama_base_url=ollama_base_url,
            embedding_model="bge-m3:latest",
            return_score="relevance",
            weight=1.0,
        )

    @pytest.fixture
    def sample_vector_results(self):
        """Create sample vector search results"""
        return pa.Table.from_pydict(
            {
                "_rowid": [1, 2, 3],
                "content": [
                    "Cream of wild mushroom recipe", 
                    "Ollama is the service to run large language models",
                    "RED6k is a comprehensive dataset containing **~6,000 samples** across **10 domains**" +
                    " created by **Aizip** for evaluating language models as summarizers in retrieval-augmented generation (RAG) systems."],
                "id": [1, 2, 3],
                "_distance": [0.1, 0.2, 0.3],
            }
        )

    @pytest.fixture
    def sample_fts_results(self):
        """Create sample FTS search results"""
        return pa.Table.from_pydict(
            {
                "_rowid": [4, 5],
                "content": [
                    "Common Mistakes in Vector Search (and How to Avoid Them) Neglecting Evaluations from the Get-Go", 
                    "Reduce heat to medium and simmer for 20 minutes."],
                "id": [4, 5],
                "_score": [0.9, 0.8],
            }
        )

    def test_rerank_hybrid_successful(
        self, reranker, sample_vector_results, sample_fts_results
    ):
        """Test successful hybrid reranking with both vector and FTS results"""

        # Call rerank_hybrid
        result = reranker.rerank_hybrid(
            "how to perform evaluation for RAG", sample_vector_results, sample_fts_results
        )

        # Assertions
        assert isinstance(result, pa.Table)
        assert "_relevance_score" in result.column_names
        assert result.num_rows == 5  # 3 vector + 2 FTS results

        # Check that scores are within valid range
        scores = result["_relevance_score"].to_pylist()
        assert all(0 <= score <= 1 for score in scores)

        # Checl that scores for selected rows as expected
        rowids = result["_rowid"].to_pylist()
        rowid_to_score = dict(zip(rowids, scores))
        # log all rowid to score mappings for debugging
        logger.info(f"RowID to Score Mapping: {rowid_to_score}")
        # expected negatives
        assert rowid_to_score[1] < 0.5  # receipe should not match
        assert rowid_to_score[5] < 0.5  # receipe should not match
        # expected negatives
        assert rowid_to_score[3] > 0.5
        assert rowid_to_score[4] > 0.5
        # related but not close enough
        assert rowid_to_score[2] > 0.2  # ollama is
        assert rowid_to_score[2] < 0.6  # ollama is

    def test_rerank_hybrid_empty_results(self, reranker):
        """Test reranking with empty vector and FTS results"""
        empty_vector = pa.table(
            {
                "content": pa.array([], type=pa.string()),
                "_rowid": pa.array([], type=pa.int64()),
            }
        )
        empty_fts = pa.table(
            {
                "content": pa.array([], type=pa.string()),
                "_rowid": pa.array([], type=pa.int64()),
            }
        )

        result = reranker.rerank_hybrid("test query", empty_vector, empty_fts)

        assert isinstance(result, pa.Table)
        assert "_relevance_score" in result.column_names
        assert result.num_rows == 0
