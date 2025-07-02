import os
from dotenv import load_dotenv, find_dotenv
import logging

import numpy as np
import pyarrow as pa
import pytest

from mcps.rag.ollama_reranker import OllamaReranker

# Configure logging for detailed test output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv(find_dotenv())
load_dotenv(find_dotenv(usecwd=True))

# Skip all tests if OLLAMA_BASE_URL is not set
pytestmark = pytest.mark.skipif(
    os.getenv("OLLAMA_BASE_URL") is None,
    reason="OLLAMA_BASE_URL environment variable not set. Set it to run Ollama reranker tests."
)


class TestOllamaRerankerHybrid:
    """Test class for OllamaReranker.rerank_hybrid method"""
    
    @pytest.fixture
    def ollama_base_url(self):
        """Get Ollama base URL from environment"""
        return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    
    @pytest.fixture
    def reranker(self, ollama_base_url):
        """Create OllamaReranker instance with mocked client"""
        return OllamaReranker(
            model_name="phi4-mini:latest",
            ollama_base_url=ollama_base_url,
            embedding_model="bge-m3:latest",
            return_score="relevance",
            weight=1.0
        )
    
    @pytest.fixture
    def sample_vector_results(self):
        """Create sample vector search results"""
        return pa.table({
            'text': ['Vector result 1', 'Vector result 2', 'Vector result 3'],
            'id': [1, 2, 3],
            '_distance': [0.1, 0.2, 0.3]
        })
    
    @pytest.fixture
    def sample_fts_results(self):
        """Create sample FTS search results"""
        return pa.table({
            'text': ['FTS result 1', 'FTS result 2'],
            'id': [4, 5],
            '_score': [0.9, 0.8]
        })
    
    def test_rerank_hybrid_successful(self, reranker, sample_vector_results, sample_fts_results):
        """Test successful hybrid reranking with both vector and FTS results"""
        
        # Call rerank_hybrid
        result = reranker.rerank_hybrid("test query", sample_vector_results, sample_fts_results)
        
        # Assertions
        assert isinstance(result, pa.Table)
        assert '_relevance_score' in result.column_names
        assert result.num_rows == 5  # 3 vector + 2 FTS results
        
        # Check that scores are within valid range
        scores = result['_relevance_score'].to_pylist()
        assert all(0 <= score <= 1 for score in scores)
        
        # Verify calls to Ollama client
    
    def test_rerank_hybrid_empty_results(self, reranker):
        """Test reranking with empty vector and FTS results"""
        empty_vector = pa.table({'text': pa.array([], type=pa.string())})
        empty_fts = pa.table({'text': pa.array([], type=pa.string())})
        
        result = reranker.rerank_hybrid("test query", empty_vector, empty_fts)
        
        assert isinstance(result, pa.Table)
        assert '_relevance_score' in result.column_names
        assert result.num_rows == 0
    