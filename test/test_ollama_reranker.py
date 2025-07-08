import os
from dotenv import load_dotenv, find_dotenv
import logging

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
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


class TestOllamaRerankerEvaluation:
    """Evaluation test class for OllamaReranker with real content fixtures"""

    @pytest.fixture
    def ollama_base_url(self):
        """Get Ollama base URL from environment"""
        return os.getenv("OLLAMA_API_BASE", "http://localhost:11434")

    @pytest.fixture
    def reranker(self, ollama_base_url):
        """Create OllamaReranker instance"""
        return OllamaReranker(
            model_name="phi4-mini:latest",
            ollama_base_url=ollama_base_url,
            embedding_model="bge-m3:latest",
            return_score="relevance",
            weight=1.0,
        )

    @pytest.fixture
    def vector_results(self):
        """Create vector search results from real content"""
        return pa.Table.from_pydict(
            {
                "_rowid": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                "content": [
                    "Vector databases are a relatively new way for interacting with abstract data representations derived from opaque machine learning models such as deep learning architectures. These representations are often called vectors or embeddings and they are a compressed version of the data used to train a machine learning model.",
                    "Neglecting Evaluations from the Get-Go: Create a small, reliable eval set: Even 50–100 labeled queries is enough to reveal huge gaps. Use standard metrics: NDCG, MRR, recall — whatever. Start with something, then refine it. Monitor improvements: Each time you tweak chunking or switch embeddings, run the eval again.",
                    "Faiss is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM. It also contains supporting code for evaluation and parameter tuning.",
                    "The study found that a decline in HbA1c, and key markers of long-term blood sugar levels, are associated with significant positive changes in specific brain regions commonly affected by age-related atrophy. Brain MRI results showed that lower HbA1c levels corresponded to greater deviations in the thalamus, caudate nucleus, and cerebellum.",
                    "Record notes anything that 'resonates', e.g interesting. Save contradictory ideas as well. All notes came to some inbox, sorted about once a week into PARA folders. Organize by context, not a topic: Projects, Areas, Resources, Archive.",
                    "The path to $10M ARR in AI is clear, but only if you start automating the workflows nobody wants to touch. Not consumer toys. Not vague productivity tools. I'm talking about the industries where digital transformation peaked with Windows XP.",
                    "MCP server that exposes a customizable prompt templates, resources, and tools. It uses FastMCP to run as server application. Dependencies, build, and run managed by uv tool.",
                    "You can enhance the quality of Copilot's responses by using effective prompts. A well-crafted prompt can help Copilot understand your requirements better and generate more relevant code suggestions. Start general, then get specific.",
                    "Classification intent_examples with OpenAI embeddings: response = openai.embeddings.create(input=[e['text'] for e in intent_examples], model='text-embedding-3-small'). Add embeddings to Faiss index for similarity search.",
                    "Charles Schwab $226000, %7 / yr. Convert to more aggressive? Fidelity 401-k $312000, %10 / yr Ok. E-trade $1755000, no grow. Have to sell some CRM at peak value, convert to S&P 500 ETF.",
                    "In terms of perfecting workflows, here's an example. This is my current coding workflow that I run on my Mac Studio: Step 1: Command-R 08-2024 breaks down requirements from the user's most recent messages. Step 2: Qwen 32b-Coder takes a swing at implementation.",
                    "LanceDB provides support for full-text search via Lance, allowing you to incorporate keyword-based search (based on BM25) in your retrieval solutions. Currently, the Lance full text search is missing some features that are in the Tantivy full text search."
                ],
                "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                "_distance": [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65],
            }
        )

    @pytest.fixture
    def fts_results(self):
        """Create FTS search results from real content"""
        return pa.Table.from_pydict(
            {
                "_rowid": [13, 14, 15, 16, 17, 18, 19, 20],
                "content": [
                    "If you've ran off-the-shelf evals for your tasks, you may have found that most don't work. They barely correlate with application-specific performance and aren't discriminative enough to use in production. As a result, we could spend weeks and still not have evals that reliably measure how we're doing on our tasks.",
                    "Qdrant is a vector similarity search engine that provides a production-ready service with a convenient API to store, search, and manage points (i.e. vectors) with an additional payload. You can think of the payloads as additional pieces of information that can help you hone in on your search.",
                    "VTEB Leaderboard: VANGUARD MUNI BND TAX EXEMPT ETF as bonds. Consider to use on Etrade as well. Review investments, convert what's possible (401k, schwab, brokerage) into bonds, gold, and ETFs resilient for recession.",
                    "The Green Mediterranean (Green-Med) diet is rich in polyphenols from plant-based sources like Mankai (a high-protein aquatic plant) and green tea, while being low in red and processed meats. The current study further strengthens this connection by suggesting that the Green-Med diet may not only support metabolic health but also exert protective effects on brain structure and function.",
                    "Create a prompt file with the Create Prompt command from the Command Palette. This command creates a .prompt.md file in the .github/prompts folder at the root of your workspace. Describe your prompt and relevant context in Markdown format.",
                    "Request access to Calendar API for google Oauth. Full text search in backend. Configure network access for MongoDb. Resize applicant image to small size. Configure domain for Websocket API Gateway startup context coding.",
                    "Reusable prompts enable you to save a prompt for a specific task with its context and instructions in a file. You can then attach and reuse that prompt in chat. If you store the prompt in your workspace, you can also share it with your team.",
                    "Северокорейский хакер, Ушедший Род, Полуварвар, Танго фрезерных станков. Online library HathiTrust: we are stewards of the largest digitized collection of knowledge allowable by copyright law. Bobiverse release Brothers of the Line (The Karus Saga Book 5)"
                ],
                "id": [13, 14, 15, 16, 17, 18, 19, 20],
                "_score": [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6],
            }
        )

    def evaluate_reranking_accuracy(self, reranker, vector_results, fts_results, question, expected_top_5):
        """Helper to evaluate reranker accuracy and return metrics"""
        result = reranker.rerank_hybrid(question, vector_results, fts_results)
        sort_indices = pc.sort_indices(result, sort_keys=[('_relevance_score', 'descending')])
        sorted_result = pc.take(result, sort_indices)
        actual_top_5 = sorted_result['_rowid'].slice(0, 5).to_pylist()
        correct_count = sum(1 for actual, expected in zip(actual_top_5, expected_top_5) if actual == expected)
        position_accuracy = correct_count / 5
        overlap_count = len(set(actual_top_5) & set(expected_top_5))
        overlap_accuracy = overlap_count / 5
        logger.info(f"Question: {question}")
        logger.info(f"Expected top 5: {expected_top_5}")
        logger.info(f"Actual top 5: {actual_top_5}")
        logger.info(f"Position accuracy: {position_accuracy:.2f}")
        logger.info(f"Overlap accuracy: {overlap_accuracy:.2f}")
        return {
            'question': question,
            'position_accuracy': position_accuracy,
            'overlap_accuracy': overlap_accuracy,
            'expected': expected_top_5,
            'actual': actual_top_5
        }

    def test_calculate_overall_metrics(self, reranker, vector_results, fts_results):
        """Calculate and report overall evaluation metrics for all questions"""
        test_cases = [
            ("How to evaluate vector search systems?", [2, 13, 3, 1, 14]),
            ("What are vector databases and how do they work?", [1, 3, 14, 2, 12]),
            ("How to improve AI model performance and evaluation?", [2, 13, 1, 3, 8]),
            ("What are the best practices for prompt engineering?", [8, 18, 19, 7, 2]),
            ("How to manage personal finances and investments?", [10, 15, 9, 6, 4]),
            ("What is the Mediterranean diet and health benefits?", [4, 16, 5, 6, 1]),
        ]
        metrics = [self.evaluate_reranking_accuracy(reranker, vector_results, fts_results, question, expected_top_5)
                    for question, expected_top_5 in test_cases]
        avg_position_accuracy = sum(m['position_accuracy'] for m in metrics) / len(metrics)
        avg_overlap_accuracy = sum(m['overlap_accuracy'] for m in metrics) / len(metrics)
        total_correct_positions = sum(m['position_accuracy'] * 5 for m in metrics)
        total_possible_positions = len(metrics) * 5
        overall_position_ratio = total_correct_positions / total_possible_positions
        logger.info(f"\n=== OVERALL EVALUATION METRICS ===")
        logger.info(f"Total test questions: {len(metrics)}")
        logger.info(f"Average position accuracy: {avg_position_accuracy:.3f}")
        logger.info(f"Average overlap accuracy: {avg_overlap_accuracy:.3f}")
        logger.info(f"Overall correct position ratio: {overall_position_ratio:.3f}")
        logger.info(f"Total correctly ranked items: {total_correct_positions:.1f}/{total_possible_positions}")
        logger.info(f"\n=== PER-QUESTION BREAKDOWN ===")
        for i, metric in enumerate(metrics, 1):
            logger.info(f"{i}. {metric['question'][:50]}...")
            logger.info(f"   Position accuracy: {metric['position_accuracy']:.2f}, Overlap accuracy: {metric['overlap_accuracy']:.2f}")
        assert overall_position_ratio >= 0.2, f"Overall position ratio {overall_position_ratio:.3f} below minimum threshold 0.2"
