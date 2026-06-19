"""Integration tests for `LlmReranker` (hybrid rerank) with different model pairs."""

import logging
import os
from dataclasses import replace
from typing import AsyncIterator

import httpx
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from lancedb.rerankers import Reranker
from langchain_core.embeddings import Embeddings

from mcps.config import ServerConfig, create_config
from mcps.rag.llm_reranker import LlmReranker
from mcps.rag.vault import create_reranker

logger = logging.getLogger(__name__)

# (inference_model, embedding_model, embedding_dimensions)
MODEL_CASES: list[tuple[str, str, int]] = [
    ("gemini-flash-lite", "nomic-embed", 768),
    ("gemini-flash-lite", "gemma-embed", 768),
    ("gpt-5-nano", "nomic-embed", 768),
    ("local-gemma", "gemma-embed", 768),
    ("", "gemma-embed", 768),
]



def _reranker_label(reranker: Reranker) -> str:
    """Return a human-readable label for the active model pair (logging only)."""
    chat_model = getattr(reranker, "chat_model", None)
    chat = chat_model.model_name if chat_model is not None else "none"
    embed = getattr(getattr(reranker, "embeddings", None), "model", "?")
    return f"{chat}+{embed}"


class FakeEmbeddings(Embeddings):
    """Deterministic in-memory embeddings implementation for unit tests."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def embed_query(self, text: str) -> list[float]:
        return self.vectors.get(text, [1.0, 0.0])

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.vectors.get(text, [1.0, 0.0]) for text in texts]


@pytest.fixture
async def async_client() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient() as client:
        yield client


@pytest.fixture
def server_config() -> ServerConfig:
    config = create_config()
    if not config.litellm_router or not config.litellm_router_key:
        pytest.skip(
            "LITELLM_ROUTER / LITELLM_API_KEY are not set; "
            "reranker tests need them to build a chat model + embeddings."
        )
    return config


@pytest.fixture(params=MODEL_CASES)
def reranker(
    request,
    server_config: ServerConfig,
    async_client: httpx.AsyncClient,
) -> Reranker:
    infer_model, embed_model, dimensions = request.param
    config = replace(
        server_config,
        rag_reranker_infer_model=infer_model,
        rag_embedding_model=embed_model,
        rag_embedding_dimensions=dimensions,
    )
    return create_reranker(config, async_client)


class TestLlmRerankerHybrid:
    """Test `LlmReranker.rerank_hybrid` across model pairs."""

    @pytest.fixture
    def sample_vector_results(self) -> pa.Table:
        return pa.Table.from_pydict(
            {
                "_rowid": [1, 2, 3],
                "content": [
                    "Cream of wild mushroom recipe",
                    "Ollama is the service to run large language models",
                    "RED6k is a comprehensive dataset containing **~6,000 samples** "
                    "across **10 domains** created by **Aizip** for evaluating "
                    "language models as summarizers in retrieval-augmented "
                    "generation (RAG) systems.",
                ],
                "id": [1, 2, 3],
                "_distance": [0.1, 0.2, 0.3],
            }
        )

    @pytest.fixture
    def sample_fts_results(self) -> pa.Table:
        return pa.Table.from_pydict(
            {
                "_rowid": [4, 5],
                "content": [
                    "Common Mistakes in Vector Search (and How to Avoid Them) "
                    "Neglecting Evaluations from the Get-Go",
                    "Reduce heat to medium and simmer for 20 minutes.",
                ],
                "id": [4, 5],
                "_score": [0.9, 0.8],
            }
        )

    def test_rerank_hybrid_successful(
        self,
        reranker: Reranker,
        sample_vector_results: pa.Table,
        sample_fts_results: pa.Table,
    ) -> None:
        result = reranker.rerank_hybrid(
            "how to perform evaluation for RAG",
            sample_vector_results,
            sample_fts_results,
        )

        assert isinstance(result, pa.Table)
        assert "_relevance_score" in result.column_names
        assert result.num_rows == 5

        scores = result["_relevance_score"].to_pylist()
        assert all(0 <= score <= 1 for score in scores)

        rowids = result["_rowid"].to_pylist()
        rowid_to_score = dict(zip(rowids, scores))
        logger.info(
            "Model pair %s | RowID to Score: %s",
            _reranker_label(reranker),
            rowid_to_score,
        )
        # expected negatives (unrelated content)
        assert rowid_to_score[1] < 0.5
        assert rowid_to_score[5] < 0.5
        # expected positives
        assert rowid_to_score[3] > 0.5
        assert rowid_to_score[4] > 0.5
        # related but not close enough
        assert 0.2 < rowid_to_score[2] < 0.6

    def test_rerank_hybrid_empty_results(self, reranker: Reranker) -> None:
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

    def test_score_with_llm(
        self,
        reranker: Reranker,
        sample_vector_results: pa.Table,
        sample_fts_results: pa.Table,
    ) -> None:
        all_results = reranker.merge_results(sample_vector_results,sample_fts_results)
        documents: list[str] = reranker._table_to_documents(all_results) # type: ignore
        scores: list[float] = reranker._score_with_llm("how to perform evaluation for RAG",documents) # type: ignore
        assert scores[0] < 0.5
        assert scores[1] < 0.5
        assert scores[2] > 0.5
        assert scores[3] >= 0.5
        assert scores[4] < 0.5

class TestLlmRerankerEvaluation:
    """Evaluate `LlmReranker` with real content fixtures across model pairs.

    The same fixtures and the same expected top-5 are used for every model pair
    so the resulting metrics are directly comparable across backends.
    """

    @pytest.fixture
    def vector_results(self) -> pa.Table:
        return pa.Table.from_pydict(
            {
                "_rowid": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                "content": [
                    "Vector databases are a relatively new way for interacting with abstract data representations derived from opaque machine learning models such as deep learning architectures. These representations are often called vectors or embeddings and they are a compressed version of the data used to train a machine learning model.",
                    "Neglecting Evaluations from the Get-Go: Create a small, reliable eval set: Even 50-100 labeled queries is enough to reveal huge gaps. Use standard metrics: NDCG, MRR, recall - whatever. Start with something, then refine it. Monitor improvements: Each time you tweak chunking or switch embeddings, run the eval again.",
                    "Faiss is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM. It also contains supporting code for evaluation and parameter tuning.",
                    "The study found that a decline in HbA1c, and key markers of long-term blood sugar levels, are associated with significant positive changes in specific brain regions commonly affected by age-related atrophy. Brain MRI results showed that lower HbA1c levels corresponded to greater deviations in the thalamus, caudate nucleus, and cerebellum.",
                    "Record notes anything that 'resonates', e.g interesting. Save contradictory ideas as well. All notes came to some inbox, sorted about once a week into PARA folders. Organize by context, not a topic: Projects, Areas, Resources, Archive.",
                    "The path to $10M ARR in AI is clear, but only if you start automating the workflows nobody wants to touch. Not consumer toys. Not vague productivity tools. I'm talking about the industries where digital transformation peaked with Windows XP.",
                    "MCP server that exposes a customizable prompt templates, resources, and tools. It uses FastMCP to run as server application. Dependencies, build, and run managed by uv tool.",
                    "You can enhance the quality of Copilot's responses by using effective prompts. A well-crafted prompt can help Copilot understand your requirements better and generate more relevant code suggestions. Start general, then get specific.",
                    "Classification intent_examples with OpenAI embeddings: response = openai.embeddings.create(input=[e['text'] for e in intent_examples], model='text-embedding-3-small'). Add embeddings to Faiss index for similarity search.",
                    "Charles Schwab $226000, %7 / yr. Convert to more aggressive? Fidelity 401-k $312000, %10 / yr Ok. E-trade $1755000, no grow. Have to sell some CRM at peak value, convert to S&P 500 ETF.",
                    "In terms of perfecting workflows, here's an example. This is my current coding workflow that I run on my Mac Studio: Step 1: Command-R 08-2024 breaks down requirements from the user's most recent messages. Step 2: Qwen 32b-Coder takes a swing at implementation.",
                    "LanceDB provides support for full-text search via Lance, allowing you to incorporate keyword-based search (based on BM25) in your retrieval solutions. Currently, the Lance full text search is missing some features that are in the Tantivy full text search.",
                ],
                "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                "_distance": [
                    0.1, 0.15, 0.2, 0.25, 0.3, 0.35,
                    0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
                ],
            }
        )

    @pytest.fixture
    def fts_results(self) -> pa.Table:
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
                    "\u0421\u0435\u0432\u0435\u0440\u043e\u043a\u043e\u0440\u0435\u0439\u0441\u043a\u0438\u0439 \u0445\u0430\u043a\u0435\u0440, \u0423\u0448\u0435\u0434\u0448\u0438\u0439 \u0420\u043e\u0434, \u041f\u043e\u043b\u0443\u0432\u0430\u0440\u0432\u0430\u0440, \u0422\u0430\u043d\u0433\u043e \u0444\u0440\u0435\u0437\u0435\u0440\u043d\u044b\u0445 \u0441\u0442\u0430\u043d\u043a\u043e\u0432. Online library HathiTrust: we are stewards of the largest digitized collection of knowledge allowable by copyright law. Bobiverse release Brothers of the Line (The Karus Saga Book 5)",
                ],
                "id": [13, 14, 15, 16, 17, 18, 19, 20],
                "_score": [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6],
            }
        )

    def evaluate_reranking_accuracy(
        self,
        reranker: Reranker,
        vector_results: pa.Table,
        fts_results: pa.Table,
        question: str,
        expected_top_5: list[int],
    ) -> dict:
        result = reranker.rerank_hybrid(question, vector_results, fts_results)
        sorted_result = result.sort_by([("_relevance_score", "descending")])
        actual_top_5 = sorted_result["_rowid"].slice(0, 5).to_pylist()
        correct_count = sum(
            1
            for actual, expected in zip(actual_top_5, expected_top_5)
            if actual == expected
        )
        position_accuracy = correct_count / 5
        overlap_count = len(set(actual_top_5) & set(expected_top_5))
        overlap_accuracy = overlap_count / 5
        logger.info(
            "Model %s | Q: %s | expected: %s | actual: %s | "
            "pos_acc: %.2f | overlap_acc: %.2f",
            _reranker_label(reranker),
            question,
            expected_top_5,
            actual_top_5,
            position_accuracy,
            overlap_accuracy,
        )
        return {
            "question": question,
            "position_accuracy": position_accuracy,
            "overlap_accuracy": overlap_accuracy,
            "expected": expected_top_5,
            "actual": actual_top_5,
        }

    def test_calculate_overall_metrics(
        self,
        reranker: Reranker,
        vector_results: pa.Table,
        fts_results: pa.Table,
    ) -> None:
        test_cases = [
            ("How to evaluate vector search systems?", [2, 13, 3, 1, 14]),
            ("What are vector databases and how do they work?", [1, 3, 14, 2, 12]),
            ("How to improve AI model performance and evaluation?", [2, 13, 1, 3, 8]),
            ("What are the best practices for prompt engineering?", [8, 18, 19, 7, 2]),
            ("How to manage personal finances and investments?", [10, 15, 9, 6, 4]),
            ("What is the Mediterranean diet and health benefits?", [4, 16, 5, 6, 1]),
        ]
        metrics = [
            self.evaluate_reranking_accuracy(
                reranker, vector_results, fts_results, question, expected_top_5
            )
            for question, expected_top_5 in test_cases
        ]
        avg_position_accuracy = sum(
            m["position_accuracy"] for m in metrics
        ) / len(metrics)
        avg_overlap_accuracy = sum(
            m["overlap_accuracy"] for m in metrics
        ) / len(metrics)
        total_correct_positions = sum(m["position_accuracy"] * 5 for m in metrics)
        total_possible_positions = len(metrics) * 5
        overall_position_ratio = total_correct_positions / total_possible_positions
        logger.info(
            "=== EVAL [%s] | Qs: %d | avg_pos: %.3f | "
            "avg_overlap: %.3f | overall_pos_ratio: %.3f ===",
            _reranker_label(reranker),
            len(metrics),
            avg_position_accuracy,
            avg_overlap_accuracy,
            overall_position_ratio,
        )
        assert overall_position_ratio >= 0.2, (
            f"Overall position ratio {overall_position_ratio:.3f} "
            f"below minimum threshold 0.2 for {_reranker_label(reranker)}"
        )


class TestLlmRerankerEmbeddingOnly:
    """Unit tests for `LlmReranker` when no chat model is configured."""

    @pytest.fixture
    def reranker(self) -> LlmReranker:
        vectors = {
            "query": [1.0, 0.0],
            "direct answer": [1.0, 0.0],
            "somewhat related": [0.7, 0.7],
            "unrelated": [0.0, 1.0],
        }
        return LlmReranker(chat_model=None, embeddings=FakeEmbeddings(vectors))

    def test_score_with_llm_returns_zeros_without_chat_model(
        self, reranker: LlmReranker
    ) -> None:
        scores = reranker._score_with_llm("query", ["direct answer", "unrelated"])

        assert scores == [0.0, 0.0]

    def test_rerank_vector_uses_only_embedding_similarity(
        self, reranker: LlmReranker
    ) -> None:
        results = pa.Table.from_pydict(
            {
                "_rowid": [1, 2, 3],
                "content": ["direct answer", "somewhat related", "unrelated"],
            }
        )

        result = reranker.rerank_vector("query", results)

        scores = result["_relevance_score"].to_pylist()
        assert scores[0] > scores[1] > scores[2]


class TestCreateRerankerEmbeddingOnly:
    """Integration tests for factory-created embedding-only `LlmReranker`."""

    @pytest.fixture
    def embedding_only_config(self, server_config: ServerConfig) -> ServerConfig:
        return replace(server_config, rag_reranker_infer_model="")

    def test_factory_creates_llm_reranker_without_chat_model(
        self,
        embedding_only_config: ServerConfig,
        async_client: httpx.AsyncClient,
    ) -> None:
        reranker = create_reranker(embedding_only_config, async_client)

        assert isinstance(reranker, LlmReranker)
        assert reranker.chat_model is None

    def test_factory_rerank_hybrid_uses_embeddings_only(
        self,
        embedding_only_config: ServerConfig,
        async_client: httpx.AsyncClient,
    ) -> None:
        reranker = create_reranker(embedding_only_config, async_client)
        vector_results = pa.Table.from_pydict(
            {
                "_rowid": [1, 2],
                "content": ["direct answer", "unrelated topic"],
                "id": [1, 2],
                "_distance": [0.1, 0.9],
            }
        )
        fts_results = pa.Table.from_pydict(
            {
                "_rowid": [3, 4],
                "content": ["another direct answer", "cooking recipe"],
                "id": [3, 4],
                "_score": [0.9, 0.1],
            }
        )

        result = reranker.rerank_hybrid("query", vector_results, fts_results)

        assert isinstance(result, pa.Table)
        assert "_relevance_score" in result.column_names
        assert result.num_rows == 4
        assert all(0 <= score <= 1 for score in result["_relevance_score"].to_pylist())
