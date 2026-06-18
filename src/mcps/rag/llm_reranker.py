"""OpenAI-compatible reranker for LanceDB search results."""

import itertools
import logging
from collections.abc import Sequence

import numpy as np
import openai
import pyarrow as pa
from lancedb.rerankers import Reranker
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

logger = logging.getLogger("mcps.rag")

SYSTEM_PROMPT="""
Given the query and document below, rate how relevant the document is to answering
the query. Output a single word:
PERFECT if they are relevant
GOOD if they are close but not exact, like both about programming but different
    languages or libraries
SOME if they are relevant in broad sense, like both about programming but one about
    coding practices and another about algorithms
BAD if there is only little similarity, like one about programming and another about
    job interviews
NONE if document unrelevant to question, like one about astronomy and another about
    cooking recipes."""


class LlmReranker(Reranker):
    """Rerank search results using an OpenAI-compatible API."""

    def __init__(
        self,
        chat_model: BaseChatModel,
        embeddings: Embeddings,
        return_score: str = "relevance",
        column: str = "content",
        weight: float = 1.0,
        embedding_batch_size: int = 8,
    ):
        """Initialize an OpenAI-compatible reranker.

        Args:
            model_name: Chat model name for LLM-based scoring.
            api_base: Base URL for the OpenAI-compatible API.
            api_key: API key for the provider. Ollama accepts any value.
            embedding_model: Embedding model name for similarity scoring.
            return_score: Score return type passed to LanceDB.
            column: Result table column containing document text.
            weight: Weight for score fusion. Values above 1.0 favor the LLM.
            embedding_batch_size: Embedding request batch size.
        """
        super().__init__(return_score)
        self.chat_model = chat_model
        self.embeddings = embeddings
        self.weight = weight
        self.column = column
        self.embedding_batch_size = embedding_batch_size

    def _score_with_llm(self, query: str, documents: list[str]) -> list[float]:
        """Score documents using the configured chat completion model."""
        return [self._score_document_with_llm(query, document) for document in documents]

    def _score_document_with_llm(self, query: str, document: str) -> float:
        prompt = self._create_relevance_prompt(query, document)
        try:
            response = self.chat_model.invoke(
                [{"role": "system", "content": SYSTEM_PROMPT},{"role": "user", "content": prompt}],
                # temperature=0.1,
                # max_tokens=100,
            )
        except Exception:
            logger.exception("Failed to score document with OpenAI-compatible API")
            return 0.0
        score_text = str(response.content) or ""
        logger.info("Llm score text is %.10s for query %.10s and doc %.10s",score_text,query,document)
        return self._parse_score(score_text)

    @staticmethod
    def _create_relevance_prompt(query: str, document: str) -> str:
        return f""" Query: {query}

Document: {document}

Relevance:"""

    @staticmethod
    def _parse_score(score_text: str) -> float:
        score_text = score_text.lower()
        if "perfect" in score_text:
            return 1.0
        if "good" in score_text:
            return 0.75
        if "some" in score_text:
            return 0.5
        if "bad" in score_text:
            return 0.25
        return 0.0

    def _score_with_embeddings(self, query: str, documents: list[str]) -> list[float]:
        """Score documents using embedding cosine similarity."""
        query_embedding = np.asarray(self.embeddings.embed_query(query), dtype=np.float64)
        doc_embeddings = np.asarray(self.embeddings.embed_documents(documents), dtype=np.float64)

        if query_embedding.ndim != 1 or doc_embeddings.ndim != 2:
            raise ValueError(
                "Embedding vectors have inconsistent dimensions; expected "
                "1-D query and 2-D documents."
            )

        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        doc_norms = doc_embeddings / (
            np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-10
        )
        similarities = np.dot(doc_norms, query_norm)
        logger.info("Max similarities: %s", similarities.max())

        normalized_similarities = self._normalize_similarities(similarities)
        return normalized_similarities.tolist()

    @staticmethod
    def _normalize_similarities(similarities: np.ndarray) -> np.ndarray:
        if len(similarities) > 1:
            min_sim = similarities.min()
            max_sim = similarities.max()
            if max_sim > min_sim:
                return (similarities - min_sim) / (max_sim - min_sim)
            return np.full_like(similarities, 1.0)
        if len(similarities) == 1:
            return np.array([1.0])
        return similarities

    def _rerank_results(self, query: str, results: pa.Table) -> pa.Table:
        """Rerank table rows and append a relevance score column."""
        num_rows = results.num_rows
        if num_rows == 0:
            return results.append_column(
                "_relevance_score",
                pa.array([], type=pa.float32()),
            )

        documents = self._table_to_documents(results)
        llm_scores = np.array(self._score_with_llm(query, documents))
        embedding_scores = np.array(self._score_with_embeddings(query, documents))

        llm_scores = np.clip(llm_scores, 0, 1)
        embedding_scores = np.clip(embedding_scores, 0, 1)
        llm_weight = self.weight / (1 + self.weight)
        embedding_weight = 1 / (1 + self.weight)
        combined_scores = llm_weight * llm_scores + embedding_weight * embedding_scores

        return results.append_column(
            "_relevance_score",
            pa.array(combined_scores.tolist()),
        )

    def _table_to_documents(self, results):
        """Extract text column to a list of strings"""
        text_column = results[self.column]
        documents = [str(text_column[index].as_py()) for index in range(results.num_rows)]
        return documents

    def rerank_hybrid(
        self,
        query: str,
        vector_results: pa.Table,
        fts_results: pa.Table,
    ) -> pa.Table:
        """Rerank hybrid search results."""
        combined_results = self.merge_results(vector_results, fts_results)
        return self._rerank_results(query, combined_results)

    def rerank_vector(self, query: str, vector_results: pa.Table) -> pa.Table:
        """Rerank vector search results."""
        return self._rerank_results(query, vector_results)

    def rerank_fts(self, query: str, fts_results: pa.Table) -> pa.Table:
        """Rerank full-text search results."""
        return self._rerank_results(query, fts_results)
