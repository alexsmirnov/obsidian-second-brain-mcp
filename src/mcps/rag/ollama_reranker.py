import numpy as np
import ollama
import pyarrow as pa
from lancedb.rerankers import Reranker


class OllamaReranker(Reranker):
    def __init__(
        self,
        model_name: str = "phi4-mini:latest",
        ollama_base_url: str = "http://localhost:11434",
        embedding_model: str = "bge-m3:latest",
        return_score: str = "relevance",
        column: str = "content",
        weight: float = 1.0
    ):
        """
        Initialize Ollama-based reranker
        
        Args:
            model_name: Name of the Ollama model for LLM-based scoring
            ollama_base_url: Base URL for Ollama API
            embedding_model: Embedding model name for similarity-based reranking
            return_score: Score return type ("relevance" or "all")
            weight: Weight for combining scores (1.0 = equal, >1.0 = LLM favored,
                <1.0 = embedding favored)
        """
        super().__init__(return_score)
        self.model_name = model_name
        self.ollama_base_url = ollama_base_url
        self.embedding_model = embedding_model
        self.client = ollama.Client(host=ollama_base_url)
        self.weight = weight
        self.column = column
        
    def _get_ollama_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for *all* ``texts`` with a single Ollama call.

        The upstream Ollama python client supports batching by simply passing a
        list of strings to the ``input`` parameter.  This implementation takes
        advantage of that capability, dramatically reducing network overhead
        compared to the previous one-request-per-text approach.

        Args:
            texts: Sequence of input strings.

        Returns:
            A list where each element is the embedding (``list[float]``) for the
            corresponding input text.
        """
        # Request all embeddings in **one** API call.
        try:
            resp = self.client.embed(model=self.embedding_model, input=texts)
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch embeddings from Ollama: {exc}") from exc

        # The response can be either a plain dict or a typed object exposing the
        # same fields as attributes.  Prefer the modern "embeddings" field.  If
        # only a single embedding was requested the field may be "embedding".
        if isinstance(resp, dict):
            vectors = resp.get("embeddings") or resp.get("embedding")
        else:
            vectors = getattr(resp, "embeddings", None) or getattr(resp, "embedding", None)

        if vectors is None:
            raise ValueError("Ollama embed response did not contain 'embeddings'")

        # Normalize the shape: single-text requests may return a single vector
        # instead of a list of vectors.
        if vectors and isinstance(vectors[0], (float, int)):
            vectors = [vectors]

        # Convert to NumPy arrays for consistency and validation.
        embeddings_np = [np.asarray(vec, dtype=np.float32) for vec in vectors]

        if not embeddings_np:
            raise ValueError("Received empty embeddings from Ollama")

        expected_dim = embeddings_np[0].shape[0]
        cleaned: list[np.ndarray] = []
        for vec in embeddings_np:
            # Ensure dimensionality is correct; pad/replace invalid vectors.
            if vec.size != expected_dim:
                vec = np.zeros(expected_dim, dtype=np.float32)
            if not np.isfinite(vec).all() or np.linalg.norm(vec) == 0:
                vec = np.zeros(expected_dim, dtype=np.float32)
            cleaned.append(vec)

        # Return Python lists for Arrow/JSON compatibility.
        return [v.tolist() for v in cleaned]
    
    def _score_with_llm(self, query: str, documents: list[str]) -> list[float]:
        """Score documents using Ollama LLM"""
        scores = []
        
        for doc in documents:
            prompt = f"""
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
    cooking recipes.

Query: {query}

Document: {doc}

Relevance:"""
            
            try:
                response = self.client.generate(
                    model=self.model_name,
                    prompt=prompt,
                    stream=False,
                    options={
                        "temperature": 0.1,  # Low temperature for consistent scoring
                        "num_predict": 10    # Limit output length
                    }
                )
                
                score_text = response.response.strip()
                if  "perfect" in score_text.lower():
                    scores.append(1.0)
                elif  "good" in score_text.lower():
                    scores.append(0.75)
                elif  "some" in score_text.lower():
                    scores.append(0.5)
                elif  "bad" in score_text.lower():
                    scores.append(0.25)
                else:
                    scores.append(0.0)  # Default score
            except Exception:
                scores.append(0.0)
                
        return scores
    
    def _score_with_embeddings(self, query: str, documents: list[str]) -> list[float]:
        """Score documents using embedding similarity"""
        # Get embeddings for query and documents
        all_texts = [query, *documents]
        embeddings = self._get_ollama_embeddings(all_texts)
        
        # Ensure deterministic dtypes and shapes
        query_embedding = np.asarray(embeddings[0], dtype=np.float32)
        doc_embeddings = np.asarray(embeddings[1:], dtype=np.float32)

        if query_embedding.ndim != 1 or doc_embeddings.ndim != 2:
            raise ValueError("Embedding vectors have inconsistent dimensions; expected 1-D query and 2-D documents.")
        
        # Calculate cosine similarity using NumPy operations
        # Normalize embeddings
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        doc_norms = doc_embeddings / (
            np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-10
        )
        
        # Calculate similarities
        similarities = np.dot(doc_norms, query_norm)
        
        # Normalize to 0-1 range
        if len(similarities) > 1:
            min_sim, max_sim = similarities.min(), similarities.max()
            if max_sim > min_sim:
                similarities = (similarities - min_sim) / (max_sim - min_sim)
            else:
                # All similarities are the same, set to 0.5
                similarities = np.full_like(similarities, 0.5)
        elif len(similarities) == 1:
            # Single document, set to 1.0
            similarities = np.array([1.0])
        
        return similarities.tolist()
    
    def _rerank_results(self, query: str, results: pa.Table) -> pa.Table:
        """Common reranking logic for all query types"""
        # Get the number of rows
        num_rows = results.num_rows
        
        if num_rows == 0:
            return results.append_column(
                "_relevance_score", pa.array([], type=pa.float32())
            )
            
        # Extract text content using PyArrow (assuming 'text' column exists)
        text_column = results[self.column]
        documents = [str(text_column[i].as_py()) for i in range(num_rows)]
        
        # Get relevance scores
        llm_scores = np.array(self._score_with_llm(query, documents))
        embedding_scores = np.array(self._score_with_embeddings(query, documents))
        
        # Normalize scores to ensure they're in 0-1 range
        llm_scores = np.clip(llm_scores, 0, 1)
        embedding_scores = np.clip(embedding_scores, 0, 1)
        
        # Combine scores using the weight parameter
        # Weight = 1.0: Equal contribution
        # Weight > 1.0: LLM scores have higher influence
        # Weight < 1.0: Embedding scores have higher influence
        llm_weight = self.weight / (1 + self.weight)
        embedding_weight = 1 / (1 + self.weight)
        
        combined_scores = llm_weight * llm_scores + embedding_weight * embedding_scores
        
        # Create a new column with the relevance scores
        score_array = pa.array(combined_scores.tolist())
        results = results.append_column('_relevance_score', score_array)
        
        return results
    
    def rerank_hybrid(
        self, query: str, vector_results: pa.Table, fts_results: pa.Table
    ) -> pa.Table:
        """Rerank hybrid search results"""
        # Merge vector and FTS results
        combined_results = self.merge_results(vector_results, fts_results)
        return self._rerank_results(query, combined_results)
    
    def rerank_vector(self, query: str, vector_results: pa.Table) -> pa.Table:
        """Rerank vector search results"""
        return self._rerank_results(query, vector_results)
    
    def rerank_fts(self, query: str, fts_results: pa.Table) -> pa.Table:
        """Rerank FTS search results"""
        return self._rerank_results(query, fts_results)
