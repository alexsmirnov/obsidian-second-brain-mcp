from lancedb.rerankers import Reranker
import pyarrow as pa
import ollama
from typing import List, Optional, Union
import numpy as np

def cosine_similarity(a, b):
    # Calculate dot product
    dot_product = sum(x * y for x, y in zip(a, b))
    # Calculate magnitudes
    magnitude_a = sum(x ** 2 for x in a) ** 0.5
    magnitude_b = sum(y ** 2 for y in b) ** 0.5
    if magnitude_a == 0 or magnitude_b == 0:
        return 0  # Avoid division by zero
    return dot_product / (magnitude_a * magnitude_b)


class OllamaReranker(Reranker):
    def __init__(
        self,
        model_name: str = "llama3.1",
        ollama_base_url: str = "http://localhost:11434",
        method: str = "llm_scoring",  # or "embedding_similarity"
        embedding_model: str = "mxbai-embed-large",
        return_score: str = "_relevance_score"
    ):
        """
        Initialize Ollama-based reranker
        
        Args:
            model_name: Name of the Ollama model for LLM-based scoring
            ollama_base_url: Base URL for Ollama API
            method: Reranking method ("llm_scoring" or "embedding_similarity")
            embedding_model: Embedding model name for similarity-based reranking
            return_score: Score return type ("relevance" or "all")
        """
        super().__init__(return_score)
        self.model_name = model_name
        self.ollama_base_url = ollama_base_url
        self.method = method
        self.embedding_model = embedding_model
        self.client = ollama.Client(host=ollama_base_url)
        
    def _get_ollama_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings from Ollama embedding model"""
        embeddings = []
        for text in texts:
            try:
                response = self.client.embed(
                    model=self.embedding_model,
                    input=text
                )
                embeddings.append(response.embeddings[0])
            except Exception as e:
                raise Exception(f"Failed to get embedding: {e}")
        return embeddings
    
    def _score_with_llm(self, query: str, documents: List[str]) -> List[float]:
        """Score documents using Ollama LLM"""
        scores = []
        
        for doc in documents:
            prompt = f"""
Given the query and document below, rate how relevant the document is to answering the query.
Output a single word: 
PERFECT if they are relevant
GOOD if they are close but not exact, like both about programming but different languages or libraries
SOME if they are relevant in broad sense, like both about programming but one about coding practices and another about algorithms
BAD if there is only little similarity, like one about programming and another about job interviews
NONE if document unrelevant to question, like one about astronomy and another about cooking recipes.

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
    
    def _score_with_embeddings(self, query: str, documents: List[str]) -> List[float]:
        """Score documents using embedding similarity"""
        # Get embeddings for query and documents
        all_texts = [query] + documents
        embeddings = self._get_ollama_embeddings(all_texts)
        
        query_embedding = np.array(embeddings[0]).reshape(1, -1)
        doc_embeddings = np.array(embeddings[1:])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        
        # Normalize to 0-1 range
        if len(similarities) > 1:
            min_sim, max_sim = similarities.min(), similarities.max()
            if max_sim > min_sim:
                similarities = (similarities - min_sim) / (max_sim - min_sim)
        
        return similarities.tolist()
    
    def _rerank_results(self, query: str, results: pa.Table) -> pa.Table:
        """Common reranking logic for all query types"""
        df = results.to_pandas()
        
        if len(df) == 0:
            return results
            
        # Extract text content (assuming 'text' column exists)
        documents = df['text'].tolist()
        
        # Get relevance scores
        if self.method == "llm_scoring":
            scores = self._score_with_llm(query, documents)
        elif self.method == "embedding_similarity":
            scores = self._score_with_embeddings(query, documents)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Add relevance scores
        df['_relevance_score'] = scores
        
        # Sort by relevance score (descending)
        df = df.sort_values('_relevance_score', ascending=False)
        
        return pa.Table.from_pandas(df)
    
    def rerank_hybrid(self, query: str, vector_results: pa.Table, fts_results: pa.Table) -> pa.Table:
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
