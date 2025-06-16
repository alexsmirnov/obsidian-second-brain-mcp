"""
Search engine and result formatting implementations for the RAG search system.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from .interfaces import (
    SearchQuery, SearchResult, Chunk, ISearchEngine, IResultFormatter, 
    IEmbeddingService, IVectorStore
)

logger = logging.getLogger("mcps")


class SemanticSearchEngine(ISearchEngine):
    """Semantic search engine using embeddings and vector similarity."""
    
    def __init__(
        self, 
        embedding_service: IEmbeddingService,
        vector_store: IVectorStore,
        reranker: Optional['IReranker'] = None
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.reranker = reranker
    
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform a semantic search operation."""
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.generate_embedding(query.text)
            
            # Search vector store
            similar_chunks = await self.vector_store.search(
                query_embedding, 
                top_k=query.top_k * 2  # Get more results for potential reranking
            )
            
            # Convert to search results with scores
            results = []
            for chunk in similar_chunks:
                if chunk.embeddings:
                    # Calculate similarity score
                    score = self._calculate_similarity(query_embedding, chunk.embeddings)
                    
                    # Apply similarity threshold
                    if score >= query.similarity_threshold:
                        result = SearchResult(chunk=chunk, score=score)
                        results.append(result)
            
            # Apply filters if specified
            if query.filters:
                results = self._apply_filters(results, query.filters)
            
            # Rerank if reranker is available
            if self.reranker and results:
                results = await self.reranker.rerank(results, query)
            
            # Sort by score and limit to top_k
            results.sort(key=lambda x: x.score, reverse=True)
            results = results[:query.top_k]
            
            logger.debug(f"Found {len(results)} search results for query: {query.text}")
            return results
            
        except Exception as e:
            logger.error(f"Search failed for query '{query.text}': {e}")
            return []
    
    def _calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            import numpy as np
            
            v1 = np.array(vec1)
            v2 = np.array(vec2)
            
            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
            
        except ImportError:
            # Fallback implementation without numpy
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
    
    def _apply_filters(self, results: List[SearchResult], filters: Dict[str, Any]) -> List[SearchResult]:
        """Apply filters to search results."""
        filtered_results = []
        
        for result in results:
            chunk = result.chunk
            include = True
            
            # Filter by tags
            if 'tags' in filters:
                required_tags = filters['tags']
                if isinstance(required_tags, str):
                    required_tags = [required_tags]
                if not any(tag in chunk.tags for tag in required_tags):
                    include = False
            
            # Filter by source path pattern
            if 'source_pattern' in filters and include:
                import re
                pattern = filters['source_pattern']
                if not re.search(pattern, str(chunk.source_path)):
                    include = False
            
            # Filter by date range
            if 'date_from' in filters and include:
                date_from = datetime.fromisoformat(filters['date_from'])
                if chunk.modified_at < date_from:
                    include = False
            
            if 'date_to' in filters and include:
                date_to = datetime.fromisoformat(filters['date_to'])
                if chunk.modified_at > date_to:
                    include = False
            
            # Filter by metadata
            if 'metadata' in filters and include:
                metadata_filters = filters['metadata']
                for key, value in metadata_filters.items():
                    if key not in chunk.metadata or chunk.metadata[key] != value:
                        include = False
                        break
            
            if include:
                filtered_results.append(result)
        
        logger.debug(f"Filtered {len(results)} results to {len(filtered_results)}")
        return filtered_results


class HybridSearchEngine(ISearchEngine):
    """Hybrid search engine combining semantic and keyword search."""
    
    def __init__(
        self,
        semantic_engine: SemanticSearchEngine,
        keyword_weight: float = 0.3,
        semantic_weight: float = 0.7
    ):
        self.semantic_engine = semantic_engine
        self.keyword_weight = keyword_weight
        self.semantic_weight = semantic_weight
    
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform hybrid search combining semantic and keyword matching."""
        # Get semantic results
        semantic_results = await self.semantic_engine.search(query)
        
        # Perform keyword search
        keyword_results = self._keyword_search(query)
        
        # Combine and rerank results
        combined_results = self._combine_results(semantic_results, keyword_results)
        
        # Sort by combined score and limit
        combined_results.sort(key=lambda x: x.score, reverse=True)
        return combined_results[:query.top_k]
    
    def _keyword_search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform simple keyword-based search."""
        # This is a simplified implementation
        # In practice, you might use a proper text search engine like Elasticsearch
        results = []
        query_terms = query.text.lower().split()
        
        # This would need access to all chunks - simplified for now
        # In a real implementation, you'd maintain a keyword index
        
        return results
    
    def _combine_results(
        self, 
        semantic_results: List[SearchResult], 
        keyword_results: List[SearchResult]
    ) -> List[SearchResult]:
        """Combine semantic and keyword search results."""
        # Create a map of chunk ID to combined score
        score_map: Dict[str, float] = {}
        chunk_map: Dict[str, Chunk] = {}
        
        # Add semantic scores
        for result in semantic_results:
            chunk_id = result.chunk.id
            score_map[chunk_id] = result.score * self.semantic_weight
            chunk_map[chunk_id] = result.chunk
        
        # Add keyword scores
        for result in keyword_results:
            chunk_id = result.chunk.id
            if chunk_id in score_map:
                score_map[chunk_id] += result.score * self.keyword_weight
            else:
                score_map[chunk_id] = result.score * self.keyword_weight
                chunk_map[chunk_id] = result.chunk
        
        # Create combined results
        combined_results = [
            SearchResult(chunk=chunk_map[chunk_id], score=score)
            for chunk_id, score in score_map.items()
        ]
        
        return combined_results


class MarkdownResultFormatter(IResultFormatter):
    """Markdown result formatter for search results."""
    
    def __init__(self, max_content_length: int = 500, include_metadata: bool = True):
        self.max_content_length = max_content_length
        self.include_metadata = include_metadata
    
    async def format(self, results: List[SearchResult], query: SearchQuery) -> str:
        """Format search results as markdown."""
        if not results:
            return f"No results found for query: **{query.text}**"
        
        formatted_parts = []
        
        # Header
        formatted_parts.append(f"# Search Results for: {query.text}")
        formatted_parts.append(f"Found {len(results)} relevant results\n")
        
        # Results
        for i, result in enumerate(results, 1):
            chunk = result.chunk
            
            # Result header
            formatted_parts.append(f"## Result {i} (Score: {result.score:.3f})")
            
            # Source information
            formatted_parts.append(f"**Source:** `{chunk.source_path.name}`")
            if chunk.position > 0:
                formatted_parts.append(f"**Section:** {chunk.position}")
            
            # Content preview
            content = chunk.content.strip()
            if len(content) > self.max_content_length:
                content = content[:self.max_content_length] + "..."
            
            formatted_parts.append("**Content:**")
            formatted_parts.append(f"```\n{content}\n```")
            
            # Tags and links
            if chunk.tags:
                formatted_parts.append(f"**Tags:** {', '.join(f'#{tag}' for tag in chunk.tags)}")
            
            if chunk.outgoing_links:
                formatted_parts.append(f"**Links:** {', '.join(f'[[{link}]]' for link in chunk.outgoing_links)}")
            
            # Metadata
            if self.include_metadata and chunk.metadata:
                metadata_items = []
                for key, value in chunk.metadata.items():
                    if key not in ['filename', 'file_size', 'file_extension']:  # Skip basic metadata
                        metadata_items.append(f"{key}: {value}")
                
                if metadata_items:
                    formatted_parts.append(f"**Metadata:** {', '.join(metadata_items)}")
            
            # Modified date
            formatted_parts.append(f"**Modified:** {chunk.modified_at.strftime('%Y-%m-%d %H:%M')}")
            
            formatted_parts.append("")  # Empty line between results
        
        # Citations
        formatted_parts.append("---")
        formatted_parts.append("## Sources")
        unique_sources = list(set(result.chunk.source_path for result in results))
        for source in unique_sources:
            formatted_parts.append(f"- `{source}`")
        
        return "\n".join(formatted_parts)


class CompactResultFormatter(IResultFormatter):
    """Compact result formatter for brief search results."""
    
    def __init__(self, max_results: int = 3, snippet_length: int = 150):
        self.max_results = max_results
        self.snippet_length = snippet_length
    
    async def format(self, results: List[SearchResult], query: SearchQuery) -> str:
        """Format search results in a compact format."""
        if not results:
            return f"No results found for: {query.text}"
        
        formatted_parts = []
        
        # Limit results for compact display
        display_results = results[:self.max_results]
        
        formatted_parts.append(f"**Found {len(results)} results for:** {query.text}\n")
        
        for i, result in enumerate(display_results, 1):
            chunk = result.chunk
            
            # Create snippet
            content = chunk.content.strip()
            if len(content) > self.snippet_length:
                content = content[:self.snippet_length] + "..."
            
            # Format result
            source_name = chunk.source_path.name
            formatted_parts.append(f"**{i}.** {source_name} (Score: {result.score:.2f})")
            formatted_parts.append(f"   {content}")
            
            if chunk.tags:
                formatted_parts.append(f"   Tags: {', '.join(f'#{tag}' for tag in chunk.tags[:3])}")
            
            formatted_parts.append("")
        
        if len(results) > self.max_results:
            formatted_parts.append(f"... and {len(results) - self.max_results} more results")
        
        return "\n".join(formatted_parts)


class JSONResultFormatter(IResultFormatter):
    """JSON result formatter for API responses."""
    
    async def format(self, results: List[SearchResult], query: SearchQuery) -> str:
        """Format search results as JSON."""
        import json
        
        formatted_results = []
        
        for result in results:
            chunk = result.chunk
            formatted_result = {
                "score": result.score,
                "content": chunk.content,
                "source_path": str(chunk.source_path),
                "position": chunk.position,
                "tags": chunk.tags,
                "outgoing_links": chunk.outgoing_links,
                "metadata": chunk.metadata,
                "created_at": chunk.created_at.isoformat(),
                "modified_at": chunk.modified_at.isoformat(),
            }
            formatted_results.append(formatted_result)
        
        response = {
            "query": query.text,
            "total_results": len(results),
            "results": formatted_results
        }
        
        return json.dumps(response, indent=2, ensure_ascii=False)