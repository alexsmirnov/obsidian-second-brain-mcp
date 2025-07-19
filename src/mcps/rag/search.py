"""
Search engine and result formatting implementations for the RAG search system.
"""

import logging
from datetime import datetime
from typing import Any, Optional

from .interfaces import (
    Chunk,
    IEmbeddingService,
    IResultFormatter,
    ISearchEngine,
    IVectorStore,
    SearchQuery,
    SearchResult,
)

logger = logging.getLogger("mcps.search")


class SemanticSearchEngine(ISearchEngine):
    """Semantic search engine using embeddings and vector similarity."""
    
    def __init__(
        self, 
        vector_store: IVectorStore,
        formatter: IResultFormatter,
        limit: int = 25,
        min_score = 0.5
    ):
        self.vector_store = vector_store
        self.formatter = formatter
        self.limit = limit
        self.min_score = min_score
    
    async def search(self, query: SearchQuery) -> str:
        """Perform a semantic search operation."""
        chunks = await self.vector_store.search(
            query=query.text,
            tags=query.tags,
            file_path=query.path,
            scope=query.scope,
            limit=self.limit
        )
        relevant_chunks = [chunk for chunk in chunks if getattr(chunk, '_relevance_score', 1.0) >= self.min_score]
        logger.info(f"Search completed: found {len(relevant_chunks)} relevant results out of {len(chunks)} total")
        return await self.formatter.format(relevant_chunks, query)
    

class MarkdownResultFormatter(IResultFormatter):
    """Markdown result formatter for search results."""
    
    def __init__(self, max_content_length: int = 1000):
        self.max_content_length = max_content_length
    
    async def format(self, results: list[Chunk], query: SearchQuery) -> str:
        """Format search results as markdown."""
        if not results:
            return f"No results found for query: **{query.text}**"
        
        formatted_parts = []
        
        # Results
        for i, chunk in enumerate(results, 1):
            
            # Result header
            score_text = f" (Score: {getattr(chunk, '_relevance_score', 0.0):.3f})" if hasattr(chunk, '_relevance_score') else ""
            formatted_parts.append(f"# Result {i}{score_text} ")
            formatted_parts.append(f"**Source:** `{chunk.source_path}`")
            
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
            
            formatted_parts.append(f"**Modified:** {chunk.modified_at.strftime('%Y-%m-%d %H:%M')}")
            
            formatted_parts.append("")  # Empty line between results
        
        
        return "\n".join(formatted_parts)


class CompactResultFormatter(IResultFormatter):
    """Compact result formatter for brief search results."""
    
    def __init__(self, max_results: int = 3, snippet_length: int = 150):
        self.max_results = max_results
        self.snippet_length = snippet_length
    
    async def format(self, results: list[SearchResult], query: SearchQuery) -> str:
        """Format search results in a compact format."""
        if not results:
            return f"No results found for: {query.text}"
        
        formatted_parts = []
        
        # Limit results for compact display
        display_results = results[:self.max_results]
        
        # formatted_parts.append(f"**Found {len(results)} results for:** {query.text}\n")
        
        for i, result in enumerate(display_results, 1):
            chunk = result.chunk
            
            # Create snippet
            content = chunk.content.strip()
            if len(content) > self.snippet_length:
                content = content[:self.snippet_length] + "..."
            
            # Format result
            source_name = chunk.source_path
            score_text = f" (Score: {getattr(chunk, '_relevance_score', 0.0):.3f})" if hasattr(chunk, '_relevance_score') else ""
            formatted_parts.append(f"**{i}.** Source: {source_name}{score_text})")
            formatted_parts.append(f"   {content}")
            
            if chunk.tags:
                formatted_parts.append(f"Tags: {', '.join(f'#{tag}' for tag in chunk.tags[:10])}")
            if chunk.outgoing_links:
                formatted_parts.append(f"Links: {', '.join(f'[[{link}]]' for link in chunk.outgoing_links[:10])}")
            formatted_parts.append("")
        
        if len(results) > self.max_results:
            formatted_parts.append(f"... and {len(results) - self.max_results} more results")
        
        return "\n".join(formatted_parts)

