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

logger = logging.getLogger("mcps")


class SemanticSearchEngine(ISearchEngine):
    """Semantic search engine using embeddings and vector similarity."""
    
    def __init__(
        self, 
        vector_store: IVectorStore,
    ):
        self.vector_store = vector_store
    
    async def search(self, query: SearchQuery) -> list[Chunk]:
        """Perform a semantic search operation."""
        return await self.vector_store.search(
            query=query.text,
            tags=query.tags,
            file_path=query.path,
            scope=query.scope,
            limit=5
        )
    

class MarkdownResultFormatter(IResultFormatter):
    """Markdown result formatter for search results."""
    
    def __init__(self, max_content_length: int = 500, include_metadata: bool = True):
        self.max_content_length = max_content_length
        self.include_metadata = include_metadata
    
    async def format(self, results: list[Chunk], query: SearchQuery) -> str:
        """Format search results as markdown."""
        if not results:
            return f"No results found for query: **{query.text}**"
        
        formatted_parts = []
        
        # Header
        formatted_parts.append(f"# Search Results for: {query.text}")
        formatted_parts.append(f"Found {len(results)} relevant results\n")
        
        # Results
        for i, chunk in enumerate(results, 1):
            
            # Result header
            score_text = f" (Score: {getattr(chunk, 'score', 0.0):.3f})" if hasattr(chunk, 'score') else ""
            formatted_parts.append(f"## Result {i}{score_text}")
            
            # Source information
            formatted_parts.append(f"**Source:** `{chunk.source_path}`")
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
            if self.include_metadata :
                metadata_items = []
                
                if metadata_items:
                    formatted_parts.append(f"**Metadata:** {', '.join(metadata_items)}")
            
            # Modified date
            formatted_parts.append(f"**Modified:** {chunk.modified_at.strftime('%Y-%m-%d %H:%M')}")
            
            formatted_parts.append("")  # Empty line between results
        
        # Citations
        formatted_parts.append("---")
        formatted_parts.append("## Sources")
        unique_sources = list(set(chunk.source_path for chunk in results))
        for source in unique_sources:
            formatted_parts.append(f"- `{source}`")
        
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

