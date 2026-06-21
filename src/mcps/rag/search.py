"""
Search engine and result formatting implementations for the RAG search system.
"""

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .interfaces import (
    Chunk,
    IResultFormatter,
    ISearchEngine,
    IVectorStore,
    SearchQuery,
    SearchResult,
)
from .reranking import IRerankingService

logger = logging.getLogger("mcps.search")

QUERY_SYSTEM_PROMPT = """ Write a concise hypothetical text that would answer
the search query below. Include likely terminology, entities, and concepts, but
do not answer conversationally. Generate one paragraph with 5 sentences"""


class HypotheticalDocumentGenerator:
    """Generate HyDE documents for vector search query expansion."""

    def __init__(self, model: BaseChatModel) -> None:
        self.model = model

    async def generate(self, query: str) -> str:
        prompt = self._create_prompt(query)
        response = await self.model.ainvoke(
            [SystemMessage(content=QUERY_SYSTEM_PROMPT), HumanMessage(content=prompt)]
        )
        content = response.content
        if isinstance(content, str):
            return content.strip()
        return str(content).strip()

    @staticmethod
    def _create_prompt(query: str) -> str:
        return f""" Query: {query}

Hypothetical document:"""


class SemanticSearchEngine(ISearchEngine):
    """Semantic search engine using embeddings and vector similarity."""

    def __init__(
        self,
        vector_store: IVectorStore,
        limit: int = 25,
        min_score: float = 0.5,
        hypothetical_document_generator: HypotheticalDocumentGenerator | None = None,
        reranker: IRerankingService | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.limit = limit
        self.min_score = min_score
        self.hypothetical_document_generator = hypothetical_document_generator
        self.reranker = reranker

    async def search(self, query: SearchQuery) -> list[Chunk]:
        """Perform a semantic search operation."""
        hypothetical_document = await self._generate_hypothetical_document(query.text)
        logger.info(
            "Search query:  %s , hypotetical document: %s",
            query,
            hypothetical_document,
        )
        chunks = await self.vector_store.search(
            query=query.text,
            hypotetical_document=hypothetical_document,
            tags=query.tags,
            file_path=query.path,
            scope=query.scope,
            limit=self.limit,
        )
        relevant_chunks = self._filter_by_min_score(chunks)
        if self.reranker is not None:
            try:
                relevant_chunks = await self.reranker.rerank(
                    query.text, relevant_chunks
                )
            except Exception:
                logger.exception(
                    "Search result reranking failed; returning vector results"
                )
        logger.info(
            "Search completed: found %s relevant results out of %s total",
            len(relevant_chunks),
            len(chunks),
        )
        return relevant_chunks

    async def _generate_hypothetical_document(self, query: str) -> str | None:
        if self.hypothetical_document_generator is None:
            return None
        try:
            hypothetical_document = await self.hypothetical_document_generator.generate(
                query
            )
        except Exception:
            logger.exception(
                "Hypothetical document generation failed; using original query"
            )
            return None
        hypothetical_document = hypothetical_document.strip()
        if not hypothetical_document:
            return None
        return hypothetical_document

    def _filter_by_min_score(self, chunks: list[Chunk]) -> list[Chunk]:
        return [
            chunk
            for chunk in chunks
            if getattr(chunk, "_relevance_score", 1.0) >= self.min_score
        ]


class MarkdownResultFormatter(IResultFormatter):
    """Markdown result formatter for search results."""

    def __init__(self, max_content_length: int = 1000):
        self.max_content_length = max_content_length

    async def format(self, results: list[Chunk], query: SearchQuery) -> str:
        """Format search results as markdown."""
        if not results:
            return f"No results found for query: **{query.text}**"

        formatted_parts = []

        for i, chunk in enumerate(results, 1):
            score_text = self._format_score(chunk)
            formatted_parts.append(f"# Result {i}{score_text} ")
            formatted_parts.append(f"**Source:** `{chunk.source_path}`")

            content = chunk.content.strip()
            if len(content) > self.max_content_length:
                content = content[: self.max_content_length] + "..."

            formatted_parts.append("**Content:**")
            formatted_parts.append(f"```\n{content}\n```")

            if chunk.tags:
                tags = ", ".join(f"#{tag}" for tag in chunk.tags)
                formatted_parts.append(f"**Tags:** {tags}")

            if chunk.outgoing_links:
                links = ", ".join(f"[[{link}]]" for link in chunk.outgoing_links)
                formatted_parts.append(f"**Links:** {links}")

            modified_at = chunk.modified_at.strftime("%Y-%m-%d %H:%M")
            formatted_parts.append(f"**Modified:** {modified_at}")

            formatted_parts.append("")  # Empty line between results

        return "\n".join(formatted_parts)

    @staticmethod
    def _format_score(chunk: Chunk) -> str:
        if not hasattr(chunk, "_relevance_score"):
            return ""
        return f" (Score: {getattr(chunk, '_relevance_score', 0.0):.3f})"


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

        display_results = results[: self.max_results]

        for i, result in enumerate(display_results, 1):
            chunk = result.chunk

            content = chunk.content.strip()
            if len(content) > self.snippet_length:
                content = content[: self.snippet_length] + "..."

            source_name = chunk.source_path
            score_text = MarkdownResultFormatter._format_score(chunk)
            formatted_parts.append(f"**{i}.** Source: {source_name}{score_text})")
            formatted_parts.append(f"   {content}")

            if chunk.tags:
                tags = ", ".join(f"#{tag}" for tag in chunk.tags[:10])
                formatted_parts.append(f"Tags: {tags}")
            if chunk.outgoing_links:
                links = ", ".join(f"[[{link}]]" for link in chunk.outgoing_links[:10])
                formatted_parts.append(f"Links: {links}")
            formatted_parts.append("")

        if len(results) > self.max_results:
            remaining_count = len(results) - self.max_results
            formatted_parts.append(f"... and {remaining_count} more results")

        return "\n".join(formatted_parts)
