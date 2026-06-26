"""
Search engine and result formatting implementations for the RAG search system.
"""

import logging
import re

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

QUERY_SYSTEM_PROMPT = """Act as a technical knowledge base. Generate a short,
3-to-5 line mock document chunk that directly resolves the following search query.

- If the query implies code or implementation: Write a raw markdown code block showing function definitions, key libraries, and syntax architecture.
- If the query is scientific or theoretical: Use precise equations, variables, and data dense technical notation.
- If query is finacial or legal, use professional language
- Do not explain the code/math. Do not write a conversational intro. Keep it under 120 tokens total."""


class HypotheticalDocumentGenerator:
    """Generate HyDE documents for vector search query expansion."""

    def __init__(self, model: BaseChatModel) -> None:
        self.model = model

    async def generate(self, query: str) -> str:
        prompt = self._create_prompt(query)
        response = await self.model.ainvoke(
            [SystemMessage(content=QUERY_SYSTEM_PROMPT), HumanMessage(content=prompt)]
        )
        content = response.text
        return self._clean_hyde_output(content)

    @staticmethod
    def _create_prompt(query: str) -> str:
        return f""" Query: {query}

Hypothetical Chunk:"""


    @staticmethod
    def _clean_hyde_output(raw_text: str) -> str:
        """
        Cleans and standardizes the hypothetical text from LLM
        before sending it to the BGE-M3 embedding engine.
        """
        if not raw_text:
            return ""
            
        # 1. Standard text cleanup (strip leading/trailing whitespace & newlines)
        cleaned = raw_text.strip()
        
        # 2. Defensive check: Remove conversational opening phrases if the LLM slips up
        conversational_prefixes = [
            r"^(here is|here's|this is|a hypothetical|mock snippet|sure, here|snippet:)\b.*?\n",
            r"^(certainly|absolutely|hope this helps)[.,!:\s]*"
        ]
        for pattern in conversational_prefixes:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
        
        cleaned = cleaned.strip()
        
        # 3. Code Block Normalization (Crucial for BGE-M3 code chunk matching)
        # If the LLM wrapped the entire output in triple backticks, extract JUST the contents.
        # We want BGE-M3 to match raw syntax, not the backtick strings themselves.
        code_block_match = re.search(r"```(?:[a-zA-Z0-9+#-]+)?\n(.*?)```", cleaned, re.DOTALL)
        if code_block_match:
            cleaned = code_block_match.group(1).strip()
        else:
            # If it's a mixed snippet, just clean up stray, unclosed backticks
            cleaned = cleaned.replace("```", "")
            
        # 4. Final Truncation Guard
        # Ensure any unexpected runaway output doesn't dilute our 256-token target size.
        # Split by whitespace to approximate word limits, or feed into your model's tokenizer.
        words = cleaned.split()
        if len(words) > 120:
            cleaned = " ".join(words[:120])
            
        return cleaned.strip()

class SemanticSearchEngine(ISearchEngine):
    """Semantic search engine using embeddings and vector similarity."""

    def __init__(
        self,
        vector_store: IVectorStore,
        limit: int = 25,
        min_score: float = 0.5,
        hypothetical_document_generator: HypotheticalDocumentGenerator | None = None,
        reranker: IRerankingService | None = None,
        neighbor_offset: int = 1,
    ) -> None:
        self.vector_store = vector_store
        self.limit = limit
        self.min_score = min_score
        self.hypothetical_document_generator = hypothetical_document_generator
        self.reranker = reranker
        self.neighbor_offset = neighbor_offset

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
        if self.neighbor_offset > 0:
            relevant_chunks = await self._merge_with_neighbors(relevant_chunks)
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
        """Filter all chunks with score less than min_score and removes duplicates"""
        seen = set()
        def _has_seen(id: str) -> bool:
            if id in seen:
                return True
            else:
                seen.add(id)
                return False
        return [
            chunk
            for chunk in chunks
            if getattr(chunk, "_relevance_score", 1.0) >= self.min_score
            and not _has_seen(chunk.id)
        ]

    async def _merge_with_neighbors(self, chunks: list[Chunk]) -> list[Chunk]:
        """Expand each result chunk into its neighbor window and merge overlaps."""
        windows_by_document = self._merged_windows_by_document(chunks)
        if not windows_by_document:
            return chunks
        neighbor_map = await self._fetch_neighbors(windows_by_document)
        return self._assemble_windows(chunks, windows_by_document, neighbor_map)

    def _merged_windows_by_document(
        self, chunks: list[Chunk]
    ) -> dict[str, list[tuple[int, int]]]:
        """Group each result's neighbor window per document, merging overlaps."""
        raw_windows: dict[str, list[tuple[int, int]]] = {}
        for chunk in chunks:
            document_id = self._document_id(chunk.id)
            start = max(0, chunk.position - self.neighbor_offset)
            end = chunk.position + self.neighbor_offset
            raw_windows.setdefault(document_id, []).append((start, end))
        return {
            document_id: self._merge_overlapping_windows(windows)
            for document_id, windows in raw_windows.items()
        }

    async def _fetch_neighbors(
        self, windows_by_document: dict[str, list[tuple[int, int]]]
    ) -> dict[str, Chunk]:
        """Fetch every chunk covered by the merged windows, keyed by id.
        """
        neighbor_ids = [
            f"{document_id}_{position}"
            for document_id in sorted(windows_by_document)
            for start, end in windows_by_document[document_id]
            for position in range(start, end + 1)
        ]
        neighbors = await self.vector_store.get_chunks_by_ids(neighbor_ids)
        return {chunk.id: chunk for chunk in neighbors}

    def _assemble_windows(
        self,
        chunks: list[Chunk],
        windows_by_document: dict[str, list[tuple[int, int]]],
        neighbor_map: dict[str, Chunk],
    ) -> list[Chunk]:
        """Emit one merged chunk per window, in result relevance order."""
        merged_results: list[Chunk] = []
        seen_windows: set[tuple[str, tuple[int, int]]] = set()
        for chunk in chunks:
            document_id = self._document_id(chunk.id)
            window = self._window_for_position(
                windows_by_document[document_id], chunk.position
            )
            if window is None or (document_id, window) in seen_windows:
                continue
            merged = self._merge_window(document_id, window, neighbor_map)
            if merged is None:
                continue
            seen_windows.add((document_id, window))
            merged_results.append(merged)
        return merged_results

    @staticmethod
    def _document_id(chunk_id: str) -> str:
        """Strip the trailing position from a chunk id."""
        return chunk_id.rsplit("_", 1)[0]

    @staticmethod
    def _merge_overlapping_windows(
        windows: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """Merge overlapping or touching windows into continuous ranges."""
        if not windows:
            return []
        sorted_windows = sorted(windows, key=lambda window: window[0])
        merged = [list(sorted_windows[0])]
        for start, end in sorted_windows[1:]:
            last = merged[-1]
            if start <= last[1] + 1:
                last[1] = max(last[1], end)
            else:
                merged.append([start, end])
        return [(start, end) for start, end in merged]

    @staticmethod
    def _window_for_position(
        windows: list[tuple[int, int]], position: int
    ) -> tuple[int, int] | None:
        """Return the merged window containing the position, if any."""
        for start, end in windows:
            if start <= position <= end:
                return (start, end)
        return None

    @classmethod
    def _merge_window(
        cls,
        document_id: str,
        window: tuple[int, int],
        neighbor_map: dict[str, Chunk],
    ) -> Chunk | None:
        """Combine every fetched chunk inside a window into a single chunk."""
        start, end = window
        window_chunks = [
            neighbor_map[chunk_id]
            for position in range(start, end + 1)
            if (chunk_id := f"{document_id}_{position}") in neighbor_map
        ]
        if not window_chunks:
            return None
        primary = window_chunks[0]
        min_position = window_chunks[0].position
        max_position = window_chunks[-1].position
        content = "\n\n".join(chunk.content for chunk in window_chunks)
        offset = min(chunk.offset for chunk in window_chunks)
        tags = sorted({tag for chunk in window_chunks for tag in chunk.tags})
        outgoing_links = sorted(
            {link for chunk in window_chunks for link in chunk.outgoing_links}
        )
        relevance_score = max(
            (
                getattr(chunk, "_relevance_score", 0.0)
                for chunk in window_chunks
                if hasattr(chunk, "_relevance_score")
            ),
            default=None,
        )
        merged = Chunk(
            id=f"{document_id}_{min_position}",
            content=content,
            title=primary.title,
            description=primary.description,
            source=primary.source,
            outgoing_links=list(outgoing_links),
            tags=list(tags),
            source_path=primary.source_path,
            wikilink_name=primary.wikilink_name,
            modified_at=primary.modified_at,
            position=min_position,
            offset=offset,
            file_size=primary.file_size,
        )
        if relevance_score is not None:
            object.__setattr__(merged, "_relevance_score", relevance_score)
        return merged


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
