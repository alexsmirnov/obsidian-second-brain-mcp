"""Agentic search engine for the RAG system."""

import logging
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from .interfaces import Chunk, ISearchEngine, IVectorStore, SearchQuery

logger = logging.getLogger("mcps.rag.search_agent")


class SearchAgent(ISearchEngine):
    def __init__(self, vector_store: IVectorStore, llm: BaseChatModel):
        self.vector_store = vector_store
        self.llm = llm

    async def search(self, query: SearchQuery) -> str:
        """Run query expansion, graph expansion, and answer generation."""
        full_query = await self._rewrite_query(query)
        tags = await self._infer_tags(query)
        search_results = await self._initial_search(full_query, tags)

        outgoing_links = {
            link
            for chunk in search_results
            for link in chunk.outgoing_links
        }
        linked_notes = await self._collect_linked_notes(query, outgoing_links)
        incoming_notes = await self._collect_incoming_notes(query, search_results)
        all_chunks = set(search_results + linked_notes + incoming_notes)
        return await self._generate_answer(query, all_chunks)

    async def _rewrite_query(self, query: SearchQuery) -> str:
        prompt = f"Rewrite this search query for semantic retrieval: {query.text}"
        try:
            rewritten = (await self._generate_text(prompt)).strip()
        except Exception:
            logger.exception("Failed to rewrite search query")
            return query.text
        return rewritten or query.text

    async def _infer_tags(self, query: SearchQuery) -> list[str]:
        prompt = (
            "Infer relevant Obsidian tags for this search query. "
            "Return comma-separated tag names only.\n"
            f"Query: {query.text}"
        )
        try:
            inferred_tags = _parse_tags(await self._generate_text(prompt))
        except Exception:
            logger.exception("Failed to infer search tags")
            return query.tags

        return list(dict.fromkeys([*query.tags, *inferred_tags]))

    async def _initial_search(self, full_query: str, tags: list[str]) -> list[Chunk]:
        try:
            chunks = await self.vector_store.search(query=full_query, tags=tags)
        except Exception:
            logger.exception("Initial vector search failed")
            return []
        return _dedupe_chunks(chunks)

    async def _collect_linked_notes(
        self,
        query: SearchQuery,
        links: set[str],
    ) -> list[Chunk]:
        if not links:
            return []

        try:
            chunks = [
                chunk
                for link in links
                for chunk in await self.vector_store.search(
                    query=query.text,
                    file_path=link,
                )
            ]
        except Exception:
            logger.exception("Failed to collect linked notes")
            return []
        return _dedupe_chunks(chunks)

    async def _collect_incoming_notes(
        self,
        query: SearchQuery,
        chunks: list[Chunk],
    ) -> list[Chunk]:
        try:
            incoming = [
                incoming_chunk
                for chunk in chunks
                for incoming_chunk in await self.vector_store.search(
                    query=f"[[{Path(chunk.source_path).stem}]] {query.text}",
                )
            ]
        except Exception:
            logger.exception("Failed to collect incoming notes")
            return []
        return _dedupe_chunks(incoming)

    async def _generate_answer(self, query: SearchQuery, chunks: set[Chunk]) -> str:
        context = "\n\n".join(
            f"Source: {chunk.source_path}\n{chunk.content}"
            for chunk in sorted(chunks, key=lambda chunk: chunk.id)
        )
        prompt = (
            "Answer the user query using only the context below. "
            "If the context is insufficient, say so.\n\n"
            f"Query: {query.text}\n\nContext:\n{context}"
        )
        try:
            answer = (await self._generate_text(prompt)).strip()
        except Exception:
            logger.exception("Failed to generate search answer")
            return "Unable to generate an answer from the retrieved context."
        return answer or "Unable to generate an answer from the retrieved context."

    async def _generate_text(self, prompt: str) -> str:
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        if isinstance(response.content, str):
            return response.content
        return str(response.content)


def _parse_tags(text: str) -> list[str]:
    raw_tags = text.replace("\n", ",").split(",")
    return [tag.strip().lstrip("#") for tag in raw_tags if tag.strip()]


def _dedupe_chunks(chunks: list[Chunk]) -> list[Chunk]:
    return list({chunk.id: chunk for chunk in chunks}.values())
