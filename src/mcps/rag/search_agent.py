"""
Search agent for the RAG system.
"""

from typing import List

from .interfaces import (
    Chunk,
    SearchQuery,
    IEmbeddingService,
    IResultFormatter,
    ISearchEngine,
    IVectorStore,
    ILLM,
)

class SearchAgent(ISearchEngine):
    def __init__(self, vector_store: IVectorStore, llm: ILLM):
        self.vector_store = vector_store
        self.llm = llm

    async def search(self, query: SearchQuery) -> str:
        """High-level search routine orchestrating all internal steps.

        The method delegates concrete work to private helper methods.  Each
        helper returns the data required by the subsequent one so that the
        pipeline is explicit and easy to follow/extend.
        """

        # Step 1 – rewrite/expand the original query text if necessary
        full_query: str = await self._rewrite_query(query)

        # Step 2 – infer tags that can narrow the search scope
        #  from thefull query and instructions note
        tags: list[str] = await self._infer_tags(query)

        # Step 3 – run the initial vector search using the prepared query & tags
        search_results: list[Chunk] = await self._initial_search(full_query, tags)

        # Step 4 – fetch notes from vector_store linked *from* the initially found chunks
        outgoing_links: set[str] = { link for link in chunk.outgoing_links for chunk in search_results}
        linked_notes: list[Chunk] = await self._collect_linked_notes(query, outgoing_links)

        # Step 5 – fetch notes from vector_store that link *to* the search results
        incoming_notes: list[Chunk] = await self._collect_incoming_notes(query, search_results)

        # Step 6 – combine every piece of context we have gathered so far
        all_chunks: set[Chunk] = set(search_results + linked_notes + incoming_notes)

        # Step 7 – finally generate answer for the user
        answer: str = await self._generate_answer(query, all_chunks)

        return answer

    # ---------------------------------------------------------------------
    # Private helper methods – implementation will be provided later.  For
    # now they only declare the contract so that the orchestration above is
    # type-safe and explicit.
    # ---------------------------------------------------------------------

    async def _rewrite_query(self, query: SearchQuery) -> str:  # noqa: D401
        """Rewrite/expand the original query using an LLM or heuristics.
        Args:
            query (SearchQuery): The original query.
        Returns:
            str: The rewritten query.
        """
        pass

    async def _infer_tags(self, query: SearchQuery) -> list[str]:
        """Infer relevant tag filters from the query text,
        requested tags, and instructions note by LLM.
        Args:
            query (str): The user query.
        Returns:
            List[str]: The inferred tags.
        """
        pass

    async def _initial_search(self, full_query: str, tags: list[str]) -> list[Chunk]:
        """Search the vector store for chunks matching the query & tags."""
        pass

    async def _collect_linked_notes(self, query: SearchQuery, links: set[str]) -> list[Chunk]:
        """Retrieve notes linked *from* the given links.
           Only return chunks that are relevant to the query.
        Args:
            query (SearchQuery): The user query.
            chunks (List[Chunk]): The chunks to collect linked notes from.
        Returns:
            List[Chunk]: The linked notes.
        """
        pass

    async def _collect_incoming_notes(self, query: SearchQuery, chunks: List[Chunk]) -> list[Chunk]:
        """Retrieve notes that link *to* the given chunks (incoming links)
           and relevant to the user query
        Args:
            query (SearchQuery): The user query.
            chunks (List[Chunk]): The chunks to collect incoming notes from.
        Returns:
            List[Chunk]: The incoming notes.
        """
        pass

    async def _generate_answer(self, query: SearchQuery, chunks: set[Chunk]) -> str:
        """Generate the final answer using all collected context chunks."""
        pass