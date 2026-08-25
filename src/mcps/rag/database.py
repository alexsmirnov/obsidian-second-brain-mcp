"""
Vector database implementations for the RAG search system.
"""

import logging
from datetime import timedelta
from pathlib import Path

import lancedb
import pyarrow as pa
from lancedb import AsyncConnection, AsyncTable
from lancedb.index import FTS, LabelList
from lancedb.pydantic import pydantic_to_schema
from lancedb.rerankers import Reranker, RRFReranker

from .interfaces import (
    Chunk,
    IEmbeddingService,
    IVectorStore,
    Link,
    NoteLinks,
    NotInitializedError,
    SearchScope,
    dedupe_links,
)

logger = logging.getLogger("mcps.database")

_LINK_QUERY_COLUMNS = [
    "wikilink_name",
    "title",
    "description",
    "position",
    "links",
    "link_types",
]


class LanceDBStore(IVectorStore):
    """
    LanceDB vector store implementation with full text search (FTS) capabilities.

    This class provides both vector similarity search and full text search using
    LanceDB's Tantivy-based FTS engine. It supports:
    - Vector embeddings for semantic search
    - Full text search with English stemming
    - Hybrid search combining both approaches
    - Configurable FTS indexing on multiple columns

    Example usage:
        store = LanceDBStore(Path("./db"), "chunks")
        await store.initialize(create_fts_index=True)

        # Hybrid search
        results = await store.search("deep learning")
    """

    db: AsyncConnection
    table: AsyncTable

    def __init__(
        self,
        db_path: Path,
        embedding_service: IEmbeddingService,
        table_name: str = "chunks",
        reranker: Reranker | None = None,
    ):
        self.db_path = db_path
        self.table_name = table_name
        self.reranker = reranker or RRFReranker(return_score="all")
        self.embedding_service = embedding_service
        self._initialized = False

    async def initialize(self) -> None:
        """
        Initialize the LanceDB vector store.
        """
        try:
            logger.info(f"Initializing LanceDB at {self.db_path}")

            # Create database directory if it doesn't exist
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            # Connect to database
            self.db = await lancedb.connect_async(
                self.db_path, read_consistency_interval=timedelta(seconds=1)
            )

            # Create or open table using Pydantic schema
            try:
                self.table = await self.db.open_table(self.table_name)
                logger.info(f"Opened existing table: {self.table_name}")
            except Exception:
                # Table doesn't exist, create it using Pydantic schema
                # Append or replace embeddings field with correct dimension
                schema: pa.Schema = pydantic_to_schema(Chunk)
                emb_field = pa.field(
                    "embeddings",
                    pa.list_(pa.float16(), self.embedding_service.ndims()),
                )
                embeddings_idx = schema.get_field_index("embeddings")
                if embeddings_idx < 0:
                    schema = schema.append(emb_field)
                else:
                    schema = schema.set(embeddings_idx, emb_field)
                self.table = await self.db.create_table(self.table_name, schema=schema)
                logger.info(f"Created new table: {self.table_name}")
                # Create indexes
                await self.reindex(True)

            self._initialized = True
            logger.info("LanceDB initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize LanceDB: {e}")
            raise

    async def cleanup(self) -> None:
        if self._initialized:
            self.table.close()
            self.db.close()
            del self.table
            del self.db
        self._initialized = False

    async def store(self, chunks: list[Chunk]) -> None:
        """Store chunks with their embeddings."""
        if not self._initialized:
            raise NotInitializedError(
                "LanceDBStore is not initialized. Call await store.initialize() first."
            )
        logger.info(f"Storing {len(chunks)} chunks in LanceDB")
        if not chunks:
            return

        try:
            # Process chunks and generate embeddings if needed
            texts = [
                # c.content
                # f"Description: {c.wikilink_name} Content:{c.content}"
                f"Doc: {c.wikilink_name} Content:{c.content}"
                for c in chunks
            ]
            embeddings = await self.embedding_service.documents_embeddings(texts)
            processed_chunks = [
                chunk.model_copy(update={"embeddings": embedding}).model_dump()
                for embedding, chunk in zip(embeddings, chunks, strict=True)
            ]
            await self.table.add(processed_chunks)
            logger.info(f"Added {len(processed_chunks)} chunks to LanceDB")
        except Exception as e:
            logger.error(f"Failed to store chunks in LanceDB: {e}")
            raise

    @staticmethod
    def _escape_sql_string(val: str) -> str:
        return val.replace("'", "''")

    async def search(
        self,
        query: str,
        hypotetical_document: str | None = None,
        tags: list[str] | None = None,
        file_path: str | None = None,
        scope: SearchScope = SearchScope.ALL,
        limit: int = 5,
    ) -> list[Chunk]:
        """Search for chunks that match query and filters.

        Args:
            query (str): The search query text.
            hypotetical_document: Expected result document, if present used for
                vector search instead of query.
            tags: List of tags to filter by. All must be present.
            file_path: Substring of source_path to filter results.
            scope: Where to search: content, title, description, or all.
            limit (int, optional): Maximum number of results to return. Defaults to 5.
        """
        if not self._initialized:
            raise NotInitializedError(
                "LanceDBStore is not initialized. Call await store.initialize() first."
            )

        # Calculate embedding for the query
        query_embedding = await self.embedding_service.query_embeddings(
            hypotetical_document or query
        )

        # Apply scope filter
        if scope == SearchScope.CONTENT:
            columns = ["content"]
        elif scope == SearchScope.TITLE:
            columns = ["title"]
        elif scope == SearchScope.DESCRIPTION:
            columns = ["description"]
        else:
            columns = ["content", "title", "description"]
        # Start the search query
        try:
            query_builder = self.table.query()

            query_builder = query_builder.nearest_to(query_embedding).column(
                "embeddings"
            )  # .distance_range(upper_bound=1000.0)
            query_builder = query_builder.nearest_to_text(query, columns=columns)

            # Apply filters: collect all predicates and issue a single .where()
            # call, because chained .where() calls replace each other on the
            # underlying async Rust query builder.
            predicates: list[str] = []
            if tags:
                tags_array = ",".join(
                    [f"'{self._escape_sql_string(t)}'" for t in tags]
                )
                predicates.append(f"array_has_all(tags, [{tags_array}])")
            if file_path:
                escaped_file_path = self._escape_sql_string(file_path)
                predicates.append(f"source_path LIKE '{escaped_file_path}%'")
            if predicates:
                query_builder = query_builder.where(" AND ".join(predicates))

            query_builder = query_builder.rerank(self.reranker)
            # Go!
            results = await query_builder.limit(limit).to_list()
            logger.info(
                f"Found {len(results)} results for query '{query}' with tags "
                f"{tags} and file path '{file_path}' and limit {limit}"
            )
            return [Chunk.model_validate(result) for result in results]

        except Exception as e:
            logger.error(f"Failed to search chunks in LanceDB: {e}")
            raise

    async def delete(self, source_paths: list[str]) -> None:
        """Delete chunks by their paths.
        WARNING: delete operaton in lancedb may corrupt indexes, so it should
        be followed by reindexing.
        """
        if not self._initialized:
            raise NotInitializedError(
                "LanceDBStore is not initialized. Call await store.initialize() first."
            )

        try:
            in_list = ",".join(
                [f"'{self._escape_sql_string(p)}'" for p in source_paths]
            )
            delete_clause = f"source_path IN ({in_list})"
            logger.info(
                f"Deleting chunks with source paths: {source_paths} using "
                f"clause: {delete_clause}"
            )
            await self.table.delete(delete_clause)

            logger.info(f"Deleted {len(source_paths)} paths from LanceDB")

        except Exception as e:
            logger.error(f"Failed to delete chunks from LanceDB: {e}")
            raise

    async def reindex(
        self,
        replace: bool = True
    ) -> None:
        """
        Create database indexes

        Args:
            replace: Replace existing index if it exists (default: True)

        Raises:
            Exception: If FTS index creation fails

        Example:
            # Create FTS index on content column
            await store.create_fts_index()

        """
        wait_time = timedelta(seconds=5)
        await self.table.optimize(cleanup_older_than=wait_time,delete_unverified=True) 
        if replace:

            for column in ["content", "title", "description"]:
                try:
                    await self.table.create_index(
                        column,
                        config=FTS(
                            base_tokenizer="simple",
                            max_token_length=30, # Drops huge base64 strings or logs from masking true chunk size
                            ),
                        replace=replace,
                        wait_timeout=wait_time,
                    )

                except Exception as e:
                    logger.error(f"Failed to create FTS index for column {column}: {e}")
                    raise
            try:
                pass
                # await self.table.create_index(
                #     column="embeddings",
                #     config=IvfPq()
                # )
            except Exception as e:
                logger.error(f"Failed to create IvPf index for embeddings: {e}")
                raise
            try:
                await self.table.create_index(
                    column="tags", config=LabelList(), wait_timeout=wait_time, replace=replace
                )
            except Exception as e:
                logger.error(f"Failed to create list index for tags: {e}")
                raise
            for column in ["links", "link_types"]:
                try:
                    await self.table.create_index(
                        column=column,
                        config=LabelList(),
                        wait_timeout=wait_time,
                        replace=replace,
                    )
                except Exception as e:
                    # Link-column indexes are a fetch-volume optimization;
                    # a failure must not break indexing.
                    logger.error(f"Failed to create list index for {column}: {e}")
        indices = await self.table.list_indices()
        logger.info([f"index {idx}," for idx in indices])

    async def sources(self) -> dict[str, float]:
        """Get last updates to source documents."""
        if not self._initialized:
            raise NotInitializedError(
                "LanceDBStore is not initialized. Call await store.initialize() first."
            )
        try:
            # Fetch only the required columns as a pyarrow Table
            arrow_table = (
                await self.table.query()
                .select(["source_path", "modified_at"])
                .to_arrow()
            )
            if arrow_table.num_rows == 0:
                return {}
            grouped = arrow_table.group_by(["source_path"]).aggregate(
                [("modified_at", "min")]
            )
            # grouped is a pyarrow Table with columns: source_path, modified_at_min
            return {
                grouped["source_path"][i].as_py(): grouped["modified_at_min"][i].as_py()
                for i in range(grouped.num_rows)
            }
        except Exception as e:
            logger.error(f"Failed to fetch source updates from LanceDB: {e}")
            raise

    async def get_chunks_by_ids(self, ids: list[str]) -> list[Chunk]:
        """Return chunks whose id is in the provided list.

        Args:
            ids: List of chunk ids to fetch.

        Returns:
            list[Chunk]: Matching chunks. Missing ids are silently ignored.
        """
        if not self._initialized:
            raise NotInitializedError(
                "LanceDBStore is not initialized. Call await store.initialize() first."
            )
        if not ids:
            return []
        try:
            id_list = ",".join(
                [f"'{self._escape_sql_string(id_)}'" for id_ in ids]
            )
            rows = await self.table.query().where(f"id IN ({id_list})").to_list()
            return [Chunk.model_validate(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to fetch chunks by ids from LanceDB: {e}")
            raise

    async def get_sources_by_name(self, wikilink_name: str) -> list[str]:
        """Return distinct source paths matching an exact wikilink note name."""
        if not self._initialized:
            raise NotInitializedError(
                "LanceDBStore is not initialized. Call await store.initialize() first."
            )
        try:
            escaped_name = self._escape_sql_string(wikilink_name)
            rows = (
                await self.table.query()
                .where(f"wikilink_name = '{escaped_name}'")
                .select(["source_path"])
                .to_list()
            )
            return sorted({row["source_path"] for row in rows})
        except Exception as e:
            logger.error(f"Failed to fetch source paths from LanceDB: {e}")
            raise

    @staticmethod
    def _group_rows_by_note(rows: list[dict]) -> dict[str, list[dict]]:
        """Group chunk rows by note, each group sorted by position."""
        groups: dict[str, list[dict]] = {}
        for row in rows:
            groups.setdefault(row["wikilink_name"], []).append(row)
        return {
            note: sorted(group, key=lambda r: r["position"])
            for note, group in groups.items()
        }

    @staticmethod
    def _links_from_row(row: dict, note: str) -> list[Link]:
        """Reconstruct typed links from one chunk row, skipping corrupt rows."""
        targets = row["links"] or []
        types = row["link_types"] or []
        if len(targets) != len(types):
            logger.warning(
                f"Skipping corrupt chunk row for note '{note}': "
                f"links ({len(targets)}) and link_types ({len(types)}) misaligned"
            )
            return []
        return [Link(type=t, target=n) for t, n in zip(types, targets, strict=True)]

    @staticmethod
    def _first_non_none(rows: list[dict], column: str) -> str | None:
        return next((row[column] for row in rows if row[column] is not None), None)

    async def get_notes_with_links(
        self, wikilink_names: list[str]
    ) -> list[NoteLinks]:
        """Return outgoing typed links for the requested notes."""
        if not self._initialized:
            raise NotInitializedError(
                "LanceDBStore is not initialized. Call await store.initialize() first."
            )
        if not wikilink_names:
            return []
        name_list = ",".join(
            f"'{self._escape_sql_string(n)}'" for n in wikilink_names
        )
        rows = (
            await self.table.query()
            .where(f"wikilink_name IN ({name_list})")
            .select(_LINK_QUERY_COLUMNS)
            .to_list()
        )
        return [
            NoteLinks(
                note=note,
                title=self._first_non_none(group, "title"),
                description=self._first_non_none(group, "description"),
                links=dedupe_links(
                    link for row in group for link in self._links_from_row(row, note)
                ),
            )
            for note, group in self._group_rows_by_note(rows).items()
        ]

    async def get_notes_linking_to(
        self, targets: list[str], relation_types: list[str] | None = None
    ) -> list[NoteLinks]:
        """Return notes holding at least one link to a requested target.

        The array_has_any SQL predicate is a sound-but-inexact pre-filter:
        it cannot correlate positions across the links/link_types arrays.
        The exact (type, target) match below in Python is the filter of
        record; the SQL predicate only limits transferred rows.
        """
        if not self._initialized:
            raise NotInitializedError(
                "LanceDBStore is not initialized. Call await store.initialize() first."
            )
        if not targets:
            return []
        targets_array = ",".join(
            f"'{self._escape_sql_string(t)}'" for t in targets
        )
        predicates = [f"array_has_any(links, [{targets_array}])"]
        if relation_types:
            types_array = ",".join(
                f"'{self._escape_sql_string(t)}'" for t in relation_types
            )
            predicates.append(f"array_has_any(link_types, [{types_array}])")
        rows = (
            await self.table.query()
            .where(" AND ".join(predicates))
            .select(_LINK_QUERY_COLUMNS)
            .to_list()
        )
        wanted_targets = set(targets)
        wanted_types = set(relation_types) if relation_types else None
        result = []
        for note, group in self._group_rows_by_note(rows).items():
            links = dedupe_links(
                link
                for row in group
                for link in self._links_from_row(row, note)
                if link.target in wanted_targets
                and (wanted_types is None or link.type in wanted_types)
            )
            if not links:
                continue
            result.append(
                NoteLinks(
                    note=note,
                    title=self._first_non_none(group, "title"),
                    description=self._first_non_none(group, "description"),
                    links=links,
                )
            )
        return result
