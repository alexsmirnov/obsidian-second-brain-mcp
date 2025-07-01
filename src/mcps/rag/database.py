"""
Vector database implementations for the RAG search system.
"""

from datetime import datetime, timedelta
import json
import logging
from math import log
from pathlib import Path
import time
from typing import Any, Dict

import lancedb
from lancedb import AsyncConnection, AsyncTable
from lancedb.embeddings import EmbeddingFunction
from lancedb.pydantic import pydantic_to_schema
from lancedb.index import FTS, IvfPq, LabelList
from lancedb.rerankers import RRFReranker, VoyageAIReranker, Reranker
import pyarrow as pa

from .interfaces import Chunk, IVectorStore, NotInitializedError, SearchScope, SourceUpdates

logger = logging.getLogger("mcps")


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
    
    def __init__(self, db_path: Path, embedding_function: EmbeddingFunction, table_name: str = "chunks", reranker: Reranker = RRFReranker(return_score='all')):
        self.db_path = db_path
        self.table_name = table_name
        self.reranker = reranker
        self.db: None | AsyncConnection = None
        self.table: AsyncTable | None = None
        self.embedding_function: EmbeddingFunction = embedding_function
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
            self.db = await lancedb.connect_async(self.db_path, read_consistency_interval= timedelta(seconds=1))
            
            # Create or open table using Pydantic schema
            try:
                self.table = await self.db.open_table(self.table_name)
                logger.info(f"Opened existing table: {self.table_name}")
            except Exception:
                # Table doesn't exist, create it using Pydantic schema
                schema: pa.Schema = pydantic_to_schema(Chunk)
                schema = schema.append(pa.field("embeddings", pa.list_(pa.float16(), self.embedding_function.ndims())))
                self.table = await self.db.create_table(self.table_name, schema=schema )
                logger.info(f"Created new table: {self.table_name}")
                # Create indexes
                await self.reindex()
            
            self._initialized = True
            logger.info("LanceDB initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LanceDB: {e}")
            raise
    
    async def store(self, chunks: list[Chunk]) -> None:
        """Store chunks with their embeddings."""
        if not self._initialized:
            raise NotInitializedError("LanceDBStore is not initialized. Call await store.initialize() first.")
        
        if not chunks:
            return
        try:
            # Process chunks and generate embeddings if needed
            processed_chunks = [ self._dump_with_embeddings(chunk) for chunk in chunks if isinstance(chunk, Chunk) ]
            
            if processed_chunks:
                await self.table.add(processed_chunks)
        except Exception as e:
            logger.error(f"Failed to store chunks in LanceDB: {e}")
            raise
    
    def _dump_with_embeddings(self, chunk: Chunk) -> dict[str, Any]:
        """Helper to dump chunk with embeddings."""
        chunk_dict = chunk.model_dump()
        if "embeddings" not in chunk_dict:
            chunk_dict['embeddings'] = self.embedding_function.compute_query_embeddings(chunk.content)[0]
        return chunk_dict

    async def search(self, query: str, tags: list[str] = [], file_path: str | None = None, scope: SearchScope = SearchScope.ALL, limit: int = 5) -> list[Chunk]:
        """Search for chunks that match query and filters.
        
        Args:
            query (str): The search query text.
            tags (list[str], optional): List of tags to filter by (all must be present). Defaults to None.
            file_path (str | None, optional): Substring of source_path to filter results. Defaults to None.
            scope (ScopeEnum, optional): Where to search (CONTENT, TITLE, DESCRIPTION, or ALL). Defaults to ScopeEnum.ALL.
            limit (int, optional): Maximum number of results to return. Defaults to 5.
        """
        if not self._initialized:
            raise NotInitializedError("LanceDBStore is not initialized. Call await store.initialize() first.")
            
        # Calculate embedding for the query
        query_embedding = self.embedding_function.compute_query_embeddings(query)[0]
        
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

            query_builder = query_builder.nearest_to(query_embedding).column("embeddings").distance_range(upper_bound=1000.0)
            query_builder = query_builder.nearest_to_text(query, columns = columns)
            
            # Apply filters based on parameters
            # Filter by tags if provided
            if tags:
                query_builder = query_builder.where(f"array_has_all(tags, {json.dumps(tags)})")
            
            # Filter by file path if provided
            if file_path:
                query_builder = query_builder.where(f"source_path LIKE '%{file_path}%'")
            
            query_builder = query_builder.rerank(self.reranker)
            # Go!
            results = await query_builder.limit(limit).to_list()
            return [Chunk.model_validate(result) for result in results]
        
        except Exception as e:
            logger.error(f"Failed to search chunks in LanceDB: {e}")
            raise
        
    
    async def delete(self, source_paths: list[str]) -> None:
        """Delete chunks by their IDs.
        WARNING: delete operaton in lancedb may corrupt indexes, so it will be followed by reindexing.
        """
        if not self._initialized:
            raise NotInitializedError("LanceDBStore is not initialized. Call await store.initialize() first.")
        
        try:

            in_list = ','.join([f"'{p}'" for p in source_paths])
            delete_clause = f"source_path IN ({in_list})"
            logger.info(f"Deleting chunks with source paths: {source_paths} using clause: {delete_clause}")
            result = await self.table.delete(delete_clause)
            
            logger.info(f"Deleted {len(source_paths)} paths from LanceDB")

        except Exception as e:
            logger.error(f"Failed to delete chunks from LanceDB: {e}")
            raise
    
    async def reindex(
        self,
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
        replace: bool = True
        wait_time=timedelta(seconds=5)
        for column in ["content", "title", "description"]: 
            try:
                await self.table.create_index(
                        column,
                        config=FTS(base_tokenizer='simple'),
                        replace=replace,
                        wait_timeout=wait_time
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
                column="tags",
                config=LabelList(),
                wait_timeout=wait_time
            )
        except Exception as e:
            logger.error(f"Failed to create list index for tags: {e}")
            raise
        indices = await self.table.list_indices()
        logger.info([f"index {idx}," for idx in indices])
    
    async def sources(self) -> Dict[str, datetime]:
        """Get last updates to source documents (by minimal modification time per source_path) using pyarrow Table.group_by and aggregate."""
        if not self._initialized:
            raise NotInitializedError("LanceDBStore is not initialized. Call await store.initialize() first.")
        try:
            # Fetch only the required columns as a pyarrow Table
            arrow_table = await self.table.query().select(["source_path", "modified_at"]).to_arrow()
            if arrow_table.num_rows == 0:
                return {}
            # Use pyarrow group_by and aggregate to get minimal modified_at per source_path
            grouped = arrow_table.group_by(["source_path"]).aggregate([("modified_at", "min")])
            # grouped is a pyarrow Table with columns: source_path, modified_at_min
            return {grouped["source_path"][i].as_py() : grouped["modified_at_min"][i].as_py() for i in range(grouped.num_rows)}
        except Exception as e:
            logger.error(f"Failed to fetch source updates from LanceDB: {e}")
            raise
