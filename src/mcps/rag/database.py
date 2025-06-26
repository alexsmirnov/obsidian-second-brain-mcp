"""
Vector database implementations for the RAG search system.
"""

import asyncio
import json
import logging
import pickle
from pathlib import Path
from typing import Optional, Any

from lancedb import AsyncConnection, AsyncTable
from lancedb.embeddings import EmbeddingFunction
from lancedb.pydantic import pydantic_to_schema
from lancedb.index import FTS
import pyarrow as pa

from .interfaces import Chunk, IVectorStore

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
        
        # Text search
        results = await store.search_text("machine learning", limit=10)
        
        # Hybrid search
        results = await store.hybrid_search("deep learning", top_k=5)
    """
    
    def __init__(self, db_path: Path, embedding_function: EmbeddingFunction, table_name: str = "chunks"):
        self.db_path = db_path
        self.table_name = table_name
        self.db: None | AsyncConnection = None
        self.table: AsyncTable | None = None
        self.embedding_function: EmbeddingFunction = embedding_function
        self._initialized = False

    
    async def initialize(self, create_fts_index: bool = True) -> None:
        """
        Initialize the LanceDB vector store.
        
        Args:
            create_fts_index: Whether to automatically create FTS index on 'content' column
        """
        try:
            import lancedb
            
            logger.info(f"Initializing LanceDB at {self.db_path}")
            
            # Create database directory if it doesn't exist
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Connect to database
            self.db = await lancedb.connect_async(self.db_path)
            
            # Create or open table using Pydantic schema
            try:
                self.table = await self.db.open_table(self.table_name)
                logger.info(f"Opened existing table: {self.table_name}")
            except Exception:
                # Table doesn't exist, create it using Pydantic schema
                schema: pa.Schema = pydantic_to_schema(Chunk)
                schema = schema.append(pa.field("embeddings", pa.list_(pa.float16(), self.embedding_function.ndims())))
                self.table = await self.db.create_table(self.table_name, schema=schema)
                logger.info(f"Created new table: {self.table_name}")
            
            # Create FTS index if requested
            if create_fts_index:
                try:
                    await self.create_fts_index()
                except Exception as e:
                    logger.warning(f"Failed to create FTS index during initialization: {e}")
                    logger.warning("FTS functionality will not be available")
            
            self._initialized = True
            logger.info("LanceDB initialized successfully")
            
        except ImportError:
            logger.error("lancedb not installed. Install with: pip install lancedb")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize LanceDB: {e}")
            raise
    
    async def store(self, chunks: list[Chunk]) -> None:
        """Store chunks with their embeddings."""
        if not self._initialized:
            await self.initialize()
        
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

    async def search(self, query: str, where: None | str = None, limit: int = 5) -> list[Chunk]:
        """Search for similar chunks."""
        if not self._initialized:
            await self.initialize()
        # Vector similarity search using embeddings
        # Calculate embedding for the query
        query_embedding = self.embedding_function.compute_query_embeddings(query)[0]
        # Perform the search
        try:
            result = await self.table.search(
                query_embedding
            )
            if where:
                result = result.where(where)
            results = await result.limit(limit).to_list()
            # Convert results to Chunk objects
            return [Chunk.model_validate(result) for result in results]
        
        except Exception as e:
            logger.error(f"Failed to search chunks in LanceDB: {e}")
            raise
        return []
    
    async def delete(self, chunk_ids: list[str]) -> None:
        """Delete chunks by their IDs."""
        if not self._initialized:
            await self.initialize()
        
        try:
            await self.table.delete(f"id IN ({', '.join(map(lambda x: f'\"{x}\"', chunk_ids))})")
            
            logger.info(f"Deleted {len(chunk_ids)} chunks from LanceDB")
            
        except Exception as e:
            logger.error(f"Failed to delete chunks from LanceDB: {e}")
            raise
    
    async def create_fts_index(
        self,
        column: str = "content",
        replace: bool = True
    ) -> None:
        """
        Create a full text search (FTS) index on specified columns.
        
        Args:
            column: column to index (default: ["content"])
            replace: Replace existing index if it exists (default: True)
            
        Raises:
            Exception: If FTS index creation fails
            
        Example:
            # Create FTS index on content column
            await store.create_fts_index()
            
        """
        
        try:
                await self.table.create_index(
                        column,
                        config=FTS(base_tokenizer='simple'),
                        replace=replace
                    )
                logger.info(f"Created FTS index on column '{column}'")
            
        except Exception as e:
            logger.error(f"Failed to create FTS index: {e}")
            raise
    