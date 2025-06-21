"""
Vector database implementations for the RAG search system.
"""

import asyncio
import json
import logging
import pickle
from pathlib import Path

from .interfaces import Chunk, IVectorStore

logger = logging.getLogger("mcps")


class LanceDBStore(IVectorStore):
    """LanceDB vector store implementation."""
    
    def __init__(self, db_path: Path, table_name: str = "chunks"):
        self.db_path = db_path
        self.table_name = table_name
        self.db = None
        self.table = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the LanceDB vector store."""
        try:
            import lancedb
            import pyarrow as pa
            
            logger.info(f"Initializing LanceDB at {self.db_path}")
            
            # Create database directory if it doesn't exist
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Connect to database
            loop = asyncio.get_event_loop()
            self.db = await loop.run_in_executor(None, lancedb.connect, str(self.db_path))
            
            # Define schema
            schema = pa.schema([
                pa.field("id", pa.string()),
                pa.field("content", pa.string()),
                pa.field("metadata", pa.string()),  # JSON string
                pa.field("outgoing_links", pa.string()),  # JSON string
                pa.field("tags", pa.string()),  # JSON string
                pa.field("source_path", pa.string()),
                pa.field("created_at", pa.string()),
                pa.field("modified_at", pa.string()),
                pa.field("position", pa.int64()),
                pa.field("embeddings", pa.list_(pa.float64())),
            ])
            
            # Create or open table
            try:
                self.table = self.db.open_table(self.table_name)
                logger.info(f"Opened existing table: {self.table_name}")
            except Exception:
                # Table doesn't exist, create it
                self.table = await loop.run_in_executor(
                    None,
                    self.db.create_table,
                    self.table_name,
                    [],  # Empty data
                    schema
                )
                logger.info(f"Created new table: {self.table_name}")
            
            self._initialized = True
            logger.info("LanceDB initialized successfully")
            
        except ImportError:
            logger.error("lancedb or pyarrow not installed. Install with: pip install lancedb pyarrow")
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
            # Convert chunks to records
            records = []
            for chunk in chunks:
                if chunk.embeddings is None:
                    logger.warning(f"Chunk {chunk.id} has no embeddings, skipping")
                    continue
                
                record = {
                    "id": chunk.id,
                    "content": chunk.content,
                    "metadata": json.dumps(chunk.metadata),
                    "outgoing_links": json.dumps(chunk.outgoing_links),
                    "tags": json.dumps(chunk.tags),
                    "source_path": str(chunk.source_path),
                    "created_at": chunk.created_at.isoformat(),
                    "modified_at": chunk.modified_at.isoformat(),
                    "position": chunk.position,
                    "embeddings": chunk.embeddings,
                }
                records.append(record)
            
            if records:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self.table.add, records)
                logger.info(f"Stored {len(records)} chunks in LanceDB")
            
        except Exception as e:
            logger.error(f"Failed to store chunks in LanceDB: {e}")
            raise
    
    async def search(self, query_embedding: list[float], top_k: int = 5) -> list[Chunk]:
        """Search for similar chunks."""
        if not self._initialized:
            await self.initialize()
        
        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self.table.search(query_embedding).limit(top_k).to_pandas()
            )
            
            chunks = []
            for _, row in results.iterrows():
                chunk = Chunk(
                    id=row["id"],
                    content=row["content"],
                    metadata=json.loads(row["metadata"]),
                    outgoing_links=json.loads(row["outgoing_links"]),
                    tags=json.loads(row["tags"]),
                    source_path=Path(row["source_path"]),
                    created_at=row["created_at"],
                    modified_at=row["modified_at"],
                    position=row["position"],
                    embeddings=row["embeddings"]
                )
                chunks.append(chunk)
            
            logger.debug(f"Found {len(chunks)} similar chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to search in LanceDB: {e}")
            return []
    
    async def delete(self, chunk_ids: list[str]) -> None:
        """Delete chunks by their IDs."""
        if not self._initialized:
            await self.initialize()
        
        try:
            for chunk_id in chunk_ids:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: self.table.delete(f"id = '{chunk_id}'")
                )
            
            logger.info(f"Deleted {len(chunk_ids)} chunks from LanceDB")
            
        except Exception as e:
            logger.error(f"Failed to delete chunks from LanceDB: {e}")
            raise


class InMemoryVectorStore(IVectorStore):
    """Simple in-memory vector store for testing and small datasets."""
    
    def __init__(self):
        self.chunks: dict[str, Chunk] = {}
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the in-memory vector store."""
        self._initialized = True
        logger.info("In-memory vector store initialized")
    
    async def store(self, chunks: list[Chunk]) -> None:
        """Store chunks with their embeddings."""
        if not self._initialized:
            await self.initialize()
        
        for chunk in chunks:
            if chunk.embeddings is None:
                logger.warning(f"Chunk {chunk.id} has no embeddings, skipping")
                continue
            self.chunks[chunk.id] = chunk
        
        logger.info(f"Stored {len(chunks)} chunks in memory")
    
    async def search(self, query_embedding: list[float], top_k: int = 5) -> list[Chunk]:
        """Search for similar chunks using cosine similarity."""
        if not self._initialized:
            await self.initialize()
        
        if not self.chunks:
            return []
        
        # Calculate cosine similarity for all chunks
        similarities = []
        for chunk in self.chunks.values():
            if chunk.embeddings:
                similarity = self._cosine_similarity(query_embedding, chunk.embeddings)
                similarities.append((similarity, chunk))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_chunks = [chunk for _, chunk in similarities[:top_k]]
        
        logger.debug(f"Found {len(top_chunks)} similar chunks")
        return top_chunks
    
    async def delete(self, chunk_ids: list[str]) -> None:
        """Delete chunks by their IDs."""
        if not self._initialized:
            await self.initialize()
        
        deleted_count = 0
        for chunk_id in chunk_ids:
            if chunk_id in self.chunks:
                del self.chunks[chunk_id]
                deleted_count += 1
        
        logger.info(f"Deleted {deleted_count} chunks from memory")
    
    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
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
            
            return dot_product / (norm1 * norm2)
            
        except ImportError:
            # Fallback implementation without numpy
            dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=False))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)


class FileBasedVectorStore(IVectorStore):
    """File-based vector store using pickle for persistence."""
    
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.chunks: dict[str, Chunk] = {}
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the file-based vector store."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing data if available
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'rb') as f:
                    self.chunks = pickle.load(f)
                logger.info(f"Loaded {len(self.chunks)} chunks from {self.storage_path}")
            except Exception as e:
                logger.warning(f"Failed to load existing data: {e}")
                self.chunks = {}
        
        self._initialized = True
        logger.info("File-based vector store initialized")
    
    async def store(self, chunks: list[Chunk]) -> None:
        """Store chunks with their embeddings."""
        if not self._initialized:
            await self.initialize()
        
        for chunk in chunks:
            if chunk.embeddings is None:
                logger.warning(f"Chunk {chunk.id} has no embeddings, skipping")
                continue
            self.chunks[chunk.id] = chunk
        
        # Persist to file
        await self._save_to_file()
        logger.info(f"Stored {len(chunks)} chunks to file")
    
    async def search(self, query_embedding: list[float], top_k: int = 5) -> list[Chunk]:
        """Search for similar chunks using cosine similarity."""
        if not self._initialized:
            await self.initialize()
        
        if not self.chunks:
            return []
        
        # Use same similarity calculation as InMemoryVectorStore
        similarities = []
        for chunk in self.chunks.values():
            if chunk.embeddings:
                similarity = self._cosine_similarity(query_embedding, chunk.embeddings)
                similarities.append((similarity, chunk))
        
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_chunks = [chunk for _, chunk in similarities[:top_k]]
        
        logger.debug(f"Found {len(top_chunks)} similar chunks")
        return top_chunks
    
    async def delete(self, chunk_ids: list[str]) -> None:
        """Delete chunks by their IDs."""
        if not self._initialized:
            await self.initialize()
        
        deleted_count = 0
        for chunk_id in chunk_ids:
            if chunk_id in self.chunks:
                del self.chunks[chunk_id]
                deleted_count += 1
        
        if deleted_count > 0:
            await self._save_to_file()
        
        logger.info(f"Deleted {deleted_count} chunks from file")
    
    async def _save_to_file(self):
        """Save chunks to file."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._save_sync
            )
        except Exception as e:
            logger.error(f"Failed to save chunks to file: {e}")
            raise
    
    def _save_sync(self):
        """Synchronous save operation."""
        with open(self.storage_path, 'wb') as f:
            pickle.dump(self.chunks, f)
    
    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        # Same implementation as InMemoryVectorStore
        try:
            import numpy as np
            
            v1 = np.array(vec1)
            v2 = np.array(vec2)
            
            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
            
        except ImportError:
            dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=False))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)