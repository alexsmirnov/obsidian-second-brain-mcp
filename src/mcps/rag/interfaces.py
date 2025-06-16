"""
Abstract interfaces for the RAG search system components.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Document:
    """Represents a document with its content and metadata."""
    id: str
    content: str
    metadata: Dict[str, Any]
    outgoing_links: List[str]  # Wikilinks
    tags: List[str]
    source_path: Path
    created_at: datetime
    modified_at: datetime


@dataclass
class Chunk:
    """Represents a chunk of text with metadata from document."""
    id: str
    content: str
    metadata: Dict[str, Any]
    outgoing_links: List[str]  # Wikilinks
    tags: List[str]
    source_path: Path
    created_at: datetime
    modified_at: datetime
    position: int
    embeddings: Optional[List[float]] = None

    def with_embeddings(self, embeddings: List[float]) -> 'Chunk':
        """Create a new chunk with the same fields and embeddings"""
        return Chunk(
            id=self.id,
            content=self.content,
            metadata=self.metadata,
            outgoing_links=self.outgoing_links,
            tags=self.tags,
            source_path=self.source_path,
            created_at=self.created_at,
            modified_at=self.modified_at,
            position=self.position,
            embeddings=embeddings
        )


@dataclass
class SearchQuery:
    """Represents a search query."""
    text: str
    filters: Optional[Dict[str, Any]] = None
    top_k: int = 5
    similarity_threshold: float = 0.7


@dataclass
class SearchResult:
    """Represents a search result."""
    chunk: Chunk
    score: float


class IDocumentProcessor(ABC):
    """Interface for document processing."""
    
    @abstractmethod
    async def process(self, file_path: Path) -> Document:
        """Process a single document file."""
        pass
    
    @abstractmethod
    def supports_file_type(self, file_path: Path) -> bool:
        """Check if this processor supports the given file type."""
        pass


class IChunker(ABC):
    """Interface for text chunking strategies."""
    
    @abstractmethod
    async def chunk(self, document: Document) -> List[Chunk]:
        """Split a document into chunks."""
        pass


class IEmbeddingService(ABC):
    """Interface for embedding generation."""
    
    @abstractmethod
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass


class IVectorStore(ABC):
    """Interface for vector storage operations."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the vector store."""
        pass
    
    @abstractmethod
    async def store(self, chunks: List[Chunk]) -> None:
        """Store chunks with their embeddings."""
        pass
    
    @abstractmethod
    async def search(self, query_embedding: List[float], top_k: int = 5) -> List[Chunk]:
        """Search for similar chunks."""
        pass
    
    @abstractmethod
    async def delete(self, chunk_ids: List[str]) -> None:
        """Delete chunks by their IDs."""
        pass


class ISearchEngine(ABC):
    """Interface for search operations."""
    
    @abstractmethod
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform a search operation."""
        pass


class IResultFormatter(ABC):
    """Interface for formatting search results."""
    
    @abstractmethod
    async def format(self, results: List[SearchResult], query: SearchQuery) -> str:
        """Format search results for display."""
        pass


class IFileTraversal(ABC):
    """Interface for file discovery and traversal."""
    
    @abstractmethod
    async def find_files(self, start_folder: Optional[str] = None, skip_patterns: Optional[List[str]] = None) -> List[Path]:
        """Find files to process."""
        pass