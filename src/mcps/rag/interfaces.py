"""
Abstract interfaces for the RAG search system components.
"""

from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Annotated

from pydantic import Field, ConfigDict, BaseModel


class Metadata(BaseModel):
    """Represents metadata with fixed fields compatible with LanceDB."""
    
    source: str | None = Field(default=None)
    description: str | None = Field(default=None)
    title: str | None = Field(default=None)


class Document(BaseModel):
    """Represents a document with its content and metadata."""
    
    id: str
    content: str
    metadata: Metadata
    outgoing_links: list[str] = Field(default_factory=list)  # Wikilinks
    tags: list[str] = Field(default_factory=list)
    source_path: str  # file path relative to vault root
    created_at: datetime
    modified_at: datetime


class Chunk(BaseModel):
    """Represents a chunk of text with metadata from document."""
    model_config = ConfigDict(extra='allow') 
    
    id: str
    content: str
    metadata: Metadata
    outgoing_links: list[str] = Field(default_factory=list)  # Wikilinks
    tags: list[str] = Field(default_factory=list)
    source_path: str  # file path relative to vault root
    created_at: datetime
    modified_at: datetime
    position: int
    

@dataclass
class SearchQuery:
    """Represents a search query."""
    text: str
    filters: dict[str, Any] | None = None
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



class IChunker(ABC):
    """Interface for text chunking strategies."""

    @abstractmethod
    def chunk(self, document: Document) -> Generator[Chunk, None, None]:
        """Split a document into chunks."""
        pass


class IEmbeddingService(ABC):
    """Interface for embedding generation."""

    @abstractmethod
    async def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        pass


class IVectorStore(ABC):
    """Interface for vector storage operations."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the vector store."""
        pass

    @abstractmethod
    async def store(self, chunks: list[Chunk]) -> None:
        """Store chunks with their embeddings."""
        pass

    @abstractmethod
    async def search(self, query: str, where: None | str = None, limit: int = 5) -> list[Chunk]:
        """Search for chunks that contain text from query.

        Args:
            query (str): The search query text.
            where (None | str, optional): Optional filter condition. Defaults to None.
            limit (int, optional): Maximum number of results to return. Defaults to 5.
        """
        pass

    @abstractmethod
    async def delete(self, chunk_ids: list[str]) -> None:
        """Delete chunks by their IDs."""
        pass


class ISearchEngine(ABC):
    """Interface for search operations."""

    @abstractmethod
    async def search(self, query: SearchQuery) -> list[SearchResult]:
        """Perform a search operation."""
        pass


class IResultFormatter(ABC):
    """Interface for formatting search results."""

    @abstractmethod
    async def format(self, results: list[SearchResult], query: SearchQuery) -> str:
        """Format search results for display."""
        pass


class IFileTraversal(ABC):
    """Interface for file discovery and traversal."""

    @abstractmethod
    def find_files(self ) -> Generator[Path]:
        """Find files to process in vault."""
        pass
