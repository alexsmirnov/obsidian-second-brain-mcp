"""
Abstract interfaces for the RAG search system components.
"""

from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Annotated

from pydantic import Field, ConfigDict, BaseModel


class SearchScope(Enum):
    """Enum for search scope options."""
    CONTENT = auto()    # Search only in content
    TITLE = auto()      # Search only in title
    DESCRIPTION = auto() # Search only in description
    ALL = auto()        # Search in all fields


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
    tags: list[str] = Field(default_factory=list)  # Tags from frontmatter only
    source_path: str  # file path relative to vault root
    modified_at: datetime


class Chunk(BaseModel):
    """Represents a chunk of text with metadata from document."""
    model_config = ConfigDict(extra='allow') 
    
    id: str
    content: str
    title: str | None
    description: str | None
    source: str | None = None
    outgoing_links: list[str] = Field(default_factory=list)  # Wikilinks
    tags: list[str] = Field(default_factory=list)
    source_path: str  # file path relative to vault root
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

class NotInitializedError(RuntimeError):
    """Exception raised when an operation is attempted on an uninitialized object."""
    def __init__(self, message="Service has not been initialized. Please call initialize() first."):
        self.message = message
        super().__init__(self.message)

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
    async def search(self, query: str, tags: list[str] = [], file_path: str | None = None, scope: SearchScope = SearchScope.ALL, limit: int = 5) -> list[Chunk]:
        """Search for chunks that contain text from query.

        Args:
            query (str): The search query text.
            tags (list[str], optional): List of tags to filter by (all must be present). Defaults to empty list.
            file_path (str | None, optional): Substring of source_path to filter results. Defaults to None.
            scope (SearchScope enum, optional): Where to search (CONTENT, TITLE, DESCRIPTION, or ALL). Defaults to ALL.
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
