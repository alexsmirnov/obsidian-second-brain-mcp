"""
Abstract interfaces for the RAG search system components.
"""

from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


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
    
class SourceUpdates(BaseModel):
    """Represents last  updates to a source document."""
    source_path: str
    modified_at: datetime

@dataclass
class SearchQuery:
    """Represents a search query."""
    text: str
    tags: list[str]
    scope: SearchScope = SearchScope.ALL
    path: str | None = None


@dataclass
class SearchResult:
    """Represents a search result."""
    chunk: Chunk
    score: float

class NotInitializedError(RuntimeError):
    """Exception raised when an operation is attempted on an uninitialized object."""

    def __init__(
        self,
        message="Service has not been initialized. Please call initialize() first.",
    ):
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
    def chunk(self, document: Document) -> Generator[Chunk]:
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
    async def search(
        self,
        query: str,
        tags: list[str] | None = None,
        file_path: str | None = None,
        scope: SearchScope = SearchScope.ALL,
        limit: int = 5,
    ) -> list[Chunk]:
        """Search for chunks that contain text from query.

        Args:
            query (str): The search query text.
            tags (list[str], optional): List of tags to filter by (all must be
                present). Defaults to None.
            file_path (str | None, optional): Substring of source_path to filter
                results. Defaults to None.
            scope (SearchScope enum, optional): Where to search (CONTENT, TITLE,
                DESCRIPTION, or ALL). Defaults to ALL.
            limit (int, optional): Maximum number of results to return. Defaults to 5.
        """
        pass

    @abstractmethod
    async def delete(self, source_paths: list[str]) -> None:
        """Delete chunks by their `source_path`s."""
        pass
    
    @abstractmethod
    async def reindex(self) -> None:
        """Recreate indexes for the vector store.
        
        Due to LanceDB implementation, indexes have to be recreated after any
        data modification.
        """
        pass

    @abstractmethod
    async def sources(self) -> dict[str, datetime]:
        """Get last updates to source documents."""
        pass

class ISearchEngine(ABC):
    """Interface for search operations."""

    @abstractmethod
    async def search(self, query: SearchQuery) -> list[Chunk]:
        """Perform a search operation."""
        pass


class IResultFormatter(ABC):
    """Interface for formatting search results."""

    @abstractmethod
    async def format(self, results: list[Chunk], query: SearchQuery) -> str:
        """Format search results for display."""
        pass


class IFileTraversal(ABC):
    """Interface for file discovery and traversal."""

    @abstractmethod
    def find_files(self) -> Generator[Path]:
        """Find files to process in vault."""
        pass


class IVault(ABC):
    """Interface for managing vault operations.
    
    Supports operations to search information from vault, read whole file by name
    or partial path, and get directory content.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the vault manager."""
        pass

    @abstractmethod
    async def update_index(self) -> None:
        """Update the index of the vault.
        
        The method gets the list of files in the vault and saved in the storage.
        It deletes all chunks for source_paths that are not in the vault anymore
        or were modified, and adds new chunks for files that were added or modified.
        This should be called before any search operation, if there were more than
        1 minute after the last search.
        """
        pass

    @abstractmethod
    async def search(self, query: str) -> str:
        """search files in vault by query. Return formatted string with results."""
        pass

    @abstractmethod
    async def get_file(self, file_name: str) -> str:
        """get file content by its name without extension or partial path.
        If multiple files match, return the first one found."""
        pass

    @abstractmethod
    async def list_files(self, directory: str) -> list[str]:
        """List all files in a directory.

        Args:
            directory (str): Directory path relative to vault root.

        Returns:
            list[str]: List of file names without .md extension in the directory,
            plus directory names ended with `/`.
        """
        pass