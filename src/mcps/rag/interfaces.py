"""
Abstract interfaces for the RAG search system components.
"""

from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


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


DEFAULT_LINK_TYPE = "related"


class Link(BaseModel):
    """A typed link from a note to a wikilink target."""
    model_config = ConfigDict(frozen=True)

    type: str
    target: str


def dedupe_links(links: Iterable[Link]) -> list[Link]:
    """Return links in first-occurrence order, dropping redundant entries.

    Drops exact (type, target) duplicates. Also drops a default-typed link
    when the same target carries an explicit type elsewhere in the input;
    two distinct explicit types to one target both survive.
    """
    links = list(links)
    explicitly_typed_targets = {
        link.target for link in links if link.type != DEFAULT_LINK_TYPE
    }
    deduped: list[Link] = []
    seen: set[Link] = set()
    for link in links:
        if (
            link.type == DEFAULT_LINK_TYPE
            and link.target in explicitly_typed_targets
        ):
            continue
        if link not in seen:
            seen.add(link)
            deduped.append(link)
    return deduped


class NoteLinks(BaseModel):
    """Typed outgoing links of one note, aggregated across all its chunks."""

    note: str  # source path of the note, e.g. "Folder/Note.md"
    title: str | None = None
    description: str | None = None
    links: list[Link] = Field(default_factory=list)


class TraversalNode(BaseModel):
    """A note reached while traversing typed relations."""

    note: str = Field(
        description=(
            "Full vault path of the reached note, starting with '/', "
            "without extension."
        )
    )
    title: str | None = None
    description: str | None = None
    depth: int
    direction: str
    relation: str
    via: str = Field(
        description=(
            "Full vault path of the already-visited note this edge connects to."
        )
    )


class TraversalResult(BaseModel):
    """The bounded typed-relation graph traversal result."""

    origin: str = Field(
        description="Full vault path the requested reference resolved to."
    )
    nodes: list[TraversalNode] = Field(default_factory=list)
    truncated: bool = False
    warning: str | None = None


class Document(BaseModel):
    """Represents a document with its content and metadata."""
    
    id: str
    content: str
    metadata: Metadata
    tags: list[str] = Field(default_factory=list)  # Tags from frontmatter only
    source_path: str  # file path relative to vault root
    wikilink_name: str # file name without extension
    file_size: int # file size in characters
    modified_at: float


class Chunk(BaseModel):
    """Represents a chunk of text with metadata from document."""
    model_config = ConfigDict(populate_by_name=True) 
    
    id: str
    content: str
    title: str | None
    description: str | None
    source: str | None = None
    links: list[str] = Field(default_factory=list)  # Wikilink targets
    link_types: list[str] = Field(default_factory=list)  # Aligned with links
    tags: list[str] = Field(default_factory=list)
    source_path: str  # file path relative to vault root
    wikilink_name: str  # file name without .md (the stem, no folders)
    modified_at: float
    position: int
    offset: int  # zero-based line index of the chunk start within the document
    file_size: int
    embeddings: list[float] | None = None
    # Search-only reranker relevance, read from the `_relevance_score` column
    # of reranked query results; never persisted or serialized.
    score: float | None = Field(
        default=None,
        validation_alias=AliasChoices("score", "_relevance_score"),
        exclude=True,
    )

    @model_validator(mode="after")
    def _link_arrays_aligned(self) -> "Chunk":
        if len(self.links) != len(self.link_types):
            raise ValueError(
                f"links ({len(self.links)}) and link_types "
                f"({len(self.link_types)}) must be aligned"
            )
        return self

    @property
    def typed_links(self) -> list[Link]:
        """Links reconstructed from the aligned links/link_types arrays."""
        return [
            Link(type=link_type, target=target)
            for link_type, target in zip(self.link_types, self.links, strict=True)
        ]

    def __hash__(self) -> int:                      # hash/id only, it's primary key
        return hash(self.id)

    def __eq__(self, other: object) -> bool:        # equality/id only
        if isinstance(other, Chunk):
            return self.id == other.id
        return NotImplemented
    
class SourceUpdates(BaseModel):
    """Represents last  updates to a source document."""
    source_path: str
    modified_at: float

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
    async def documents_embeddings(
        self, texts: list[str]
    ) -> list[list[float]]:
        """Generate embedding for a list of texts
        Args:
            texts (list[str]): List of texts to generate embeddings for.
        Returns:
            list[list[float]]: List of embeddings arrays for the texts."""
        pass

    @abstractmethod
    async def query_embeddings(
        self, query: str
    ) -> list[float]:
        """Generate embedding for a query
        Args:
            query (str): Query text to generate embeddings for.
        Returns:
            list[list[float]]: List of embeddings arrays for the texts."""
        pass
    @abstractmethod
    def ndims(self) -> int:
        """Return the number of dimensions of the embeddings."""
        pass


class IDocumentSummaryGenerator(ABC):
    """Interface for whole-document summary generation."""

    @abstractmethod
    async def generate(self, document: Document) -> str:
        """Generate a summary for the whole source document."""
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
        hypotetical_document: str | None = None,
        tags: list[str] | None = None,
        file_path: str | None = None,
        scope: SearchScope = SearchScope.ALL,
        limit: int = 5,
    ) -> list[Chunk]:
        """Search for chunks that contain text from query.

        Args:
            query (str): The search query text. If empty, return chunks by
                tags and file_path only, without vector or full-text search.
            tags (list[str], optional): List of tags to filter by (all must be
                present). Defaults to None.
            file_path (str | None, optional): Substring of source_path to filter
                results. Defaults to None.
            scope (SearchScope enum, optional): Where to search (CONTENT, TITLE,
                DESCRIPTION, or ALL). Defaults to ALL.
            limit (int, optional): Maximum number of results to return. Defaults to 5.

        Raises:
            ValueError: If the query is empty and neither tags nor file_path
                filters are provided.
        """
        pass

    @abstractmethod
    async def delete(self, source_paths: list[str]) -> None:
        """Delete chunks by their `source_path`s."""
        pass
    
    @abstractmethod
    async def reindex(self,replace: bool = True) -> None:
        """Recreate indexes for the vector store.
        
        Due to LanceDB implementation, indexes have to be recreated after any
        data modification.
        """
        pass

    @abstractmethod
    async def get_chunks_by_ids(self, ids: list[str]) -> list[Chunk]:
        """Return chunks whose id is in the provided list."""
        pass

    @abstractmethod
    async def sources(self) -> dict[str, float]:
        """Get last updates to source documents."""
        pass

    @abstractmethod
    async def get_sources_by_name(self, wikilink_name: str) -> list[str]:
        """Return source paths of notes whose file name (stem) matches."""
        pass

    @abstractmethod
    async def get_notes_with_links(
        self, source_paths: list[str]
    ) -> list[NoteLinks]:
        """Return outgoing typed links for the notes at the given source paths.

        Aggregates the links of all chunk rows of each note, deduplicated.
        Each entry's `note` is the note's source path. Notes absent from the
        index are silently omitted. An empty input returns an empty list.
        """
        pass

    @abstractmethod
    async def get_notes_linking_to(
        self, targets: list[str], relation_types: list[str] | None = None
    ) -> list[NoteLinks]:
        """Return notes holding at least one link to a requested target.

        Targets are raw wikilink strings as authored. Each entry's `note` is
        the linking note's source path. Each returned entry carries only the
        edges pointing at a requested target, optionally restricted to the
        given relation types (OR-combined). An empty input returns an empty
        list. Matching is exact on the (type, target) pair, never on the SQL
        pre-filter alone.
        """
        pass

class ISearchEngine(ABC):
    """Interface for search operations."""

    @abstractmethod
    async def search(self, query: SearchQuery) -> list[Chunk]:
        """Perform a search operation."""
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
        """Synchronize the vault file system with the vector store.

        The method gets the list of files in the vault and saved in the storage.
        It deletes all chunks for source_paths that are not in the vault anymore
        or were modified, adds new chunks for files that were added or modified,
        and rebuilds the vector-store indexes. The Obsidian lifespan runs it at
        startup, every few minutes, and before index-backed tool calls through
        ``IndexScheduler``. Vault search and read methods never call it.
        """
        pass

    @abstractmethod
    async def search(
        self,
        query: str,
        tags: list[str] | None = None,
        path: str | None = None,
    ) -> list[Chunk]:
        """Search files in vault by query. Return list of matching chunks."""
        pass

    @abstractmethod
    async def resolve_references(
        self, references: Iterable[str]
    ) -> dict[str, list[str]]:
        """Map each reference to the sorted indexed note paths it matches.

        References are resolved like in `get_file`; one that matches nothing
        maps to an empty list.
        """
        pass

    @abstractmethod
    async def get_file(
        self,
        reference: str,
        offset: int | None = None,
        limit: int | None = None,
    ) -> str:
        """Get note content by reference, without the .md extension.

        A reference is a full path ('/Folder/Note'), a partial path
        ('Folder/Note') matching any note whose path ends with those
        segments, or a bare name ('Note'). Raises FileNotFoundError when
        nothing matches. When several notes match, the one with the shortest
        path wins, then alphabetical order.

        offset is a zero-based line index; limit is a maximum number of lines.
        """
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

    @abstractmethod
    async def get_backlinks(
        self, note_paths: list[str]
    ) -> dict[str, list[Link]]:
        """Return incoming links for the requested notes, keyed by note path.

        Each returned Link has `type` set to the incoming edge's type and
        `target` set to the source note's path ('/Folder/Note'). A link
        written with a name shared by several notes counts only for the note
        Obsidian resolves it to from the linking note: shortest path, then
        closest to the linking note, then alphabetical order. Notes with no incoming links map to an empty list;
        every requested path is present as a key. An empty input returns an
        empty dict.
        """
        pass

    @abstractmethod
    async def traverse_relations(
        self,
        reference: str,
        depth: int = 1,
        relation_types: list[str] | None = None,
    ) -> TraversalResult:
        """Walk typed relations in both directions from a note.

        The reference is resolved like in `get_file`, with the same rules.
        The walk is breadth-first, includes at most ``depth`` hops, and
        follows only the requested relation types when provided. Every note
        in the result is identified by its path ('/Folder/Note').
        """
        pass
