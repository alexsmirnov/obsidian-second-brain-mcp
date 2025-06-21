"""
RAG (Retrieval-Augmented Generation) search system package.
"""

from .database import (
    FileBasedVectorStore,
    InMemoryVectorStore,
    LanceDBStore,
)
from .document_processing import (
    FixedSizeChunker,
    MarkdownFileTraversal,
    MarkdownProcessor,
    SemanticChunker,
)
from .embeddings import (
    BatchEmbeddingService,
    HuggingFaceEmbedding,
    MockEmbedding,
    OpenAIEmbedding,
)
from .interfaces import (
    Chunk,
    Document,
    IChunker,
    IDocumentProcessor,
    IEmbeddingService,
    IFileTraversal,
    IResultFormatter,
    ISearchEngine,
    IVectorStore,
    SearchQuery,
    SearchResult,
)
from .search import (
    CompactResultFormatter,
    HybridSearchEngine,
    JSONResultFormatter,
    MarkdownResultFormatter,
    SemanticSearchEngine,
)

__all__ = [
    # Interfaces
    "Document",
    "Chunk",
    "SearchQuery",
    "SearchResult",
    "IDocumentProcessor",
    "IChunker",
    "IEmbeddingService",
    "IVectorStore",
    "ISearchEngine",
    "IResultFormatter",
    "IFileTraversal",
    
    # Document Processing
    "MarkdownFileTraversal",
    "MarkdownProcessor",
    "FixedSizeChunker",
    "SemanticChunker",
    
    # Embeddings
    "HuggingFaceEmbedding",
    "OpenAIEmbedding",
    "MockEmbedding",
    "BatchEmbeddingService",
    
    # Database
    "LanceDBStore",
    "InMemoryVectorStore",
    "FileBasedVectorStore",
    
    # Search
    "SemanticSearchEngine",
    "HybridSearchEngine",
    "MarkdownResultFormatter",
    "CompactResultFormatter",
    "JSONResultFormatter",
]