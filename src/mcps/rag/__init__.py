"""
RAG (Retrieval-Augmented Generation) search system package.
"""

from .interfaces import (
    Document,
    Chunk,
    SearchQuery,
    SearchResult,
    IDocumentProcessor,
    IChunker,
    IEmbeddingService,
    IVectorStore,
    ISearchEngine,
    IResultFormatter,
    IFileTraversal,
)

from .document_processing import (
    MarkdownFileTraversal,
    MarkdownProcessor,
    FixedSizeChunker,
    SemanticChunker,
)

from .embeddings import (
    HuggingFaceEmbedding,
    OpenAIEmbedding,
    MockEmbedding,
    BatchEmbeddingService,
)

from .database import (
    LanceDBStore,
    InMemoryVectorStore,
    FileBasedVectorStore,
)

from .search import (
    SemanticSearchEngine,
    HybridSearchEngine,
    MarkdownResultFormatter,
    CompactResultFormatter,
    JSONResultFormatter,
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