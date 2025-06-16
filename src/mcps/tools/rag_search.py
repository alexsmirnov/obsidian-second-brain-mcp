import logging
import os
from pathlib import Path
import re
from typing import List, Optional
import asyncio

from mcps.config import ServerConfig
from mcps.rag import (
    SearchQuery,
    MarkdownFileTraversal,
    MarkdownProcessor,
    FixedSizeChunker,
    SemanticChunker,
    HuggingFaceEmbedding,
    MockEmbedding,
    InMemoryVectorStore,
    FileBasedVectorStore,
    SemanticSearchEngine,
    MarkdownResultFormatter,
    CompactResultFormatter,
    BatchEmbeddingService,
    IChunker,
    IEmbeddingService,
    IVectorStore,
    IResultFormatter,
)

logger = logging.getLogger("mcps")


class ComponentFactory:
    """Factory for creating RAG components."""
    
    @staticmethod
    def create_file_traversal(config: Optional[ServerConfig] = None) -> MarkdownFileTraversal:
        """Create file traversal component."""
        base_path = Path.cwd()
        if config and hasattr(config, 'vault_dir') and config.vault_dir:
            base_path = config.vault_dir
        return MarkdownFileTraversal(base_path)
    
    @staticmethod
    def create_document_processor() -> MarkdownProcessor:
        """Create document processor."""
        return MarkdownProcessor()
    
    @staticmethod
    def create_chunker(chunk_type: str = "fixed") -> IChunker:
        """Create chunker component."""
        if chunk_type == "semantic":
            return SemanticChunker(max_chunk_size=2000, min_chunk_size=100)
        else:
            return FixedSizeChunker(chunk_size=1000, overlap=200)
    
    @staticmethod
    def create_embedding_service(use_mock: bool = False) -> IEmbeddingService:
        """Create embedding service."""
        if use_mock:
            return MockEmbedding(dimension=384)
        else:
            try:
                return HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
            except Exception as e:
                logger.warning(f"Failed to create HuggingFace embedding service: {e}. Using mock embedding.")
                return MockEmbedding(dimension=384)
    
    @staticmethod
    def create_vector_store(config: Optional[ServerConfig] = None, use_memory: bool = False) -> IVectorStore:
        """Create vector store."""
        if use_memory:
            return InMemoryVectorStore()
        else:
            # Use file-based storage in cache directory
            cache_dir = Path("cache/rag")
            cache_dir.mkdir(parents=True, exist_ok=True)
            storage_path = cache_dir / "vector_store.pkl"
            return FileBasedVectorStore(storage_path)
    
    @staticmethod
    def create_search_engine(
        embedding_service,
        vector_store
    ) -> SemanticSearchEngine:
        """Create search engine."""
        return SemanticSearchEngine(
            embedding_service=embedding_service,
            vector_store=vector_store
        )
    
    @staticmethod
    def create_result_formatter(format_type: str = "markdown") -> IResultFormatter:
        """Create result formatter."""
        if format_type == "compact":
            return CompactResultFormatter(max_results=5, snippet_length=200)
        else:
            return MarkdownResultFormatter(max_content_length=600, include_metadata=True)


class RAGSearchOrchestrator:
    """Orchestrates the entire RAG search process."""
    
    def __init__(
        self,
        file_traversal: MarkdownFileTraversal,
        document_processor: MarkdownProcessor,
        chunker: IChunker,
        embedding_service: IEmbeddingService,
        vector_store: IVectorStore,
        search_engine: SemanticSearchEngine,
        result_formatter: IResultFormatter,
        config: Optional[ServerConfig] = None
    ):
        self.file_traversal = file_traversal
        self.document_processor = document_processor
        self.chunker = chunker
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.search_engine = search_engine
        self.result_formatter = result_formatter
        self.config = config
        self.batch_embedding_service = BatchEmbeddingService(embedding_service, batch_size=16)
        
    async def index_documents(self, file_paths: List[Path]) -> None:
        """Index a list of documents."""
        if not file_paths:
            logger.info("No files to index")
            return
        
        logger.info(f"Starting indexing of {len(file_paths)} documents")
        
        try:
            # Initialize vector store
            await self.vector_store.initialize()
            
            # Process documents
            processed_docs = []
            for file_path in file_paths:
                if self.document_processor.supports_file_type(file_path):
                    try:
                        doc = await self.document_processor.process(file_path)
                        processed_docs.append(doc)
                    except Exception as e:
                        logger.warning(f"Failed to process {file_path}: {e}")
            
            logger.info(f"Successfully processed {len(processed_docs)} documents")
            
            # Chunk documents
            all_chunks = []
            for doc in processed_docs:
                try:
                    chunks = await self.chunker.chunk(doc)
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.warning(f"Failed to chunk document {doc.id}: {e}")
            
            logger.info(f"Created {len(all_chunks)} chunks")
            
            if not all_chunks:
                logger.warning("No chunks created from documents")
                return
            
            # Generate embeddings in batches
            chunk_texts = [chunk.content for chunk in all_chunks]
            embeddings = await self.batch_embedding_service.generate_embeddings(chunk_texts)
            
            # Add embeddings to chunks
            chunks_with_embeddings = []
            for chunk, embedding in zip(all_chunks, embeddings):
                chunk_with_emb = chunk.with_embeddings(embedding)
                chunks_with_embeddings.append(chunk_with_emb)
            
            # Store in vector database
            await self.vector_store.store(chunks_with_embeddings)
            
            logger.info(f"Successfully indexed {len(chunks_with_embeddings)} chunks")
            
        except Exception as e:
            logger.error(f"Failed to index documents: {e}")
            raise
        
    async def search(self, query: str, **kwargs) -> str:
        """Perform a search operation."""
        try:
            # Ensure vector store is initialized
            await self.vector_store.initialize()
            
            # Create search query
            search_query = SearchQuery(
                text=query,
                top_k=kwargs.get('top_k', 5),
                similarity_threshold=kwargs.get('similarity_threshold', 0.3),
                filters=kwargs.get('filters')
            )
            
            # Perform search
            results = await self.search_engine.search(search_query)
            
            # Format results
            formatted_results = await self.result_formatter.format(results, search_query)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return f"Search failed: {str(e)}"


class RagSearch:
    """Main RAG search class."""
    
    def __init__(
        self,
        config: Optional[ServerConfig] = None,
        skip_patterns: Optional[List[str]] = None,
        force_reindex: bool = False,
        use_mock_embedding: bool = False,
        use_memory_store: bool = False
    ):
        self.config = config
        self.skip_patterns = skip_patterns or []
        self.force_reindex = force_reindex
        self.use_mock_embedding = use_mock_embedding
        self.use_memory_store = use_memory_store
        
        # Create components using factory
        self.file_traversal = ComponentFactory.create_file_traversal(config)
        self.document_processor = ComponentFactory.create_document_processor()
        self.chunker = ComponentFactory.create_chunker()
        self.embedding_service = ComponentFactory.create_embedding_service(use_mock_embedding)
        self.vector_store = ComponentFactory.create_vector_store(config, use_memory_store)
        self.search_engine = ComponentFactory.create_search_engine(
            self.embedding_service,
            self.vector_store
        )
        self.result_formatter = ComponentFactory.create_result_formatter()
        
        # Create orchestrator
        self.orchestrator = RAGSearchOrchestrator(
            file_traversal=self.file_traversal,
            document_processor=self.document_processor,
            chunker=self.chunker,
            embedding_service=self.embedding_service,
            vector_store=self.vector_store,
            search_engine=self.search_engine,
            result_formatter=self.result_formatter,
            config=config
        )
        
        self._initialized = False
    
    async def _ensure_initialized(self, start_folder: Optional[str] = None):
        """Ensure the RAG system is initialized and indexed."""
        if self._initialized and not self.force_reindex:
            return
        
        try:
            # Check if we need to index
            if self.force_reindex or await self._needs_indexing():
                logger.info("Indexing documents...")
                files = await self.file_traversal.find_files(start_folder, self.skip_patterns)
                await self.orchestrator.index_documents(files)
            
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise
    
    async def _needs_indexing(self) -> bool:
        """Check if indexing is needed."""
        # Simple check - if vector store is empty, we need indexing
        try:
            await self.vector_store.initialize()
            # Try a dummy search to see if there's any data
            dummy_embedding = [0.0] * 384  # Default dimension
            results = await self.vector_store.search(dummy_embedding, top_k=1)
            return len(results) == 0
        except Exception:
            return True
    
    async def search_markdown_files(
        self,
        query: str,
        start_folder: Optional[str] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.3
    ) -> str:
        """Main function to perform RAG search in markdown files."""
        try:
            # Ensure system is initialized
            await self._ensure_initialized(start_folder)
            
            # Perform search
            return await self.orchestrator.search(
                query,
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )
            
        except Exception as e:
            logger.error(f"RAG search failed: {e}")
            return f"Search failed: {str(e)}"


async def search_markdown_files(query: str, start_folder: Optional[str] = None, config: Optional[ServerConfig] = None) -> str:
    """
    Performs a RAG search in markdown files for content related to the query.
    
    Args:
        query: The search query.
        start_folder: Optional starting folder path. If not provided, searches all markdown files.
        config: Server configuration.
        
    Returns:
        Content from markdown files that may contain answers to the query.
    """
    logger.info(f"Performing RAG search with query: {query}, start_folder: {start_folder}")
    
    try:
        # Create RAG search instance
        rag_search = RagSearch(
            config=config,
            use_mock_embedding=True,  # Use mock embedding by default to avoid dependency issues
            use_memory_store=True     # Use memory store for simplicity
        )
        
        # Perform search
        return await rag_search.search_markdown_files(
            query=query,
            start_folder=start_folder,
            top_k=5,
            similarity_threshold=0.3
        )
        
    except Exception as e:
        logger.error(f"RAG search failed: {e}")
        return f"RAG search failed: {str(e)}"