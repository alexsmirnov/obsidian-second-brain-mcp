"""
Production-ready Vault implementation for managing RAG operations.

This module provides a comprehensive Vault class that implements the IVault interface,
managing all aspects of document indexing, searching, and retrieval in a RAG system.
"""

import asyncio
from importlib.metadata import files
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from lancedb.embeddings import EmbeddingFunction, OllamaEmbeddings
from lancedb.rerankers import RRFReranker

from mcps.rag.ollama_reranker import OllamaReranker

from .database import LanceDBStore
from .document_processing import FixedSizeChunker, MarkdownFileTraversal, MarkdownProcessor, SemanticChunker
from .interfaces import (
    IChunker,
    IDocumentProcessor,
    IFileTraversal,
    IResultFormatter,
    ISearchEngine,
    IVault,
    IVectorStore,
    NotInitializedError,
    SearchQuery,
    SearchScope,
)
from .search import MarkdownResultFormatter, SemanticSearchEngine
from lancedb.embeddings import EmbeddingFunction, get_registry

logger = logging.getLogger("mcps")


def _create_embedding_function() -> EmbeddingFunction:
    """Create and configure embedding function."""
    ollama_base_url = os.getenv("OLLAMA_API_BASE")
    voyage_api_key = os.getenv("VOYAGE_API_KEY")
    if voyage_api_key:
        return get_registry().get("voyageai").create(name='voyage-3-lite')
    elif ollama_base_url:
        return get_registry().get("ollama").create(name="bge-m3:latest", host=ollama_base_url)
    else:
        raise RuntimeError("No embedding service configured. Set OLLAMA_API_BASE or VOYAGE_API_KEY environment variable.")


def _create_file_traversal(vault_path: Path, skip_patterns: Optional[list[str]] = None) -> IFileTraversal:
    """Create and configure file traversal service."""
    return MarkdownFileTraversal(base_path=vault_path)


def _create_document_processor(vault_path: Path) -> IDocumentProcessor:
    """Create and configure document processor service."""
    return MarkdownProcessor(base_path=vault_path)


def _create_chunker() -> IChunker:
    """Create and configure text chunker service."""
    # return FixedSizeChunker(chunk_size=chunk_size, overlap=chunk_overlap)
    return SemanticChunker()


def _create_reranker():
    """Create and configure reranker."""
    if os.environ.get("VOYAGE_API_KEY"):
        from lancedb.rerankers import VoyageAIReranker
        return VoyageAIReranker(model_name="rerank-2-lite", column="content", api_key=os.environ.get("VOYAGE_API_KEY"))
    elif os.environ.get("OLLAMA_API_BASE"):
        return OllamaReranker(
            model_name="phi4-mini:latest",
            ollama_base_url=os.environ.get("OLLAMA_API_BASE", "http://localhost:11434"),
            embedding_model="bge-m3:latest",
            return_score='relevance',
            column='content',
            weight=1.0
        )
    else:
        from lancedb.rerankers import RRFReranker
        return RRFReranker(return_score='all')


def _create_vector_store(db_path: Path, table_name: str) -> IVectorStore:
    """Create and configure vector store service."""
    return LanceDBStore(
        db_path=db_path,
        embedding_function=_create_embedding_function(),
        table_name=table_name,
        reranker=_create_reranker()
    )


def _create_search_engine(vector_store: IVectorStore) -> ISearchEngine:
    """Create and configure search engine service."""
    return SemanticSearchEngine(vector_store=vector_store, limit=10)


def _create_result_formatter(max_content_length: int = 4000, include_metadata: bool = True) -> IResultFormatter:
    """Create and configure result formatter service."""
    return MarkdownResultFormatter(max_content_length=max_content_length, include_metadata=include_metadata)


class Vault(IVault):
    """
    Production-ready Vault implementation for managing RAG operations.
    
    This class provides a comprehensive implementation of the IVault interface,
    managing document indexing, searching, and retrieval operations using
    dependency injection for all services.
    
    Features:
    - Dependency injection for all services
    - Thread-safe operations with proper synchronization
    - Comprehensive error handling and logging
    - Resource management and cleanup
    - Incremental index updates based on file modification times
    - Support for various search scopes and filtering options
    
    Example:
        vault = create_vault(Path("/path/to/documents"))
        await vault.initialize()
        await vault.update_index()
        results = await vault.search("machine learning")
    """
    
    def __init__(
        self,
        vault_path: Path,
        file_traversal: IFileTraversal,
        document_processor: IDocumentProcessor,
        chunker: IChunker,
        vector_store: IVectorStore,
        search_engine: ISearchEngine,
        result_formatter: IResultFormatter,
        batch_size: int = 8,
    ):
        """
        Initialize the Vault with injected services.
        
        Args:
            vault_path: Path to the vault directory containing documents
            file_traversal: Service for traversing files
            document_processor: Service for processing documents
            chunker: Service for chunking text
            vector_store: Service for storing and retrieving vectors
            search_engine: Service for semantic search
            result_formatter: Service for formatting search results
            batch_size: Number of files to process in each batch
        """
        self.vault_path = Path(vault_path)
        self.db_path = self.vault_path / '.vault_db'
        self._initialized = False
        self._lock = asyncio.Lock()
        self._last_update_check: Optional[datetime] = None
        self._update_interval = timedelta(minutes=1)
        self.batch_size = batch_size
        
        # Inject services
        self.file_traversal = file_traversal
        self.document_processor = document_processor
        self.chunker = chunker
        self.vector_store = vector_store
        self.search_engine = search_engine
        self.result_formatter = result_formatter
        
        logger.info(f"Vault initialized with injected services for path: {vault_path}")
    
    async def initialize(self) -> None:
        """
        Initialize the vault manager and all its services.
        
        This method performs the following operations:
        - Validates the vault path exists
        - Initializes the vector store and creates necessary indexes
        - Sets up the embedding service
        - Prepares all services for operation
        
        Raises:
            RuntimeError: If initialization fails
            FileNotFoundError: If vault path doesn't exist
        """
        async with asyncio.Lock():
            if self._initialized:
                logger.debug("Vault already initialized")
                return
            
            try:
                logger.info("Starting Vault initialization")
                
                # Validate vault path
                if not self.vault_path.exists():
                    raise FileNotFoundError(f"Vault path does not exist: {self.vault_path}")
                
                if not self.vault_path.is_dir():
                    raise ValueError(f"Vault path is not a directory: {self.vault_path}")
                
                # Initialize vector store (this will create the database and indexes)
                logger.info("Initializing vector store")
                await self.vector_store.initialize()
                
                # Test embedding service by generating a test embedding
                logger.info("Testing embedding service")
                
                self._initialized = True
                logger.info("Vault initialization completed successfully")
                
            except Exception as e:
                logger.error(f"Vault initialization failed: {e}")
                self._initialized = False
                raise RuntimeError(f"Failed to initialize Vault: {e}") from e
    
    async def update_index(self) -> None:
        """
        Update the index of the vault.
        1. Getting the list of current files in the vault
        2. Comparing with files stored in the database
        3. Removing chunks for deleted or modified files
        4. Processing and adding chunks for new or modified files
        5. Rebuilding indexes for optimal performance
        Raises:
            NotInitializedError: If vault is not initialized
            RuntimeError: If index update fails
        """
        if not self._initialized:
            raise NotInitializedError("Vault must be initialized before updating index")
        
        async with self._lock:
            try:
                logger.info("Starting index update")
                start_time = datetime.now()
                stored_files = await self.vector_store.sources()
                logger.info(f"Found {len(stored_files)} files in database")
                
                file_count = 0
                files_to_delete = []
                files_to_add = []
                modified = False
                for file_path in self.file_traversal.find_files():
                    try:
                        file_count += 1
                        stat = file_path.stat()
                        modified_at = datetime.fromtimestamp(stat.st_mtime)
                        relative_path = file_path.relative_to(self.vault_path).as_posix()
                        if relative_path not in stored_files:
                            files_to_add.append(file_path)
                        elif modified_at > stored_files.get(relative_path, datetime.min):
                            files_to_add.append(file_path)
                            files_to_delete.append(relative_path)
                            del stored_files[relative_path]
                        else:
                            del stored_files[relative_path]
                        if len(files_to_add) >= self.batch_size:
                            modified = True
                            if files_to_delete:
                                await self.vector_store.delete(files_to_delete)
                                files_to_delete.clear()
                            await self._batch_process_files(files_to_add)
                    except (OSError, ValueError) as e:
                        logger.warning(f"Failed to get stats for file {file_path}: {e}")
                        continue
                logger.info(f"Found {file_count} files in vault")
                
                
                if stored_files:
                    await self.vector_store.delete(list(stored_files.keys()))
                    modified = True
                
                if files_to_add:
                    await self._batch_process_files(files_to_add)
                    modified = True
                # Rebuild indexes if any changes were made
                if modified:
                    logger.info("Rebuilding indexes")
                    await self.vector_store.reindex()
                
                self._last_update_check = datetime.now()
                duration = (self._last_update_check - start_time).total_seconds()
                logger.info(f"Index update completed in {duration:.2f} seconds")
                
            except Exception as e:
                logger.error(f"Index update failed: {e}")
                raise RuntimeError(f"Failed to update index: {e}") from e

    async def _batch_process_files(self, files_to_add: list[Path]) -> None:
            # Process files in batches
            add_results = await asyncio.gather(
                *[self._process_file(path) for path in files_to_add],
                return_exceptions=True
            )
            files_to_add.clear()

    async def _process_file(self, file_path: Path) -> None:
            document = await self.document_processor.process(file_path)
            await self.vector_store.store([ chunk for chunk in self.chunker.chunk(document)])
    
    async def search(self, query: str) -> str:
        """
        Search files in vault by query and return formatted results.
        Args:
            query: The search query string
            
        Returns:
            Formatted string with search results
            
        Raises:
            NotInitializedError: If vault is not initialized
            RuntimeError: If search operation fails
        """
        if not self._initialized:
            raise NotInitializedError("Vault must be initialized before searching")
        
        try:
            # Check if index needs updating
            if (self._last_update_check is None or 
                datetime.now() - self._last_update_check > self._update_interval):
                logger.info("Index update needed before search")
                await self.update_index()
            
            logger.info(f"Searching for: {query}")
            
            chunks = await self.vector_store.search(
                query=query,
                tags=[],
                file_path=None,
                scope=SearchScope.ALL,
                limit=5
            )
            
            # Create search query object for formatter
            search_query = SearchQuery(
                text=query,
                tags=[],
                scope=SearchScope.ALL,
                path=None
            )
            
            formatted_results = await self.result_formatter.format(chunks, search_query)
            
            logger.info(f"Search completed: found {len(chunks)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search operation failed: {e}")
            raise RuntimeError(f"Failed to search vault: {e}") from e
    
    async def get_file(self, file_name: str) -> str:
        """
        Get file content by its name without extension or partial path.
        
        This method searches for files matching the given name and returns
        the content of the first match found.
        
        Args:
            file_name: File name without extension or partial path
            
        Returns:
            Content of the matched file
            
        Raises:
            NotInitializedError: If vault is not initialized
            FileNotFoundError: If no matching file is found
            RuntimeError: If file reading fails
        """
        if not self._initialized:
            raise NotInitializedError("Vault must be initialized before getting files")
        
        try:
            logger.debug(f"Looking for file: {file_name}")
            
            # Search for matching files
            matching_files = []
            for file_path in self.file_traversal.find_files():
                # Check if file name matches (with or without extension)
                file_stem = file_path.stem.lower()
                file_name_lower = file_name.lower()
                relative_path = file_path.relative_to(self.vault_path).as_posix()
                
                if (file_stem == file_name_lower or 
                    file_name_lower in relative_path.lower()):
                    matching_files.append(file_path)
            
            if not matching_files:
                raise FileNotFoundError(f"No file found matching: {file_name}")
            
            # Return content of first match
            target_file = matching_files[0]
            logger.info(f"Found file: {target_file.relative_to(self.vault_path)}")
            
            try:
                content = target_file.read_text(encoding='utf-8')
                return content
            except UnicodeDecodeError:
                # Try with error handling for problematic files
                content = target_file.read_text(encoding='utf-8', errors='replace')
                logger.warning(f"File {target_file} had encoding issues, some characters may be corrupted")
                return content
            
        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to get file {file_name}: {e}")
            raise RuntimeError(f"Failed to get file: {e}") from e
    
    async def list_files(self, directory: str) -> list[str]:
        """
        List all files in a directory.
        
        Args:
            directory: Directory path relative to vault root
            
        Returns:
            List of file names without .md extension, plus directory names ended with '/'
            
        Raises:
            NotInitializedError: If vault is not initialized
            RuntimeError: If directory listing fails
        """
        if not self._initialized:
            raise NotInitializedError("Vault must be initialized before listing files")
        
        try:
            logger.debug(f"Listing files in directory: {directory}")
            
            # Resolve target directory
            if directory == "" or directory == "/":
                target_dir = self.vault_path
            else:
                # Remove leading/trailing slashes and resolve path
                clean_dir = directory.strip('/')
                target_dir = self.vault_path / clean_dir
            
            if not target_dir.exists():
                logger.warning(f"Directory does not exist: {target_dir}")
                return []
            
            if not target_dir.is_dir():
                logger.warning(f"Path is not a directory: {target_dir}")
                return []
            
            files = []
            
            try:
                for item in target_dir.iterdir():
                    if item.is_file() and item.suffix.lower() == '.md':
                        # Add file name without extension
                        files.append(item.stem)
                    elif item.is_dir() and not item.name.startswith('.'):
                        # Add directory name with trailing slash
                        files.append(f"{item.name}/")
            except PermissionError as e:
                logger.warning(f"Permission denied accessing directory {target_dir}: {e}")
                return []
            
            # Sort files for consistent output
            files.sort()
            
            logger.debug(f"Found {len(files)} items in {directory}")
            return files
            
        except Exception as e:
            logger.error(f"Failed to list files in directory {directory}: {e}")
            raise RuntimeError(f"Failed to list directory: {e}") from e
    
    async def cleanup(self) -> None:
        """
        Clean up resources and close connections.
        
        This method should be called when the vault is no longer needed
        to ensure proper resource cleanup.
        """
        try:
            logger.info("Cleaning up Vault resources")
            
            # Close database connections if needed
            if hasattr(self.vector_store, 'db') and self.vector_store.db:
                # LanceDB connections are automatically managed
                pass
            
            self._initialized = False
            logger.info("Vault cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during vault cleanup: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup on garbage collection."""
        if self._initialized:
            try:
                # Note: We can't call async cleanup from __del__
                # This is just a safety net for resource cleanup
                logger.debug("Vault destructor called")
            except Exception:
                pass  # Ignore errors in destructor


def create_vault(
    vault_path: Path,
    db_table_name: str = "documents",
) -> Vault:
    """
    Factory method to create a fully configured Vault instance.
    
    This is the main entry point for creating Vault instances with all
    required services properly configured and injected.
    
    Args:
        vault_path: Path to the vault directory containing documents
        skip_patterns: List of patterns to skip during file traversal
        chunk_size: Size of text chunks for processing
        chunk_overlap: Overlap between consecutive chunks
        embedding_model: Name of the embedding model to use
        db_table_name: Name of the database table for storing chunks
        max_content_length: Maximum content length for result formatting
        include_metadata: Whether to include metadata in formatted results
        batch_size: Number of files to process in each batch
        
    Returns:
        Configured Vault instance ready for use
        
    Raises:
        RuntimeError: If service creation fails
    """
    try:
        logger.info(f"Creating Vault for path: {vault_path}")
        
        # Create all services using factory methods
        file_traversal = _create_file_traversal(Path(vault_path) )
        document_processor = _create_document_processor(Path(vault_path))
        chunker = _create_chunker()
        
        db_path = Path(vault_path) / '.vault_db'
        vector_store = _create_vector_store(db_path, db_table_name)
        search_engine = _create_search_engine(vector_store)
        result_formatter = _create_result_formatter()
        
        # Create and return Vault instance with injected services
        vault = Vault(
            vault_path=Path(vault_path),
            file_traversal=file_traversal,
            document_processor=document_processor,
            chunker=chunker,
            vector_store=vector_store,
            search_engine=search_engine,
            result_formatter=result_formatter,
        )
        
        logger.info("Vault factory method completed successfully")
        return vault
        
    except Exception as e:
        logger.error(f"Failed to create Vault: {e}")
        raise RuntimeError(f"Vault creation failed: {e}") from e
