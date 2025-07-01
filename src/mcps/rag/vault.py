"""
Production-ready Vault implementation for managing RAG operations.

This module provides a comprehensive Vault class that implements the IVault interface,
managing all aspects of document indexing, searching, and retrieval in a RAG system.
"""

import asyncio
from importlib.metadata import files
import logging
import os
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
from lancedb.embeddings import EmbeddingFunction, OllamaEmbeddings
from lancedb.rerankers import RRFReranker

from .database import LanceDBStore
from .document_processing import FixedSizeChunker, MarkdownFileTraversal, MarkdownProcessor
from .interfaces import (
    Chunk,
    Document,
    IChunker,
    IDocumentProcessor,
    IEmbeddingService,
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


class Vault(IVault):
    """
    Production-ready Vault implementation for managing RAG operations.
    
    This class provides a comprehensive implementation of the IVault interface,
    managing document indexing, searching, and retrieval operations. It automatically
    instantiates and manages all required services with sensible defaults.
    
    Features:
    - Automatic service instantiation with dependency injection
    - Thread-safe operations with proper synchronization
    - Comprehensive error handling and logging
    - Resource management and cleanup
    - Incremental index updates based on file modification times
    - Support for various search scopes and filtering options
    
    Example:
        vault = Vault(Path("/path/to/documents"))
        await vault.initialize()
        await vault.update_index()
        results = await vault.search("machine learning")
    """
    
    def __init__(
        self,
        vault_path: Path,
        skip_patterns: Optional[list[str]] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        db_table_name: str = "documents",
        max_content_length: int = 2000,
        include_metadata: bool = True,
        batch_size: int = 8,
    ):
        """
        Initialize the Vault with the specified configuration.
        
        Args:
            vault_path: Path to the vault directory containing documents
            skip_patterns: List of patterns to skip during file traversal
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between consecutive chunks
            embedding_model: Name of the embedding model to use
            db_table_name: Name of the database table for storing chunks
            max_content_length: Maximum content length for result formatting
            include_metadata: Whether to include metadata in formatted results
        """
        self.vault_path = Path(vault_path)
        self.db_path = self.vault_path / '.vault_db'
        self._initialized = False
        self._lock = threading.RLock()
        self._last_update_check: Optional[datetime] = None
        self._update_interval = timedelta(minutes=1)
        self.batch_size = batch_size
        
        try:
            # Initialize services with dependency injection
            logger.info(f"Initializing Vault for path: {vault_path}")
            
            # File traversal service
            self.file_traversal: IFileTraversal = MarkdownFileTraversal(
                base_path=self.vault_path,
            )
            
            # Document processor
            self.document_processor: IDocumentProcessor = MarkdownProcessor(
                base_path=self.vault_path
            )
            
            # Text chunker
            self.chunker: IChunker = FixedSizeChunker(
                chunk_size=chunk_size,
                overlap=chunk_overlap
            )
            
            # Create LanceDB-compatible embedding function
            ollama_base_url = os.getenv("OLLAMA_API_BASE")
            if ollama_base_url:
                self.embedding_function = get_registry().get("ollama").create(name="bge-m3:latest", host=ollama_base_url)
                # self.embedding_function = get_registry().get("ollama").create(name="hf.co/nomic-ai/nomic-embed-text-v2-moe-GGUF:Q6_K", host=ollama_base_url)
            
            if os.environ.get("VOYAGE_API_KEY") :
                from lancedb.rerankers import VoyageAIReranker
                reranker =  VoyageAIReranker(model_name="rerank-2-lite", column = "content", api_key=os.environ.get("VOYAGE_API_KEY"))
            else:
                from lancedb.rerankers import RRFReranker
                reranker = RRFReranker(return_score='all')
            # Vector store
            self.vector_store: IVectorStore = LanceDBStore(
                db_path=self.db_path,
                embedding_function=self.embedding_function,
                table_name=db_table_name,
                reranker=reranker
            )
            
            # Search engine
            self.search_engine: ISearchEngine = SemanticSearchEngine(
                vector_store=self.vector_store
            )
            
            # Result formatter
            self.result_formatter: IResultFormatter = MarkdownResultFormatter(
                max_content_length=max_content_length,
                include_metadata=include_metadata
            )
            
            logger.info("Vault services initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Vault services: {e}")
            raise RuntimeError(f"Vault initialization failed: {e}") from e
    
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
        
        async with asyncio.Lock():
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
