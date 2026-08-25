"""
This module provides a comprehensive Vault class that implements the IVault interface,
managing all aspects of document indexing, searching, and retrieval in a RAG system.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import httpx
from lancedb.rerankers import Reranker, RRFReranker
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

from mcps.config import ServerConfig
from mcps.rag.embeddings import LangChainEmbeddingService
from mcps.rag.llm_reranker import LlmReranker
from mcps.rag.proxy_reranker import ProxyReranker

from .database import LanceDBStore
from .document_processing import (
    SUMMARY_CHUNK_POSITION,
    MarkdownFileTraversal,
    MarkdownProcessor,
    SemanticChunker,
    create_chunk,
)
from .interfaces import (
    Chunk,
    IChunker,
    IDocumentProcessor,
    IDocumentSummaryGenerator,
    IEmbeddingService,
    IFileTraversal,
    ISearchEngine,
    IVault,
    IVectorStore,
    Link,
    NoteLinks,
    NotInitializedError,
    SearchQuery,
    SearchScope,
    TraversalNode,
    TraversalResult,
    dedupe_links,
)
from .reranking import LangChainReranker
from .search import (
    HypotheticalDocumentGenerator,
    SemanticSearchEngine,
)
from .summarization import LangChainDocumentSummaryGenerator

logger = logging.getLogger("mcps.vault")
MAX_TRAVERSAL_NODES = 100


def _create_file_traversal(vault_path: Path) -> IFileTraversal:
    """Create and configure file traversal service."""
    return MarkdownFileTraversal(base_path=vault_path)


def _create_document_processor(vault_path: Path) -> IDocumentProcessor:
    """Create and configure document processor service."""
    return MarkdownProcessor(base_path=vault_path)


def _create_chunker() -> IChunker:
    """Create and configure text chunker service."""
    # return FixedSizeChunker(chunk_size=chunk_size, overlap=chunk_overlap)
    return SemanticChunker()


def create_embeddings(
        config: ServerConfig,
        http_client: httpx.AsyncClient
    ) -> IEmbeddingService:
    base_url = config.router_api_base
    api_key = SecretStr(config.router_api_key)
    embeddings = OpenAIEmbeddings(
        model=config.rag_embedding_model,
        dimensions=config.rag_embedding_dimensions,
        base_url=base_url,
        api_key=api_key,
        http_async_client=http_client,
        check_embedding_ctx_length=False
    )
    logger.info("Use embeddings with model %s and dimensions %d",
                config.rag_embedding_model, 
                config.rag_embedding_dimensions)
    return LangChainEmbeddingService(
                embeddings=embeddings,
                dimensions=config.rag_embedding_dimensions,
            )

def create_reranker(
        config: ServerConfig,
        http_client: httpx.AsyncClient
    ) -> Reranker:
    if config.rag_reranker_model:
        logger.info("Use reranker API with model %s",config.rag_reranker_model)
        return ProxyReranker(
            model_name=config.rag_reranker_model,
            proxy_url=config.router_api_base,
            api_key=config.router_api_key
        )
    if not (config.rag_reranker_embedding_model and config.rag_reranker_embedding_dimensions):
        logger.info("Use RRF reranker")
        return RRFReranker(return_score="relevance")

    base_url = config.router_api_base
    api_key = SecretStr(config.router_api_key)
    embed = OpenAIEmbeddings(
        model=config.rag_reranker_embedding_model,
        dimensions=config.rag_reranker_embedding_dimensions,
        base_url=base_url,
        api_key=api_key,
        http_async_client=http_client,
        check_embedding_ctx_length=False
    )
    if not config.rag_reranker_infer_model:
        logger.info("Use Embeddings reranker with model %s",config.rag_reranker_embedding_model)
        return LlmReranker(chat_model=None, embeddings=embed)

    infer_model = ChatOpenAI(
        model=config.rag_reranker_infer_model,
        base_url=base_url,
        api_key=api_key,
        http_async_client=http_client
    )
    logger.info("Use LLM reranker with embeddings model %s and chat model",config.rag_reranker_embedding_model, config.rag_reranker_infer_model)
    return LlmReranker(infer_model, embed)

@asynccontextmanager
async def _create_vector_store(
    config: ServerConfig,
    embedding_service: IEmbeddingService,
    reranker: Reranker | None = None,
) -> AsyncIterator[IVectorStore]:
    """Create and configure vector store service."""
    store = LanceDBStore(
        db_path=config.vault_dir / ".vault_db",  # type: ignore
        embedding_service=embedding_service,
        table_name=config.table_name,
        reranker=reranker,
    )
    try:
        await store.initialize()
        yield store
    finally:
        await store.cleanup()


def _create_search_engine(
    vector_store: IVectorStore,
    config: ServerConfig,
    http_client: httpx.AsyncClient,
) -> ISearchEngine:
    """Create and configure search engine service."""
    if not config.rag_infer_model:
        logger.info("Use plain search engine")
        return SemanticSearchEngine(
            vector_store,
            limit=config.search_limit,
        )

    search_model = ChatOpenAI(
        model=config.rag_infer_model,
        base_url=config.router_api_base,
        api_key=SecretStr(config.router_api_key),
        http_async_client=http_client,
    )
    logger.info("Use semantic search engine with model %s",config.rag_infer_model)
    return SemanticSearchEngine(
        vector_store,
        limit=config.search_limit,
        hypothetical_document_generator=HypotheticalDocumentGenerator(search_model),
        reranker=LangChainReranker(search_model),
    )


def _create_document_summary_generator(
    config: ServerConfig,
    http_client: httpx.AsyncClient,
) -> IDocumentSummaryGenerator | None:
    if not config.rag_summary_model:
        return None

    summary_model = ChatOpenAI(
        model=config.rag_summary_model,
        base_url=config.router_api_base,
        api_key=SecretStr(config.router_api_key),
        http_async_client=http_client,
        max_retries=2,
    )
    return LangChainDocumentSummaryGenerator(summary_model)



class Vault(IVault):
    """
    Vault implementation for managing RAG operations.
    
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
        document_summary_generator: IDocumentSummaryGenerator | None = None,
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
            batch_size: Number of files to process in each batch
        """
        self.vault_path = Path(vault_path)
        self.db_path = self.vault_path / '.vault_db'
        self._initialized = False
        self._lock = asyncio.Lock()
        self.batch_size = batch_size
        
        # Inject services
        self.file_traversal = file_traversal
        self.document_processor = document_processor
        self.chunker = chunker
        self.vector_store = vector_store
        self.search_engine = search_engine
        self.document_summary_generator = document_summary_generator
        
        logger.info(f"Vault initialized with injected services for path: {vault_path}")

    async def initialize(self) -> None:
        """
        Initialize the vault manager and all its services.
        
        This method performs the following operations:
        - Validates the vault path exists
        - Sets up the embedding service
        - Prepares all services for operation
        
        Raises:
            RuntimeError: If initialization fails
            FileNotFoundError: If vault path doesn't exist
        """
        async with self._lock:
            if self._initialized:
                logger.debug("Vault already initialized")
                return

            self._initialized = True
            logger.info(
                f"Vault {self.vault_path} initialization completed successfully"
            )
    
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
                        modified_at = stat.st_mtime
                        relative_path = file_path.relative_to(
                            self.vault_path
                        ).as_posix()
                        if relative_path not in stored_files:
                            logger.info("File %s not found in database",relative_path)
                            files_to_add.append(file_path)
                        elif modified_at > stored_files.get(
                            relative_path, 0
                        ):
                            logger.info("File %s has updated at %s, stored time %s",
                                        relative_path,
                                        datetime.fromtimestamp(modified_at).isoformat(),
                                        datetime.fromtimestamp(stored_files.get(relative_path,0)).isoformat()
                                        )
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
                        logger.warning(
                            f"Failed to get stats for file {file_path}: {e}"
                        )
                        continue
                logger.info(f"Found {file_count} files in vault")
                
                
                if stored_files:
                    logger.info("Delete %d removed files: %s",len(stored_files),list(stored_files.keys())[:4])
                    await self.vector_store.delete(list(stored_files.keys()))
                    modified = True
                
                if files_to_delete:
                    await self.vector_store.delete(files_to_delete)
                if files_to_add:
                    await self._batch_process_files(files_to_add)
                    modified = True
                # Rebuild indexes if any changes were made
                if modified:
                    logger.info("Rebuilding indexes")
                    await self.vector_store.reindex(False)
                
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                logger.info(
                    f"Index update completed in {duration:.2f} seconds, "
                    f"processed {file_count} files"
                )
                
            except Exception as e:
                logger.error(f"Index update failed: {e}")
                raise RuntimeError(f"Failed to update index: {e}") from e

    async def _batch_process_files(self, files_to_add: list[Path]) -> None:
        logger.info(f"Add {len(files_to_add)} files in batch")
        results = await asyncio.gather(
            *[self._process_file(path) for path in files_to_add],
            return_exceptions=True,
        )
        for path, result in zip(files_to_add, results, strict=True):
            if isinstance(result, Exception):
                logger.error("Failed to process file %s: %s", path, result)
        files_to_add.clear()

    async def _process_file(self, file_path: Path) -> None:
        document = await self.document_processor.process(file_path)
        chunks = list(self.chunker.chunk(document))
        if self.document_summary_generator is not None and document.content.strip() and len(chunks) > 2:
            try:
                summary = await self.document_summary_generator.generate(document)
                if summary.strip():
                    logger.info(
                        "Generated summary chunk for %s",
                        document.source_path,
                    )
                    chunks = [create_chunk(document, summary,SUMMARY_CHUNK_POSITION), *chunks]
            except Exception as e:
                logger.warning(
                    "Failed to generate summary chunk for %s: %s",
                    document.source_path,
                    e,
                )

        await self.vector_store.store(chunks)
    
    async def search(
        self,
        query: str,
        tags: list[str] | None = None,
        path: str | None = None,
    ) -> list[Chunk]:
        """
        Search files in vault by query and return matching chunks.

        Args:
            query: The search query string
            tags: Optional list of tags to filter by (all must match)
            path: Optional file path prefix to limit search scope

        Returns:
            List of matching Chunk objects

        Raises:
            NotInitializedError: If vault is not initialized
            RuntimeError: If search operation fails
        """
        if not self._initialized:
            raise NotInitializedError("Vault must be initialized before searching")

        try:
            logger.info(f"Searching for: {query}")
            search_query = SearchQuery(
                text=query,
                tags=tags or [],
                scope=SearchScope.ALL,
                path=path,
            )

            return await self.search_engine.search(query=search_query)

        except Exception as e:
            logger.error(f"Search operation failed: {e}")
            raise RuntimeError(f"Failed to search vault: {e}") from e

    async def get_backlinks(
        self, wikilink_names: list[str]
    ) -> dict[str, list[Link]]:
        """Return incoming links for the requested notes, keyed by note name.

        A single store query fetches every note linking to any requested
        name; each returned Link carries the incoming edge's type and the
        source note's wikilink_name as target. Per-note buckets are
        deduplicated on (type, target): summary chunks carry the whole
        document's links and would otherwise report every edge twice.
        """
        if not self._initialized:
            raise NotInitializedError(
                "Vault must be initialized before fetching backlinks"
            )
        if not wikilink_names:
            return {}
        backlinks: dict[str, list[Link]] = {
            name: [] for name in wikilink_names
        }
        sources = await self.vector_store.get_notes_linking_to(wikilink_names)
        for source in sources:
            for link in source.links:
                if link.target in backlinks:
                    backlinks[link.target].append(
                        Link(type=link.type, target=source.note)
                    )
        return {
            name: dedupe_links(links) for name, links in backlinks.items()
        }
    
    async def traverse_relations(
        self,
        wikilink_name: str,
        depth: int = 1,
        relation_types: list[str] | None = None,
    ) -> TraversalResult:
        """Walk typed note relations breadth-first in both directions."""
        if not self._initialized:
            raise NotInitializedError(
                "Vault must be initialized before traversing relations"
            )

        origin_notes = await self.vector_store.get_notes_with_links(
            [wikilink_name]
        )
        if not origin_notes:
            raise FileNotFoundError(wikilink_name)

        nodes: list[TraversalNode] = []
        visited = {wikilink_name}
        frontier = [wikilink_name]
        frontier_notes = origin_notes

        for current_depth in range(1, depth + 1):
            incoming_notes = await self.vector_store.get_notes_linking_to(
                frontier, relation_types
            )
            proposals = self._relation_proposals(
                frontier_notes,
                incoming_notes,
                relation_types,
            )
            candidates = self._unvisited_proposals(proposals, visited)
            candidate_names = list(dict.fromkeys(node.note for node in candidates))
            if not candidate_names:
                break
            resolved_notes = await self.vector_store.get_notes_with_links(
                candidate_names
            )
            resolved_by_name = {note.note: note for note in resolved_notes}
            resolved_nodes = self._resolved_nodes(
                candidates,
                resolved_by_name,
                current_depth,
            )

            remaining = MAX_TRAVERSAL_NODES - len(nodes)
            if len(resolved_nodes) > remaining:
                nodes.extend(resolved_nodes[:remaining])
                return TraversalResult(
                    origin=wikilink_name,
                    nodes=nodes,
                    truncated=True,
                    warning=(
                        f"Traversal reached the {MAX_TRAVERSAL_NODES}-note cap "
                        f"at depth {current_depth}; narrow relation_types or "
                        "reduce depth."
                    ),
                )

            nodes.extend(resolved_nodes)
            frontier = [node.note for node in resolved_nodes]
            if not frontier:
                break
            frontier_notes = [
                resolved_by_name[name] for name in frontier
            ]

        return TraversalResult(origin=wikilink_name, nodes=nodes)

    @staticmethod
    def _relation_proposals(
        outgoing_notes: list[NoteLinks],
        incoming_notes: list[NoteLinks],
        relation_types: list[str] | None,
    ) -> list[TraversalNode]:
        proposals = [
            TraversalNode(
                note=link.target,
                depth=0,
                direction="outgoing",
                relation=link.type,
                via=source.note,
            )
            for source in outgoing_notes
            for link in source.links
            if relation_types is None or link.type in relation_types
        ]
        proposals.extend(
            TraversalNode(
                note=source.note,
                depth=0,
                direction="incoming",
                relation=link.type,
                via=link.target,
            )
            for source in incoming_notes
            for link in source.links
        )
        return proposals

    @staticmethod
    def _unvisited_proposals(
        proposals: list[TraversalNode], visited: set[str]
    ) -> list[TraversalNode]:
        candidates: list[TraversalNode] = []
        for proposal in proposals:
            if proposal.note in visited:
                continue
            visited.add(proposal.note)
            candidates.append(proposal)
        return candidates

    @staticmethod
    def _resolved_nodes(
        candidates: list[TraversalNode],
        resolved_by_name: dict[str, NoteLinks],
        depth: int,
    ) -> list[TraversalNode]:
        return [
            proposal.model_copy(
                update={
                    "title": resolved_by_name[proposal.note].title,
                    "description": resolved_by_name[proposal.note].description,
                    "depth": depth,
                }
            )
            for proposal in candidates
            if proposal.note in resolved_by_name
        ]

    async def get_file(
        self,
        wikilink_name: str,
        offset: int | None = None,
        limit: int | None = None,
    ) -> str:
        """
        Get file content by wikilink name.
        
        Args:
            wikilink_name: Full relative wikilink name without extension.
            offset: Zero-based line index to start reading from.
            limit: Maximum number of lines to return.
            
        Returns:
            Content of the matched file
            
        Raises:
            NotInitializedError: If vault is not initialized
            FileNotFoundError: If no matching file is found
            RuntimeError: If file reading fails
        """
        if not self._initialized:
            raise NotInitializedError("Vault must be initialized before getting files")

        logger.debug(f"Looking for file: {wikilink_name}")
        source_path = await self._resolve_source_path(wikilink_name)
        target_file = self.vault_path / source_path
        if not target_file.exists():
            raise FileNotFoundError(f"No file found matching: {wikilink_name}")
        logger.info(f"Found file: {source_path}")

        content = target_file.read_text(encoding="utf-8", errors="replace")
        return self._slice_content(content, offset, limit)

    async def _resolve_source_path(self, wikilink_name: str) -> str:
        if "/" in wikilink_name:
            return f"{wikilink_name}.md"

        source_paths = await self.vector_store.get_sources_by_name(wikilink_name)
        if not source_paths:
            raise FileNotFoundError(f"No file found matching: {wikilink_name}")
        if len(source_paths) > 1:
            raise ValueError(
                f"Ambiguous wikilink name '{wikilink_name}'. Matches: "
                f"{', '.join(source_paths)}"
            )
        return source_paths[0]

    @staticmethod
    def _slice_content(
        content: str,
        offset: int | None,
        limit: int | None,
    ) -> str:
        lines = content.splitlines(keepends=True)
        start = offset or 0
        selected = lines[start:] if limit is None else lines[start:start + limit]
        return "".join(selected)

    async def list_files(self, directory: str) -> list[str]:
        """
        List all files in a directory.
        
        Args:
            directory: Directory path relative to vault root
            
        Returns:
            List of file names without .md extension, plus directory names ended
            with '/'
            
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
                clean_dir = directory.strip("/")
                target_dir = self.vault_path / clean_dir
            
            if not target_dir.exists():
                logger.warning(f"Directory does not exist: {target_dir}")
                return []
            
            if not target_dir.is_dir():
                logger.warning(f"Path is not a directory: {target_dir}")
                return []
            
            try:
                files = [
                    item.stem if item.is_file() else f"{item.name}/"
                    for item in target_dir.iterdir()
                    if (
                        (item.is_file() and item.suffix.lower() == ".md")
                        or item.is_dir()
                    )
                    and not item.name.startswith(".")
                ]
                files.sort()
                logger.debug(f"Found {len(files)} items in {directory}")
                return files
            except PermissionError as e:
                logger.warning(
                    f"Permission denied accessing directory {target_dir}: {e}"
                )
                return []
            
            
        except Exception as e:
            logger.error(f"Failed to list files in directory {directory}: {e}")
            raise RuntimeError(f"Failed to list directory: {e}") from e
    
    async def cleanup(self) -> None:
        """
        Clean up resources and close connections.
        
        This method should be called when the vault is no longer needed
        to ensure proper resource cleanup.
        """
        self._initialized = False
        logger.info("Vault cleanup completed")
    


@asynccontextmanager
async def create_vault(
    config: ServerConfig,
    http_client: httpx.AsyncClient
) -> AsyncIterator[Vault]:
    
    """
    Factory method to create a fully configured Vault instance.
    
    This is the main entry point for creating Vault instances with all
    required services properly configured and injected.
    
    Args:
        config - MCP server configuration
        
    Returns:
        Configured Vault instance ready for use
        
    Raises:
        RuntimeError: If service creation fails
    """
    try:
        vault_path : Path = config.vault_dir # type: ignore checked in server.py
        logger.info(f"Creating Vault for path: {vault_path}")
        
        # Create all services using factory methods
        file_traversal = _create_file_traversal(vault_path)
        document_processor = _create_document_processor(vault_path)
        chunker = _create_chunker()
        embeddings = create_embeddings(config,http_client)
        reranker = create_reranker(config,http_client)
        async with _create_vector_store(config,embeddings,reranker) as vector_store:
            search_engine = _create_search_engine(
                vector_store,
                config,
                http_client,
            )
            document_summary_generator = _create_document_summary_generator(
                config,
                http_client,
            )
            
            # Create and return Vault instance with injected services
            vault = Vault(
                vault_path=Path(vault_path),
                file_traversal=file_traversal,
                document_processor=document_processor,
                chunker=chunker,
                vector_store=vector_store,
                search_engine=search_engine,
                document_summary_generator=document_summary_generator,
            )
            await vault.initialize()
            logger.info("Vault factory method completed successfully")
            yield vault
            await vault.cleanup()
        
    except Exception as e:
        logger.error(f"Failed to create Vault: {e}")
        raise RuntimeError(f"Vault creation failed: {e}") from e
