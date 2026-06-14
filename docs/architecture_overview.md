# MCPS Architecture

FastMCP-based Model Context Protocol server with RAG capabilities for Obsidian vault search. Implements dependency injection pattern with interface abstractions for all components. #architecture #design #python #mcp

## Technology Stack

### Core Framework
- **Python 3.12+** - Runtime environment
- **FastMCP v2.8.1** - Model Context Protocol server implementation [src/mcps/server.py:28-34](../src/mcps/server.py#L28-L34)
- **Pydantic v2.x** - Data validation and modeling (via FastMCP dependency)

### Database and Search
- **LanceDB v0.23.0** - Vector database with full-text search capabilities [src/mcps/rag/database.py:32-308](../src/mcps/rag/database.py#L32-L308)
- **rank-bm25** - BM25 algorithm for keyword-based ranking

### AI Services
- **LiteLLM Router** - Central model gateway for web research and Obsidian RAG models.
- **LangChain core interfaces** - Provider-neutral `BaseChatModel` and `Embeddings` contracts inside RAG.
- **langchain-openai adapters** - OpenAI-compatible adapters outside RAG, configured against LiteLLM Router.

### Document Processing
- **markdown v3.4.0** - Markdown parsing
- **python-frontmatter v1.1.0** - YAML frontmatter extraction [src/mcps/rag/document_processing.py:118](../src/mcps/rag/document_processing.py#L118)

## System Architecture

### Design Pattern: Dependency Injection with Interface Abstractions

The architecture follows a **Dependency Injection** pattern with clear interface definitions in [src/mcps/rag/interfaces.py:95-268](../src/mcps/rag/interfaces.py#L95-L268). All RAG components implement abstract interfaces enabling flexible implementation swapping.

#### Core Interfaces

**IDocumentProcessor** [src/mcps/rag/interfaces.py:95-101](../src/mcps/rag/interfaces.py#L95-L101)
- Document file processing and metadata extraction

**IChunker** [src/mcps/rag/interfaces.py:105-111](../src/mcps/rag/interfaces.py#L105-L111)
- Text chunking strategies (FixedSizeChunker, SemanticChunker)

**IEmbeddingService** [src/mcps/rag/interfaces.py:114-130](../src/mcps/rag/interfaces.py#L114-L130)
- Embedding generation from multiple providers

**IVectorStore** [src/mcps/rag/interfaces.py:140-193](../src/mcps/rag/interfaces.py#L140-L193)
- Vector storage and hybrid search operations

**ISearchEngine** [src/mcps/rag/interfaces.py:195-201](../src/mcps/rag/interfaces.py#L195-L201)
- Search execution and result retrieval

**IResultFormatter** [src/mcps/rag/interfaces.py:204-210](../src/mcps/rag/interfaces.py#L204-L210)
- Search result formatting

**IFileTraversal** [src/mcps/rag/interfaces.py:213-219](../src/mcps/rag/interfaces.py#L213-L219)
- File system discovery

**IVault** [src/mcps/rag/interfaces.py:222-268](../src/mcps/rag/interfaces.py#L222-L268)
- High-level vault management orchestrating all components

### Component Architecture

#### Entry Point Flow

```
main() [src/mcps/__init__.py:15-30]
  ↓
create_config() [src/mcps/config.py:38-75]
  ↓
create_server() [src/mcps/server.py:75-87]
  ↓
DevAutomationServer initialization [src/mcps/server.py:28-34]
  ↓
Tool/Resource registration [src/mcps/server.py:38-68]
  ↓
Async server start [src/mcps/server.py:70-72]
```

#### RAG Pipeline Architecture

The RAG pipeline in [src/mcps/rag/vault.py:129-360](../src/mcps/rag/vault.py#L129-L360) implements the `IVault` interface:

**Index Update Flow** [src/mcps/rag/vault.py:236-306](../src/mcps/rag/vault.py#L236-L306):
1. Retrieve stored file metadata from vector store
2. Traverse file system for current files
3. Compare modification timestamps
4. Delete outdated chunks from removed/modified files
5. Process new/modified files in batches
6. Rebuild search indexes if changes detected

**File Processing Pipeline** [src/mcps/rag/vault.py:316-318](../src/mcps/rag/vault.py#L316-L318):
1. Document processing extracts content and metadata
2. Chunking splits document into semantic segments
3. Vector store stores chunks with embeddings

**Search Flow** [src/mcps/rag/search.py:22-70](../src/mcps/rag/search.py#L22-L70):
1. Generate query embedding
2. Execute hybrid search (vector + full-text)
3. Apply tag and path filters
4. Rerank results using selected strategy
5. Format results for output

#### Document Processing Components

**MarkdownProcessor** [src/mcps/rag/document_processing.py:108-168](../src/mcps/rag/document_processing.py#L108-L168):
- Parses YAML frontmatter metadata
- Extracts title, description, source, tags
- Generates MD5-based document IDs
- Extracts wikilinks and hashtags

**SemanticChunker** [src/mcps/rag/document_processing.py:213-329](../src/mcps/rag/document_processing.py#L213-L329):
- Splits by markdown headers (H1-H2)
- Merges small sections to meet minimum chunk size
- Splits large sections by paragraphs
- Default: max_chunk_size=2000, min_chunk_size=100

**Chunk Creation** [src/mcps/rag/document_processing.py:49-71](../src/mcps/rag/document_processing.py#L49-L71):
- Extracts outgoing wikilinks using regex patterns
- Extracts hashtags from content
- Combines document-level and content-level tags

#### Vector Store Implementation

**LanceDBStore** [src/mcps/rag/database.py:32-308](../src/mcps/rag/database.py#L32-L308) provides:
- Hybrid search combining vector similarity and full-text search
- Tag filtering with LanceDB `array_has_all` operator
- Path filtering with SQL LIKE clauses
- LanceDB RRF reranking plus optional async LangChain post-retrieval reranking

**Index Types** [src/mcps/rag/database.py:231-279](../src/mcps/rag/database.py#L231-L279):
- Full-text search indexes on content, title, description columns
- LabelList index on tags column for efficient tag filtering

#### Reranking Strategies

**LangChainReranker** [src/mcps/rag/reranking.py](../src/mcps/rag/reranking.py):
- Async LLM-based relevance scoring with categories (PERFECT, GOOD, SOME, BAD, NONE)
- Uses provider-neutral `BaseChatModel`

**RRF (Reciprocal Rank Fusion)**:
- LanceDB reranking strategy combining vector and full-text signals

## MCP Server Features

### Registered Tools

**ObsidianTools** [src/mcps/tools/obsidian_vault.py:35-64](../src/mcps/tools/obsidian_vault.py#L35-L64):
- `obsidian_list_files` - List files in vault folder
- `obsidian_get_content` - Retrieve file content
- `obsidian_rename_move` - Rename/move notes with wikilink updates
- `obsidian_search` - Semantic search in vault with tag/path filtering

### Registered Resources

**Resource Handlers** [src/mcps/server.py:39-64](../src/mcps/server.py#L39-L64):
- `url://{encoded_url}` - URL content fetching
- `doc://{library_name}` - Documentation resource access
- `project://{project_name}` - Project content access
- `resource://test` - Test resource
- `documentation://test/docs` - Test documentation

### Prompt Templates

**File-based Prompts** [src/mcps/prompts/file_prompts.py:16-20](../src/mcps/prompts/file_prompts.py#L16-L20):
- Dynamic prompt templates loaded from markdown files in prompts directory
- Template variable substitution with `{{variable}}` format
- Initial prompts: code review, readability checks

## Key Design Decisions

### Interface Segregation
All components implement specific interfaces enabling:
- Flexible component replacement without code changes
- Provider-neutral embedding and chat-model adapters
- RRF plus optional async LangChain reranking
- Multiple chunking strategies (FixedSize, Semantic)

### Async Context Managers
Tools use async context manager pattern [src/mcps/common.py:26-42](../src/mcps/common.py#L26-L42) for proper vault lifecycle management and cleanup.

### Factory Pattern
Component creation isolated in factory functions:
- `create_vault()` [src/mcps/rag/vault.py:502-555](../src/mcps/rag/vault.py#L502-L555)
- `create_server()` [src/mcps/server.py:75-87](../src/mcps/server.py#L75-L87)

### Thread-Safe Operations
Vault operations protected with `asyncio.Lock` [src/mcps/rag/vault.py:178](../src/mcps/rag/vault.py#L178) for concurrent access safety.

### Batch Processing
File processing and embedding generation use configurable batch sizes [src/mcps/rag/vault.py:181](../src/mcps/rag/vault.py#L181) for performance optimization.
