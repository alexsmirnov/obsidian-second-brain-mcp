# MCPS Architecture

FastMCP-based Model Context Protocol server with RAG capabilities for Obsidian vault search and a LangGraph-powered web research agent. Implements dependency injection pattern with interface abstractions for RAG components. #architecture #design #python #mcp

## Technology Stack

### Core Framework
- **Python 3.13+** - Runtime environment [pyproject.toml:9](../pyproject.toml#L9)
- **FastMCP 3.4.2** - Model Context Protocol server implementation [src/mcps/server.py:8](../src/mcps/server.py#L8), [src/mcps/server.py:82-94](../src/mcps/server.py#L82-L94)
- **Pydantic v2.x** - Data validation and modeling (via FastMCP and lancedb dependencies) [src/mcps/rag/interfaces.py:12](../src/mcps/rag/interfaces.py#L12)

### Database and Search
- **LanceDB 0.25.3** - Vector database with full-text and hybrid search capabilities [src/mcps/rag/database.py:9-14](../src/mcps/rag/database.py#L9-L14)
- **pyarrow** - Columnar data format for LanceDB schemas and table operations
- **rank-bm25** - BM25 algorithm for keyword-based ranking (reserved for future use)

### AI Services
- **OpenAI-compatible model router** - Central gateway for web research and Obsidian RAG chat/embedding models.
- **LangChain core interfaces** - Provider-neutral `BaseChatModel` and `Embeddings` contracts inside RAG.
- **langchain-openai adapters** - OpenAI-compatible adapters configured against the model router.
- **langchain-google-genai** - Google Gemini adapter for the research agent.
- **LangGraph** - Agent graph orchestration for deep web research [src/mcps/research/deep_research.py:15](../src/mcps/research/deep_research.py#L15).

### Document Processing
- **markdown** - Markdown parsing
- **python-frontmatter** - YAML frontmatter extraction [src/mcps/rag/document_processing.py:12](../src/mcps/rag/document_processing.py#L12)

## System Architecture

### Design Pattern: Dependency Injection with Interface Abstractions

The architecture follows a **Dependency Injection** pattern with clear interface definitions in [src/mcps/rag/interfaces.py:101-307](../src/mcps/rag/interfaces.py#L101-L307). All RAG components implement abstract interfaces enabling flexible implementation swapping.

#### Core Interfaces

**IDocumentProcessor** [src/mcps/rag/interfaces.py:101-107](../src/mcps/rag/interfaces.py#L101-L107)
- Document file processing and metadata extraction

**IChunker** [src/mcps/rag/interfaces.py:111-117](../src/mcps/rag/interfaces.py#L111-L117)
- Text chunking strategies (FixedSizeChunker, SemanticChunker)

**IEmbeddingService** [src/mcps/rag/interfaces.py:120-147](../src/mcps/rag/interfaces.py#L120-L147)
- Embedding generation from multiple providers

**IDocumentSummaryGenerator** [src/mcps/rag/interfaces.py:150-156](../src/mcps/rag/interfaces.py#L150-L156)
- Whole-document summary generation

**IVectorStore** [src/mcps/rag/interfaces.py:158-223](../src/mcps/rag/interfaces.py#L158-L223)
- Vector storage and hybrid search operations

**ISearchEngine** [src/mcps/rag/interfaces.py:225-231](../src/mcps/rag/interfaces.py#L225-L231)
- Search execution and result retrieval

**IResultFormatter** [src/mcps/rag/interfaces.py:234-240](../src/mcps/rag/interfaces.py#L234-L240)
- Search result formatting

**IFileTraversal** [src/mcps/rag/interfaces.py:243-249](../src/mcps/rag/interfaces.py#L243-L249)
- File system discovery

**IVault** [src/mcps/rag/interfaces.py:252-307](../src/mcps/rag/interfaces.py#L252-L307)
- High-level vault management orchestrating all components

### Component Architecture

#### Entry Point Flow

```
main() [src/mcps/server.py:192-235]
  ↓
create_config() [src/mcps/config.py:52-107]
  ↓
create_server() [src/mcps/server.py:134-146]
  ↓
DevAutomationServer initialization [src/mcps/server.py:82-94]
  ↓
Tool registration [src/mcps/server.py:96-121]
  ↓
Async server start [src/mcps/server.py:125-131]
```

#### RAG Pipeline Architecture

The RAG pipeline in [src/mcps/rag/vault.py:252-556](../src/mcps/rag/vault.py#L252-L556) implements the `IVault` interface:

**Index Update Flow** [src/mcps/rag/vault.py:285-364](../src/mcps/rag/vault.py#L285-L364):
1. Retrieve stored file metadata from vector store
2. Traverse file system for current files
3. Compare modification timestamps
4. Delete outdated chunks from removed/modified files
5. Process new/modified files in batches
6. Rebuild search indexes if changes detected

**File Processing Pipeline** [src/mcps/rag/vault.py:377-396](../src/mcps/rag/vault.py#L377-L396):
1. Document processing extracts content and metadata
2. Chunking splits document into semantic segments
3. Optional whole-document summary generation
4. Vector store stores chunks with embeddings

**Search Flow** [src/mcps/rag/search.py:113-146](../src/mcps/rag/search.py#L113-L146):
1. Generate hypothetical document (optional)
2. Generate query embedding
3. Execute hybrid search (vector + full-text)
4. Apply tag and path filters
5. Rerank results using selected strategy
6. Merge results with neighbor chunks
7. Format results for output

#### Document Processing Components

**MarkdownProcessor** [src/mcps/rag/document_processing.py:151-215](../src/mcps/rag/document_processing.py#L151-L215):
- Parses YAML frontmatter metadata
- Extracts title, description, source, tags
- Generates MD5-based document IDs
- Extracts wikilinks and hashtags

**SemanticChunker** [src/mcps/rag/document_processing.py:264-460](../src/mcps/rag/document_processing.py#L264-L460):
- Splits by markdown headers (H1-H3)
- Merges small sections to meet minimum chunk size
- Splits large sections by paragraphs
- Default: max_chunk_size=1000, min_chunk_size=500

**Chunk Creation** [src/mcps/rag/document_processing.py:56-93](../src/mcps/rag/document_processing.py#L56-L93):
- Extracts outgoing wikilinks using regex patterns
- Extracts hashtags from content
- Combines document-level and content-level tags

#### Vector Store Implementation

**LanceDBStore** [src/mcps/rag/database.py:27-375](../src/mcps/rag/database.py#L27-L375) provides:
- Hybrid search combining vector similarity and full-text search
- Tag filtering with LanceDB `array_has_all` operator
- Path filtering with SQL LIKE clauses
- LanceDB RRF reranking plus optional proxy/LLM rerankers

**Index Types** [src/mcps/rag/database.py:250-304](../src/mcps/rag/database.py#L250-L304):
- Full-text search indexes on content, title, description columns
- LabelList index on tags column for efficient tag filtering

#### Reranking Strategies

**LangChainReranker** [src/mcps/rag/reranking.py](../src/mcps/rag/reranking.py):
- Async LLM-based relevance scoring with categories (PERFECT, GOOD, SOME, BAD, NONE)
- Uses provider-neutral `BaseChatModel`

**ProxyReranker** [src/mcps/rag/proxy_reranker.py](../src/mcps/rag/proxy_reranker.py):
- OpenAI-compatible `/v1/rerank` endpoint wrapper with RRF fallback

**LlmReranker** [src/mcps/rag/llm_reranker.py](../src/mcps/rag/llm_reranker.py):
- Fuses LLM structured relevance ratings with embedding cosine similarity

**RRF (Reciprocal Rank Fusion)**:
- LanceDB reranking strategy combining vector and full-text signals

## MCP Server Features

### Registered Tools

**web_research** [src/mcps/server.py:110-118](../src/mcps/server.py#L110-L118) - Always registered. Runs the LangGraph deep-research agent through the shared lifespan context `researcher`.

**ObsidianTools** [src/mcps/tools/obsidian_vault.py:208-244](../src/mcps/tools/obsidian_vault.py#L208-L244) - Registered only when `config.vault_dir` is set:
- `obsidian_list_files` - List files and subfolders in a vault folder
- `obsidian_get_content` - Retrieve note content by wikilink name with optional offset/limit
- `obsidian_rename_move` - Rename/move notes and rewrite outgoing wikilinks
- `obsidian_search` - Semantic search in vault with tag/path filtering

### Registered Resources

Resource handlers are currently disabled in `register()` [src/mcps/server.py:97-107](../src/mcps/server.py#L97-L107). The placeholder modules live under `src/mcps/resources/`.

### Prompt Templates

Prompt loading is commented out in `register()` [src/mcps/server.py:123](../src/mcps/server.py#L123). [src/mcps/prompts/file_prompts.py](../src/mcps/prompts/file_prompts.py) contains file-based prompt template utilities.

## Key Design Decisions

### Interface Segregation
All components implement specific interfaces enabling:
- Flexible component replacement without code changes
- Provider-neutral embedding and chat-model adapters
- RRF plus optional async LangChain reranking
- Multiple chunking strategies (FixedSize, Semantic)

### Async Context Managers
The Obsidian vault lifespan uses async context managers for HTTP client and vault lifecycle management [src/mcps/tools/obsidian_vault.py:181-203](../src/mcps/tools/obsidian_vault.py#L181-L203).

### Factory Pattern
Component creation isolated in factory functions:
- `create_vault()` [src/mcps/rag/vault.py:570-629](../src/mcps/rag/vault.py#L570-L629)
- `create_server()` [src/mcps/server.py:134-146](../src/mcps/server.py#L134-L146)
- `create_researcher()` [src/mcps/research/agent.py](../src/mcps/research/agent.py)

### Thread-Safe Operations
Vault operations protected with `asyncio.Lock` [src/mcps/rag/vault.py:249](../src/mcps/rag/vault.py#L249) for concurrent access safety.

### Batch Processing
File processing and embedding generation use configurable batch sizes [src/mcps/rag/vault.py:329-375](../src/mcps/rag/vault.py#L329-L375) for performance optimization.
