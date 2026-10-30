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
- **rank-bm25** - BM25L keyword ranking over source-mapped fetch-filter windows [src/mcps/research/tools/filtering.py](../src/mcps/research/tools/filtering.py)

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

The architecture follows a **Dependency Injection** pattern with clear interface definitions in [src/mcps/rag/interfaces.py:184-436](../src/mcps/rag/interfaces.py#L184-L436). All RAG components implement abstract interfaces enabling flexible implementation swapping.

#### Core Interfaces

**IDocumentProcessor** [src/mcps/rag/interfaces.py:184-190](../src/mcps/rag/interfaces.py#L184-L190)
- Document file processing and metadata extraction

**IChunker** [src/mcps/rag/interfaces.py:194-200](../src/mcps/rag/interfaces.py#L194-L200)
- Text chunking strategies (FixedSizeChunker, SemanticChunker)

**IEmbeddingService** [src/mcps/rag/interfaces.py:203-231](../src/mcps/rag/interfaces.py#L203-L231)
- Embedding generation from multiple providers

**IDocumentSummaryGenerator** [src/mcps/rag/interfaces.py:233-239](../src/mcps/rag/interfaces.py#L233-L239)
- Whole-document summary generation

**IVectorStore** [src/mcps/rag/interfaces.py:241-331](../src/mcps/rag/interfaces.py#L241-L331)
- Vector storage and hybrid search operations

**ISearchEngine** [src/mcps/rag/interfaces.py:333-339](../src/mcps/rag/interfaces.py#L333-L339)
- Search execution and result retrieval

**IFileTraversal** [src/mcps/rag/interfaces.py:342-348](../src/mcps/rag/interfaces.py#L342-L348)
- File system discovery

**IVault** [src/mcps/rag/interfaces.py:351-436](../src/mcps/rag/interfaces.py#L351-L436)
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

**Index Update Flow** [src/mcps/rag/vault.py:293-381](../src/mcps/rag/vault.py#L293-L381). The [IndexScheduler](../src/mcps/rag/index_scheduler.py) triggers it at startup, every 5 minutes, and before index-backed tool calls:
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
- Default: max_chunk_size=2000, min_chunk_size=1000 (250-500 tokens)

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

**web_research** [src/mcps/server.py:154-166](../src/mcps/server.py#L154-L166) - Registered, but enabled by the research lifespan only when a CDP browser is available (`BROWSER_CDP_URL` set and reachable); otherwise it is disabled while other tools stay available. Runs the LangGraph deep-research agent as a background job (hybrid polling): the call starts a job and waits up to `research_wait_seconds` (default 25 s, under the 30 s MCP client timeout), forwarding progress. If the job finishes in time the call returns `status="completed"` with the result; otherwise it returns `status="running"`, a job name of three random words (for example `amber-falcon-river`) and instructions to call `web_research_status`. If the client abandons the first call before it returns a job name, the job is cancelled.

**web_research_status** [src/mcps/server.py](../src/mcps/server.py) - Takes the job name, waits up to `research_wait_seconds` for the job, and returns the same job record: `status`, `result` or `error`, and the last progress message while the job is running. An unknown or expired name is a tool error. It is enabled and disabled together with `web_research`.

Jobs are handled by `ResearchJobs` in [src/mcps/research/jobs.py](../src/mcps/research/jobs.py) and stored behind the async `JobStore` protocol. The default `InMemoryJobStore` keeps only the last progress message per job and drops finished jobs `research_job_ttl_seconds` after their last update (checked on each create and get; there is no cleanup task). Waiting reads only the stored record, so a persistent store can replace the in-memory one without changing the tools. A persistent store must mark `running` jobs as `failed` at startup, because their tasks do not survive a restart. If a client cancels right after a call returned, the job keeps running until it ends on its own. Running jobs are cancelled when the server shuts down, before the browser closes.

**ObsidianTools** [src/mcps/tools/obsidian_vault.py:208-244](../src/mcps/tools/obsidian_vault.py#L208-L244) - Registered only when `config.vault_dir` is set:
- `obsidian_list_files` - List files and subfolders in a vault folder
- `obsidian_read_note` - Retrieve note content by full path, partial path, or name with optional line-based offset/limit
- `obsidian_rename_move` - Rename/move notes and rewrite outgoing wikilinks
- `obsidian_search` - Semantic search in vault with tag/path filtering

**MemoryTools** [src/mcps/tools/vault_memory.py](../src/mcps/tools/vault_memory.py) - Registered together with ObsidianTools. They store agent memories as notes in the vault folder named by `memory_dir`. The existing search and read tools retrieve them.
- `add_memory` - Create a note under `General/<topic>/` or `Projects/<project>/`. It fails if the note exists.
- `delete_memory` - Permanently delete one note in the memory folder. It is marked as destructive.

**ExposedSkillsProvider** [src/mcps/tools/vault_skills.py](../src/mcps/tools/vault_skills.py) - Added when `config.vault_dir` is set. It serves each `Claude/skills/<name>/SKILL.md` in the vault whose frontmatter has `exposed: true` as the MCP resource `skill://<name>/SKILL.md`. It scans the folder again on every request.

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
Lifespans use async context managers for connection lifecycle management: the Obsidian lifespan owns the HTTP client, the vault, and the index scheduler [src/mcps/tools/obsidian_vault.py:191-230](../src/mcps/tools/obsidian_vault.py#L191-L230), and the research lifespan owns the HTTP client and the whole browser stack through nested context managers (`build_research_config` → `build_fetch_tool` → `create_browser_fetch`) so fetches reuse one CDP connection [src/mcps/research/lifespan.py:14](../src/mcps/research/lifespan.py#L14).

### Factory Pattern
Component creation isolated in factory functions:
- `create_vault()` [src/mcps/rag/vault.py:570-629](../src/mcps/rag/vault.py#L570-L629)
- `create_server()` [src/mcps/server.py:134-146](../src/mcps/server.py#L134-L146)
- `create_researcher()` [src/mcps/research/agent.py](../src/mcps/research/agent.py)

### Thread-Safe Operations
Vault operations protected with `asyncio.Lock` [src/mcps/rag/vault.py:249](../src/mcps/rag/vault.py#L249) for concurrent access safety.

### Batch Processing
File processing and embedding generation use configurable batch sizes [src/mcps/rag/vault.py:329-375](../src/mcps/rag/vault.py#L329-L375) for performance optimization.
