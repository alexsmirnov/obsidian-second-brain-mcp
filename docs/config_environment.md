# Configuration and Environment

Environment variables, configuration options, and server settings for the MCPS Model Context Protocol server. #config #environment #python

## Configuration Class

**ServerConfig** [src/mcps/config.py:14-50](../src/mcps/config.py#L14-L50) - Dataclass containing all server configuration.

## Environment Variables

### Router Configuration

#### `ROUTER_API_BASE` (required for AI tools) #env
OpenAI-compatible model router base URL used by web research and Obsidian RAG model adapters.
**Format**: `http://host:port`
**Used by**: [src/mcps/config.py:83](../src/mcps/config.py#L83), [src/mcps/research/config.py](../src/mcps/research/config.py), [src/mcps/rag/vault.py](../src/mcps/rag/vault.py)

#### `ROUTER_API_KEY` (required for AI tools) #env
Authentication token for the model router.
**Format**: Router token string
**Used by**: [src/mcps/config.py:84](../src/mcps/config.py#L84), [src/mcps/research/config.py](../src/mcps/research/config.py), [src/mcps/rag/vault.py](../src/mcps/rag/vault.py)

### API Keys

#### `ANTHROPIC_API_KEY` (optional) #env
Anthropic API key for Claude models.
**Format**: Anthropic API key string
**Used by**: Server configuration

### Vault Configuration

#### `VAULT` (optional) #env
Path to Obsidian vault root directory.
**Format**: Absolute or relative file system path
**Example**: `/Users/username/Documents/MyVault`
**Default**: None (vault features disabled)
**Used by**: [src/mcps/config.py:97-101](../src/mcps/config.py#L97-L101)

## Configuration Options

### Directory Paths

#### `prompts_dir` #config
Directory containing prompt template markdown files.
**Type**: Path
**Default**: `Path(__file__).parent / "prompts"` [src/mcps/config.py:16](../src/mcps/config.py#L16)
**Used by**: [src/mcps/prompts/file_prompts.py](../src/mcps/prompts/file_prompts.py)

#### `cache_dir` #config
Cache directory for temporary files.
**Type**: Path
**Default**: `Path(__file__).parent / "cache"` [src/mcps/config.py:17](../src/mcps/config.py#L17)

#### `tests_dir` #config
Test directory location.
**Type**: Path
**Default**: `Path(__file__).parent / "tests"` [src/mcps/config.py:18](../src/mcps/config.py#L18)

### Vector Database Settings

#### `table_name` #config
LanceDB table name for document storage.
**Type**: str
**Default**: `"documents"`
**Used by**: [src/mcps/rag/database.py](../src/mcps/rag/database.py)

### Document Processing Settings

#### `skip_patterns` #config
Regex patterns for files to skip during vault indexing.
**Type**: list[str]
**Default**: [src/mcps/rag/document_processing.py:97-105](../src/mcps/rag/document_processing.py#L97-L105)
```python
[
    r'^\..*',           # Hidden files
    r'node_modules/',
    r'__pycache__/',
    r'/worktrees/',
    r'^scripts/',
    r'^templates/',
    r'^prompts/',
]
```

#### `batch_size` #config
Number of files to process in a single batch.
**Type**: int
**Default**: `8` [src/mcps/config.py:27](../src/mcps/config.py#L27)
**Used by**: [src/mcps/rag/vault.py:329-375](../src/mcps/rag/vault.py#L329-L375)

#### `max_chunk_size` #config
Maximum size of text chunks in characters.
**Type**: int
**Default**: `4000`
**Used by**: [src/mcps/rag/document_processing.py](../src/mcps/rag/document_processing.py)

### Model Configuration

#### `rag_embedding_model` #config
OpenAI-compatible model name for Obsidian RAG embeddings.
**Type**: str
**Default**: `""`
**Environment**: `RAG_EMBEDDING_MODEL`
**Used by**: [src/mcps/rag/embeddings.py](../src/mcps/rag/embeddings.py)

#### `rag_embedding_dimensions` #config
Embedding vector dimension used for LanceDB schema.
**Type**: int
**Default**: `0`
**Environment**: `RAG_EMBEDDING_DIMENSIONS`
**Used by**: [src/mcps/rag/embeddings.py](../src/mcps/rag/embeddings.py), [src/mcps/rag/database.py](../src/mcps/rag/database.py)

#### `rag_reranker_model` #config
OpenAI-compatible model name for DB-level LanceDB reranking. Empty uses the local reranking path.
**Type**: str
**Default**: `""`
**Environment**: `RAG_RERANKER_MODEL`
**Used by**: [src/mcps/rag/proxy_reranker.py](../src/mcps/rag/proxy_reranker.py), [src/mcps/rag/vault.py](../src/mcps/rag/vault.py)

#### `rag_infer_model` #config
OpenAI-compatible chat model name for search-level HyDE generation and one-call structured reranking. Empty disables search-level LLM inference and preserves vector-only search behavior.
**Type**: str
**Default**: `""`
**Environment**: `RAG_INFER_MODEL`
**Used by**: [src/mcps/rag/search.py](../src/mcps/rag/search.py), [src/mcps/rag/reranking.py](../src/mcps/rag/reranking.py), [src/mcps/rag/vault.py](../src/mcps/rag/vault.py)

#### `rag_summary_model` #config
OpenAI-compatible chat model name for whole-document summary generation during indexing. Empty disables summary chunks.
**Type**: str
**Default**: `""`
**Environment**: `RAG_SUMMARY_MODEL`
**Used by**: [src/mcps/rag/summarization.py](../src/mcps/rag/summarization.py), [src/mcps/rag/vault.py:181-195](../src/mcps/rag/vault.py#L181-L195)

### Search Configuration

#### `search_limit` #config
Default maximum number of search results.
**Type**: int
**Default**: `30`
**Used by**: [src/mcps/rag/search.py](../src/mcps/rag/search.py)

#### `rag_reranker_embedding_model` #config
OpenAI-compatible embedding model name used by the local LLM reranker.
**Type**: str
**Default**: `""`
**Environment**: `RAG_RERANKER_EMBEDDING_MODEL`
**Used by**: [src/mcps/rag/llm_reranker.py](../src/mcps/rag/llm_reranker.py)

#### `rag_reranker_embedding_dimensions` #config
Vector dimension for `rag_reranker_embedding_model`.
**Type**: int
**Default**: `0`
**Environment**: `RAG_RERANKER_EMBEDDING_DIMENSIONS`
**Used by**: [src/mcps/rag/llm_reranker.py](../src/mcps/rag/llm_reranker.py)

#### `rag_reranker_infer_model` #config
OpenAI-compatible chat model used by the local LLM reranker when no proxy reranker is configured.
**Type**: str
**Default**: `""`
**Environment**: `RAG_RERANKER_INFER_MODEL`
**Used by**: [src/mcps/rag/llm_reranker.py](../src/mcps/rag/llm_reranker.py)

#### `research_fast_model` #config
Fast/cheap chat model for web research query generation and result cleanup.
**Type**: str
**Default**: `""`
**Environment**: `RESEARCH_FAST_MODEL`
**Used by**: [src/mcps/config.py:95](../src/mcps/config.py#L95), [src/mcps/research/config.py](../src/mcps/research/config.py)

#### `research_infer_model` #config
Stronger chat model for web research reflection and final answer synthesis.
**Type**: str
**Default**: `""`
**Environment**: `RESEARCH_INFER_MODEL`
**Used by**: [src/mcps/config.py:96](../src/mcps/config.py#L96), [src/mcps/research/config.py](../src/mcps/research/config.py)

#### `google_api_key` #config
Google Custom Search API key. Required only when using Google instead of DuckDuckGo for web research.
**Type**: str
**Default**: `""`
**Environment**: `GOOGLE_API_KEY`
**Used by**: [src/mcps/config.py:103](../src/mcps/config.py#L103), [src/mcps/research/tools.py](../src/mcps/research/tools.py)

#### `google_search_id` #config
Google Custom Search Engine ID. Required only when `GOOGLE_API_KEY` is set.
**Type**: str
**Default**: `""`
**Environment**: `GOOGLE_SEARCH_ID`
**Used by**: [src/mcps/config.py:104](../src/mcps/config.py#L104), [src/mcps/research/tools.py](../src/mcps/research/tools.py)

## Environment File Loading

**Loading Order** [src/mcps/config.py:49-56](../src/mcps/config.py#L49-L56):

1. Project root `.env` file (highest priority)
2. User home directory `.env` file

Environment variables are loaded from the first `.env` file found in this order.

## Configuration Factory

**create_config()** [src/mcps/config.py:38-75](../src/mcps/config.py#L38-L75):
Creates ServerConfig instance with:
- Environment variable loading
- Directory creation and validation
- Default value handling
- API key extraction from environment

## Logging Configuration

### Log Files

#### Log Directory #config
**Path**: `~/Library/Logs/Mcps/` [src/mcps/logs.py:10](../src/mcps/logs.py#L10)
**File**: `mcps.log`

#### Log Rotation #config
**Max Size**: 10 MB per file
**Backup Count**: 5 files
**Implementation**: [src/mcps/logs.py:15-17](../src/mcps/logs.py#L15-L17)

### Log Levels

#### Default Log Level #config
**Level**: INFO
**Configured in**: [src/mcps/logs.py](../src/mcps/logs.py)

#### HTTP Client Logging #config
**httpx logging**: WARNING level (reduced noise)
**Configured in**: [src/mcps/logs.py:44](../src/mcps/logs.py#L44)

## Test Configuration

**pytest Configuration** [pyproject.toml:54-63](../pyproject.toml#L45-L54):

#### `log_cli` #config
Enable CLI logging during tests.
**Value**: `true`

#### `log_cli_level` #config
CLI log level for tests.
**Value**: `"INFO"`

#### `addopts` #config
Additional pytest options.
**Value**: `["--import-mode=importlib", "--durations=0"]`

#### `testpaths` #config
Test directory location.
**Value**: `["tests"]`

#### `pythonpath` #config
Python path for test imports.
**Value**: `["src"]`

#### `asyncio_mode` #config
Asyncio test mode.
**Value**: `"auto"`

#### `asyncio_default_fixture_loop_scope` #config
Asyncio fixture loop scope.
**Value**: `"function"`

## Example .env File

See [`env.example`](../env.example) for a complete, commented environment file. A minimal version is shown below:

```bash
# OpenAI-compatible model router
ROUTER_API_BASE=http://localhost:4000
ROUTER_API_KEY=sk-router-token

# RAG Models
RAG_EMBEDDING_MODEL=text-embedding-3-small
RAG_EMBEDDING_DIMENSIONS=1536

# Vault Configuration
VAULT=/Users/username/Documents/ObsidianVault
```

## Configuration Validation

Configuration validation is performed in [src/mcps/config.py:110-132](../src/mcps/config.py#L110-L132) by `validate_config()`:
- Missing `ROUTER_API_BASE` produces warnings when research or RAG embedding models are configured.
- Missing `RAG_EMBEDDING_DIMENSIONS` produces a warning when `RAG_EMBEDDING_MODEL` is set.
- Validation is non-fatal so the server can start with a subset of features enabled.
