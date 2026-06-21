# Configuration and Environment

Environment variables, configuration options, and server settings for the MCPS Model Context Protocol server. #config #environment #python

## Configuration Class

**ServerConfig** [src/mcps/config.py:11-37](../src/mcps/config.py#L11-L37) - Dataclass containing all server configuration.

## Environment Variables

### Router Configuration

#### `LITELLM_ROUTER` (required for AI tools) #env
LiteLLM Router base URL used by web research and Obsidian RAG model adapters.
**Format**: `http://host:port`
**Used by**: [src/mcps/tools/research/config.py](../src/mcps/tools/research/config.py), [src/mcps/tools/obsidian_models.py](../src/mcps/tools/obsidian_models.py)

#### `LITELLM_ROUTER_KEY` (required for AI tools) #env
Authentication token for the LiteLLM Router.
**Format**: Router token string
**Used by**: [src/mcps/tools/research/config.py](../src/mcps/tools/research/config.py), [src/mcps/tools/obsidian_models.py](../src/mcps/tools/obsidian_models.py)

### API Keys

#### `OPENAI_API_KEY` (optional) #env
OpenAI API key retained in server configuration for non-RAG integrations.
RAG model access goes through `LITELLM_ROUTER`.

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
**Used by**: [src/mcps/config.py:58-62](../src/mcps/config.py#L58-L62)

## Configuration Options

### Directory Paths

#### `prompts_dir` #config
Directory containing prompt template markdown files.
**Type**: Path
**Default**: `Path(__file__).parent / "prompts"`
**Used by**: [src/mcps/prompts/file_prompts.py](../src/mcps/prompts/file_prompts.py)

#### `cache_dir` #config
Cache directory for temporary files.
**Type**: Path
**Default**: `Path(__file__).parent / "cache"`

#### `tests_dir` #config
Test directory location.
**Type**: Path
**Default**: `Path(__file__).parent / "tests"`

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
**Default**: [src/mcps/rag/document_processing.py:75-82](../src/mcps/rag/document_processing.py#L75-L82)
```python
[
    r'^\..*',           # Hidden files
    r'node_modules/',
    r'__pycache__/',
    r'^scripts/',
    r'^templates/',
    r'^prompts/',
]
```

#### `batch_size` #config
Number of files to process in a single batch.
**Type**: int
**Default**: `8`
**Used by**: [src/mcps/rag/vault.py:181](../src/mcps/rag/vault.py#L181)

#### `max_chunk_size` #config
Maximum size of text chunks in characters.
**Type**: int
**Default**: `4000`
**Used by**: [src/mcps/rag/document_processing.py](../src/mcps/rag/document_processing.py)

### Model Configuration

#### `rag_embedding_model` #config
LiteLLM Router model name for Obsidian RAG embeddings.
**Type**: str
**Default**: `"text-embedding-3-small"`
**Environment**: `RAG_EMBEDDING_MODEL`
**Used by**: [src/mcps/tools/obsidian_models.py](../src/mcps/tools/obsidian_models.py)

#### `rag_embedding_dimensions` #config
Embedding vector dimension used for LanceDB schema.
**Type**: int
**Default**: `1536`
**Environment**: `RAG_EMBEDDING_DIMENSIONS`
**Used by**: [src/mcps/rag/embeddings.py](../src/mcps/rag/embeddings.py), [src/mcps/rag/database.py](../src/mcps/rag/database.py)

#### `rag_reranker_model` #config
LiteLLM Router model name for DB-level LanceDB reranking. Empty uses the local LanceDB reranking path.
**Type**: str
**Default**: `""`
**Environment**: `RAG_RERANKER_MODEL`
**Used by**: [src/mcps/rag/proxy_reranker.py](../src/mcps/rag/proxy_reranker.py), [src/mcps/rag/vault.py](../src/mcps/rag/vault.py)

#### `rag_infer_model` #config
LiteLLM Router chat model name for search-level HyDE generation and one-call structured reranking. Empty disables search-level LLM inference and preserves vector-only search behavior.
**Type**: str
**Default**: `""`
**Environment**: `RAG_INFER_MODEL`
**Used by**: [src/mcps/rag/search.py](../src/mcps/rag/search.py), [src/mcps/rag/reranking.py](../src/mcps/rag/reranking.py), [src/mcps/rag/vault.py](../src/mcps/rag/vault.py)

### Search Configuration

#### `search_limit` #config
Default maximum number of search results.
**Type**: int
**Default**: `20`
**Used by**: [src/mcps/rag/search.py](../src/mcps/rag/search.py)

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

**pytest Configuration** [pyproject.toml:45-54](../pyproject.toml#L45-L54):

#### `log_cli` #config
Enable CLI logging during tests.
**Value**: `true`

#### `log_cli_level` #config
CLI log level for tests.
**Value**: `"INFO"`

#### `addopts` #config
Additional pytest options.
**Value**: `["--import-mode=importlib"]`

#### `testpaths` #config
Test directory location.
**Value**: `["test"]`

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

```bash
# LiteLLM Router
LITELLM_ROUTER=http://localhost:4000
LITELLM_ROUTER_KEY=sk-router-token

# Optional API Keys
ANTHROPIC_API_KEY=sk-ant-...

# RAG Models
RAG_EMBEDDING_MODEL=text-embedding-3-small
RAG_EMBEDDING_DIMENSIONS=1536
RAG_RERANKER_MODEL=gpt-4o-mini

# Vault Configuration
VAULT=/Users/username/Documents/ObsidianVault
```

## Configuration Validation

Configuration validation performed in [src/mcps/config.py:38-75](../src/mcps/config.py#L38-L75):
- Directory existence checks
- Environment variable presence validation
- Path resolution and normalization
- Default value assignment
