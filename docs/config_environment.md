# Configuration and Environment

Environment variables, configuration options, and server settings for the MCPS Model Context Protocol server. #config #environment #python

## Configuration Class

**ServerConfig** [src/mcps/config.py:11-36](../src/mcps/config.py#L11-L36) - Dataclass containing all server configuration.

## Environment Variables

### API Keys

#### `OPENAI_API_KEY` (required) #env
OpenAI API key for embedding service.
**Format**: `sk-...` (OpenAI API key format)
**Used by**: [src/mcps/rag/embeddings.py](../src/mcps/rag/embeddings.py)

#### `VOYAGE_API_KEY` (required) #env
VoyageAI API key for embedding and reranking services.
**Format**: VoyageAI API key string
**Used by**: [src/mcps/rag/vault.py:52-58](../src/mcps/rag/vault.py#L52-L58)

#### `OLLAMA_API_BASE` (optional) #env
Ollama API base URL for local embedding and reranking services.
**Format**: `http://host:port` URL format
**Default**: Ollama default endpoint
**Example**: `http://localhost:11434`
**Used by**: [src/mcps/rag/vault.py:65-70](../src/mcps/rag/vault.py#L65-L70)

#### `ANTHROPIC_API_KEY` (optional) #env
Anthropic API key for Claude models.
**Format**: Anthropic API key string
**Used by**: Server configuration

#### `PERPLEXITY_API_KEY` (optional) #env
Perplexity AI API key for deep research functionality.
**Format**: Perplexity API key string
**Used by**: [src/mcps/tools/deep_research.py](../src/mcps/tools/deep_research.py)

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

#### `ollama_embedding_model` #config
Ollama model name for embeddings.
**Type**: str
**Default**: `"bge-m3:latest"`
**Used by**: [src/mcps/rag/vault.py:65-70](../src/mcps/rag/vault.py#L65-L70)

#### `voyage_embedding_model` #config
VoyageAI model name for embeddings.
**Type**: str
**Default**: `"voyage-3-lite"`
**Used by**: [src/mcps/rag/vault.py:52-58](../src/mcps/rag/vault.py#L52-L58)

#### `ollama_reranker_model` #config
Ollama model name for reranking.
**Type**: str
**Default**: `"phi4-mini:latest"`
**Used by**: [src/mcps/rag/ollama_reranker.py](../src/mcps/rag/ollama_reranker.py)

#### `voyage_reranker_model` #config
VoyageAI model name for reranking.
**Type**: str
**Default**: `"rerank-2-lite"`
**Used by**: [src/mcps/rag/vault.py:93-94](../src/mcps/rag/vault.py#L93-L94)

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
# Required API Keys
OPENAI_API_KEY=sk-...
VOYAGE_API_KEY=pa-...

# Optional API Keys
ANTHROPIC_API_KEY=sk-ant-...
PERPLEXITY_API_KEY=pplx-...

# Ollama Configuration
OLLAMA_API_BASE=http://localhost:11434

# Vault Configuration
VAULT=/Users/username/Documents/ObsidianVault
```

## Configuration Validation

Configuration validation performed in [src/mcps/config.py:38-75](../src/mcps/config.py#L38-L75):
- Directory existence checks
- Environment variable presence validation
- Path resolution and normalization
- Default value assignment
