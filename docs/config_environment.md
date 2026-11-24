# Configuration and Environment

Environment variables, configuration options, and server settings for the MCPS Model Context Protocol server. #config #environment #python

## Configuration Class

**ServerConfig** [src/mcps/config.py:291](../src/mcps/config.py#L291) is a frozen pydantic model. It holds one settings group for each feature. Each group is a pydantic-settings `BaseSettings` class that reads its own environment variables and validates them. Code reads a value as `config.<group>.<field>`, for example `config.router.api_base`.

| Attribute | Class | Environment variables |
|-----------|-------|-----------------------|
| `router` | `RouterSettings` [src/mcps/config.py:70](../src/mcps/config.py#L70) | `ROUTER_*` |
| `rag` | `RagSettings` [src/mcps/config.py:102](../src/mcps/config.py#L102) | `RAG_*` |
| `vault` | `VaultSettings` [src/mcps/config.py:132](../src/mcps/config.py#L132) | `VAULT`, `MEMORY_DIR`, `VAULT_TABLE_NAME` |
| `chunking` | `ChunkingSettings` [src/mcps/config.py:143](../src/mcps/config.py#L143) | `MIN_CHUNK_SIZE`, `MAX_CHUNK_SIZE` |
| `search` | `SearchSettings` [src/mcps/config.py:159](../src/mcps/config.py#L159) | `SEARCH_LIMIT`, `MIN_SCORE` |
| `google` | `GoogleSettings` [src/mcps/config.py:172](../src/mcps/config.py#L172) | `GOOGLE_*` |
| `browser` | `BrowserSettings` [src/mcps/config.py:181](../src/mcps/config.py#L181) | `BROWSER_*` |
| `scraper` | `ScraperSettings` [src/mcps/config.py:203](../src/mcps/config.py#L203) | `SCRAPER_PROVIDER`, `SCRAPE_DO_TOKEN`, `BRIGHT_DATA_*` |
| `fetch` | `FetchSettings` [src/mcps/config.py:227](../src/mcps/config.py#L227) | `DECISION_MODEL`, `FETCH_RESTRICTED_DOMAINS` |
| `research` | `ResearchSettings` [src/mcps/config.py:260](../src/mcps/config.py#L260) | `RESEARCH_FAST_MODEL`, `RESEARCH_INFER_MODEL`, `RESEARCH_WAIT_SECONDS`, `RESEARCH_JOB_TTL_SECONDS` |
| `research_limits` | `ResearchLimits` [src/mcps/config.py:245](../src/mcps/config.py#L245) | `RESEARCH_ROUND_START_BUDGET`, `RESEARCH_MAX_*` |

An empty environment variable selects the default value. The server reads only the variable names that this page lists, so a variable such as `PROVIDER` or `VAULT_DIR` has no effect. A group or `ServerConfig` rejects an unknown argument, so a misspelled field name fails at once. The list `ENV_GROUPS` [src/mcps/config.py:276](../src/mcps/config.py#L276) names every group class.

## Environment Variables

### Router Configuration

#### `ROUTER_API_BASE` (required for AI tools on the host) #env
OpenAI-compatible model router base URL used by web research and Obsidian RAG model adapters when the server runs outside a container.
**Attribute**: `router.api_base`
**Format**: `http://host:port`. It must be an http or https URL with a host and no user name, password, query, or fragment. Any other value stops startup.
**Used by**: [src/mcps/research/config.py](../src/mcps/research/config.py), [src/mcps/rag/vault.py](../src/mcps/rag/vault.py), [src/mcps/research/tools/fetch.py](../src/mcps/research/tools/fetch.py)

#### `ROUTER_DOCKER_API_BASE` (optional) #env
OpenAI-compatible model router base URL used when the server runs in a container. If `/.dockerenv` exists and this variable is set, its value replaces `ROUTER_API_BASE`. In a container without this variable, the server uses `ROUTER_API_BASE`.
**Attribute**: `router.docker_api_base`
**Format**: same as `ROUTER_API_BASE`
**Used by**: [src/mcps/config.py:70](../src/mcps/config.py#L70)

#### `ROUTER_API_KEY` (required for AI tools) #env
Authentication token for the model router.
**Attribute**: `router.api_key`
**Format**: Router token string
**Used by**: [src/mcps/research/config.py](../src/mcps/research/config.py), [src/mcps/rag/vault.py](../src/mcps/rag/vault.py)

### API Keys

#### `ANTHROPIC_API_KEY` (optional) #env
Anthropic API key for Claude models. The server configuration does not read it.
**Format**: Anthropic API key string

### Vault Configuration

#### `VAULT` (optional) #env
Path to the Obsidian vault root directory. The `--vault` command line option wins over this variable.
**Attribute**: `vault.dir`
**Format**: Absolute or relative file system path
**Example**: `/Users/username/Documents/MyVault`
**Default**: None (vault features disabled)
**Used by**: [src/mcps/server.py](../src/mcps/server.py), [src/mcps/rag/vault.py](../src/mcps/rag/vault.py), [src/mcps/tools/obsidian_vault.py](../src/mcps/tools/obsidian_vault.py)

#### `MEMORY_DIR` (optional) #env
Vault folder where `add_memory` writes agent memories and where `delete_memory` can remove them.
**Attribute**: `vault.memory_dir`
**Format**: Relative folder path inside the vault, with `/` separators
**Example**: `Notes/Memory`
**Default**: `Memory`
**Used by**: [src/mcps/tools/vault_memory.py](../src/mcps/tools/vault_memory.py)

#### `VAULT_TABLE_NAME` (optional) #env
LanceDB table name for document storage.
**Attribute**: `vault.table_name`
**Default**: `documents`
**Used by**: [src/mcps/rag/vault.py](../src/mcps/rag/vault.py)

## Configuration Options

### Document Processing Settings

#### `MIN_CHUNK_SIZE` #env
Minimum size of text sections in characters. The chunker merges smaller sections.
**Attribute**: `chunking.min_size`
**Type**: int, greater than 0
**Default**: `500`
**Validation**: it must not be larger than `MAX_CHUNK_SIZE`.
**Used by**: [src/mcps/rag/vault.py](../src/mcps/rag/vault.py), [src/mcps/rag/document_processing.py](../src/mcps/rag/document_processing.py)

#### `MAX_CHUNK_SIZE` #env
Maximum size of text chunks in characters.
**Attribute**: `chunking.max_size`
**Type**: int, greater than 0
**Default**: `1000`
**Used by**: [src/mcps/rag/vault.py](../src/mcps/rag/vault.py), [src/mcps/rag/document_processing.py](../src/mcps/rag/document_processing.py)

The indexer skips files that match `default_skip_patterns` [src/mcps/rag/document_processing.py:132](../src/mcps/rag/document_processing.py#L132). The patterns are code, not configuration.

### Model Configuration

All model names are empty by default. An empty model name turns its feature off.

#### `RAG_EMBEDDING_MODEL` #env
OpenAI-compatible model name for Obsidian RAG embeddings.
**Attribute**: `rag.embedding_model`
**Default**: `""`
**Validation**: if it is set, `RAG_EMBEDDING_DIMENSIONS` must be 1 or more.
**Used by**: [src/mcps/rag/vault.py](../src/mcps/rag/vault.py)

#### `RAG_EMBEDDING_DIMENSIONS` #env
Embedding vector dimension used for the LanceDB schema. It must match the model.
**Attribute**: `rag.embedding_dimensions`
**Type**: int, 0 or more
**Default**: `0`
**Used by**: [src/mcps/rag/vault.py](../src/mcps/rag/vault.py), [src/mcps/rag/database.py](../src/mcps/rag/database.py)

#### `RAG_RERANKER_MODEL` #env
OpenAI-compatible model name for DB-level LanceDB reranking. Empty uses the local reranking path.
**Attribute**: `rag.reranker_model`
**Default**: `""`
**Used by**: [src/mcps/rag/proxy_reranker.py](../src/mcps/rag/proxy_reranker.py), [src/mcps/rag/vault.py](../src/mcps/rag/vault.py)

#### `RAG_RERANKER_EMBEDDING_MODEL` #env
OpenAI-compatible embedding model name used by the local LLM reranker.
**Attribute**: `rag.reranker_embedding_model`
**Default**: `""`
**Used by**: [src/mcps/rag/llm_reranker.py](../src/mcps/rag/llm_reranker.py)

#### `RAG_RERANKER_EMBEDDING_DIMENSIONS` #env
Vector dimension for `RAG_RERANKER_EMBEDDING_MODEL`.
**Attribute**: `rag.reranker_embedding_dimensions`
**Type**: int, 0 or more
**Default**: `0`
**Used by**: [src/mcps/rag/llm_reranker.py](../src/mcps/rag/llm_reranker.py)

#### `RAG_RERANKER_INFER_MODEL` #env
OpenAI-compatible chat model used by the local LLM reranker when no proxy reranker is configured.
**Attribute**: `rag.reranker_infer_model`
**Default**: `""`
**Used by**: [src/mcps/rag/llm_reranker.py](../src/mcps/rag/llm_reranker.py)

#### `RAG_INFER_MODEL` #env
OpenAI-compatible chat model name for search-level HyDE generation and one-call structured reranking. Empty disables search-level LLM inference and keeps vector-only search.
**Attribute**: `rag.infer_model`
**Default**: `""`
**Used by**: [src/mcps/rag/search.py](../src/mcps/rag/search.py), [src/mcps/rag/reranking.py](../src/mcps/rag/reranking.py), [src/mcps/rag/vault.py](../src/mcps/rag/vault.py)

#### `RAG_SUMMARY_MODEL` #env
OpenAI-compatible chat model name for whole-document summary generation during indexing. Empty disables summary chunks.
**Attribute**: `rag.summary_model`
**Default**: `""`
**Used by**: [src/mcps/rag/summarization.py](../src/mcps/rag/summarization.py), [src/mcps/rag/vault.py](../src/mcps/rag/vault.py)

#### `RESEARCH_FAST_MODEL` #env
Fast and cheap chat model for web research query generation and result cleanup.
**Attribute**: `research.fast_model`
**Default**: `""`
**Used by**: [src/mcps/research/config.py](../src/mcps/research/config.py)

#### `RESEARCH_INFER_MODEL` #env
Stronger chat model for web research reflection and final answer synthesis.
**Attribute**: `research.infer_model`
**Default**: `""`
**Used by**: [src/mcps/research/config.py](../src/mcps/research/config.py)

#### `RESEARCH_EVAL_MODEL` (evaluation only) #env
LLM-judge model for the DRACO web research evaluation. The evaluation code reads it directly from the environment. It is not part of `ServerConfig`.
**Type**: str
**Default**: falls back to `RESEARCH_INFER_MODEL`
**Used by**: [tests/evaluation/judge.py](../tests/evaluation/judge.py)

### Search Configuration

#### `SEARCH_LIMIT` (optional) #env
Default maximum number of search results.
**Attribute**: `search.limit`
**Type**: int, 1 or more
**Default**: `30`
**Used by**: [src/mcps/rag/vault.py](../src/mcps/rag/vault.py), [src/mcps/rag/search.py](../src/mcps/rag/search.py)

#### `MIN_SCORE` #env
Minimum reranker relevance score for a search result to be returned. RRF scores are about 0.016, and LLM or proxy rerankers score from 0 to 1. The default of 0.0 keeps all results.
**Attribute**: `search.min_score`
**Type**: float, from 0 to 1
**Default**: `0.0`
**Used by**: [src/mcps/rag/vault.py](../src/mcps/rag/vault.py)

#### `GOOGLE_API_KEY` #env
Google Custom Search API key. Web research needs it only to use Google instead of the free Yahoo fallback.
**Attribute**: `google.api_key`
**Default**: `""`
**Used by**: [src/mcps/research/config.py](../src/mcps/research/config.py), [src/mcps/research/tools/google.py](../src/mcps/research/tools/google.py)

#### `GOOGLE_SEARCH_ID` #env
Google Custom Search Engine ID. Web research needs it when `GOOGLE_API_KEY` is set.
**Attribute**: `google.search_id`
**Default**: `""`
**Used by**: [src/mcps/research/config.py](../src/mcps/research/config.py), [src/mcps/research/tools/google.py](../src/mcps/research/tools/google.py)

### Web Fetch

`fetch(url, query)` returns a `FetchResult` (see [Packages & Modules](packages_modules.md)). Routing: restricted domains return status `RESTRICTED` without I/O. GitHub and arXiv use specialized fetchers (no fallback). URLs with a registered binary extension (`.pdf` today) use the httpx extractor. All other URLs render in the CDP browser via crawl4ai. If the browser reports a registered binary type, such as a PDF at a URL without `.pdf`, the fetch retries the URL once with the httpx extractor. The formats are listed in `extractors.py`, and a new format needs one new entry there. The retry does not share browser cookies. Sources reach the filter natively: HTML is rendered once to Markdown, Markdown keeps its source structure, and other textual types (including raw PDF text) stay plain. A nonblank query selects source passages and resolves web links absolutely. It uses local BM25L by default, or per-fragment router decision scores when `DECISION_MODEL` is set. Selection units are windows that target 300 characters with no overlap for BM25L, and 5,000 characters with a 1,500-character overlap for decision scores. Each cut moves up to a tenth of the window size to the nearest H1-H6 heading or the line after a blank line. BM25L blends each window score with its neighbors (weight 0.2) and keeps matching windows that score at least 0.075 of the best window on the page. A window holds the exact source slice, so blank lines stay in place. Local selection ranks matching windows by score and keeps the best ones that fit the 15,000-char output budget, in source order. When matching windows are dropped, the result ends with `[Content truncated]`. A blank query keeps the prepared whole document without any model call. The result is truncated to 15,000 chars at a safe Markdown/word boundary. When the browser or PDF result is 401/403/429, empty, timeout, or browser-unavailable, the configured `SCRAPER_PROVIDER` is used. If the provider is unavailable, the original result is kept.

#### `BROWSER_CDP_URL` #env
CDP endpoint of a running browser (for example `ws://127.0.0.1:9222`). It must be unauthenticated. The browser runs outside the server (Obscura service in docker compose). If the variable is unset or the browser is unreachable at startup, only the `web_research` tool is disabled.
**Attribute**: `browser.cdp_url`
**Default**: `""`
**Used by**: [src/mcps/research/tools/browser.py](../src/mcps/research/tools/browser.py)

#### `BROWSER_POOL_SIZE` #env
Number of named browser sessions. Each session keeps one tab that later fetches reuse. The pool size also limits how many pages render at the same time. It replaces `FETCH_CONCURRENCY`.
**Attribute**: `browser.pool_size`
**Type**: int, 1 or more
**Default**: `8`
**Used by**: [src/mcps/research/tools/browser.py](../src/mcps/research/tools/browser.py)

#### `BROWSER_RECONNECT_ATTEMPTS` #env
Number of times the server tries to start a new browser connection after it loses the current one. All fetches that failed on the lost connection share one round of attempts.
**Attribute**: `browser.reconnect_attempts`
**Type**: int, 1 or more
**Default**: `5`
**Used by**: [src/mcps/research/tools/browser.py](../src/mcps/research/tools/browser.py)

#### `BROWSER_RECONNECT_DELAY_SECONDS` #env
Pause before each reconnect attempt, in seconds.
**Attribute**: `browser.reconnect_delay_seconds`
**Type**: float, 0 or more
**Default**: `2.0`
**Used by**: [src/mcps/research/tools/browser.py](../src/mcps/research/tools/browser.py)

#### `BROWSER_CONNECT_TIMEOUT_SECONDS` #env
Time limit for one connection attempt, in seconds.
**Attribute**: `browser.connect_timeout_seconds`
**Type**: float, greater than 0
**Default**: `30.0`
**Used by**: [src/mcps/research/tools/browser.py](../src/mcps/research/tools/browser.py)

#### `BROWSER_FETCH_RETRIES` #env
Number of extra tries for a fetch that lost the browser connection. The server does not retry HTTP errors or other failed pages.
**Attribute**: `browser.fetch_retries`
**Type**: int, 0 or more
**Default**: `1`
**Used by**: [src/mcps/research/tools/browser.py](../src/mcps/research/tools/browser.py)

#### `BROWSER_CLEANUP_TIMEOUT_SECONDS` #env
Time limit for resetting one tab and for closing one connection, in seconds. If the reset takes longer, the server closes the tab.
**Attribute**: `browser.cleanup_timeout_seconds`
**Type**: float, greater than 0
**Default**: `5.0`
**Used by**: [src/mcps/research/tools/browser.py](../src/mcps/research/tools/browser.py)

#### `RESEARCH_WAIT_SECONDS` #env
How long `web_research` and `web_research_status` wait for a research job before they return its current state. It must stay below the MCP client timeout (30 s by default).
**Attribute**: `research.wait_seconds`
**Type**: float, greater than 0
**Default**: `25.0`
**Used by**: [src/mcps/research/lifespan.py](../src/mcps/research/lifespan.py)

#### `RESEARCH_JOB_TTL_SECONDS` #env
How long a finished research job stays readable after its last update.
**Attribute**: `research.job_ttl_seconds`
**Type**: float, greater than 0
**Default**: `3600.0`
**Used by**: [src/mcps/research/lifespan.py](../src/mcps/research/lifespan.py)

#### `SCRAPER_PROVIDER` #env
Commercial unblocking provider used for blocked, empty, timed-out, or browser-unavailable pages.
**Attribute**: `scraper.provider`
**Type**: one of `scrape_do`, `bright_data`, or empty
**Default**: `""` (disabled)
**Validation**: `scrape_do` needs `SCRAPE_DO_TOKEN`. `bright_data` needs `BRIGHT_DATA_API_KEY` and `BRIGHT_DATA_ZONE`. Any other value stops startup.
**Used by**: [src/mcps/research/tools/fetch.py](../src/mcps/research/tools/fetch.py)

#### `SCRAPE_DO_TOKEN` #env
Scrape.do API token. Required when `SCRAPER_PROVIDER=scrape_do`.
**Attribute**: `scraper.scrape_do_token`
**Default**: `""`
**Used by**: [src/mcps/research/tools/scrape_do.py](../src/mcps/research/tools/scrape_do.py)

#### `BRIGHT_DATA_API_KEY`, `BRIGHT_DATA_ZONE` #env
Bright Data API key and Web Unlocker zone name. Both are required when `SCRAPER_PROVIDER=bright_data`.
**Attribute**: `scraper.bright_data_api_key`, `scraper.bright_data_zone`
**Default**: `""`
**Used by**: [src/mcps/research/tools/bright_data.py](../src/mcps/research/tools/bright_data.py)

#### `DECISION_MODEL` #env
Router decision model (TypeSafe System One, for example `jev-latest`) that scores each fragment with one yes/no question: "does the passage help answer the query?". Fragments with a probability of at least 0.6 are kept, in source order. It needs a router URL. Without one, the server logs a startup warning and the fetch uses lexical filtering. Requests go to `POST /v1/systemone` on the router root, with `ROUTER_API_KEY` as the bearer token when set. Any decision failure returns `FILTER_FAILED` and never falls back to lexical output.
**Attribute**: `fetch.decision_model`
**Default**: `""`
**Used by**: [src/mcps/research/tools/fetch.py](../src/mcps/research/tools/fetch.py), [src/mcps/research/tools/decision_ranker.py](../src/mcps/research/tools/decision_ranker.py), [src/mcps/common/decisions/client.py](../src/mcps/common/decisions/client.py)

#### `FETCH_RESTRICTED_DOMAINS` #env
Comma-separated hostnames. The server never fetches the hostname itself or its subdomains (status `RESTRICTED`). The server trims each entry and converts it to lowercase.
**Attribute**: `fetch.restricted_domains`
**Type**: tuple[str, ...]
**Default**: `()`
**Used by**: [src/mcps/research/tools/fetch.py](../src/mcps/research/tools/fetch.py)

### Deep Research Limits

The deep research agent takes its bounds from `config.research_limits`. The lifespan passes them to `create_researcher` [src/mcps/research/lifespan.py](../src/mcps/research/lifespan.py), and the agent keeps them as `ResearchAgent.limits` [src/mcps/research/deep_research.py](../src/mcps/research/deep_research.py).

#### `RESEARCH_ROUND_START_BUDGET` #env
Seconds since the start of a run. No new research round begins after this time.
**Attribute**: `research_limits.round_start_budget`
**Type**: float, greater than 0
**Default**: `180.0`

#### `RESEARCH_MAX_PAGES_PER_ROUND` #env
Maximum number of pages the agent fetches in one round.
**Attribute**: `research_limits.max_pages_per_round`
**Type**: int, 1 or more
**Default**: `10`

#### `RESEARCH_MAX_INITIAL_QUERIES` #env
Maximum number of search queries in the first round.
**Attribute**: `research_limits.max_initial_queries`
**Type**: int, 1 or more
**Default**: `3`

#### `RESEARCH_MAX_FOLLOW_UPS` #env
Maximum number of follow-up search queries, and separately of follow-up links, after a round.
**Attribute**: `research_limits.max_follow_ups`
**Type**: int, 1 or more
**Default**: `5`

#### `RESEARCH_MAX_PARSE_ATTEMPTS` #env
Maximum number of model calls for one structured answer. The first call counts. The agent repeats the prompt when the output is invalid.
**Attribute**: `research_limits.max_parse_attempts`
**Type**: int, 1 or more
**Default**: `3`

## Environment File Loading

**DOTENV_PATHS** [src/mcps/config.py:37](../src/mcps/config.py#L37) lists the `.env` files. `create_config()` loads them in this order:

1. Project root `.env` file
2. User home directory `.env` file

A variable that is already set in the real environment always wins over a `.env` value. If both files set a variable, the project root file wins.

## Configuration Factory

**create_config(vault_dir=None)** [src/mcps/config.py:336](../src/mcps/config.py#L336) loads the `.env` files and returns a `ServerConfig`. It does not create directories. The `vault_dir` argument comes from the `--vault` command line option and wins over `VAULT`.

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

#### Environment isolation #config
The autouse fixture `clean_config_env` in [tests/conftest.py](../tests/conftest.py) removes every configuration variable from the environment for each test. It also turns off `.env` loading. A test sets the variables it needs with `monkeypatch.setenv`. A test marked `real_env` skips the fixture and reads the real environment and `.env` files. The integration modules for embeddings, rerankers, and the evaluation judge use this marker.

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

Pydantic validates every value when `create_config()` builds the `ServerConfig`. An invalid value raises a `pydantic.ValidationError` that names the field, and the server does not start. The rules are:
- A number must parse and stay inside the range that its variable lists above.
- `MIN_CHUNK_SIZE` must not be larger than `MAX_CHUNK_SIZE`.
- If `RAG_EMBEDDING_MODEL` is set, `RAG_EMBEDDING_DIMENSIONS` must be 1 or more.
- `SCRAPER_PROVIDER` must be `scrape_do`, `bright_data`, or empty, and the provider needs its credentials.
- A router URL must be an http or https URL with a host and no credentials, query, or fragment.

One combination only logs a warning, so the server can start with a subset of features. If a research, decision, or RAG embedding model is set while `ROUTER_API_BASE` is empty, `ServerConfig` [src/mcps/config.py:309](../src/mcps/config.py#L309) logs one warning for each feature. That feature then does not work until a router URL is set.
