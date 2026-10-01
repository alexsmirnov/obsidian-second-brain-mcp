# Tests and Coverage

Test files and coverage areas for the MCPS Model Context Protocol server. #tests #testing #python

## Test Organization

Tests located in `/work/tests/` directory using pytest framework.

## Unit Tests

### [tests/test_search_agent.py](../tests/test_search_agent.py)
Tests for agentic search functionality including query rewriting and search parameter estimation.

**Coverage**:
- Query rewriting with LLM integration
- File path search parameter estimation
- Tag search parameter estimation
- Combined file and tag parameter estimation
- Error handling for LLM failures

**Key Test Cases**:
- `test_rewrite_query_success` - LLM query rewriting happy path
- `test_rewrite_query_llm_error` - LLM error handling
- `test_estimate_search_params_files_only` - File path extraction
- `test_estimate_search_params_tags_only` - Tag extraction
- `test_estimate_search_params_mixed` - Combined file and tag extraction

### [tests/test_embedding_service.py](../tests/test_embedding_service.py)
Tests for LangChain-backed embedding service.

**Coverage**:
- Empty input without model calls
- Document embedding order preservation
- Query vs document embedding modes
- Configured dimension reporting

### [tests/test_document_processing.py](../tests/test_document_processing.py)
Tests for document processing, file traversal, markdown parsing, and chunking.

**Coverage**:
- File traversal and discovery
- Skip pattern filtering
- Markdown frontmatter parsing
- Wikilink extraction
- Hashtag extraction
- Document ID generation
- Semantic chunking with header splitting
- Chunk size management

**Key Test Cases**:
- `TestMarkdownFileTraversal` - File system traversal tests
- `TestMarkdownProcessor` - Document parsing and metadata extraction
- `TestSemanticChunker` - Text chunking strategies

### [tests/test_lancedb_store.py](../tests/test_lancedb_store.py)
Tests for LanceDB vector store implementation.

**Coverage**:
- Vector storage operations
- Hybrid search (vector + full-text)
- Tag filtering
- Path filtering
- Reranking strategies
- Index creation and management
- Chunk retrieval and deletion

### [tests/test_chunk.py](../tests/test_chunk.py)
Tests for chunk data models and validation.

**Coverage**:
- Chunk model creation
- Field validation
- Metadata handling
- Serialization and deserialization

### [tests/test_langchain_reranker.py](../tests/test_langchain_reranker.py)
Tests for provider-neutral async LangChain reranking.

**Coverage**:
- LLM-based relevance scoring
- Relevance category classification (PERFECT, GOOD, SOME, BAD, NONE)
- Empty result handling
- Invalid score handling

### [tests/test_config.py](../tests/test_config.py)
Tests for `ServerConfig` creation, environment variable precedence, CLI overrides, and validation warnings.

### [tests/test_deep_research.py](../tests/test_deep_research.py)
Contract tests for the LangGraph deep research agent with mocked models and HTTP calls.

### [tests/test_fetch_combinators.py](../tests/test_fetch_combinators.py)
Contract tests for `FetchResult` (`ok`, `is_retryable`) and the combinators `Filtered`, `Fallback`, `UrlSelector`, `FilterSelector`, `FilterChain`, `Throttled`, `Blocked`, `Truncate`, using recording stubs.

### [tests/test_research_tools.py](../tests/test_research_tools.py)
Contract tests for search parsing (Google, DuckDuckGo), the composed `create_fetch` (routing, restricted domains, escalation, truncation, concurrency), per-source status mapping and mime, and exceptions becoming `UNSUPPORTED` results (HTTP mocked with pytest-httpx).

### [tests/test_research_fallback_fetchers.py](../tests/test_research_fallback_fetchers.py)
Contract tests for the browser (fake crawl4ai crawler), Scrape.do and Bright Data fallbacks: request shape, target vs. provider status mapping, provider selection from `ServerConfig`, and crawler connection lifetime (reuse across repeated and overlapping fetches, no replacement on failure, single open/close, startup failure yielding no crawler).

### [tests/test_fetch_filtering.py](../tests/test_fetch_filtering.py)
Contract tests for `RelevanceFilter` (BM25 keyword guarantee, LLM filter with mocked completion, `FILTER_FAILED`, link resolution against the requested URL) and the `MarkdownToHtml` and `PreTextToHtml` filters.

### [tests/test_browser_runtime.py](../tests/test_browser_runtime.py)
Tests for `create_browser_fetch` (configured CDP, local Obscura spawn), the lifespan enabling/disabling only `web_research`, and the research lifespan owning one browser crawler (reused across fetches, closed before the local Obscura process is terminated).

### [tests/test_web_fetch_evaluation.py](../tests/test_web_fetch_evaluation.py)
Contract tests for `load_cases` and `summarize` of the fetch evaluation script, including loading the shipped case files.

### [tests/test_llm_reranker.py](../tests/test_llm_reranker.py)
Integration tests for `LlmReranker` with different model pairs and embedding fusion.

### [tests/test_obsidian_vault_lifespan.py](../tests/test_obsidian_vault_lifespan.py)
Tests for periodic vault indexing updates and error boundaries in the Obsidian lifespan.

### [tests/test_obsidian_vault_tools.py](../tests/test_obsidian_vault_tools.py)
Tests verifying registrations, execution, and error handling of Obsidian tool definitions, including per-document search-result consolidation, gap markers, typed links/backlinks, and the 25,000-character response budget.

### [tests/test_proxy_reranker.py](../tests/test_proxy_reranker.py)
Integration tests for `ProxyReranker` with OpenAI-compatible rerank endpoints.

### [tests/test_search.py](../tests/test_search.py)
Tests for `SemanticSearchEngine` querying, filtering, and field restrictions.

### [tests/test_summarization.py](../tests/test_summarization.py)
Tests for the LangChain-based document summary generator.

### [tests/test_vault.py](../tests/test_vault.py)
Tests for high-level `Vault` initialization, indexing, updates, and searching.

### [tests/test_vault_search_engine.py](../tests/test_vault_search_engine.py)
Tests for search engine instantiation logic and semantic search model selection.

### [tests/test_vault_summary_chunks.py](../tests/test_vault_summary_chunks.py)
Tests verifying document summary chunk injection during document processing.

### [tests/test_vault_summary_wiring.py](../tests/test_vault_summary_wiring.py)
Tests ensuring proper wiring and optional fallback of document summary generation in the vault.

### [tests/test_evaluation_scoring.py](../tests/test_evaluation_scoring.py)
Tests for DRACO rubric scoring (positive/negative weights, normalization).

### [tests/test_evaluation_runner.py](../tests/test_evaluation_runner.py)
Tests for DRACO evaluation summary aggregation (averages, errors, per-domain scores).

### [tests/test_evaluation_judge_integration.py](../tests/test_evaluation_judge_integration.py)
Integration tests for the DRACO LLM criterion judge against a real router model. Skipped unless `ROUTER_API_BASE` and `ROUTER_API_KEY` are set.

## Evaluation Scripts

### [tests/web_research_evaluation.py](../tests/web_research_evaluation.py)
DRACO benchmark (`perplexity-ai/draco`, Technology + Academic domains) for the deep-research agent behind `web_research`. Each rubric criterion is scored by an LLM judge (`RESEARCH_EVAL_MODEL`, falling back to `RESEARCH_INFER_MODEL`). Support package: [tests/evaluation/](../tests/evaluation/).

**Run**: `uv run python tests/web_research_evaluation.py` from the repository root. Log and HTML report are written to `tmp/`.

### [tests/web_fetch_evaluation.py](../tests/web_fetch_evaluation.py)
Replays JSONL fetch cases through the production `fetch(url, query)` and prints failure counts (restricted domains are not failures) and mean response size versus the saved baseline. Cases: `tests/evaluation/data/fetch-cases-smoke.jsonl` (8, default) and `fetch-cases-baseline.jsonl` (279).

**Run**: `uv run tests/web_fetch_evaluation.py --output tmp/fetch-smoke.jsonl [--cases tests/evaluation/data/fetch-cases-baseline.jsonl]`. Requires a CDP browser or Obscura on `PATH`.

### [tests/vault_evaluation.py](../tests/vault_evaluation.py)
Comprehensive evaluation test for vault search functionality measuring precision, recall, and F-score.

**Coverage**:
- Search result quality metrics
- Expected word presence validation
- Unwanted word absence validation
- Multi-query test scenarios
- Performance benchmarking

**Evaluation Metrics**:
- Precision: percentage of expected words found
- Recall: percentage of unwanted words not found
- F-score: harmonic mean of precision and recall

**Test Scenarios**:
- AI/ML topic searches
- Programming language searches
- Project-specific searches
- Personal knowledge management queries

## Test Configuration

**pytest Configuration** [pyproject.toml:54-63](../pyproject.toml#L54-L63):
- Log CLI enabled at INFO level
- Import mode: importlib with durations
- Test paths: `tests` directory
- Python path: `src` directory
- Asyncio mode: auto
- Asyncio fixture scope: function

## Test Fixtures

### Mock Fixtures
Tests use pytest fixtures for mocking external dependencies:
- `vector_store_mock` - Mock IVectorStore implementation
- `llm_mock` - Mock LLM client for testing agentic features
- `embedding_service_mock` - Mock embedding service

### Data Fixtures
Helper functions create test data:
- `_make_chunk()` - Create valid Chunk instances
- `_make_document()` - Create valid Document instances

## Test Dependencies

**Development Dependencies** [pyproject.toml:41-47](../pyproject.toml#L41-L47):
- pytest == 8.3.4
- pytest-asyncio == 0.25.3
- pytest-httpx >= 0.35.0
- datasets >= 4.5.0

## Coverage Areas

### Core Functionality
- Document processing and chunking
- Vector storage and retrieval
- Hybrid search operations
- Embedding generation
- Result reranking

### Integration Points
- LanceDB database operations
- Shared `httpx.AsyncClient` injection
- LangGraph deep research agent workflow
- Vault lifespan and periodic indexing

### Error Handling
- API failure scenarios
- Invalid input handling
- Missing file handling
- Configuration validation

### Performance
- Batch processing
- Search performance metrics
- Indexing efficiency

## Test Execution

Tests run using pytest with asyncio support for async function testing.

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_search_agent.py

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=src/mcps
```
