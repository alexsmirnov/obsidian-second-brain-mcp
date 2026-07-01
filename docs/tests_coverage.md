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

### [tests/test_llm_reranker.py](../tests/test_llm_reranker.py)
Integration tests for `LlmReranker` with different model pairs and embedding fusion.

### [tests/test_obsidian_vault_lifespan.py](../tests/test_obsidian_vault_lifespan.py)
Tests for periodic vault indexing updates and error boundaries in the Obsidian lifespan.

### [tests/test_obsidian_vault_tools.py](../tests/test_obsidian_vault_tools.py)
Tests verifying registrations, execution, and error handling of Obsidian tool definitions.

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

## Evaluation Scripts

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
