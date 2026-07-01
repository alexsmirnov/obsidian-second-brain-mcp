# Dependencies and Libraries

External libraries and tools used by the MCPS Model Context Protocol server, organized by role and source. #dependencies #libraries #python

## Framework

### fastmcp
Model Context Protocol server framework providing `FastMCP`, `Context`, tool/resource decorators, and lifespan management.
**Version**: `==3.4.2`
**Used by**: [src/mcps/server.py:8](../src/mcps/server.py#L8), [src/mcps/tools/obsidian_vault.py:10](../src/mcps/tools/obsidian_vault.py#L10), [src/mcps/research/lifespan.py:5](../src/mcps/research/lifespan.py#L5)
**Docs**: [github.com/jlowin/fastmcp](https://github.com/jlowin/fastmcp)

### langgraph
Orchestration framework for the deep research agent graph.
**Version**: `>=1.1.6`
**Used by**: [src/mcps/research/deep_research.py:15](../src/mcps/research/deep_research.py#L15)
**Docs**: [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/)

### langchain-core
Provider-neutral LangChain abstractions: `BaseChatModel`, `Embeddings`, messages, and runnables.
**Version**: transitive via langchain-openai/langchain-community
**Used by**: [src/mcps/rag/search.py:8](../src/mcps/rag/search.py#L8), [src/mcps/rag/embeddings.py:3](../src/mcps/rag/embeddings.py#L3), [src/mcps/research/config.py:14](../src/mcps/research/config.py#L14)

## Runtime

### lancedb
Serverless vector database with vector similarity, full-text search, hybrid search, and reranking.
**Version**: `==0.25.3`
**Used by**: [src/mcps/rag/database.py:9-14](../src/mcps/rag/database.py#L9-L14), [src/mcps/rag/vault.py:14](../src/mcps/rag/vault.py#L14)
**Docs**: [lancedb.github.io/lancedb](https://lancedb.github.io/lancedb/)

### pyarrow
Arrow columnar data format used by LanceDB for schemas and table operations.
**Version**: transitive via lancedb
**Used by**: [src/mcps/rag/database.py:10](../src/mcps/rag/database.py#L10), [src/mcps/rag/llm_reranker.py:6](../src/mcps/rag/llm_reranker.py#L6), [src/mcps/rag/proxy_reranker.py:4](../src/mcps/rag/proxy_reranker.py#L4)

### markdown
Python Markdown parser.
**Version**: `>=3.10.0`
**Used by**: document processing pipeline.
**Docs**: [python-markdown.github.io](https://python-markdown.github.io/)

### rank_bm25
BM25 keyword relevance scoring (reserved for future hybrid search integration).
**Version**: unpinned
**Docs**: [github.com/dorianbrown/rank_bm25](https://github.com/dorianbrown/rank_bm25)

### python-dotenv
Loads environment variables from `.env` files.
**Version**: `>=1.2.2`
**Used by**: [src/mcps/config.py:7](../src/mcps/config.py#L7)
**Docs**: [saurabh-kumar.com/python-dotenv](https://saurabh-kumar.com/python-dotenv/)

### python-frontmatter
YAML frontmatter parser for Obsidian markdown files.
**Version**: `>=1.3.0`
**Used by**: [src/mcps/rag/document_processing.py:12](../src/mcps/rag/document_processing.py#L12)
**Docs**: [python-frontmatter.readthedocs.io](https://python-frontmatter.readthedocs.io/)

### html2text
Converts HTML documents to clean markdown.
**Version**: `>=2025.4.15`

### lxml
HTML/XML parsing for web research content extraction.
**Version**: `>=5.0.0`
**Used by**: [src/mcps/research/tools.py:16](../src/mcps/research/tools.py#L16)

### arxiv-to-prompt
Formats arXiv papers for context-window use.
**Version**: `>=0.13.3`

### pymupdf
PDF text extraction.
**Version**: `>=1.25.0`

### httpx
Shared async HTTP client for model router and web requests.
**Version**: transitive via fastmcp
**Used by**: [src/mcps/server.py:7](../src/mcps/server.py#L7), [src/mcps/tools/obsidian_vault.py:9](../src/mcps/tools/obsidian_vault.py#L9), [src/mcps/research/tools.py:15](../src/mcps/research/tools.py#L15)

### pydantic
Data validation and modeling.
**Version**: transitive via fastmcp/lancedb
**Used by**: [src/mcps/rag/interfaces.py:12](../src/mcps/rag/interfaces.py#L12), [src/mcps/tools/obsidian_vault.py:13](../src/mcps/tools/obsidian_vault.py#L13)

## Model Providers

### langchain-openai
OpenAI-compatible chat and embedding adapters, used against the model router.
**Version**: `>=1.1.11`
**Used by**: [src/mcps/rag/vault.py:15](../src/mcps/rag/vault.py#L15), [src/mcps/research/config.py:16](../src/mcps/research/config.py#L16)

### langchain-google-genai
Google Gemini model adapter for web research.
**Version**: `>=4.2.5`
**Used by**: [src/mcps/research/config.py:15](../src/mcps/research/config.py#L15)

### langchain-community
Community LangChain integrations.
**Version**: `>=0.4.1`

### langchain-aws
AWS Bedrock LangChain integrations.
**Version**: `>=1.4.0`

## Development

### pytest
Test framework.
**Version**: `==8.3.4`
**Used by**: all `tests/test_*.py`
**Docs**: [pytest.org](https://pytest.org)

### pytest-asyncio
Async test support.
**Version**: `==0.25.3`

### pytest-httpx
HTTPX mocking for tests.
**Version**: `>=0.35.0`

### datasets
HuggingFace datasets utility (used in tests/evaluation).
**Version**: `>=4.5.0`

## Linting and Type Checking

### ruff
Linter and formatter.
**Version**: `>=0.11.10`
**Configured in**: [pyproject.toml:81-93](../pyproject.toml#L81-L93)

### pyright
Type checker.
**Version**: `>=1.1.407`
**Configured in**: [pyproject.toml:68-79](../pyproject.toml#L68-L79)

### pyrefly
Alternative type checker.
**Version**: `>=0.15.2`
**Configured in**: [pyproject.toml:113-121](../pyproject.toml#L113-L121)

## Dependency Manifest

All version constraints and groups are declared in [pyproject.toml:10-52](../pyproject.toml#L10-L52).
