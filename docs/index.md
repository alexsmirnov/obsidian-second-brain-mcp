# MCPS

Model Context Protocol (MCP) server providing AI assistants with RAG-based Obsidian vault search, customizable prompts, resources, and tools. Built with Python and FastMCP, deployed as a Python package. #index #overview #python #mcp #rag

## Documentation

- [Architecture Overview](architecture_overview.md) - System design, component architecture, and design patterns
- [Packages & Modules](packages_modules.md) - Internal package structure and dependencies
- [RAG & Obsidian Integration](rag_obsidian_integration.md) - RAG pipeline design, Obsidian features, and search mechanics
- [Deployment](deployment_infrastructure.md) - Python package deployment and build system
- [Configuration](config_environment.md) - Environment variables and configuration options
- [Tests](tests_coverage.md) - Test files and coverage areas

## Dependencies

- **fastmcp** - Model Context Protocol server implementation, [docs](https://github.com/jlowin/fastmcp)
- **lancedb** - Vector database with full-text search and hybrid search capabilities, [docs](https://lancedb.github.io/lancedb/)
- **markdown** - Markdown parsing for document processing, [docs](https://python-markdown.github.io/)
- **rank_bm25** - BM25 algorithm implementation for keyword search ranking
- **python-frontmatter** - YAML frontmatter parser for markdown files, [docs](https://python-frontmatter.readthedocs.io/)
- **langchain-core/langchain-openai** - Provider-neutral model interfaces and LiteLLM Router-compatible adapters
- **httpx** - Shared async HTTP client for model and web requests
