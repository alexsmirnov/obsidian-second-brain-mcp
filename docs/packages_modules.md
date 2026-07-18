# Packages and Modules

Internal package structure and dependencies for the MCPS Model Context Protocol server. #packages #modules #python

## Main Package: mcps

Root package containing server core, configuration, and logging.

### [src/mcps/server.py](../src/mcps/server.py) #module
MCP server implementation using FastMCP framework. Composes research and Obsidian lifespans and registers tools conditionally.
**Uses**: config, logs, tools.obsidian_vault, research.agent, research.lifespan
**Used by**: (entry point)

### [src/mcps/config.py](../src/mcps/config.py) #module
Configuration management and environment variable loading.
**Uses**: dotenv, rag.document_processing (default skip patterns)
**Used by**: server, rag.vault, research.config, tools, tests

### [src/mcps/logs.py](../src/mcps/logs.py) #module
Logging configuration and setup.
**Uses**: (standard library only)
**Used by**: server

## Sub-package: mcps.rag

Retrieval-Augmented Generation functionality including document processing, vector storage, search, reranking, and summarization.

### [src/mcps/rag/interfaces.py](../src/mcps/rag/interfaces.py) #module
Abstract interface definitions for all RAG components using ABC and Pydantic models.
**Uses**: pydantic
**Used by**: all rag modules, tests

### [src/mcps/rag/vault.py](../src/mcps/rag/vault.py) #module
High-level vault management orchestrating document processing, storage, search, reranking, and embeddings.
**Uses**: interfaces, document_processing, database, search, embeddings, reranking, llm_reranker, proxy_reranker
**Used by**: tools.obsidian_vault

### [src/mcps/rag/document_processing.py](../src/mcps/rag/document_processing.py) #module
Document file discovery, markdown processing, and text chunking strategies.
**Uses**: interfaces, frontmatter
**Used by**: vault

### [src/mcps/rag/database.py](../src/mcps/rag/database.py) #module
LanceDB vector store implementation with hybrid search and index management.
**Uses**: interfaces, lancedb, pyarrow
**Used by**: vault 

### [src/mcps/rag/embeddings.py](../src/mcps/rag/embeddings.py) #module
Provider-neutral LangChain embedding adapter.
**Uses**: interfaces, langchain_core
**Used by**: vault

### [src/mcps/rag/search.py](../src/mcps/rag/search.py) #module
Search engine, hypothetical document generation, and result formatting implementations.
**Uses**: interfaces, langchain_core
**Used by**: vault

### [src/mcps/rag/search_agent.py](../src/mcps/rag/search_agent.py) #module
Agentic search with query rewriting and search parameter estimation.
**Uses**: interfaces, langchain_core
**Used by**: (currently unused)

### [src/mcps/rag/reranking.py](../src/mcps/rag/reranking.py) #module
Provider-neutral async LangChain reranking using structured output.
**Uses**: interfaces, pydantic, langchain_core
**Used by**: search

### [src/mcps/rag/llm_reranker.py](../src/mcps/rag/llm_reranker.py) #module
LanceDB reranker that fuses LLM relevance ratings with embedding cosine similarity.
**Uses**: interfaces, lancedb, langchain_core, pyarrow
**Used by**: vault

### [src/mcps/rag/proxy_reranker.py](../src/mcps/rag/proxy_reranker.py) #module
HTTP-based proxy reranker for OpenAI-compatible `/v1/rerank` endpoints with RRF fallback.
**Uses**: lancedb, httpx, pyarrow
**Used by**: vault

### [src/mcps/rag/summarization.py](../src/mcps/rag/summarization.py) #module
Whole-document summary generator using a LangChain chat model.
**Uses**: interfaces, langchain_core
**Used by**: vault

## Sub-package: mcps.tools

MCP tool implementations exposing functionality to AI assistants.

### [src/mcps/tools/obsidian_vault.py](../src/mcps/tools/obsidian_vault.py) #module
Obsidian vault operations: file listing, content retrieval, rename/move, and semantic search. Builds the Obsidian lifespan and registers vault tools conditionally.
**Uses**: config, rag.interfaces, rag.vault, fastmcp, httpx
**Used by**: server

### [src/mcps/tools/internet_search.py](../src/mcps/tools/internet_search.py) #module
Internet search placeholder stub.
**Uses**: config
**Used by**: (currently unused)

## Sub-package: mcps.research

LangGraph-based web deep research agent.

### [src/mcps/research/agent.py](../src/mcps/research/agent.py) #module
Research agent public interface and factory.
**Uses**: research.deep_research
**Used by**: server, research.lifespan

### [src/mcps/research/deep_research.py](../src/mcps/research/deep_research.py) #module
Iterative deep research StateGraph: query generation, fetching, extraction, reflection, and final answer synthesis.
**Uses**: langgraph, langchain_core, pydantic, research.tools
**Used by**: research.agent

### [src/mcps/research/config.py](../src/mcps/research/config.py) #module
Research model configuration factory for LangChain chat models and the shared HTTP client.
**Uses**: config, httpx, langchain_core, langchain_openai, langchain_google_genai, pydantic
**Used by**: research.lifespan

### [src/mcps/research/lifespan.py](../src/mcps/research/lifespan.py) #module
FastMCP lifespan handler that creates the shared HTTP client and builds the researcher.
**Uses**: config, fastmcp, httpx, research.agent, research.config
**Used by**: server

### [src/mcps/research/tools.py](../src/mcps/research/tools.py) #module
Async web search (DuckDuckGo, Google) and content fetching utilities.
**Uses**: httpx, lxml, pydantic
**Used by**: research.deep_research

## Sub-package: mcps.resources

MCP resource handler placeholders. The handlers are imported but commented out in `server.register()` and are not currently active.

### [src/mcps/resources/project_resource.py](../src/mcps/resources/project_resource.py) #module
Project content resource handler placeholder.
**Uses**: (standard library)
**Used by**: (currently unused)

### [src/mcps/resources/doc_resource.py](../src/mcps/resources/doc_resource.py) #module
Documentation resource handler placeholder.
**Uses**: (standard library)
**Used by**: (currently unused)

### [src/mcps/resources/url_resource.py](../src/mcps/resources/url_resource.py) #module
URL content resource handler placeholder.
**Uses**: (standard library)
**Used by**: (currently unused)

## Sub-package: mcps.prompts

Prompt template management utilities. Prompt registration is currently commented out in `server.register()`.

### [src/mcps/prompts/file_prompts.py](../src/mcps/prompts/file_prompts.py) #module
File-based prompt template loading with variable substitution.
**Uses**: (standard library)
**Used by**: (currently unused)

## Package Dependency Flow

```
Entry Point (server.main)
├── config
│   └── rag.document_processing (default skip patterns)
├── server
│   ├── config
│   ├── logs
│   ├── research.lifespan
│   │   ├── research.agent
│   │   │   └── research.deep_research
│   │   │       └── research.tools
│   │   └── research.config
│   └── tools.obsidian_vault (conditional)
│       └── rag.vault
│           ├── interfaces
│           ├── document_processing
│           ├── database
│           ├── search
│           ├── embeddings
│           ├── reranking
│           ├── llm_reranker
│           ├── proxy_reranker
│           └── summarization
└── prompts (currently unused)
```

## Cross-Package Dependencies

Only internal project packages are listed. External library dependencies are documented in [Dependencies and Libraries](dependencies_libraries.md).

### Most Depended Upon
- **rag.interfaces** - Used by all RAG components and tests for interface contracts
- **config** - Used by server, vault, research, and tools for configuration

### Leaf Modules (No Internal Dependencies)
- logs
- rag.interfaces
- resources modules
- prompts.file_prompts
- research.tools
