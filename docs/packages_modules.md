# Packages and Modules

Internal package structure and dependencies for the MCPS Model Context Protocol server. #packages #modules #python

## Main Package: mcps

Root package containing server core, configuration, and common utilities.

### [src/mcps/server.py](../src/mcps/server.py) #module
MCP server implementation using FastMCP framework.
**Uses**: config, common, tools, resources, prompts
**Used by**: (entry point)

### [src/mcps/config.py](../src/mcps/config.py) #module
Configuration management and environment variable loading.
**Uses**: (standard library only)
**Used by**: server, rag.vault

### [src/mcps/common.py](../src/mcps/common.py) #module
Common utilities and base classes for async context manager tools.
**Uses**: (standard library only)
**Used by**: tools

### [src/mcps/logs.py](../src/mcps/logs.py) #module
Logging configuration and setup.
**Uses**: (standard library only)
**Used by**: server

## Sub-package: mcps.rag

Retrieval-Augmented Generation functionality including document processing, vector storage, and search.

### [src/mcps/rag/interfaces.py](../src/mcps/rag/interfaces.py) #module
Abstract interface definitions for all RAG components using ABC and Pydantic models.
**Uses**: (Pydantic BaseModel, ABC)
**Used by**: all rag modules

### [src/mcps/rag/vault.py](../src/mcps/rag/vault.py) #module
High-level vault management orchestrating document processing, storage, and search.
**Uses**: interfaces, document_processing, database, search
**Used by**: tools.obsidian_vault

### [src/mcps/rag/document_processing.py](../src/mcps/rag/document_processing.py) #module
Document file discovery, markdown processing, and text chunking strategies.
**Uses**: interfaces
**Used by**: vault

### [src/mcps/rag/database.py](../src/mcps/rag/database.py) #module
LanceDB vector store implementation with hybrid search capabilities.
**Uses**: interfaces
**Used by**: vault

### [src/mcps/rag/embeddings.py](../src/mcps/rag/embeddings.py) #module
Provider-neutral LangChain embedding adapter.
**Uses**: interfaces
**Used by**: vault

### [src/mcps/rag/search.py](../src/mcps/rag/search.py) #module
Search engine and result formatting implementations.
**Uses**: interfaces, database
**Used by**: vault

### [src/mcps/rag/search_agent.py](../src/mcps/rag/search_agent.py) #module
Agentic search with query rewriting and search parameter estimation.
**Uses**: interfaces
**Used by**: (planned future use)

### [src/mcps/rag/reranking.py](../src/mcps/rag/reranking.py) #module
Provider-neutral async reranking using LangChain chat model interfaces.
**Uses**: interfaces
**Used by**: search, tools.obsidian_vault

## Sub-package: mcps.tools

MCP tool implementations exposing functionality to AI assistants.

### [src/mcps/tools/obsidian_vault.py](../src/mcps/tools/obsidian_vault.py) #module
Obsidian vault operations: file listing, content retrieval, rename/move, and semantic search.
**Uses**: common, rag.vault
**Used by**: server

### [src/mcps/tools/internet_search.py](../src/mcps/tools/internet_search.py) #module
Internet search using Serper and Tavily APIs.
**Uses**: common
**Used by**: server

### [src/mcps/tools/deep_research.py](../src/mcps/tools/deep_research.py) #module
Deep research capabilities using Perplexity AI.
**Uses**: common
**Used by**: server

### [src/mcps/tools/rag_search.py](../src/mcps/tools/rag_search.py) #module
RAG search tool implementation.
**Uses**: common
**Used by**: server

## Sub-package: mcps.resources

MCP resource handlers for various content types.

### [src/mcps/resources/project_resource.py](../src/mcps/resources/project_resource.py) #module
Project content resource handler.
**Uses**: (standard library)
**Used by**: server

### [src/mcps/resources/doc_resource.py](../src/mcps/resources/doc_resource.py) #module
Documentation resource handler.
**Uses**: (standard library)
**Used by**: server

### [src/mcps/resources/url_resource.py](../src/mcps/resources/url_resource.py) #module
URL content resource handler.
**Uses**: (standard library)
**Used by**: server

## Sub-package: mcps.prompts

Prompt template management loading prompts from markdown files.

### [src/mcps/prompts/file_prompts.py](../src/mcps/prompts/file_prompts.py) #module
File-based prompt template loading with variable substitution.
**Uses**: (standard library)
**Used by**: server

## Package Dependency Flow

```
Entry Point (__init__.py)
├── config
│   └── (no internal dependencies)
├── server
│   ├── config
│   ├── common
│   ├── tools
│   │   ├── common
│   │   └── rag.vault
│   │       ├── interfaces
│   │       ├── document_processing
│   │       ├── database
│   │       ├── search
│   │       └── reranking
│   ├── resources
│   └── prompts
└── logs
```

## Cross-Package Dependencies

Only internal project packages are listed. External library dependencies documented in [index.md](index.md).

### Most Depended Upon
- **rag.interfaces** - Used by all RAG components for interface contracts
- **common** - Used by all tool implementations for base classes
- **config** - Used by server and vault for configuration

### Leaf Modules (No Internal Dependencies)
- config
- common
- logs
- rag.interfaces
- resources modules
- prompts.file_prompts
