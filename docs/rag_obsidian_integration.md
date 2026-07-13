# RAG System Design and Obsidian Integration

FastMCP-based RAG pipeline for semantic search over Obsidian vaults. Uses interface-based dependency injection with hybrid search (vector + full-text) via LanceDB and OpenAI-compatible model router-backed LangChain model adapters. #rag #obsidian #search #architecture

## Overview

The RAG system provides semantic search capabilities optimized for Obsidian vault structure. Key features:

- **Hybrid search** combining vector similarity and full-text search
- **Obsidian-native** wikilink extraction, hashtag parsing, YAML frontmatter
- **Provider-neutral model access** through LangChain interfaces and an OpenAI-compatible model router
- **Incremental indexing** with file change detection

## System Architecture #architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  MCP Server Layer                                                │
│    └── FastMCP DevAutomationServer                               │
├──────────────────────────────────────────────────────────────────┤
│  Tools Layer                                                     │
│    └── obsidian_search, obsidian_read_note, obsidian_list_files  │
├──────────────────────────────────────────────────────────────────┤
│  Vault Orchestrator                                              │
│    └── Vault class: initialize(), update_index(), search()       │
├──────────────────────────────────────────────────────────────────┤
│  RAG Components                                                  │
│    ├── Document Processing (parsing, chunking)                   │
│    ├── Vector Store (LanceDB hybrid search)                      │
│    ├── Search Engine (query processing)                          │
│    └── Embeddings (multi-provider)                               │
├──────────────────────────────────────────────────────────────────┤
│  Interfaces Layer                                                │
│    └── IVault, IVectorStore, ISearchEngine, IChunker, etc.       │
└──────────────────────────────────────────────────────────────────┘
```

## Interface Definitions #module

All components implement abstract interfaces enabling flexible dependency injection.

| Interface | Location | Purpose |
|-----------|----------|---------|
| **IDocumentProcessor** | [interfaces.py:101-107](../src/mcps/rag/interfaces.py#L101-L107) | Parse files to `Document` objects |
| **IChunker** | [interfaces.py:111-117](../src/mcps/rag/interfaces.py#L111-L117) | Split `Document` to `Chunk` list |
| **IEmbeddingService** | [interfaces.py:120-147](../src/mcps/rag/interfaces.py#L120-L147) | Generate vector embeddings |
| **IDocumentSummaryGenerator** | [interfaces.py:150-156](../src/mcps/rag/interfaces.py#L150-L156) | Generate whole-document summaries |
| **IVectorStore** | [interfaces.py:158-223](../src/mcps/rag/interfaces.py#L158-L223) | Store/search vectors with filtering |
| **ISearchEngine** | [interfaces.py:225-231](../src/mcps/rag/interfaces.py#L225-L231) | High-level search orchestration |
| **IResultFormatter** | [interfaces.py:234-240](../src/mcps/rag/interfaces.py#L234-L240) | Format results for output |
| **IFileTraversal** | [interfaces.py:243-249](../src/mcps/rag/interfaces.py#L243-L249) | Discover files in vault |
| **IVault** | [interfaces.py:252-307](../src/mcps/rag/interfaces.py#L252-L307) | Facade coordinating all components |

### Data Models

**Document** - Parsed markdown file before chunking:
- `id`: MD5 hash of file path
- `content`: Full document text
- `metadata`: title, description, source from frontmatter
- `tags`: Frontmatter tags only
- `source_path`: Relative to vault root
- `wikilink_name`: File name without `.md`
- `file_size`: File size in characters
- `modified_at`: For change detection

**Chunk** - Section after splitting:
- `id`: `{document_id}_{position}`
- `content`: Chunk text
- `title`, `description`, `source`: Metadata fields
- `outgoing_links`: Extracted `[[wikilinks]]`
- `tags`: Frontmatter + inline `#tags`
- `position`: Order in document
- `wikilink_name`: Vault-relative wikilink target without `.md`
- `offset`: Zero-based line index of the chunk start within document content
- `file_size`: Source file size in characters
- `embeddings`: Optional vector embedding

Implementation: [interfaces.py:31-69](../src/mcps/rag/interfaces.py#L31-L69)

## Document Processing Pipeline #module

### File Discovery

**MarkdownFileTraversal** finds `.md` files recursively, filtering by skip patterns (hidden dirs, `node_modules/`, etc.).

Implementation: [document_processing.py:108-148](../src/mcps/rag/document_processing.py#L108-L148)

### Markdown Processing

**MarkdownProcessor** parses files using `python-frontmatter`:
1. Extract YAML frontmatter (`title`, `description`, `source`, `tags`)
2. Get file modification time
3. Generate MD5-based document ID

Implementation: [document_processing.py:151-215](../src/mcps/rag/document_processing.py#L151-L215)

### Chunking Strategies

**SemanticChunker** (default):
- Splits by H1-H3 headers (pattern: `^(#{1,3}\s+.+)$`)
- Merges small sections (<500 chars) with subsequent content until `max_chunk_size`
- Enforces the character-count `max_chunk_size`: splits large sections by paragraphs, then lines, then whitespace or characters for an oversized line
- Preserves header with content

Implementation: [document_processing.py:264-460](../src/mcps/rag/document_processing.py#L264-L460)

**FixedSizeChunker** (alternative, currently unused):
- Fixed 1000-char chunks with 200-char overlap
- Breaks at word boundaries

Implementation: [document_processing.py:218-261](../src/mcps/rag/document_processing.py#L218-L261)

## Obsidian-Specific Features #obsidian

### Wikilink Extraction

Pattern handles all Obsidian wikilink formats:
- Basic: `[[Note Name]]`
- Display text: `[[Note Name|Display Text]]`
- Headers: `[[Note Name#Header]]`
- Combined: `[[Note Name#Header|Display Text]]`
- Images: `![[Note Name]]`
- Nested brackets: `[[Note [with] brackets]]`

Returns note name only, automatically deduplicated.

Implementation: [document_processing.py:27-45](../src/mcps/rag/document_processing.py#L27-L45)

### Hashtag Extraction

Pattern: `#([a-zA-Z][a-zA-Z0-9_-]*)`
- Must start with letter
- Allows letters, numbers, underscores, hyphens
- Does not match markdown headers (`# `)

Implementation: [document_processing.py:48-53](../src/mcps/rag/document_processing.py#L48-L53)

### YAML Frontmatter

Extracts properties using `python-frontmatter`:
- `title`, `description`, `source`: String metadata
- `tags`: String or list format supported
- Handles malformed YAML gracefully

Implementation: [document_processing.py:151-215](../src/mcps/rag/document_processing.py#L151-L215)

### Chunk Creation

Combines document-level and chunk-level metadata:
- Extracts wikilinks from chunk content
- Extracts inline hashtags from chunk content
- Merges with frontmatter tags
- Stores the source note wikilink name, chunk start offset, and file size for partial reads
- Summary chunks use `SUMMARY_CHUNK_POSITION = -1`

Implementation: [document_processing.py:56-93](../src/mcps/rag/document_processing.py#L56-L93)

## Vector Store and Search #search #database

### LanceDBStore

Implements hybrid search combining vector similarity and full-text search.

**Hybrid Query Construction:**
1. `nearest_to(query_embedding)` - Vector similarity on embeddings column
2. `nearest_to_text(query)` - FTS on content/title/description columns

Implementation: [database.py:147-221](../src/mcps/rag/database.py#L147-L221)

### Filtering

**Tag filtering**: `array_has_all(tags, [...])` - All specified tags must match
- Implementation: [database.py:198-208](../src/mcps/rag/database.py#L198-L208)

**Path filtering**: `source_path LIKE '%substring%'` - Substring match on file path
- Implementation: [database.py:198-208](../src/mcps/rag/database.py#L198-L208)

### Search Scopes

| Scope | Description |
|-------|-------------|
| `CONTENT` | Search chunk content only |
| `TITLE` | Search document titles only |
| `DESCRIPTION` | Search descriptions only |
| `ALL` | Search all fields (default) |

Definition: [interfaces.py:15-20](../src/mcps/rag/interfaces.py#L15-L20)

### Indexes

- **FTS indexes** on `content`, `title`, `description` columns (Tantivy-based)
- **LabelList index** on `tags` column for efficient filtering

Implementation: [database.py:250-304](../src/mcps/rag/database.py#L250-L304)

## Embedding Model Configuration #config

RAG uses `LangChainEmbeddingService` over the provider-neutral LangChain `Embeddings` interface. Provider-specific adapter construction happens in `create_vault()` and points to the OpenAI-compatible model router.

| Config | Purpose | Default |
|--------|---------|---------|
| `rag_embedding_model` | Model router embedding model name | `""` |
| `rag_embedding_dimensions` | LanceDB embedding vector dimension | `0` |

Implementation: [embeddings.py](../src/mcps/rag/embeddings.py)
Factory boundary: [vault.py:73-93](../src/mcps/rag/vault.py#L73-L93)

## Reranking Strategies #search

| Strategy | Description | Condition |
|----------|-------------|-----------|
| **RRFReranker** | LanceDB reciprocal rank fusion for vector + full-text results | Default when no reranker model configured |
| **ProxyReranker** | HTTP `/v1/rerank` call to the model router with RRF fallback | `rag_reranker_model` set |
| **LlmReranker** | Fuses LLM structured relevance ratings with embedding cosine similarity | `rag_reranker_embedding_model` and optionally `rag_reranker_infer_model` set |
| **LangChainReranker** | Async post-retrieval relevance scoring through `BaseChatModel` | `rag_infer_model` set |

**LangChainReranker** uses LLM scoring categories: PERFECT(1.0), GOOD(0.75), SOME(0.5), BAD(0.25), NONE(0.0).

Implementations: [proxy_reranker.py](../src/mcps/rag/proxy_reranker.py), [llm_reranker.py](../src/mcps/rag/llm_reranker.py), [reranking.py](../src/mcps/rag/reranking.py)
Factory boundary: [vault.py:95-131](../src/mcps/rag/vault.py#L95-L131)

## Vault Orchestrator #module

### Factory Pattern

`create_vault()` wires provider-neutral dependencies. Model construction and HTTP client ownership happen in the Obsidian tool lifespan:

```
Obsidian lifespan
    ├── shared httpx.AsyncClient
    ├── build_obsidian_model_config → LangChain adapters via model router
    ├── LangChainEmbeddingService → LanceDBStore
    ├── optional LangChainReranker → SemanticSearchEngine
    └── create_vault → Vault
```

Implementation: [vault.py:570-629](../src/mcps/rag/vault.py#L570-L629)

### Index Update Algorithm

`update_index()` performs incremental updates:
1. Get stored file metadata from database
2. Traverse current vault files
3. Compare modification timestamps:
   - **New files** → Add to processing batch
   - **Modified files** → Delete old chunks, add to batch
   - **Unchanged** → Skip
   - **Deleted** → Remove from database
4. Process in batches (default: 8 files)
5. Rebuild indexes if changes detected

Implementation: [vault.py:285-364](../src/mcps/rag/vault.py#L285-L364)

### Thread Safety

- `asyncio.Lock` protects `initialize()` and `update_index()`
- Auto-refresh triggers update every minute via the Obsidian lifespan

Implementation: [vault.py:249](../src/mcps/rag/vault.py#L249)

## Search Flow #search

```
Search Request
    │
    ▼
SemanticSearchEngine.search()
    ├── Generate hypothetical document with rag_infer_model (optional)
    │
    ▼
LanceDBStore.search()
    ├── Generate embedding from hypothetical document or original query
    ├── Build hybrid query (vector + FTS)
    ├── Apply filters (tags, path)
    └── Apply DB-level reranking
    │
    ▼
Filter by min_score (0.5 default)
    │
    ▼
One-call structured reranking by chunk IDs (optional)
    │
    ▼
MarkdownResultFormatter.format()
    │
    ▼
Formatted Markdown String
```

Search engine: [search.py](../src/mcps/rag/search.py)
Result formatter: [search.py:326-412](../src/mcps/rag/search.py#L326-L412)

If HyDE generation or search-level reranking fails, search logs the error and returns the min-score-filtered vector-store results. `rag_infer_model` controls the search-level inference path. `rag_reranker_*` settings configure DB-level LanceDB reranking.

## Configuration #config

| Setting | Default | Purpose |
|---------|---------|---------|
| `vault_dir` | - | Path to Obsidian vault |
| `table_name` | `"documents"` | LanceDB table name |
| `max_chunk_size` | `4000` | Max content in results |
| `search_limit` | `30` | Max results returned |
| `rag_embedding_model` | `""` | Model router embedding model |
| `rag_embedding_dimensions` | `0` | LanceDB vector dimension |
| `rag_infer_model` | `""` | Optional search-level HyDE and structured reranking model |
| `rag_reranker_model` | `""` | Optional DB-level LanceDB reranker model |
| `rag_summary_model` | `""` | Optional whole-document summary model |

Configuration: [config.py:14-50](../src/mcps/config.py#L14-L50)

See [Configuration](config_environment.md) for complete environment variable reference.

## Code References

### Core Files
- [src/mcps/rag/vault.py](../src/mcps/rag/vault.py) - Orchestrator and factory functions
- [src/mcps/rag/interfaces.py](../src/mcps/rag/interfaces.py) - Interface definitions and data models
- [src/mcps/rag/document_processing.py](../src/mcps/rag/document_processing.py) - Parsing and chunking
- [src/mcps/rag/database.py](../src/mcps/rag/database.py) - LanceDB vector store
- [src/mcps/rag/search.py](../src/mcps/rag/search.py) - Search engine and result formatting
- [src/mcps/rag/embeddings.py](../src/mcps/rag/embeddings.py) - Embedding service
- [src/mcps/rag/reranking.py](../src/mcps/rag/reranking.py) - Async LLM-based search-level reranking
- [src/mcps/rag/llm_reranker.py](../src/mcps/rag/llm_reranker.py) - LanceDB LLM+embedding reranker
- [src/mcps/rag/proxy_reranker.py](../src/mcps/rag/proxy_reranker.py) - OpenAI-compatible proxy reranker
- [src/mcps/rag/summarization.py](../src/mcps/rag/summarization.py) - Whole-document summary generator

### Tests
- [tests/test_document_processing.py](../tests/test_document_processing.py) - Parser and chunking tests
- [tests/test_lancedb_store.py](../tests/test_lancedb_store.py) - Vector store tests
- [tests/test_embedding_service.py](../tests/test_embedding_service.py) - Embedding tests
- [tests/test_search.py](../tests/test_search.py) - Search engine tests
- [tests/test_langchain_reranker.py](../tests/test_langchain_reranker.py) - LangChain reranker tests
- [tests/test_llm_reranker.py](../tests/test_llm_reranker.py) - LLM reranker tests
- [tests/test_proxy_reranker.py](../tests/test_proxy_reranker.py) - Proxy reranker tests
- [tests/test_vault.py](../tests/test_vault.py) - Vault orchestrator tests

## Related Documentation

- [Architecture Overview](architecture_overview.md) - System design patterns
- [Packages & Modules](packages_modules.md) - Module structure
- [Configuration](config_environment.md) - Environment variables
