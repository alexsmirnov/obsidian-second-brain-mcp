# RAG System Design and Obsidian Integration

FastMCP-based RAG pipeline for semantic search over Obsidian vaults. Uses interface-based dependency injection with hybrid search (vector + full-text) via LanceDB, supporting multiple embedding providers and reranking strategies. #rag #obsidian #search #architecture

## Overview

The RAG system provides semantic search capabilities optimized for Obsidian vault structure. Key features:

- **Hybrid search** combining vector similarity and full-text search
- **Obsidian-native** wikilink extraction, hashtag parsing, YAML frontmatter
- **Flexible providers** for embeddings (VoyageAI, OpenAI, Ollama) and reranking
- **Incremental indexing** with file change detection

## System Architecture #architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  MCP Server Layer                                                │
│    └── FastMCP DevAutomationServer                               │
├──────────────────────────────────────────────────────────────────┤
│  Tools Layer                                                     │
│    └── obsidian_search, obsidian_get_content, obsidian_list_files│
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
| **IDocumentProcessor** | [interfaces.py:95-101](../src/mcps/rag/interfaces.py#L95-L101) | Parse files to `Document` objects |
| **IChunker** | [interfaces.py:105-111](../src/mcps/rag/interfaces.py#L105-L111) | Split `Document` to `Chunk` list |
| **IEmbeddingService** | [interfaces.py:114-130](../src/mcps/rag/interfaces.py#L114-L130) | Generate vector embeddings |
| **IVectorStore** | [interfaces.py:140-193](../src/mcps/rag/interfaces.py#L140-L193) | Store/search vectors with filtering |
| **ISearchEngine** | [interfaces.py:195-201](../src/mcps/rag/interfaces.py#L195-L201) | High-level search orchestration |
| **IResultFormatter** | [interfaces.py:204-210](../src/mcps/rag/interfaces.py#L204-L210) | Format results for output |
| **IFileTraversal** | [interfaces.py:213-219](../src/mcps/rag/interfaces.py#L213-L219) | Discover files in vault |
| **IVault** | [interfaces.py:222-268](../src/mcps/rag/interfaces.py#L222-L268) | Facade coordinating all components |

### Data Models

**Document** - Parsed markdown file before chunking:
- `id`: MD5 hash of file path
- `content`: Full document text
- `metadata`: title, description, source from frontmatter
- `tags`: Frontmatter tags only
- `source_path`: Relative to vault root
- `modified_at`: For change detection

**Chunk** - Section after splitting:
- `id`: `{document_id}_{position}`
- `content`: Chunk text
- `outgoing_links`: Extracted `[[wikilinks]]`
- `tags`: Frontmatter + inline `#tags`
- `position`: Order in document

Implementation: [interfaces.py:31-63](../src/mcps/rag/interfaces.py#L31-L63)

## Document Processing Pipeline #module

### File Discovery

**MarkdownFileTraversal** finds `.md` files recursively, filtering by skip patterns (hidden dirs, `node_modules/`, etc.).

Implementation: [document_processing.py:85-106](../src/mcps/rag/document_processing.py#L85-L106)

### Markdown Processing

**MarkdownProcessor** parses files using `python-frontmatter`:
1. Extract YAML frontmatter (`title`, `description`, `source`, `tags`)
2. Get file modification time
3. Generate MD5-based document ID

Implementation: [document_processing.py:108-168](../src/mcps/rag/document_processing.py#L108-L168)

### Chunking Strategies

**SemanticChunker** (default):
- Splits by H1 and H2 headers only (pattern: `^(#{1,2}\s+.+)$`)
- Merges small sections (<100 chars) with subsequent
- Splits large sections (>2000 chars) by paragraphs
- Preserves header with content

Implementation: [document_processing.py:213-329](../src/mcps/rag/document_processing.py#L213-L329)

**FixedSizeChunker** (alternative):
- Fixed 1000-char chunks with 200-char overlap
- Breaks at word boundaries

Implementation: [document_processing.py:171-210](../src/mcps/rag/document_processing.py#L171-L210)

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

Implementation: [document_processing.py:20-38](../src/mcps/rag/document_processing.py#L20-L38)

### Hashtag Extraction

Pattern: `#([a-zA-Z][a-zA-Z0-9_-]*)`
- Must start with letter
- Allows letters, numbers, underscores, hyphens
- Does not match markdown headers (`# `)

Implementation: [document_processing.py:41-46](../src/mcps/rag/document_processing.py#L41-L46)

### YAML Frontmatter

Extracts properties using `python-frontmatter`:
- `title`, `description`, `source`: String metadata
- `tags`: String or list format supported
- Handles malformed YAML gracefully

Implementation: [document_processing.py:114-161](../src/mcps/rag/document_processing.py#L114-L161)

### Chunk Creation

Combines document-level and chunk-level metadata:
- Extracts wikilinks from chunk content
- Extracts inline hashtags from chunk content
- Merges with frontmatter tags

Implementation: [document_processing.py:49-71](../src/mcps/rag/document_processing.py#L49-L71)

## Vector Store and Search #search #database

### LanceDBStore

Implements hybrid search combining vector similarity and full-text search.

**Hybrid Query Construction:**
1. `nearest_to(query_embedding)` - Vector similarity on embeddings column
2. `nearest_to_text(query)` - FTS on content/title/description columns

Implementation: [database.py:138-205](../src/mcps/rag/database.py#L138-L205)

### Filtering

**Tag filtering**: `array_has_all(tags, [...])` - All specified tags must match
- Implementation: [database.py:186-189](../src/mcps/rag/database.py#L186-L189)

**Path filtering**: `source_path LIKE '%substring%'` - Substring match on file path
- Implementation: [database.py:192-193](../src/mcps/rag/database.py#L192-L193)

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

Implementation: [database.py:231-279](../src/mcps/rag/database.py#L231-L279)

## Embedding Providers #config

**OpenAIEmbedding** class supports multiple providers via OpenAI-compatible API.

| Provider | Model | Dimensions | Condition |
|----------|-------|------------|-----------|
| VoyageAI | `voyage-3.5-lite` | 1024 | `voyage_api_key` set |
| OpenAI | `text-embedding-3-small` | 1536 | `openai_api_key` set |
| Ollama | `bge-m3` | 1024 | `ollama_api_base` set |

Selection priority: VoyageAI > OpenAI > Ollama

Implementation: [embeddings.py:15-58](../src/mcps/rag/embeddings.py#L15-L58)
Factory: [vault.py:47-72](../src/mcps/rag/vault.py#L47-L72)

## Reranking Strategies #search

| Strategy | Description | Condition |
|----------|-------------|-----------|
| **VoyageAIReranker** | API-based relevance scoring | `voyage_api_key` set |
| **OllamaReranker** | LLM + embedding scoring | `ollama_api_base` set |
| **RRFReranker** | Reciprocal Rank Fusion | Fallback |

**OllamaReranker** combines:
- LLM scoring: PERFECT(1.0), GOOD(0.75), SOME(0.5), BAD(0.25), NONE(0.0)
- Embedding similarity (cosine)
- Weighted combination (default: equal)

Implementation: [ollama_reranker.py:12-209](../src/mcps/rag/ollama_reranker.py#L12-L209)
Factory: [vault.py:91-105](../src/mcps/rag/vault.py#L91-L105)

## Vault Orchestrator #module

### Factory Pattern

`create_vault()` wires all dependencies from `ServerConfig`:

```
ServerConfig
    ├── vault_dir → Vault.vault_path
    ├── _create_file_traversal → MarkdownFileTraversal
    ├── _create_document_processor → MarkdownProcessor
    ├── _create_chunker → SemanticChunker
    ├── _create_vector_store → LanceDBStore
    │   ├── _create_embedding_function → OpenAIEmbedding
    │   └── _create_reranker → VoyageAI/Ollama/RRF
    └── _create_search_engine → SemanticSearchEngine
        └── _create_result_formatter → MarkdownResultFormatter
```

Implementation: [vault.py:502-555](../src/mcps/rag/vault.py#L502-L555)

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

Implementation: [vault.py:236-306](../src/mcps/rag/vault.py#L236-L306)

### Thread Safety

- `asyncio.Lock` protects `initialize()` and `update_index()`
- Auto-refresh triggers update if >1 minute since last check

Implementation: [vault.py:178](../src/mcps/rag/vault.py#L178)

## Search Flow #search

```
Search Request
    │
    ▼
SemanticSearchEngine.search()
    │
    ▼
LanceDBStore.search()
    ├── Generate query embedding (query=True mode)
    ├── Build hybrid query (vector + FTS)
    ├── Apply filters (tags, path)
    └── Rerank results
    │
    ▼
Filter by min_score (0.5 default)
    │
    ▼
MarkdownResultFormatter.format()
    │
    ▼
Formatted Markdown String
```

Search engine: [search.py:37-48](../src/mcps/rag/search.py#L37-L48)
Result formatter: [search.py:57-90](../src/mcps/rag/search.py#L57-L90)

## Configuration #config

| Setting | Default | Purpose |
|---------|---------|---------|
| `vault_dir` | - | Path to Obsidian vault |
| `voyage_api_key` | `""` | VoyageAI credentials |
| `openai_api_key` | `""` | OpenAI credentials |
| `ollama_api_base` | `""` | Ollama service URL |
| `table_name` | `"documents"` | LanceDB table name |
| `max_chunk_size` | `4000` | Max content in results |
| `search_limit` | `20` | Max results returned |

Configuration: [config.py:11-36](../src/mcps/config.py#L11-L36)

See [Configuration](config_environment.md) for complete environment variable reference.

## Code References

### Core Files
- [src/mcps/rag/vault.py](../src/mcps/rag/vault.py) - Orchestrator and factory functions
- [src/mcps/rag/interfaces.py](../src/mcps/rag/interfaces.py) - Interface definitions and data models
- [src/mcps/rag/document_processing.py](../src/mcps/rag/document_processing.py) - Parsing and chunking
- [src/mcps/rag/database.py](../src/mcps/rag/database.py) - LanceDB vector store
- [src/mcps/rag/search.py](../src/mcps/rag/search.py) - Search engine and result formatting
- [src/mcps/rag/embeddings.py](../src/mcps/rag/embeddings.py) - Embedding service
- [src/mcps/rag/ollama_reranker.py](../src/mcps/rag/ollama_reranker.py) - LLM-based reranking

### Tests
- [test/test_document_processing.py](../test/test_document_processing.py) - Parser and chunking tests
- [test/test_lancedb_store.py](../test/test_lancedb_store.py) - Vector store tests
- [test/test_embedding_service.py](../test/test_embedding_service.py) - Embedding tests

## Related Documentation

- [Architecture Overview](architecture_overview.md) - System design patterns
- [Packages & Modules](packages_modules.md) - Module structure
- [Configuration](config_environment.md) - Environment variables
