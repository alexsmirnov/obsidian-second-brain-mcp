# RAG System Design and Obsidian Integration

FastMCP-based RAG pipeline for semantic search over Obsidian vaults. It uses interface-based dependency injection, LanceDB hybrid search, and OpenAI-compatible model-router-backed LangChain adapters. #rag #obsidian #search #architecture

## Overview

The RAG system provides semantic search over an Obsidian vault with:

- **Hybrid search** combining vector similarity and full-text search.
- **Obsidian-native parsing** for typed relation links, hashtags, and YAML frontmatter.
- **Provider-neutral model access** through LangChain interfaces and an OpenAI-compatible model router.
- **Incremental indexing** based on file change detection.

## System Architecture #architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│ MCP Server Layer                                                       │
│   └── FastMCP DevAutomationServer                                      │
├────────────────────────────────────────────────────────────────────────┤
│ Tools Layer                                                            │
│   └── obsidian_search, obsidian_read_note, obsidian_list_files,       │
│       obsidian_traverse_relations                                      │
├────────────────────────────────────────────────────────────────────────┤
│ Vault Orchestrator                                                     │
│   └── Vault: initialize(), update_index(), search(),                  │
│       get_backlinks(), traverse_relations()                            │
├────────────────────────────────────────────────────────────────────────┤
│ RAG Components                                                         │
│   ├── Document Processing (parsing, chunking, typed relation links)   │
│   ├── Vector Store (LanceDB hybrid search and link lookup)            │
│   ├── Search Engine (query processing)                                │
│   └── Embeddings (multi-provider)                                     │
├────────────────────────────────────────────────────────────────────────┤
│ Interfaces Layer                                                       │
│   └── IVault, IVectorStore, ISearchEngine, IChunker, and related APIs │
└────────────────────────────────────────────────────────────────────────┘
```

## Interface Definitions #module

All components implement abstract interfaces enabling flexible dependency injection.

| Interface | Location | Purpose |
|-----------|----------|---------|
| **IDocumentProcessor** | [interfaces.py:184-190](../src/mcps/rag/interfaces.py#L184-L190) | Parse files to `Document` objects |
| **IChunker** | [interfaces.py:194-200](../src/mcps/rag/interfaces.py#L194-L200) | Split `Document` into `Chunk` objects |
| **IEmbeddingService** | [interfaces.py:203-231](../src/mcps/rag/interfaces.py#L203-L231) | Generate vector embeddings |
| **IDocumentSummaryGenerator** | [interfaces.py:233-239](../src/mcps/rag/interfaces.py#L233-L239) | Generate whole-document summaries |
| **IVectorStore** | [interfaces.py:241-331](../src/mcps/rag/interfaces.py#L241-L331) | Store/search vectors and resolve typed outgoing and incoming links |
| **ISearchEngine** | [interfaces.py:333-339](../src/mcps/rag/interfaces.py#L333-L339) | High-level search orchestration |
| **IFileTraversal** | [interfaces.py:342-348](../src/mcps/rag/interfaces.py#L342-L348) | Discover files in a vault |
| **IVault** | [interfaces.py:351-436](../src/mcps/rag/interfaces.py#L351-L436) | Facade coordinating vault operations, backlinks, and relation traversal |

### Data Models

**Document** — Parsed markdown file before chunking:
- `id`: MD5 hash of the file path.
- `content`: Full document text.
- `metadata`: `title`, `description`, and `source` from frontmatter.
- `tags`: Frontmatter tags only.
- `source_path`: Relative to the vault root.
- `wikilink_name`: File name without `.md` and without folders (the stem).
- `file_size`: File size in characters.
- `modified_at`: Timestamp used for change detection.

**Chunk** — A section after splitting:
- `id`: `{document_id}_{position}`.
- `content`: Chunk text.
- `title`, `description`, `source`: Metadata fields.
- `links`: Wikilink targets.
- `link_types`: Link types positionally aligned with `links`.
- `typed_links`: Property that reconstructs aligned arrays as `Link` objects.
- `tags`: Frontmatter and inline `#tags`.
- `position`: Order in the document.
- `source_path`: Path relative to the vault root, with `.md`.
- `wikilink_name`: File name without `.md` and without folders (the stem).
- `offset`: Zero-based line index of the chunk start.
- `file_size`: Source file size in characters.
- `embeddings`: Optional vector embedding.

**Link** — A typed relation edge:
- `type`: Free-form relation type extracted from the note text.
- `target`: Destination of the link. In the store this is the raw wikilink text as written. Tool results replace it with a note path (see [Note Paths](#note-paths)).

Implementation: [interfaces.py:22-152](../src/mcps/rag/interfaces.py#L22-L152)

## Document Processing Pipeline #module

### File Discovery

**MarkdownFileTraversal** recursively finds `.md` files while applying skip patterns for hidden directories, `node_modules/`, and similar paths.

Implementation: [document_processing.py:143-183](../src/mcps/rag/document_processing.py#L143-L183)

### Markdown Processing

**MarkdownProcessor** parses files using `python-frontmatter`:
1. Extract YAML frontmatter (`title`, `description`, `source`, `tags`).
2. Get the file modification time.
3. Generate an MD5-based document ID.

Implementation: [document_processing.py:186-250](../src/mcps/rag/document_processing.py#L186-L250)

### Chunking Strategies

**SemanticChunker** (default):
- Splits by H1-H3 headers.
- Merges small sections with subsequent content until `max_chunk_size`.
- Enforces `max_chunk_size` by splitting oversized sections at paragraphs, lines, whitespace, or characters.
- Preserves headers with their content.

Implementation: [document_processing.py:307-503](../src/mcps/rag/document_processing.py#L307-L503)

**FixedSizeChunker** (alternative):
- Produces fixed-size chunks with overlap.
- Breaks at word boundaries.

Implementation: [document_processing.py:253-304](../src/mcps/rag/document_processing.py#L253-L304)

## Obsidian-Specific Features #obsidian

### Typed Relation Link Extraction

The parser recognizes standard Obsidian wikilink forms:
- Basic: `[[Note Name]]`.
- Display text: `[[Note Name|Display Text]]`.
- Headers: `[[Note Name#Header]]`.
- Combined headers and display text.
- Embeds: `![[Note Name]]`.
- Nested brackets: `[[Note [with] brackets]]`.

It also recognizes Dataview inline fields containing a wikilink in these forms:
- Standalone: `field:: [[Target]]`.
- Hidden parenthesized: `(field:: [[Target]])`.
- Visible bracketed: `[field:: [[Target]]]`.
- List-item and indented forms, such as `- field:: [[Target]]`.

A field name must match `[a-zA-Z][a-zA-Z0-9_-]*`. A recognized field types exactly the first following wikilink. Bare wikilinks have the default type `related`. Source order is preserved. Duplicate edges collapse by `(type, target)`; an explicit type absorbs a default-typed edge with the same target.

Implementation: [document_processing.py:18-80](../src/mcps/rag/document_processing.py#L18-L80)

### Note Paths #note-paths

A **note path** is the path of a note relative to the vault root, without `.md`, with a leading `/`. The file `Projects/AI/Note.md` has the note path `/Projects/AI/Note`. Search results, backlinks, and traversal results use note paths, so a caller can find the file without searching the vault.

`obsidian_read_note` and `obsidian_traverse_relations` accept a **reference** to a note. Matching is case-sensitive and never includes the `.md` extension:

| Form | Example | Matches |
|------|---------|---------|
| Full path | `/AI/Note` | Exactly that file |
| Partial path | `AI/Note` | Every note whose path ends with those whole folder names and the file name: `/AI/Note`, `/Work/AI/Note`. It does not match `/XAI/Note`. |
| Bare name | `Note` | Every note with that file name |

If a partial path or bare name matches more than one note, the tool picks one with the Obsidian rules, described below. If it matches none, the tool returns a not-found error. A full path to a file is read straight from disk, so it also works for a file that the index skipped. Partial paths and bare names are resolved through the index.

A reference is rejected when it is blank, ends with `.md` or `/`, contains `\`, `..`, or an empty folder name (`//`).

When several notes match, the rules are applied in this order until one note is left:

1. The shortest path (fewest folders).
2. The note closest to the note that holds the link: the most shared leading folders. A read or traversal start has no linking note, so this rule is skipped.
3. Alphabetical order of the path.

Link targets in results are resolved the same way, from the note that holds the link. A link written `[[Dup]]` points to one note only, and duplicates are removed after that step. A backlink or incoming traversal edge is reported only for the note its link resolves to. A link to a note that does not exist keeps its original text in search results.

Implementation: [note_paths.py](../src/mcps/rag/note_paths.py) (`reference_name`, `select_matching`, `resolve_links`), [vault.py](../src/mcps/rag/vault.py) (`Vault.resolve_references`, `get_file`, `get_backlinks`, `traverse_relations`)

### Hashtag Extraction

Pattern: `#([a-zA-Z][a-zA-Z0-9_-]*)`.
- Tags must start with a letter.
- Tags can contain letters, numbers, underscores, and hyphens.
- Markdown headers (`# `) do not match.

Implementation: [document_processing.py:83-87](../src/mcps/rag/document_processing.py#L83-L87)

### YAML Frontmatter

The processor extracts `title`, `description`, `source`, and `tags` through `python-frontmatter`. It supports string and list tag formats and handles malformed YAML gracefully.

Implementation: [document_processing.py:186-250](../src/mcps/rag/document_processing.py#L186-L250)

### Chunk Creation

Chunk creation extracts typed relation links and inline hashtags, combines inline and frontmatter tags, records source-note details, and uses `SUMMARY_CHUNK_POSITION = -1` for whole-document summary chunks.

Implementation: [document_processing.py:90-127](../src/mcps/rag/document_processing.py#L90-L127)

## Vector Store and Search #search #database

### LanceDBStore

`LanceDBStore` implements hybrid search with vector similarity and full-text search.

**Hybrid query construction:**
1. `nearest_to(query_embedding)` performs vector similarity on embeddings.
2. `nearest_to_text(query)` performs full-text search on content, title, and description.

Implementation: [database.py:156-228](../src/mcps/rag/database.py#L156-L228)

### Filtering

**Tag filtering:** `array_has_all(tags, [...])` requires every requested tag.

**Path filtering:** `source_path LIKE 'prefix%'` keeps source paths that start with the given text. The `obsidian_search` tool removes one leading `/` first, so a folder taken from a returned `note_path` works as is.

Implementation: [database.py:204-218](../src/mcps/rag/database.py#L204-L218)

### Link Lookups

`get_notes_with_links()` takes source paths (`Folder/Note.md`) and aggregates every chunk of each requested note into a deduplicated `NoteLinks` record. `get_notes_linking_to()` takes raw link targets and finds source notes that link to them, optionally restricted to OR-combined relation types. In both, `NoteLinks.note` is the note's source path, so two notes with the same file name stay separate.

The LanceDB `array_has_any` predicate over `links` and `link_types` is a sound-but-inexact pre-filter because parallel arrays cannot be correlated by SQL position. Exact `(type, target)` matching is always re-verified in Python after rows are fetched.

Implementation: [database.py:409-537](../src/mcps/rag/database.py#L409-L537)

### Search Scopes

| Scope | Description |
|-------|-------------|
| `CONTENT` | Search chunk content only |
| `TITLE` | Search document titles only |
| `DESCRIPTION` | Search descriptions only |
| `ALL` | Search all fields (default) |

Definition: [interfaces.py:14-18](../src/mcps/rag/interfaces.py#L14-L18)

### Indexes

- **FTS indexes** on `content`, `title`, and `description`.
- **LabelList indexes** on `tags`, `links`, and `link_types` for tag and relation lookup.

Implementation: [database.py:264-332](../src/mcps/rag/database.py#L264-L332)

### Relation Traversal

`obsidian_traverse_relations` walks the typed relation graph from a note in both directions. It accepts depths from 1 through 3 and optional free-form `relation_types`, which are OR-combined; omitted types follow every relation.

The walk is breadth-first. Each note appears at most once, at its shallowest depth. It is limited to 100 distinct notes and returns a truncation warning when the cap is reached. Dangling targets that are not indexed notes are silently excluded. Each returned node identifies its relation type, direction, and the note through which it was reached.

The starting note is given as a reference (see [Note Paths](#note-paths)). `origin`, `nodes[].note`, and `nodes[].via` are note paths. A link whose name matches several notes leads to the one note picked by the Obsidian rules. For incoming edges, the walk looks for links written with any name that can resolve to a note on the current level (`Note`, `Folder/Note`, or `/Folder/Note`), and keeps those that resolve to that note from the linking note.

Implementation: [vault.py:480-609](../src/mcps/rag/vault.py#L480-L609), [obsidian_vault.py:274-284](../src/mcps/tools/obsidian_vault.py#L274-L284), [obsidian_vault.py:438-451](../src/mcps/tools/obsidian_vault.py#L438-L451)

### Schema Migration

`links` and `link_types` replace the former untyped link column incompatibly. There is no automatic migration. When upgrading an existing deployment, delete `.vault_db` inside the vault root once; the next `update_index()` cycle rebuilds the table with the new schema. This is a required one-time manual operation.

## Embedding Model Configuration #config

RAG uses `LangChainEmbeddingService` over the provider-neutral LangChain `Embeddings` interface. Provider-specific adapter construction happens in `create_vault()` and targets the OpenAI-compatible model router.

| Config | Purpose | Default |
|--------|---------|---------|
| `rag_embedding_model` | Model-router embedding model name | `""` |
| `rag_embedding_dimensions` | LanceDB embedding vector dimension | `0` |

Implementation: [embeddings.py](../src/mcps/rag/embeddings.py)
Factory boundary: [vault.py:73-93](../src/mcps/rag/vault.py#L73-L93)

## Reranking Strategies #search

| Strategy | Description | Condition |
|----------|-------------|-----------|
| **RRFReranker** | LanceDB reciprocal-rank fusion for vector and full-text results | Default when no reranker model is configured |
| **ProxyReranker** | HTTP `/v1/rerank` call to the model router with RRF fallback | `rag_reranker_model` is set |
| **LlmReranker** | Fuses LLM relevance ratings with embedding cosine similarity | `rag_reranker_embedding_model` and optionally `rag_reranker_infer_model` are set |
| **LangChainReranker** | Async post-retrieval relevance scoring through `BaseChatModel` | `rag_infer_model` is set |

## Vault Orchestrator #module

### Factory Pattern

`create_vault()` wires provider-neutral dependencies. Model construction and HTTP client ownership happen in the Obsidian tool lifespan.

Implementation: [vault.py:745-807](../src/mcps/rag/vault.py#L745-L807)

### Index Update Algorithm

`update_index()` performs incremental updates:
1. Get stored file metadata from the database.
2. Traverse current vault files.
3. Add new files, replace modified files, skip unchanged files, and remove deleted files.
4. Process files in batches.
5. Rebuild indexes if changes were detected.

Implementation: [vault.py:293-381](../src/mcps/rag/vault.py#L293-L381)

### Index Update Scheduling

`IndexScheduler` owns every `update_index()` call made by the Obsidian server. It runs one update at startup and then one every 5 minutes. A caller can also request an update. The `obsidian_search`, `obsidian_read_note`, and `obsidian_traverse_relations` tools request one before they read the index. This way a note that a client just edited is visible to the next tool call.

A requested update always starts after the request, so it sees every file change made before the tool call. Calls that arrive while an update runs wait for the next update, and all of them share that one run.

A tool call waits at most 20 seconds. If the update takes longer, the tool logs a warning and answers from the current index. The update keeps running. If the update fails, the scheduler logs the error and the tool answers from the current index. The `obsidian_list_files` tool reads the file system and does not request an update.

Implementation: [index_scheduler.py:1-67](../src/mcps/rag/index_scheduler.py#L1-L67), [obsidian_vault.py:191-230](../src/mcps/tools/obsidian_vault.py#L191-L230)

## Search Flow #search

```
Search Request
    │
    ▼
SemanticSearchEngine.search()
    ├── Generate hypothetical document with rag_infer_model (optional)
    ▼
LanceDBStore.search()
    ├── Generate embedding from hypothetical document or original query
    ├── Build hybrid query (vector + FTS)
    ├── Apply filters (tags, path)
    └── Apply database-level reranking
    ▼
Filter by min_score (0.5 default)
    ▼
Merge neighboring chunks and perform structured reranking (optional)
    ▼
Combine result windows by source file
    ▼
Apply the 25,000-character response budget
    ▼
list[SearchResultItem]
```

Search engine: [search.py](../src/mcps/rag/search.py)
Tool result model: [obsidian_vault.py:161-175](../src/mcps/tools/obsidian_vault.py#L161-L175)

If HyDE generation or search-level reranking fails, search logs the error and returns min-score-filtered vector-store results.

The tool returns one record per source file, ranked by that file's first result window. Each record identifies its note by `note_path` (see [Note Paths](#note-paths)); there are no `wikilink_name` or `source_path` fields. Isolated windows from the same file are ordered by source offset and separated with `<<Gap: N lines>>`; title, description, tags, outgoing links, and backlinks appear once for the consolidated document.

Returned string values have a combined 25,000-character budget. If the response exceeds it, document content is replaced by `<<Dropped: content is too big>>`, largest content first. If metadata alone still exceeds the budget, the lowest-ranked document records are removed. The tool returns an object with `results` (the notes), `total_documents` (the count before trimming), `truncated`, and a `warning` that reports the retained and original document counts. `truncated` is `false` and `warning` is empty when nothing was removed. The size fields are outside the character budget.

## Configuration #config

| Setting | Default | Purpose |
|---------|---------|---------|
| `vault_dir` | — | Path to the Obsidian vault |
| `table_name` | `"documents"` | LanceDB table name |
| `min_chunk_size` | `1000` | Minimum section size in characters before merging (250 tokens) |
| `max_chunk_size` | `2000` | Maximum chunk size in characters (500 tokens) |
| `search_limit` | `30` | Maximum candidate chunks retrieved before document consolidation and response budgeting |
| `rag_embedding_model` | `""` | Model-router embedding model |
| `rag_embedding_dimensions` | `0` | LanceDB vector dimension |
| `rag_infer_model` | `""` | Optional search-level HyDE and structured-reranking model |
| `rag_reranker_model` | `""` | Optional LanceDB reranker model |
| `rag_summary_model` | `""` | Optional whole-document summary model |

## Code References

### Core Files
- [src/mcps/rag/vault.py](../src/mcps/rag/vault.py) — Orchestrator and factory functions.
- [src/mcps/rag/interfaces.py](../src/mcps/rag/interfaces.py) — Interfaces and data models.
- [src/mcps/rag/document_processing.py](../src/mcps/rag/document_processing.py) — Parsing and chunking.
- [src/mcps/rag/database.py](../src/mcps/rag/database.py) — LanceDB vector store.
- [src/mcps/rag/search.py](../src/mcps/rag/search.py) — Search engine.
- [src/mcps/tools/obsidian_vault.py](../src/mcps/tools/obsidian_vault.py) — Obsidian MCP tools.

### Tests
- [tests/test_document_processing.py](../tests/test_document_processing.py) — Parser and chunking tests.
- [tests/test_chunk.py](../tests/test_chunk.py) — Chunk model and chunker tests.
- [tests/test_lancedb_store.py](../tests/test_lancedb_store.py) — Vector-store and typed-link lookup tests.
- [tests/test_search.py](../tests/test_search.py) — Search-engine tests.
- [tests/test_vault.py](../tests/test_vault.py) — Vault orchestrator tests.
- [tests/test_obsidian_vault_tools.py](../tests/test_obsidian_vault_tools.py) — Obsidian MCP tool and traversal tests.
- [tests/test_index_scheduler.py](../tests/test_index_scheduler.py) — Index update scheduling tests.

## Related Documentation

- [Architecture Overview](architecture_overview.md) — System design patterns.
- [Packages & Modules](packages_modules.md) — Module structure.
- [Configuration](config_environment.md) — Environment variables.
