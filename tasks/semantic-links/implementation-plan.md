# Semantic Link Types Implementation Plan

## Goal

Replace the flat untyped `Chunk.outgoing_links: list[str]` with typed `(type, target)` link data parsed from Dataview-style inline fields, expose typed outgoing links and backlinks through `obsidian_search`, and add an MCP tool that walks the typed link graph forward and backward to a caller-supplied depth filtered by caller-supplied relation types.

**Recorded deviation from SPEC.md.** Success criterion #2 in `tasks/semantic-links/SPEC.md` says a bare `[[Some Note]]` defaults to type `reference`. The user selected `related` as the default type instead, because `reference::` is already in use as an explicit field for external sources in the vault (~1,478 occurrences) and reusing it as the implicit default would make the two indistinguishable. Every occurrence of "default type" in this plan means the literal string `related`. All other success criteria are implemented verbatim.

## Out of Scope

- Parsing relation fields from YAML frontmatter (body only)
- Extracting or typing external (non-wikilink) markdown-link targets — `implements:: [ColBERT](https://x.com)` yields no link
- Enforcing, validating, or filtering by a fixed relation-type taxonomy or reserved-field-name list
- Batch reclassification of existing untyped links in vault note files
- Query rewriting, tag inference, or other `SearchAgent` / `SemanticSearchEngine` behavior
- Automatic migration of existing LanceDB tables. The column shape changes incompatibly; the operator deletes `<vault>/.vault_db` once and lets the next `update_index()` rebuild. This is documented in Phase 5, not implemented in code.
- A persisted or in-memory backlink index. Backlinks are computed per query by SQL pushdown plus Python verification.

---

## Research Findings

Files:
- `src/mcps/rag/interfaces.py` — data models (`Chunk`, `Document`, `Metadata`) and all ABCs
- `src/mcps/rag/document_processing.py` — wikilink/tag extraction, chunk construction, chunkers
- `src/mcps/rag/database.py` — `LanceDBStore(IVectorStore)`, LanceDB schema, search, indexes
- `src/mcps/rag/search.py` — `SemanticSearchEngine`, HyDE, neighbor merging, result formatters
- `src/mcps/rag/vault.py` — `Vault(IVault)` orchestrator, index-update cycle, `create_vault` factory
- `src/mcps/tools/obsidian_vault.py` — MCP tool layer, parameter aliases, result models
- `src/mcps/server.py` — MCP tool registration entry point
- `src/mcps/config.py:25` — `table_name: str = "documents"`

Functions/Methods under change:
- `src/mcps/rag/document_processing.py:34-52` `extract_wikilinks(content: str) -> list[str]` — regex at line 48 is `r'!?\[\[((?:[^\[\]]|\[[^\[\]]*\])*?)(?:[#|][^\]]*?)?\]\]'`; matches `[[T]]`, `[[T|Display]]`, `[[T#Header]]`, `![[T]]`, `[[T [with] brackets]]`; captures only the interior target; line 52 `return list(set(filtered_matches))` deduplicates and discards source order.
- `src/mcps/rag/document_processing.py:55-60` `extract_content_tags(text: str) -> list[str]` — regex `r'#([a-zA-Z][a-zA-Z0-9_-]*)'`, returns `list(set(matches))`. Unchanged by this plan; cited because it shares the field-name character class.
- `src/mcps/rag/document_processing.py:63-100` `create_chunk(document: Document, content: str, position: int, line_offset: int = 0) -> Chunk` — line 70 `chunk_id = f"{document.id}_{position}"`; line 72 `is_summary_chunk = position == SUMMARY_CHUNK_POSITION`; line 77 `metadata_source_text = document.content if is_summary_chunk else content`; line 80 `outgoing_links = extract_wikilinks(metadata_source_text)`; line 84 `combined_tags = list(set(document.tags + content_tags))`; line 92 passes `outgoing_links=outgoing_links` into the returned `Chunk`.
- `src/mcps/rag/document_processing.py:23` `SUMMARY_CHUNK_POSITION = -1`.
- `src/mcps/rag/database.py:62-104` `LanceDBStore.initialize() -> None` — line 84 `schema: pa.Schema = pydantic_to_schema(Chunk)`; lines 85-93 build `emb_field = pa.field("embeddings", pa.list_(pa.float16(), self.embedding_service.ndims()))` and `schema.append()`/`schema.set()` it; line 94 `create_table`; line 97 `await self.reindex(True)`.
- `src/mcps/rag/database.py:114-141` `LanceDBStore.store(chunks: list[Chunk]) -> None` — lines 126-131 embed `f"Doc: {c.wikilink_name} Content:{c.content}"`; line 134 `chunk.model_copy(update={"embeddings": embedding}).model_dump()`; line 137 `await self.table.add(processed_chunks)`.
- `src/mcps/rag/database.py:143-145` `LanceDBStore._escape_sql_string(val: str) -> str` — `val.replace("'", "''")`.
- `src/mcps/rag/database.py:147-221` `LanceDBStore.search(...) -> list[Chunk]` — lines 195-197 carry the comment that chained `.where()` calls REPLACE each other on the async Rust query builder; lines 198-208 collect `predicates: list[str]` (`array_has_all(tags, [...])` line 203, `source_path LIKE '...%'` line 206) and issue exactly one `.where(" AND ".join(predicates))` at line 208; line 217 `return [Chunk.model_validate(result) for result in results]`.
- `src/mcps/rag/database.py:250-304` `LanceDBStore.reindex(replace: bool = True) -> None` — line 269 `optimize(...)`; lines 272-286 FTS indexes on `["content", "title", "description"]` with `FTS(base_tokenizer="simple", max_token_length=30)`; lines 296-302 `create_index(column="tags", config=LabelList(), wait_timeout=wait_time, replace=replace)`.
- `src/mcps/rag/database.py:333-356` `get_chunks_by_ids(ids: list[str]) -> list[Chunk]` — `self.table.query().where("id IN (...)").to_list()`.
- `src/mcps/rag/database.py:358-375` `get_sources_by_name(wikilink_name: str) -> list[str]` — exact `wikilink_name = '...'` match, returns `sorted({row["source_path"] for row in rows})`.
- `src/mcps/rag/search.py:273-323` `SemanticSearchEngine._merge_window(cls, document_id: str, window: tuple[int, int], neighbor_map: dict[str, Chunk]) -> Chunk | None` — lines 295-297 `outgoing_links = sorted({link for chunk in window_chunks for link in chunk.outgoing_links})`; lines 306-320 rebuild a `Chunk` with `outgoing_links=list(outgoing_links)` at line 312; lines 321-322 restore `_relevance_score` via `object.__setattr__`.
- `src/mcps/rag/search.py:326-367` `MarkdownResultFormatter.format(...)` — line 355 `if chunk.outgoing_links:`, line 356 `", ".join(f"[[{link}]]" for link in chunk.outgoing_links)`. Dead code: no production call site.
- `src/mcps/rag/search.py:370-410` `CompactResultFormatter.format(...)` — lines 401-402 same pattern with `[:10]`; line 394 calls `MarkdownResultFormatter._format_score`. Dead code: no production call site.
- `src/mcps/rag/vault.py:285-373` `Vault.update_index() -> None` — lines 360-362 `if modified: logger.info("Rebuilding indexes"); await self.vector_store.reindex(False)` is the sole per-cycle index hook.
- `src/mcps/rag/vault.py:386-405` `Vault._process_file(file_path: Path) -> None` — line 389 `if self.document_summary_generator is not None and document.content.strip() and len(chunks) > 2:` gates summary-chunk creation; line 397 `chunks = [create_chunk(document, summary, SUMMARY_CHUNK_POSITION), *chunks]`; line 405 `await self.vector_store.store(chunks)`.
- `src/mcps/rag/vault.py:406-443` `Vault.search(query: str, tags: list[str] | None = None, path: str | None = None) -> list[Chunk]` — builds `SearchQuery(text=query, tags=tags or [], scope=SearchScope.ALL, path=path)`.
- `src/mcps/rag/vault.py:579-634` `create_vault(...)` — DI factory; constructs no result formatter.
- `src/mcps/tools/obsidian_vault.py:209-249` `register_tools(mcp: FastMCP) -> None` — `obsidian_list_files` at 210-217, `obsidian_read_note` at 218-226, `obsidian_rename_move` registration COMMENTED OUT at 227-231, `obsidian_search` at 232-247.
- `src/mcps/tools/obsidian_vault.py:356-392` `search(query: SearchQuery, ctx: Context, tags: Tags = None, path: PathFilter = None) -> list[SearchResultItem]` — line 365 fetches chunks from the vault; line 375 `outgoing_links=c.outgoing_links`; line 381 `for c in chunks[:10]`; lines 383-386 append a truncation-warning `SearchResultItem`.
- `src/mcps/server.py:121-122` — `if self.config.vault_dir: obsidian_vault.register_tools(self.mcp)`.

Types/Classes:
- `src/mcps/rag/interfaces.py:43-68` `Chunk(BaseModel)` — line 45 `model_config = ConfigDict(populate_by_name=True)`; fields `id: str`, `content: str`, `title: str | None`, `description: str | None`, `source: str | None = None`, **line 52 `outgoing_links: list[str] = Field(default_factory=list)`**, `tags: list[str] = Field(default_factory=list)`, `source_path: str`, `wikilink_name: str`, `modified_at: float`, `position: int`, `offset: int`, `file_size: int`, `embeddings: list[float] | None = None`; lines 62-63 `__hash__` by `id`; lines 65-68 `__eq__` by `id`.
- `src/mcps/rag/interfaces.py:22-27` `Metadata(BaseModel)` — `title: str | None`, `description: str | None`, `source: str | None`.
- `src/mcps/rag/interfaces.py:30-40` `Document(BaseModel)` — `id`, `content`, `metadata`, `tags`, `source_path`, `wikilink_name`, `file_size`, `modified_at`.
- `src/mcps/rag/interfaces.py:157-222` `IVectorStore(ABC)` — `initialize` 160-163, `store` 165-168, `search` 170-193, `delete` 195-198, `reindex(replace: bool = True)` 200-207, `get_chunks_by_ids` 209-212, `sources` 214-217, `get_sources_by_name` 219-222. No backlink or link-lookup method exists.
- `src/mcps/rag/interfaces.py:233-239` `IResultFormatter(ABC)` — single abstract `format`. No production implementation is wired anywhere.
- `src/mcps/rag/interfaces.py:251-309` `IVault(ABC)` — `initialize` 258-261, `update_index` 263-273, `search` 275-283, `get_file` 285-296, `list_files` 298-309. No traversal method exists.
- `src/mcps/tools/obsidian_vault.py:141-143` `SearchResultItem(BaseModel)` — `content: str`.
- `src/mcps/tools/obsidian_vault.py:145-154` `SearchResultFullItem(SearchResultItem)` — `title: str | None`, `description: str | None`, `tags: list[str]`, **line 150 `outgoing_links: list[str] = Field(default_factory=list)`**, `source_path: str`, `wikilink_name: str`, `offset: int`, `file_size: int`. No `backlinks` field.
- `src/mcps/tools/obsidian_vault.py:21-139` — parameter aliases `FolderPath` 21-33, `WikilinkName` 35-46, `CurrentNotePath` 48-59, `NewNotePath` 61-74, `SearchQuery` 76-88, `Tags` 90-101, `PathFilter` 103-114, `ReadOffset` 116-127, `ReadLimit` 129-139. All are `Annotated[T, Field(description=...)]`.
- `src/mcps/rag/vault.py:222-231` `Vault.__init__(self, vault_path, file_traversal, document_processor, chunker, vector_store, search_engine, document_summary_generator=None, batch_size=8)` — accepts **no** formatter parameter. Line 243 of its docstring nevertheless documents a nonexistent `result_formatter` parameter.
- `src/mcps/rag/vault.py:38` imports `IResultFormatter`; `src/mcps/rag/vault.py:47-51` imports `MarkdownResultFormatter` at line 49. Neither is used in the module body.
- `src/mcps/rag/search.py:13` imports `IResultFormatter`.

Integration points:
- `src/mcps/rag/database.py:84` / `:134` / `:217` — schema derivation, `model_dump()` before `table.add()`, `Chunk.model_validate()` after `.to_list()`. Any `Chunk` field-shape change flows through all three with no separate serialization code.
- `src/mcps/rag/search.py:295-297,312` — set comprehension over `chunk.outgoing_links`; requires hashable, orderable elements.
- `src/mcps/rag/search.py:355-356,401-402` — formatters interpolate each element into `f"[[{link}]]"`.
- `src/mcps/tools/obsidian_vault.py:375` — copies `c.outgoing_links` verbatim into the tool result model.
- `src/mcps/server.py:121-122` → `src/mcps/tools/obsidian_vault.py:209-249` — the only registration path for a new MCP tool.
- `src/mcps/rag/vault.py:360-362` — the sole per-index-cycle hook.

Test files:
- `tests/test_document_processing.py:839-865` — `test_extract_wikilinks_various_patterns`, `test_extract_wikilinks_duplicates_removed`, `test_extract_wikilinks_empty_content`.
- `tests/test_document_processing.py:1005-1016` — `test_extract_wikilinks_parametrized`, params `[("[[Simple]]", ["Simple"]), ("[[Link|Display]]", ["Link"]), ("[[Multi Word Link]]", ["Multi Word Link"]), ("[[Link1]] and [[Link2]]", ["Link1","Link2"]), ("No links here", []), ("[[]]", []), ("[[Link with | pipe]]", ["Link with "])]`; all assert via `set(links) == set(expected_links)`.
- `tests/test_document_processing.py:399-424` — `test_create_summary_chunk_extracts_tags_and_links_from_whole_document`, asserts `set(chunk.outgoing_links) == {"Global Link", "Second Link"}`.
- `tests/test_chunk.py:18-36` — `test_chunk_accepts_wikilink_name_offset_and_size`, builds via `Chunk.model_validate({...})`; `:38-56` parametrized `test_chunk_requires_wikilink_name_offset_and_size`.
- `tests/test_lancedb_store.py:26-31` `temp_db_path` fixture; `:34-113` `sample_chunks` fixture — 4 `Chunk`s, `chunk_1` has `outgoing_links=["artificial_intelligence"]`, `chunk_2` `["neural_networks"]`, `chunk_3` `["language_models"]`, `chunk_4` `["python", "data_science"]`; `:116-146` `dummy_embedding_function` (16-dim SHA256-derived); `:149-153` `lancedb_store`; `:157-181` `lancedb_store_with_data`; `:476-494` module-level `make_chunk(source_path, modified_at, idx=0)` with `outgoing_links=[]`.
- `tests/test_search.py:10-37` — `make_chunk(doc_id, content="content", relevance_score=None, position=0, source_path=None, tags=None, outgoing_links=None, offset=0) -> Chunk`; `:282-307` `test_search_neighbor_merging_unions_tags_and_links` asserts `result[0].outgoing_links == ["center_link", "neighbor_link"]`.
- `tests/test_vault_summary_chunks.py:40-74` `FakeVectorStore(IVectorStore)` — implements exactly the 8 current abstract methods, records `stored_chunks`; `:104-118` `document` fixture with content `"# Note\n\nBody with [[Whole Link]] and #whole-tag."`; `:160-178` asserts `set(summary_chunk.outgoing_links) == {"Whole Link"}`.
- `tests/test_vault.py:42-75` `FakeVectorStore(IVectorStore)` — implements exactly the 8 current abstract methods; `:79` `FakeSearchEngine(ISearchEngine)`.
- `tests/test_obsidian_vault_tools.py:17-22` `OBSIDIAN_TOOL_NAMES = {"obsidian_list_files", "obsidian_read_note", "obsidian_rename_move", "obsidian_search"}`; `:25-27` `FakeContext`; `:30-73` `FakeVault` (duck-typed, does **not** subclass `IVault`); `:76-88` registration test asserting `OBSIDIAN_TOOL_NAMES <= tool_names`.
- `pyproject.toml:55-63` `[tool.pytest.ini_options]` — `testpaths = ["tests"]`, `asyncio_mode = "auto"`. No `tests/conftest.py` exists.

Missing coverage:
- No test parses any of the 3 in-scope inline-field syntax forms.
- No test exercises typed `(type, target)` construction or defaulting of bare wikilinks.
- No test covers link source-order preservation (current code deliberately discards it via `set()`).
- No test exists for a backlink query, a link-traversal tool, or `obsidian_search` backlinks — none of these exist.
- No LanceDB round-trip test for any link shape other than `list[str]`.
- `MarkdownResultFormatter` and `CompactResultFormatter` have zero test coverage and zero production call sites.
- **Pre-existing failure on a clean tree:** `test.sh tests/test_obsidian_vault_tools.py` reports `1 failed, 12 passed`. `tests/test_obsidian_vault_tools.py:88` asserts `OBSIDIAN_TOOL_NAMES <= tool_names`, but `OBSIDIAN_TOOL_NAMES` lists `obsidian_rename_move`, whose registration is commented out at `src/mcps/tools/obsidian_vault.py:227-231`. Error: `AssertionError: assert {...} <= {...}` / `Extra items in the left set: 'obsidian_rename_move'`. Phase 0 corrects this so later phases have a clean baseline.

---

## Implementation Research Findings

- **Typed-link column shape** — considered: (a) a `list[Link]` Pydantic field auto-derived by `lancedb.pydantic.pydantic_to_schema()`; (b) a manually built PyArrow `list<struct<type,target>>` field patched into the schema the way `embeddings` already is at `database.py:85-93`; (c) two positionally aligned `list[str]` columns, `links` and `link_types`. **Chosen: (c).** Why: `pydantic_to_schema()` raises `TypeError` on any `list[BaseModel]` field in the pinned `lancedb==0.25.3`, so (a) is impossible. (b) works but requires hand-maintained Arrow schema code and cannot be `LabelList`-indexed as a struct list. (c) is derived natively by `pydantic_to_schema()` with zero schema-patching code (it deletes rather than adds the schema patch the prior research artifact proposed), both columns are `LabelList`-indexable exactly like the existing `tags` column, and chunk writes are atomic per row so the two arrays cannot drift apart in storage. Cost: the alignment invariant must be enforced in Python — addressed by a `Chunk` model validator.
- **Backlink index** — considered: (a) a persisted reverse-index table rebuilt in `update_index()`; (b) an in-memory dict rebuilt at index time; (c) per-query SQL pushdown against the `links` column. **Chosen: (c).** Why: at the documented scale envelope (`src/mcps/rag/CLAUDE.md`: 2000-3000 documents, ~10k chunks, ~1 query/minute, 5s acceptable latency) a single filtered scan is well inside budget, while (a) adds a second table plus a consistency problem across the delete/add/reindex cycle and (b) is lost on every process restart and unavailable to the tool layer before the first index run.
- **Pushdown predicate** — considered: `array_has(links, 'X')`, `list_contains(links, 'X')`, `array_has_any(links, ['X','Y'])`. **All three verified working against the pinned `lancedb==0.25.3`.** **Chosen: `array_has_any`** because it expresses the whole frontier in one predicate. **Critical soundness limit, verified experimentally:** SQL cannot correlate positions across the two arrays. Given rows `A{links:[B,C], types:[requires,related]}`, `D{links:[C], types:[requires]}`, `E{links:[D,C], types:[requires,reference]}`, the predicate `array_has_any(links,['C']) AND array_has_any(link_types,['requires'])` returns `A, D, E` — including `E`, whose `requires` edge points at `D`, not at `C`. The predicate is therefore **sound but inexact**: it never drops a true match, and it may admit false ones. Every caller must re-verify the exact `(type, target)` pair in Python after fetching. Pushdown is a fetch-volume optimization only, never the filter of record.
- **Column projection** — `AsyncQuery.select([...])` verified working on the pinned version in combination with `.where(...)` and `.limit(...)`. Link and traversal queries project only `wikilink_name`, `title`, `description`, `links`, `link_types`, avoiding transfer of the embedding vectors.
- **Unlimited queries** — verified that `table.query().where(...).to_list()` without `.limit()` returns all matching rows on the pinned version. Backlink queries deliberately issue no row limit and cap results in Python after grouping by note; a SQL `LIMIT` would truncate arbitrary chunk rows and could silently drop a note's only backlink.
- **Inline-field anchor regex** — considered: (a) a permissive `(?:^|[\s(\[])` prefix; (b) an anchored `(?:^[ \t]*(?:[-*+]\s+)?|[(\[])` prefix. **Chosen: (b), with `re.MULTILINE`.** Why: (a) was empirically shown to produce false positives — it types `prose not::a field but [[Y]]` as `not` and would type `std::vector`. (b) was verified to accept all three in-scope forms plus list items and indented lines, and to reject both false positives:

  | Input | Field names matched |
  |---|---|
  | `requires:: [[RAG Basics]]` | `['requires']` |
  | `Text (requires:: [[RAG Basics]]) more` | `['requires']` |
  | `Text [requires:: [[RAG Basics]]] more` | `['requires']` |
  | `- requires:: [[RAG Basics]]` | `['requires']` |
  | `  requires:: [[Indented]]` | `['requires']` |
  | `custom-field:: [[X]]` | `['custom-field']` |
  | `see the C++ std::vector [[docs]]` | `[]` |
  | `prose not::a field but [[Y]]` | `[]` |

- **Target extraction inside a field** — the existing wikilink interior pattern is reused unchanged, verified to yield `['RAG Basics']` for `requires:: [[RAG Basics#Section|Alias]]`, `['A','B']` for `requires:: [[A]] and [[B]]`, and `[]` for `implements:: [ColBERT](https://x.com)`.
- **Traversal aggregation unit** — LanceDB returns chunk rows, not notes; one note is many rows with different `links` arrays, and a note only receives a summary chunk when it yields more than two chunks (`vault.py:389`), so the summary chunk cannot be used as a per-note shortcut. Every link query therefore groups rows by `wikilink_name` and unions their `(type, target)` pairs before any dedup, cap, or visited-set logic is applied.
- **MCP tool registration** — follow the sole existing pattern: module-level `async def`, registered by a `mcp.tool(func, name=..., description=...)` call inside `register_tools()` (`obsidian_vault.py:209-249`), reachable via `server.py:121-122`.
- **MCP parameter modeling** — `Annotated[T, Field(description=...)]` aliases, matching all nine existing aliases at `obsidian_vault.py:21-139`.
- **Traversal bounds** — depth constrained by `Field(default=1, ge=1, le=3)`; a `MAX_TRAVERSAL_NODES = 100` cap on **distinct notes**. Rationale: branching factor across ~4,781 measured vault wikilinks makes unbounded depth-3 traversal unpredictable in size, and `ge`/`le` validation is enforced by FastMCP before the tool body runs.

---

## Conventions

- **AUTO-STOP:** if an EXPECTED outcome contradicts the actual outcome → STOP, report, await direction.
- **Pause after each phase for human verification.** Template A phases additionally pause at CONFIRM_RED; Template B phases at CONFIRM_GREEN.
- Run tests only via `test.sh '<file>'`, lint only via `lint.sh '<file>'`, compile-check only via `compile.sh '<file>'`. Do not invent test or lint commands.
- Never use absolute paths inside project files.
- All `file:line` references are relative to the plan's baseline (current `HEAD`). Phases that delete or insert lines shift the references in later phases for that same file; each phase enumerates its own edits, and every insertion point is additionally identified by the enclosing symbol name, which does not shift.
- Default link type is the literal string `related` throughout (see the deviation note in Goal).
- Link filtering by relation type is **always** re-verified in Python on exact `(type, target)` pairs. SQL predicates on `link_types` are a fetch-volume optimization only and are never the filter of record.

---

## Phases

### Phase 0: Baseline correction [REFACTOR]
- **Goal:** Remove unreachable formatter classes (`IResultFormatter`, `MarkdownResultFormatter`, `CompactResultFormatter`) and correct stale test data (`obsidian_rename_move` in `OBSIDIAN_TOOL_NAMES`) to establish a clean green test baseline.
- **Details:** [implementation-phase-0.md](implementation-phase-0.md)

### Phase 1: Typed link model and extraction [NEW_FEATURE]
- **Goal:** Introduce `Link` data model, `DEFAULT_LINK_TYPE`, link deduplication logic (`dedupe_links`), and parallel aligned `links`/`link_types` columns on `Chunk`. Replace `extract_wikilinks` with `extract_typed_links` supporting Dataview inline fields while updating chunk creation, window merging, and search result models.
- **Details:** [implementation-phase-1.md](implementation-phase-1.md)

### Phase 2: LanceDB storage and link queries [NEW_FEATURE]
- **Goal:** Persist and index parallel link columns in LanceDB using `LabelList` indexes, and implement `IVectorStore.get_notes_with_links()` and `IVectorStore.get_notes_linking_to()` with SQL pushdown and exact Python pair re-verification.
- **Details:** [implementation-phase-2.md](implementation-phase-2.md)

### Phase 3: `obsidian_search` typed links and backlinks [NEW_FEATURE]
- **Goal:** Implement `IVault.get_backlinks()` to resolve incoming links for notes, add `backlinks: list[Link]` to `SearchResultFullItem`, and surface typed outgoing links and backlinks in `obsidian_search` results with batched retrieval.
- **Details:** [implementation-phase-3.md](implementation-phase-3.md)

### Phase 4: `obsidian_traverse_relations` tool [NEW_FEATURE]
- **Goal:** Implement multi-hop bidirectional graph traversal in `IVault.traverse_relations()`, add `TraversalNode` and `TraversalResult` models, define parameter aliases, and register the `obsidian_traverse_relations` FastMCP tool with bounds enforcement and type filtering.
- **Details:** [implementation-phase-4.md](implementation-phase-4.md)

### Phase 5: Documentation sync [REFACTOR]
- **Goal:** Synchronize project documentation (`docs/rag_obsidian_integration.md`, `docs/architecture_overview.md`) with the new typed links architecture, schema migration instructions, and tool descriptions while removing references to deleted formatters.
- **Details:** [implementation-phase-5.md](implementation-phase-5.md)

---

## Final Verification

After all phases are completed, execute the following full verification sequence:

### Automated Test Suite
Run every test file across the codebase:
1. `test.sh 'tests/test_document_processing.py'`
2. `test.sh 'tests/test_chunk.py'`
3. `test.sh 'tests/test_search.py'`
4. `test.sh 'tests/test_lancedb_store.py'`
5. `test.sh 'tests/test_vault.py'`
6. `test.sh 'tests/test_vault_summary_chunks.py'`
7. `test.sh 'tests/test_obsidian_vault_tools.py'`
8. `test.sh 'tests/test_server.py'`

### Compilation and Type Checking
Run compilation checks across all modified and touched modules:
`compile.sh 'src/mcps/rag/interfaces.py' 'src/mcps/rag/document_processing.py' 'src/mcps/rag/database.py' 'src/mcps/rag/search.py' 'src/mcps/rag/vault.py' 'src/mcps/tools/obsidian_vault.py' 'src/mcps/server.py'`

### Linting
Run linter across all source and test files:
`lint.sh 'src/mcps/rag/interfaces.py' 'src/mcps/rag/document_processing.py' 'src/mcps/rag/database.py' 'src/mcps/rag/search.py' 'src/mcps/rag/vault.py' 'src/mcps/tools/obsidian_vault.py' 'src/mcps/server.py' 'tests/test_document_processing.py' 'tests/test_chunk.py' 'tests/test_search.py' 'tests/test_lancedb_store.py' 'tests/test_vault.py' 'tests/test_vault_summary_chunks.py' 'tests/test_obsidian_vault_tools.py' 'tests/test_server.py'`

### Manual End-to-End Verification
1. Start server against a re-indexed vault: `uv run mcps`
2. Test `obsidian_search` to verify `outgoing_links` and `backlinks` are populated with `Link(type, target)` structures.
3. Test `obsidian_traverse_relations` with depth 1 and 2 on notes with known typed links.
4. Verify filtering with `relation_types` parameter narrows the graph walk appropriately.
