# Research: semantic-links codebase

## Summary
`outgoing_links` is a flat `list[str]` field on `Chunk` (`interfaces.py:52`), populated by
`extract_wikilinks()` (`document_processing.py:34-52`) from either the chunk's own text or,
for summary chunks, the whole document text (`document_processing.py:77,80`). It is dumped
verbatim into LanceDB (schema derived from `Chunk` via `pydantic_to_schema()`,
`database.py:84`), round-tripped through `Chunk.model_validate()` on search
(`database.py:217`), unioned/re-sorted when neighbor chunks are merged
(`search.py:295-297,312`), rendered as `[[link]]` text by both result formatters
(`search.py:355-356,401-402`), and re-exposed as `SearchResultFullItem.outgoing_links`
by the `obsidian_search` MCP tool (`obsidian_vault.py:150,375`). No reverse/backlink index
and no link-traversal MCP tool exist anywhere in the codebase. No parsing of Dataview-style
inline-field prefixes (`field:: [[Target]]` and its hidden/visible bracket variants) exists;
`extract_wikilinks` only strips headers/display text/image-prefix, and deduplicates via
`set()`, discarding link order.

Empirically confirmed against the project's pinned `lancedb==0.25.3` (`pyproject.toml`,
`uv.lock`): `pydantic_to_schema()` raises `TypeError` on a `list[SomeBaseModel]` field
(the list-of-struct case needed to replace `outgoing_links: list[str]` with typed
`{type, target}` pairs), but a manually constructed PyArrow schema field of type
`list<struct<type: string, target: string>>` stores and queries correctly through
`AsyncTable.add()` / `.where()` / `.to_list()`. The codebase already has a precedent for
this exact "derive-then-patch" pattern: `database.py:84-93` calls `pydantic_to_schema(Chunk)`
then manually replaces/appends the `embeddings` field before `create_table()`.

## Research Findings

Files:
- `src/mcps/rag/document_processing.py` — wikilink/tag extraction, chunk construction
- `src/mcps/rag/interfaces.py` — `Chunk`, `Document`, `IVectorStore`, `IVault` and other ABCs
- `src/mcps/rag/database.py` — LanceDB-backed `IVectorStore` implementation
- `src/mcps/rag/vault.py` — `Vault` orchestrator (`IVault`), index-update cycle, DI wiring
- `src/mcps/rag/search.py` — `SemanticSearchEngine`, neighbor merging, result formatters
- `src/mcps/tools/obsidian_vault.py` — MCP tool layer (`obsidian_search` and others)
- `src/mcps/server.py` — MCP tool registration entry point

Functions/Methods under change:
- `document_processing.py:34-52` `extract_wikilinks(content: str) -> list[str]` — regex
  `r'!?\[\[((?:[^\[\]]|\[[^\[\]]*\])*?)(?:[#|][^\]]*?)?\]\]'` (line 48) matches
  `[[Target]]`/`[[Target|Display]]`/`[[Target#Header]]`/`![[Target]]`; captures only the
  bracket interior, no inline-field prefix scanning; `return list(set(filtered_matches))`
  (line 52) drops duplicates and does not preserve source order.
- `document_processing.py:63-100` `create_chunk(document: Document, content: str, position:
  int, line_offset: int = 0) -> Chunk` — line 72 `is_summary_chunk = position ==
  SUMMARY_CHUNK_POSITION` (`SUMMARY_CHUNK_POSITION = -1`, line 23); line 77
  `metadata_source_text = document.content if is_summary_chunk else content` (summary chunks
  scan the whole document, not the truncated summary string); line 80
  `outgoing_links = extract_wikilinks(metadata_source_text)`; line 92
  `Chunk(..., outgoing_links=outgoing_links, ...)`.
- `database.py:62-104` `LanceDBStore.initialize() -> None` — line 84
  `schema: pa.Schema = pydantic_to_schema(Chunk)`; lines 85-93 manually build/patch an
  `embeddings` `pa.field` and `schema.append()`/`schema.set()` it before
  `self.db.create_table(self.table_name, schema=schema)` (line 94).
- `database.py:114-141` `LanceDBStore.store(chunks: list[Chunk]) -> None` — line 134
  `chunk.model_copy(update={"embeddings": embedding}).model_dump()` per chunk; line 137
  `await self.table.add(processed_chunks)`.
- `database.py:147-221` `LanceDBStore.search(...) -> list[Chunk]` — builds `predicates:
  list[str]` and joins with `" AND "` into one `.where()` call (lines 198-208); line 217
  `return [Chunk.model_validate(result) for result in results]`.
- `database.py:333-356` `get_chunks_by_ids(ids: list[str]) -> list[Chunk]` — used by
  neighbor-merge to fetch adjacent chunks by `id`; no equivalent lookup by `source_path`+
  link-target exists yet (needed for backlink/traversal queries).
- `database.py:358-375` `get_sources_by_name(wikilink_name: str) -> list[str]` — exact
  `wikilink_name = '...'` match, `sorted({row["source_path"] for row in rows})`.
- `vault.py:285-373` `Vault.update_index() -> None` — compares `vector_store.sources()`
  (line 304) against `file_traversal.find_files()` (line 311); batches deletes/adds
  (lines 335-362); line 362 `await self.vector_store.reindex(False)` runs after any
  modification — this is the single index-time integration point where a reverse/backlink
  index would need to be (re)built.
- `vault.py:375-384` `_batch_process_files(files_to_add: list[Path]) -> None` —
  `asyncio.gather` over `_process_file` per path.
- `vault.py:386-405` `_process_file(file_path: Path) -> None` — `document_processor.process`
  → `chunker.chunk` → optional summary chunk via `create_chunk(document, summary,
  SUMMARY_CHUNK_POSITION)` (line 397) → `vector_store.store(chunks)` (line 405).
- `vault.py:407-444` `Vault.search(...) -> list[Chunk]` — builds `SearchQuery(text=query,
  tags=tags or [], scope=SearchScope.ALL, path=path)`, delegates to
  `search_engine.search()`.
- `search.py:113-146` `SemanticSearchEngine.search(query: SearchQuery) -> list[Chunk]` —
  HyDE generation, `vector_store.search()`, min-score filter, optional rerank, optional
  neighbor merge (`neighbor_offset > 0`).
- `search.py:274-323` `_merge_window(cls, document_id: str, window: tuple[int, int],
  neighbor_map: dict[str, Chunk]) -> Chunk | None` — lines 295-297:
  `outgoing_links = sorted({link for chunk in window_chunks for link in
  chunk.outgoing_links})`; line 312 `outgoing_links=list(outgoing_links)` in the
  reconstructed `Chunk`. Requires `outgoing_links` elements to be hashable and orderable.
- `search.py:326-367` `MarkdownResultFormatter.format(...)` — line 355 `if
  chunk.outgoing_links:`, line 356 `links = ", ".join(f"[[{link}]]" for link in
  chunk.outgoing_links)`.
- `search.py:370-410` `CompactResultFormatter.format(...)` — line 401-402, same pattern,
  truncated to `chunk.outgoing_links[:10]`.
- `obsidian_vault.py:356-392` `search(query: SearchQuery, ctx: Context, tags: Tags = None,
  path: PathFilter = None) -> list[SearchResultItem]` — line 365 `chunks =
  await _vault_from_context(ctx).search(...)`; lines 370-380 build one
  `SearchResultFullItem` per chunk (line 375 `outgoing_links=c.outgoing_links`), truncated to
  10 results with a trailing warning `SearchResultItem` (lines 383-386).
- `obsidian_vault.py:209-249` `register_tools(mcp: FastMCP) -> None` — registers
  `obsidian_list_files`, `obsidian_read_note`, `obsidian_search` via
  `mcp.tool(func, name=..., description=...)`; `obsidian_rename_move` registration is
  commented out (lines 227-231) though `rename_move_note` still exists (lines 294-353).
- `server.py:97-122` `DevAutomationServer.register(self)` — line 121-122: `if
  self.config.vault_dir: obsidian_vault.register_tools(self.mcp)` — the call site through
  which any new tool added inside `register_tools()` becomes reachable.

Types/Classes:
- `interfaces.py:43-68` `Chunk(BaseModel)` — `model_config = ConfigDict(populate_by_name=True)`
  (line 45); field to retype: `outgoing_links: list[str] = Field(default_factory=list)  #
  Wikilinks` (line 52); other fields: `id`, `content`, `title`, `description`, `source`,
  `tags`, `source_path`, `wikilink_name`, `modified_at`, `position`, `offset`, `file_size`,
  `embeddings: list[float] | None = None`; custom `__hash__`/`__eq__` by `id` (lines 62-68).
- `interfaces.py:30-40` `Document(BaseModel)` — `id`, `content`, `metadata`, `tags`,
  `source_path`, `wikilink_name`, `file_size`, `modified_at`.
- `interfaces.py:157-222` `IVectorStore(ABC)` — `initialize`, `store`, `search(query,
  hypotetical_document, tags, file_path, scope, limit) -> list[Chunk]` (lines 170-193),
  `delete`, `reindex(replace: bool = True)`, `get_chunks_by_ids`, `sources() -> dict[str,
  float]`, `get_sources_by_name`. No backlink/reverse-index method exists.
- `interfaces.py:251-309` `IVault(ABC)` — `initialize`, `update_index()` (docstring at
  lines 264-273 explicitly describes the compare/delete/add/reindex cycle), `search`,
  `get_file`, `list_files`. No traversal method exists.
- `obsidian_vault.py:141-154` `SearchResultItem(BaseModel)` (line 141-143, `content: str`
  only) and `SearchResultFullItem(SearchResultItem)` (lines 145-154) — field to retype:
  `outgoing_links: list[str] = Field(default_factory=list)  # Wikilinks` (line 150); other
  fields: `title`, `description`, `tags`, `source_path`, `wikilink_name`, `offset`,
  `file_size`. No `backlinks` field exists.
- `database.py:27-376` `LanceDBStore(IVectorStore)` — `db: AsyncConnection`, `table:
  AsyncTable` (lines 46-47); `__init__` takes `db_path`, `embedding_service`, `table_name=
  "chunks"`, `reranker` (lines 49-60).

Integration points:
- `search.py:295-297,312` — set-comprehension/`sorted()` over `chunk.outgoing_links` across
  merged neighbor chunks; assumes hashable, orderable elements.
- `search.py:355-356,401-402` — formatters render each `outgoing_links` element as
  `f"[[{link}]]"` (an `str`, interpolated directly).
- `obsidian_vault.py:375` — `search()` tool copies `c.outgoing_links` verbatim into
  `SearchResultFullItem.outgoing_links`.
- `database.py:84,134,217` — LanceDB schema derivation, `model_dump()` before `table.add()`,
  and `Chunk.model_validate()` after `.to_list()` all operate on the `Chunk` model as a
  whole; any field-shape change to `outgoing_links` flows through all three call sites
  automatically once `Chunk` is updated, with no separate serialization code to change.
- `server.py:121-122` → `obsidian_vault.py:209-249` `register_tools()` — the only
  registration path for a new MCP tool exposed when `config.vault_dir` is set.
- `vault.py:360-362` — `Vault.update_index()`'s `if modified: await
  self.vector_store.reindex(False)` is the sole per-index-cycle hook available today; no
  separate "after store, before reindex" hook exists for building an in-memory or
  persisted backlink structure.

Test files:
- `tests/test_document_processing.py:839-1020` — `extract_wikilinks`/`extract_content_tags`
  parametrized cases, e.g. `("[[Link with | pipe]]", ["Link with "])`
  (no inline-field-prefix cases exist).
- `tests/test_document_processing.py:399-424` —
  `test_create_summary_chunk_extracts_tags_and_links_from_whole_document`: asserts
  `set(chunk.outgoing_links) == {"Global Link", "Second Link"}` when links appear in
  `document.content` but not in the truncated summary text.
- `tests/test_lancedb_store.py:35-113` — `sample_chunks` fixture, 4 `Chunk` objects with
  `outgoing_links: list[str]`, e.g. `outgoing_links=["artificial_intelligence"]`.
- `tests/test_lancedb_store.py:439-459` — `test_search_tags_and_path_both_applied`,
  regression test for the single-`.where()`-call fix (`database.py:207-208`).
- `tests/test_search.py:10-37` — `make_chunk()` helper takes `outgoing_links: list[str] |
  None = None`.
- `tests/test_search.py:282-307` — `test_search_neighbor_merging_unions_tags_and_links`:
  center chunk `outgoing_links=["center_link"]`, neighbor `outgoing_links=["neighbor_link"]`;
  asserts merged `result[0].outgoing_links == ["center_link", "neighbor_link"]` (sorted
  union of plain strings).
- `tests/test_vault_summary_chunks.py:104-178` —
  `test_process_file_stores_summary_chunk_in_addition_to_semantic_chunks`: document body
  `"# Note\n\nBody with [[Whole Link]] and #whole-tag."`; asserts
  `set(summary_chunk.outgoing_links) == {"Whole Link"}`.
- `tests/test_obsidian_vault_tools.py:17-22,78-88` — `OBSIDIAN_TOOL_NAMES = {
  "obsidian_list_files", "obsidian_read_note", "obsidian_rename_move", "obsidian_search"}`
  checked against registered tool names; does not assert on `outgoing_links` content
  directly.
- `tests/test_vault.py:24-213` — `Vault` unit tests via Fake* implementations of every
  `IVault` collaborator interface; none reference `outgoing_links`.

Missing coverage:
- No test parses any of the 3 in-scope inline-field syntax forms
  (`field:: [[Target]]` / `(field:: [[Target]])` / `[field:: [[Target]]]`).
- No test exercises typed `{type, target}` link construction or defaulting to
  `reference` for bare `[[wikilinks]]`.
- No test exists for a reverse/backlink index (none is built).
- No test exists for a link-traversal MCP tool (none is registered).
- No test covers `obsidian_search` exposing backlinks (field does not exist).
- No LanceDB round-trip test exists for a non-`list[str]` shape of `outgoing_links`/`links`.

## Implementation Research Findings

- LanceDB schema for typed links — considered: (a) rely on
  `lancedb.pydantic.pydantic_to_schema()` to auto-derive a `list[LinkItem]` Pydantic field
  into an Arrow `list<struct<...>>` column; (b) manually construct the Arrow field for the
  typed-links column and patch it into the schema after `pydantic_to_schema(Chunk)`, the
  same way `embeddings` is already patched. Chosen: (b). Why: empirically confirmed
  (`lancedb==0.25.3`, the version pinned in `pyproject.toml`/`uv.lock`) that
  `pydantic_to_schema()` raises `TypeError: Converting Pydantic type to Arrow Type:
  unsupported type <class '...'>` for any `list[SomeBaseModel]` field — the auto-derive path
  (a) does not work. A manually built PyArrow schema field of type `list<struct<type:
  string, target: string>>`, added via the same `schema.append()`/`schema.set()` pattern
  already used at `database.py:85-93`, was empirically confirmed to create a table, accept
  rows shaped as `list[dict]` (i.e., `Chunk.model_dump()`'s natural output for a
  `list[LinkItem]` field), and round-trip through both `.add()` and `.query().where(...)
  .to_list()`.
- MCP tool registration convention for a new traversal tool — considered: none (only one
  established pattern exists in this codebase). Chosen: follow the existing
  `register_tools()` pattern at `obsidian_vault.py:209-249` — define the tool function at
  module level, add a `mcp.tool(function, name="obsidian_...", description="...")` call
  inside `register_tools()`, reachable through the existing `server.py:121-122` conditional
  registration on `config.vault_dir`. Why: this is the only tool-registration mechanism
  used anywhere in the codebase (`obsidian_list_files`, `obsidian_read_note`,
  `obsidian_search` all follow it); FastMCP's own convention documentation confirms
  `mcp.tool(func, name=..., description=...)` and `@mcp.tool(name=..., description=...)`
  are equivalent registration forms.
- MCP tool parameter modeling — considered: raw parameters vs. `Annotated[..., Field(...)]`
  aliases. Chosen: `Annotated[..., Field(description=..., ...)]` aliases, matching every
  existing parameter in `obsidian_vault.py:21-139` (`FolderPath`, `WikilinkName`, `Tags`,
  `PathFilter`, etc.). Why: this is the sole parameter-declaration convention already used
  for every registered tool in this file; FastMCP's structured-output guidance confirms
  `Annotated` + `Field` metadata is the supported mechanism for parameter descriptions/
  constraints surfaced to MCP clients.
- MCP tool return-value modeling — considered: plain `dict`/`str` vs. `pydantic.BaseModel`/
  `dataclass` return types. Chosen: `pydantic.BaseModel` subclasses (as already done for
  `SearchResultItem`/`SearchResultFullItem` at `obsidian_vault.py:141-154`, returned as
  `list[SearchResultItem]` from `search()` at line 361). Why: matches the sole existing
  return-value convention in this file; FastMCP automatically emits both human-readable
  content and machine-readable `structuredContent` for `BaseModel`/`dataclass` return
  values, requiring no extra serialization code, consistent with how `search()` already
  returns typed Pydantic models.
- Reverse (backlink) index build timing — considered: none (no existing precedent for any
  index/cache built outside `vector_store.reindex()`). Chosen: not determined by research —
  the only index-time integration point in the current codebase is
  `vault.py:360-362` (`if modified: await self.vector_store.reindex(False)`), which is the
  sole hook invoked once per `update_index()` cycle after chunks are stored.
  `[UNVERIFIED: no design/implementation-approach decision was made here per the research
  phase's scope — this file documents integration points only, not chosen designs beyond
  the storage-schema and tool-registration questions above]`.

## External References
- LanceDB Python schema/patterns reference (Context7 `/lancedb/lancedb`):
  `plugins/lancedb/skills/lancedb/references/python/patterns.md` — "Schema Design and
  Validation" section, confirms `LanceModel`/Pydantic-derived schemas as the recommended
  path for simple (non-nested) models, with manual PyArrow schemas as the documented
  alternative for "complex runtime schema requirements".
- `lancedb.pydantic` source, installed package: `.venv/lib/python3.13/site-packages/
  lancedb/pydantic.py:253-333` (`_py_type_to_arrow_type`, `_pydantic_to_arrow_type`,
  `_pydantic_type_to_arrow_type`, `pydantic_to_schema`) — read directly to confirm the
  `list[BaseModel]` unsupported-type behavior empirically reproduced above.
- FastMCP best-practices skill (`fastmcp-best-practices`, local skill reference,
  `reference_part_02.md:46-147`) — "Return Values and Structured Output" section,
  confirms `BaseModel`/`dataclass` tool return types generate both `content` and
  `structuredContent` automatically.
