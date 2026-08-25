# Phase 2: LanceDB storage and link queries [NEW_FEATURE]

## Deviations recorded during implementation

1. **Two round-trip tests passed at RED for the right reason (user decision: option 1, accept as contract guards).** `test_store_and_search_round_trips_typed_links` and `test_store_and_search_round_trips_multiple_typed_links_in_order` passed immediately because Phase 1 already placed `links`/`link_types` on `Chunk` and the existing `pydantic_to_schema`/`model_dump`/`model_validate` path round-trips them with no new code. Both assert `typed_links == [Link(type=..., target=...)]` — actual `Link` objects, order and type preserved — so they are non-vacuous regression guards for SPEC criterion #4. The remaining 9 new tests failed at RED for the correct reasons (8x `AttributeError` on the missing methods, 1x assertion failure on the missing indexes).
2. **Index-failure handling follows the plan's final explicit instruction, not the contradictory one.** The plan asked both for "the same try/except-and-log handling already applied to `tags`" (which re-raises) and for failures to be "logged and swallowed ... not raised". Implemented log-and-swallow (no raise) for the `links`/`link_types` `LabelList` indexes, since a fetch-volume-optimization index failure must not break indexing. The `tags` handler is unchanged.
3. **Import adjustments per the Phase 0 deviation-2 precedent (no unused imports).** Only `NoteLinks` was added to `tests/test_vault.py` and `tests/test_vault_summary_chunks.py` (`Link` is unused there and would trip F401). `tests/test_lancedb_store.py` imports `Link` but never `NoteLinks`: the tests assert duck-typed attributes (`.note`, `.links`) on returned objects, so the module stayed importable during RED — the same pattern Phase 3's CONFIRM_RED pre-approves.
4. **Shared helpers extracted per DRY.** `_group_rows_by_note`, `_links_from_row`, `_first_non_none`, and the module-level `_LINK_QUERY_COLUMNS` constant factor the grouping, metadata-resolution, corrupt-row handling, and column projection shared by both query methods. Rubber-duck review verdict on both RED tests and GREEN implementation: APPROVED.

Persists the two link columns, indexes them, and adds the two link-lookup methods every later phase depends on.

### RED — `tests/test_lancedb_store.py` (new tests appended after line 523)

**Source under test:** `src/mcps/rag/database.py:62`, `src/mcps/rag/database.py:358`
**Functions/methods under test:** `LanceDBStore.initialize()`, `LanceDBStore.store()`, `LanceDBStore.reindex()`, `LanceDBStore.get_notes_with_links()`, `LanceDBStore.get_notes_linking_to()`

**Fixtures:**
- `temp_db_path` → `Path` — existing, `tests/test_lancedb_store.py:26-31`, unchanged.
- `dummy_embedding_function` → `IEmbeddingService` — existing, `tests/test_lancedb_store.py:116-146`, 16-dimension SHA256-derived embeddings, unchanged.
- `lancedb_store` → `IVectorStore` — existing, `tests/test_lancedb_store.py:149-153`, `LanceDBStore(temp_db_path, dummy_embedding_function, "test_chunks")`, unchanged.
- `sample_chunks` → `list[Chunk]` — existing, `tests/test_lancedb_store.py:34-113`, as amended in Phase 1.
- `link_graph_chunks` → `list[Chunk]` — **new**, appended after `sample_chunks`. Five chunks whose non-link fields follow the `sample_chunks` pattern (`source=None`, `modified_at=datetime.now().timestamp()`, `offset=0`, `file_size=len(content)`, `tags=[]`):

  | `id` | `wikilink_name` | `source_path` | `title` | `description` | `position` | `links` | `link_types` |
  |---|---|---|---|---|---|---|---|
  | `alpha_0` | `Alpha` | `/g/alpha.md` | `Alpha Note` | `About alpha` | `0` | `["Beta"]` | `["requires"]` |
  | `alpha_1` | `Alpha` | `/g/alpha.md` | `Alpha Note` | `About alpha` | `1` | `["Gamma", "Beta"]` | `["related", "requires"]` |
  | `beta_0` | `Beta` | `/g/beta.md` | `Beta Note` | `About beta` | `0` | `["Gamma"]` | `["refines"]` |
  | `gamma_0` | `Gamma` | `/g/gamma.md` | `Gamma Note` | `About gamma` | `0` | `["Delta"]` | `["requires"]` |
  | `decoy_0` | `Decoy` | `/g/decoy.md` | `Decoy Note` | `About decoy` | `0` | `["Gamma", "Beta"]` | `["requires", "related"]` |

  `Decoy` exists specifically to expose the pushdown soundness limit: it satisfies `array_has_any(links,['Beta']) AND array_has_any(link_types,['requires'])` at the SQL level, yet its `requires` edge points at `Gamma`, not `Beta`. Any implementation that trusts the SQL predicate alone will wrongly return it.

- `lancedb_store_with_links` → `IVectorStore` — **new** async fixture, same construction as `lancedb_store_with_data` (`tests/test_lancedb_store.py:157-181`) but storing `link_graph_chunks`: `await store.initialize()`, `await store.store(link_graph_chunks)`, `await store.reindex()`, `yield store`, `await store.cleanup()`.

**Test cases:**

#### `test_store_and_search_round_trips_typed_links`
- **Use Case** SPEC success criterion #4.
- **Given:** `lancedb_store` initialized, and `sample_chunks[0]` (`chunk_1`, `links=["artificial_intelligence"]`, `link_types=["related"]`) stored.
- **When:** `await lancedb_store.search("machine learning", limit=1)`.
- **Then:** `results[0].links == ["artificial_intelligence"]`, `results[0].link_types == ["related"]`, and `results[0].typed_links == [Link(type="related", target="artificial_intelligence")]`.

#### `test_store_and_search_round_trips_multiple_typed_links_in_order`
- **Use Case** SPEC criterion #4, ordering and multiplicity.
- **Given:** `lancedb_store` initialized, and `sample_chunks[3]` (`chunk_4`, `links=["python", "data_science"]`) stored with `link_types` changed to `["requires", "related"]` via `model_copy(update=...)`.
- **When:** `await lancedb_store.search("python programming", limit=1)`.
- **Then:** `results[0].typed_links == [Link(type="requires", target="python"), Link(type="related", target="data_science")]`. Array order survives the round trip.

#### `test_reindex_creates_label_list_indexes_on_link_columns`
- **Use Case** filtering performance requirement for backlink queries.
- **Given:** `lancedb_store_with_links`.
- **When:** `await lancedb_store_with_links.table.list_indices()`.
- **Then:** the returned index list includes entries covering the `links` column and the `link_types` column, alongside the existing `tags` index. Assert on the set of indexed column names, not on index object identity.

#### `test_get_notes_with_links_aggregates_all_chunks_of_a_note`
- **Use Case** traversal correctness — one note is many chunk rows (Implementation Research, "Traversal aggregation unit").
- **Given:** `lancedb_store_with_links`; note `Alpha` has two chunk rows carrying `requires→Beta`, `related→Gamma`, `requires→Beta` between them.
- **When:** `await lancedb_store_with_links.get_notes_with_links(["Alpha"])`.
- **Then:** returns exactly one `NoteLinks` with `note == "Alpha"`, `title == "Alpha Note"`, `description == "About alpha"`, and `links == [Link(type="requires", target="Beta"), Link(type="related", target="Gamma")]` — the duplicate `requires→Beta` pair collapsed, both rows' links present.

#### `test_get_notes_with_links_returns_nothing_for_unknown_note`
- **Use Case** dangling-link decision: unknown notes are silently excluded.
- **Given:** `lancedb_store_with_links`.
- **When:** `await lancedb_store_with_links.get_notes_with_links(["Nonexistent"])`.
- **Then:** returns `[]`.

#### `test_get_notes_with_links_returns_empty_list_for_empty_input`
- **Use Case** edge case, mirrors `test_get_chunks_by_ids_returns_empty_list_for_empty_input` (`tests/test_lancedb_store.py:327-335`).
- **Given:** `lancedb_store` initialized, no data.
- **When:** `await lancedb_store.get_notes_with_links([])`.
- **Then:** returns `[]`, and no query is issued against an empty table.

#### `test_get_notes_linking_to_returns_sources_with_only_matching_edges`
- **Use Case** SPEC criterion #5 (backlinks).
- **Given:** `lancedb_store_with_links`.
- **When:** `await lancedb_store_with_links.get_notes_linking_to(["Gamma"])`.
- **Then:** returns `NoteLinks` entries for `Alpha`, `Beta`, and `Decoy` only. `Alpha`'s `links` is exactly `[Link(type="related", target="Gamma")]` — its `requires→Beta` edge is **not** included, because only edges pointing at a requested target are returned. `Beta`'s is `[Link(type="refines", target="Gamma")]`. `Decoy`'s is `[Link(type="requires", target="Gamma")]`.

#### `test_get_notes_linking_to_filters_by_relation_type_without_trusting_sql`
- **Use Case** the soundness limit of `array_has_any` (Implementation Research). This is the phase's most important test.
- **Given:** `lancedb_store_with_links`, where `Decoy` has `links=["Gamma","Beta"]` and `link_types=["requires","related"]` and therefore satisfies the SQL pre-filter for target `Beta` with type `requires`, while its actual `requires` edge points at `Gamma`.
- **When:** `await lancedb_store_with_links.get_notes_linking_to(["Beta"], relation_types=["requires"])`.
- **Then:** returns exactly one `NoteLinks`, for `Alpha`, with `links == [Link(type="requires", target="Beta")]`. `Decoy` must **not** appear. If it appears, the implementation returned SQL pre-filter results without Python pair re-verification.

#### `test_get_notes_linking_to_or_combines_relation_types`
- **Use Case** SPEC: relation types are OR-combined.
- **Given:** `lancedb_store_with_links`.
- **When:** `await lancedb_store_with_links.get_notes_linking_to(["Gamma"], relation_types=["related", "refines"])`.
- **Then:** returns entries for `Alpha` (`related→Gamma`) and `Beta` (`refines→Gamma`), and not `Decoy` (whose only `Gamma` edge is `requires`).

#### `test_get_notes_linking_to_returns_empty_list_for_empty_input`
- **Use Case** edge case.
- **Given:** `lancedb_store` initialized, no data.
- **When:** `await lancedb_store.get_notes_linking_to([])`.
- **Then:** returns `[]`.

#### `test_get_notes_linking_to_escapes_quotes_in_target_names`
- **Use Case** note names legitimately contain apostrophes; mirrors `test_get_chunks_by_ids_escapes_special_characters` (`tests/test_lancedb_store.py:338-348`).
- **Given:** `lancedb_store` initialized, with one chunk stored via `sample_chunks[0].model_copy(update={"id": "q_0", "wikilink_name": "Quoter", "links": ["O'Malley"], "link_types": ["requires"]})`.
- **When:** `await lancedb_store.get_notes_linking_to(["O'Malley"])`.
- **Then:** returns one `NoteLinks` with `note == "Quoter"` and `links == [Link(type="requires", target="O'Malley")]`. No SQL error is raised.

→ **EXPECTED: FAIL** — `NoteLinks`, `get_notes_with_links`, and `get_notes_linking_to` do not exist; the `links`/`link_types` columns are not indexed.

---

### CONFIRM_RED
Run: `test.sh 'tests/test_lancedb_store.py'`

Confirm: the new tests fail with `AttributeError` / `ImportError` for the missing symbols, and `test_reindex_creates_label_list_indexes_on_link_columns` fails on its assertion rather than erroring. The pre-existing tests in the file must all pass — they were made compatible in Phase 1. If any pre-existing test fails, or any new test passes → STOP, report, await direction.

Request user to validate. Do not move to GREEN implementation phase before user approval.

---

### GREEN — `src/mcps/rag/interfaces.py`, `src/mcps/rag/database.py`, `tests/test_vault.py`, `tests/test_vault_summary_chunks.py`

**Types to create — `src/mcps/rag/interfaces.py`, inserted immediately after `dedupe_links`:**
- `NoteLinks(BaseModel)` — fields: `note: str` (the note's `wikilink_name`), `title: str | None = None`, `description: str | None = None`, `links: list[Link] = Field(default_factory=list)`.

**Types to modify — `src/mcps/rag/interfaces.py:157-222` `IVectorStore(ABC)`, two new abstract methods appended after `get_sources_by_name` (which ends at line 222):**
- `async def get_notes_with_links(self, wikilink_names: list[str]) -> list[NoteLinks]`
  Contract: for each requested note that exists in the index, return one `NoteLinks` aggregating the outgoing links of **all** its chunk rows, deduplicated by `dedupe_links`. Requested notes absent from the index are silently omitted. An empty input returns `[]`.
- `async def get_notes_linking_to(self, targets: list[str], relation_types: list[str] | None = None) -> list[NoteLinks]`
  Contract: for each note in the index holding at least one link whose `target` is in `targets` (and, when `relation_types` is given, whose `type` is in `relation_types`), return one `NoteLinks` whose `links` contains **only** those matching edges. Non-matching edges of the same note are excluded. An empty input returns `[]`. Matching is exact on the `(type, target)` pair, never on the SQL pre-filter alone.

**Methods to modify — `src/mcps/rag/database.py`:**
- `initialize()` (lines 62-104): no change. `pydantic_to_schema(Chunk)` at line 84 derives both new `list[str]` columns natively, exactly as it already does for `tags`. Do **not** add any manual PyArrow patch for the link columns; the only manual field remains `embeddings` at lines 85-93.
- `store()` (lines 114-141): no change. `model_dump()` at line 134 emits both new columns automatically.
- `search()` (lines 147-221): no change. `Chunk.model_validate()` at line 217 restores both columns, and the Phase 1 alignment validator runs there.
- `reindex(replace: bool = True)` (lines 250-304): after the existing `tags` `LabelList` index creation (lines 296-302), create a `LabelList()` index on the `links` column and one on the `link_types` column, using the identical `create_index(column=..., config=LabelList(), wait_timeout=wait_time, replace=replace)` call shape and the same try/except-and-log handling already applied to `tags`. Index-creation failure must be logged and swallowed exactly as the existing code does, not raised.

**Imports to add — `src/mcps/rag/database.py`:** add `Link`, `NoteLinks`, and `dedupe_links` to the existing `from .interfaces import ...` statement.

**Imports to add — `tests/test_lancedb_store.py`, `tests/test_vault.py`, `tests/test_vault_summary_chunks.py`:** add `Link` and `NoteLinks` to each file's existing `from mcps.rag.interfaces import (...)` statement.

**Methods to create — `src/mcps/rag/database.py`, appended after `get_sources_by_name` (which ends at line 375):**
- `async def get_notes_with_links(self, wikilink_names: list[str]) -> list[NoteLinks]`
  Behavior: return `[]` immediately for empty input. Build one predicate `wikilink_name IN ('a','b',...)` with each name passed through `_escape_sql_string` (lines 143-145). Issue a single `self.table.query().where(predicate).select(["wikilink_name", "title", "description", "position", "links", "link_types"]).to_list()` with no row limit. Group rows by `wikilink_name`; within a group, sort the rows by `position` ascending, then concatenate each row's `zip(link_types, links)` pairs into `Link` objects in order and pass the concatenation through `dedupe_links`. Take `title` and `description` from the first row in the group that has a non-`None` value. Return one `NoteLinks` per group. A row whose `links` and `link_types` lengths disagree is a corrupt row: log a warning naming the note and skip that row rather than raising.
- `async def get_notes_linking_to(self, targets: list[str], relation_types: list[str] | None = None) -> list[NoteLinks]`
  Behavior: return `[]` immediately for empty `targets`. Build the SQL pre-filter as `array_has_any(links, ['t1','t2',...])`, each target escaped via `_escape_sql_string`; when `relation_types` is non-empty, AND in `array_has_any(link_types, ['r1',...])` likewise escaped. Join both predicates into a **single** `.where()` call — chained `.where()` calls replace each other on the async query builder (see the comment at `database.py:195-197`). Project the same six columns and issue no row limit. **Then re-verify in Python:** for every row, keep only those `(type, target)` pairs where `target in targets` and, when `relation_types` is given, `type in relation_types`. Group the rows by `wikilink_name` and sort each group by `position` ascending. For each group, collect its surviving pairs in order, `dedupe_links` them, and drop groups left with no pairs. Return one `NoteLinks` per surviving group, with `title`/`description` resolved as above. The Python filter — not the SQL predicate — determines the result; the SQL predicate only limits how many rows are transferred.

**Test-double updates (required — `IVectorStore` gains two abstract methods, so every subclass fails to instantiate without them):**
- `tests/test_vault.py:42-75` `FakeVectorStore` — add `async def get_notes_with_links(self, wikilink_names: list[str]) -> list[NoteLinks]: return []` and `async def get_notes_linking_to(self, targets: list[str], relation_types: list[str] | None = None) -> list[NoteLinks]: return []`.
- `tests/test_vault_summary_chunks.py:40-74` `FakeVectorStore` — add the same two stubs.
- `tests/test_search_agent.py:35` uses an `AsyncMock`, not a subclass; no change needed.

→ **EXPECTED: PASS** — all tests from the RED phase.

---

### VERIFY_GREEN

Run, in order:
1. `test.sh 'tests/test_lancedb_store.py'`
2. `test.sh 'tests/test_vault.py'`
3. `test.sh 'tests/test_vault_summary_chunks.py'`
4. `compile.sh 'src/mcps/rag/interfaces.py' 'src/mcps/rag/database.py'`
5. `lint.sh 'src/mcps/rag/interfaces.py' 'src/mcps/rag/database.py' 'tests/test_lancedb_store.py' 'tests/test_vault.py' 'tests/test_vault_summary_chunks.py'`

Confirm: all pass, with `test_get_notes_linking_to_filters_by_relation_type_without_trusting_sql` green. Any regression → STOP, report, await direction.

**Manual check:** none. If a `.vault_db` directory built by a previous version of the code is present in a development vault, the new columns will not exist in its table and `initialize()` will bind to the old schema. Deleting it is covered in Phase 5; it does not affect these tests, which build a fresh temporary database per test.
