# Phase 3: `obsidian_search` typed links and backlinks [NEW_FEATURE]

Surfaces typed outgoing links (already carried on the chunk since Phase 1) plus newly computed backlinks in the search tool's results.

### RED — `tests/test_obsidian_vault_tools.py` (new tests appended after line 88)

**Source under test:** `src/mcps/tools/obsidian_vault.py:356`, `src/mcps/rag/vault.py:406`
**Functions/methods under test:** `obsidian_vault.search()`, `Vault.get_backlinks()`

**Fixtures:**
- `FakeContext` → existing, `tests/test_obsidian_vault_tools.py:25-27`, unchanged.
- `FakeVault` → existing, `tests/test_obsidian_vault_tools.py:30-73`, extended with:
  - a new instance attribute `self.backlinks: dict[str, list[Link]] = {}` set in `__init__`;
  - a new method `async def get_backlinks(self, wikilink_names: list[str]) -> dict[str, list[Link]]` returning `{name: self.backlinks.get(name, []) for name in wikilink_names}`;
  - a new instance attribute `self.get_backlinks_calls: list[list[str]] = []`, appended to on each call, so the batching test can assert on it.
  `FakeVault` is duck-typed and does not subclass `IVault`, so adding methods to the ABC does not break it; these additions are needed only because the tool will call them.
- `search_result_chunk` → `Chunk` — built inline per test with `id="doc_0"`, `content="body"`, `title="Doc"`, `description="A doc"`, `source_path="notes/doc.md"`, `wikilink_name="Doc"`, `modified_at=1234.5`, `position=0`, `offset=0`, `file_size=4`, `tags=[]`, and per-test `links`/`link_types`.

**Test cases:**

#### `test_search_exposes_typed_outgoing_links`
- **Use Case** SPEC success criterion #5, outgoing half.
- **Given:** a `FakeVault` whose `search_results` is a single chunk with `links=["RAG Basics", "LanceDB"]` and `link_types=["requires", "related"]`, and whose `backlinks` is empty.
- **When:** `await obsidian_vault.search("query", FakeContext(vault))`.
- **Then:** `result[0].outgoing_links == [Link(type="requires", target="RAG Basics"), Link(type="related", target="LanceDB")]`.

#### `test_search_exposes_typed_backlinks`
- **Use Case** SPEC success criterion #5, incoming half.
- **Given:** a `FakeVault` whose `search_results` is a single chunk with `wikilink_name="Doc"` and no links, and whose `backlinks` is `{"Doc": [Link(type="refines", target="RAG 2.0")]}`.
- **When:** `await obsidian_vault.search("query", FakeContext(vault))`.
- **Then:** `result[0].backlinks == [Link(type="refines", target="RAG 2.0")]` — where `target` names the note on the other end, i.e. the note that points at `Doc`.

#### `test_search_requests_backlinks_for_all_returned_notes_in_one_call`
- **Use Case** performance decision: one backlink query per search, not one per result.
- **Given:** a `FakeVault` whose `search_results` contains three chunks with `wikilink_name` values `"A"`, `"B"`, and `"A"` again (two chunks from the same note).
- **When:** `await obsidian_vault.search("query", FakeContext(vault))`.
- **Then:** `vault.get_backlinks_calls` has length 1, and its single entry contains `"A"` and `"B"` exactly once each — duplicates collapsed before the call.

#### `test_search_caps_backlinks_per_note`
- **Use Case** result-size bound decision (20 backlinks per note).
- **Given:** a `FakeVault` whose `search_results` is one chunk with `wikilink_name="Doc"`, and whose `backlinks` is `{"Doc": [Link(type="related", target=f"N{i}") for i in range(25)]}`.
- **When:** `await obsidian_vault.search("query", FakeContext(vault))`.
- **Then:** `len(result[0].backlinks) == 20`, and those 20 are the first 20 in the supplied order (`N0` … `N19`).

#### `test_search_returns_empty_backlinks_when_note_has_none`
- **Use Case** edge case.
- **Given:** a `FakeVault` with one search-result chunk and an empty `backlinks` mapping.
- **When:** `await obsidian_vault.search("query", FakeContext(vault))`.
- **Then:** `result[0].backlinks == []`. No exception, no `None`.

#### `test_search_truncation_warning_item_is_unaffected_by_backlinks`
- **Use Case** regression guard for existing behavior at `obsidian_vault.py:381-386`.
- **Given:** a `FakeVault` whose `search_results` contains 12 chunks.
- **When:** `await obsidian_vault.search("query", FakeContext(vault))`.
- **Then:** the result has 11 items — 10 `SearchResultFullItem`s plus the trailing warning `SearchResultItem` — and backlinks were requested only for the 10 notes actually returned, not all 12.

→ **EXPECTED: FAIL** — `SearchResultFullItem.backlinks` and `Vault.get_backlinks` do not exist.

---

### CONFIRM_RED
Run: `test.sh 'tests/test_obsidian_vault_tools.py'`

Confirm: the new tests fail on missing `backlinks` / `get_backlinks`. The 13 pre-existing tests must still pass. `test_search_exposes_typed_outgoing_links` must fail on its assertion, not on an import — the typed shape landed in Phase 1, so if it passes immediately, verify it is asserting `Link` objects and not strings before treating it as non-vacuous; a pass here for the *right* reason is acceptable and must be reported, not silently accepted. If any other new test passes → STOP, report, await direction.

Request user to validate. Do not move to GREEN implementation phase before user approval.

---

### GREEN — `src/mcps/rag/interfaces.py`, `src/mcps/rag/vault.py`, `src/mcps/tools/obsidian_vault.py`

**Constants to create — `src/mcps/tools/obsidian_vault.py`, beside `UPDATE_INTERVAL` (line 157):**
- `MAX_BACKLINKS_PER_NOTE = 20`.

**Types to modify — `src/mcps/tools/obsidian_vault.py:145-154` `SearchResultFullItem`:**
- Add field `backlinks: list[Link] = Field(default_factory=list)` immediately after `outgoing_links`. Each entry's `target` is the note that links **to** this result; its `type` is the type of that incoming edge.

**Types to modify — `src/mcps/rag/interfaces.py:251-309` `IVault(ABC)`, one new abstract method appended after `list_files` (which ends at line 309):**
- `async def get_backlinks(self, wikilink_names: list[str]) -> dict[str, list[Link]]`
  Contract: for each requested note name, return the incoming links pointing at it, keyed by that note name. Each returned `Link` has `type` set to the incoming edge's type and `target` set to the **source** note's `wikilink_name`. Notes with no incoming links map to `[]`; every requested name is present as a key. An empty input returns `{}`.

**Imports to add — `src/mcps/rag/vault.py`:** add `Link`, `NoteLinks`, and `dedupe_links` to the existing `from .interfaces import ...` statement at lines 31-45 (from which `IResultFormatter` was removed in Phase 0).

**Imports to add — `tests/test_obsidian_vault_tools.py`:** add `Link` to the existing `from mcps.rag.interfaces import ...` statement.

**Methods to create — `src/mcps/rag/vault.py`, appended after `Vault.search` (which ends at line 443):**
- `async def get_backlinks(self, wikilink_names: list[str]) -> dict[str, list[Link]]`
  Behavior: return `{}` for empty input. Issue exactly one `await self.vector_store.get_notes_linking_to(wikilink_names)` call — no relation-type filter, no row limit. Invert the result: for each returned `NoteLinks` (a source note) and each of its links (whose `target` is one of the requested names), append `Link(type=link.type, target=source.note)` to the bucket for `link.target`. Deduplicate each bucket by `(type, target)` via `dedupe_links`, so a source note with several chunks mentioning the same edge appears once — summary chunks carry the whole document's links and would otherwise double every edge (`document_processing.py:77`). Initialise every requested name to `[]` so all keys are present.

**Methods to modify — `src/mcps/tools/obsidian_vault.py:356-392` `search()`:**
- After line 381's `chunks[:10]` slice determines the returned set, collect the distinct `wikilink_name` values of exactly those chunks, preserving first-appearance order, and issue one `await _vault_from_context(ctx).get_backlinks(names)` call.
- Set `outgoing_links=c.typed_links` (already done in Phase 1) and `backlinks=backlink_map.get(c.wikilink_name, [])[:MAX_BACKLINKS_PER_NOTE]` when building each `SearchResultFullItem`.
- The truncation-warning `SearchResultItem` at lines 383-386 is unchanged and receives no backlinks.
- Update the `obsidian_search` tool description at `obsidian_vault.py:232-247` to state that results carry typed outgoing links and typed backlinks, each an object with `type` and `target`, and that `target` names the note on the other end of the edge.

→ **EXPECTED: PASS** — all tests from the RED phase.

---

### VERIFY_GREEN

Run, in order:
1. `test.sh 'tests/test_obsidian_vault_tools.py'`
2. `test.sh 'tests/test_vault.py'`
3. `compile.sh 'src/mcps/rag/interfaces.py' 'src/mcps/rag/vault.py' 'src/mcps/tools/obsidian_vault.py'`
4. `lint.sh 'src/mcps/rag/interfaces.py' 'src/mcps/rag/vault.py' 'src/mcps/tools/obsidian_vault.py' 'tests/test_obsidian_vault_tools.py'`

Confirm: all pass. Any regression → STOP, report, await direction.

**Manual check:** `uv run mcps` against a vault whose `.vault_db` has been deleted and re-indexed; call `obsidian_search` and confirm each result carries `outgoing_links` and `backlinks` as arrays of `{type, target}` objects.
