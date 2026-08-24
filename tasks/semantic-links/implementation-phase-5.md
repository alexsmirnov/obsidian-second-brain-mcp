# Phase 5: Documentation sync [REFACTOR]

Brings the docs mind map in line with the code. No source or test changes.

### CONFIRM_GREEN — full suite

**Source under test:** `docs/rag_obsidian_integration.md`, `docs/architecture_overview.md`, `docs/index.md`
**Functions/methods under test:** none — this phase changes documentation only.

**Fixtures:** none.

**Test cases:** the entire suite as it stands after Phase 4.

→ **EXPECTED: PASS** — run `test.sh 'tests/test_document_processing.py'`, `test.sh 'tests/test_chunk.py'`, `test.sh 'tests/test_search.py'`, `test.sh 'tests/test_lancedb_store.py'`, `test.sh 'tests/test_vault.py'`, `test.sh 'tests/test_vault_summary_chunks.py'`, `test.sh 'tests/test_obsidian_vault_tools.py'`, `test.sh 'tests/test_server.py'` and record the totals as the baseline. Any failure → STOP, fix first.

Request user to validate. Do not proceed to MAINTAIN_GREEN before user approval.

---

### MAINTAIN_GREEN — `docs/rag_obsidian_integration.md`, `docs/architecture_overview.md`

**`docs/index.md`:** no change. Every touched document is already a node in the mind map; no new node is introduced.

**`docs/rag_obsidian_integration.md`:**
- Line 10: extend the "Obsidian-native" bullet to name typed relation links alongside wikilink extraction.
- Line 22: add `obsidian_traverse_relations` to the Tools Layer list in the architecture diagram.
- Line 50: delete the `| **IResultFormatter** | ... |` table row — the interface was removed in Phase 0.
- Lines 48-52: update the `IVectorStore` and `IVault` row line ranges to their post-Phase-4 values, and note in the `IVectorStore` row that it also provides link and backlink lookup.
- Lines 66-78 (Chunk data model): replace the `outgoing_links` bullet at line 70 with two bullets — `links`: link targets, and `link_types`: positionally aligned link types — and state that `Chunk.typed_links` zips them into `Link` objects. Add a `Link` sub-entry describing `type` and `target`.
- Lines 113-127 ("Wikilink Extraction"): retitle to cover typed relation links. Document the three in-scope Dataview inline-field forms (`field:: [[T]]`, `(field:: [[T]])`, `[field:: [[T]]]`), the list-item form, the field-name character class `[a-zA-Z][a-zA-Z0-9_-]*`, that a field types exactly the first wikilink that follows it, that bare wikilinks default to type `related`, and that source order is now preserved. Replace the closing sentence "Returns note name only, automatically deduplicated." with a statement of the dedup rule: duplicates collapse on `(type, target)`, and an explicit type absorbs a default-typed edge to the same target.
- Lines 147-156 ("Chunk Creation"): replace "Extracts wikilinks from chunk content" with "Extracts typed relation links from chunk content".
- Lines 189-194 ("Indexes"): add the `LabelList` indexes on `links` and `link_types` to the existing `tags` entry.
- Add a subsection under "Vector Store and Search" documenting `get_notes_with_links` and `get_notes_linking_to`, and stating explicitly that the `array_has_any` SQL predicate is a sound-but-inexact pre-filter and that exact `(type, target)` matching always happens in Python afterwards.
- Line 284 of the Search Flow diagram: remove the `MarkdownResultFormatter.format()` box and the "Formatted Markdown String" terminal, replacing them with the structured `list[SearchResultItem]` the tool actually returns.
- Line 291: delete the "Result formatter: [search.py:326-412]" reference.
- Add a "Relation Traversal" subsection documenting `obsidian_traverse_relations`: both directions always, depth 1-3, OR-combined free-form `relation_types`, 100-distinct-note cap with a truncation warning, each note reported once at its shallowest depth, dangling targets excluded silently.
- Lines 327-335 (Tests): add `tests/test_chunk.py` and `tests/test_obsidian_vault_tools.py`.
- Add a "Schema Migration" note: the `links`/`link_types` columns replace `outgoing_links` incompatibly; there is no automatic migration. An operator upgrading an existing deployment deletes the `.vault_db` directory inside the vault once, after which the next `update_index()` cycle rebuilds the table with the new schema. State this as a required one-time manual step, with the path expressed relative to the vault root, never as an absolute path.

**`docs/architecture_overview.md`:**
- Line 54: delete the `**IResultFormatter** [src/mcps/rag/interfaces.py:234-240]` entry.

→ **EXPECTED: PASS** — the same baseline test totals, unchanged. Documentation edits cannot affect them; re-running is a guard against accidental source edits.

---

### VERIFY_GREEN

Run the same eight `test.sh` invocations from CONFIRM_GREEN.

Confirm: identical totals to the baseline. Any regression means a source file was edited in this phase → STOP, report, await direction.

**Manual check:** follow `docs/index.md` to `docs/rag_obsidian_integration.md` and confirm every `file:line` reference it now contains resolves to the symbol it claims, and that no reference to `IResultFormatter`, `MarkdownResultFormatter`, `CompactResultFormatter`, `extract_wikilinks`, or `outgoing_links` remains anywhere under `docs/`.
