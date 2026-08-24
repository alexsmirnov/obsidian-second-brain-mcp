# Phase 1: Typed link model and extraction [NEW_FEATURE]

Introduces the `Link` type, the dedup rule, the two aligned `Chunk` columns with their alignment invariant, and the Dataview inline-field parser. This phase does not touch the database or the tool layer beyond one mechanical field rename.

### RED — `tests/test_document_processing.py` (new tests appended after line 1016), `tests/test_chunk.py` (new tests appended after line 56), `tests/test_search.py` (modification at lines 282-307)

**Source under test:** `src/mcps/rag/document_processing.py:34`, `src/mcps/rag/interfaces.py:43`, `src/mcps/rag/search.py:273`
**Functions/methods under test:** `extract_typed_links()`, `dedupe_links()`, `Chunk` model validation, `Chunk.typed_links`, `create_chunk()`, `SemanticSearchEngine._merge_window()`

**Fixtures:**
- No new pytest fixtures are required for the extraction tests — inputs are string literals passed directly.
- `document` → `Document` — reuse the existing construction style at `tests/test_document_processing.py:399-424`. For `test_create_chunk_defaults_bare_wikilinks_to_related` build a `Document` with `id="doc"`, `content="Body with [[Alpha]] and requires:: [[Beta]]."`, `metadata=Metadata(title=None, description=None, source=None)`, `tags=[]`, `source_path="notes/doc.md"`, `wikilink_name="doc"`, `file_size=48`, `modified_at=1234.5`.
- `tests/test_search.py:10-37` `make_chunk` — change its `outgoing_links: list[str] | None = None` parameter to `links: list[Link] | None = None`, and have it populate `links=[l.target for l in links or []]` and `link_types=[l.type for l in links or []]` on the constructed `Chunk`. Every existing call site that passes `outgoing_links=[...]` must be updated; the only ones are at `tests/test_search.py:289` and `:296`.

**Test cases:**

#### `test_extract_typed_links_standalone_field`
- **Use Case** SPEC success criterion #1, standalone form.
- **Given:** the string `"requires:: [[RAG Basics]]"`.
- **When:** `extract_typed_links("requires:: [[RAG Basics]]")`.
- **Then:** returns `[Link(type="requires", target="RAG Basics")]`.

#### `test_extract_typed_links_hidden_parenthesised_field`
- **Use Case** SPEC success criterion #1, hidden form.
- **Given:** the string `"Prose (requires:: [[RAG Basics]]) continues."`.
- **When:** `extract_typed_links("Prose (requires:: [[RAG Basics]]) continues.")`.
- **Then:** returns `[Link(type="requires", target="RAG Basics")]`.

#### `test_extract_typed_links_visible_bracketed_field`
- **Use Case** SPEC success criterion #1, visible form.
- **Given:** the string `"Prose [requires:: [[RAG Basics]]] continues."`.
- **When:** `extract_typed_links("Prose [requires:: [[RAG Basics]]] continues.")`.
- **Then:** returns `[Link(type="requires", target="RAG Basics")]`.

#### `test_extract_typed_links_list_item_field`
- **Use Case** Dataview list-item convention used throughout the vault.
- **Given:** the string `"- requires:: [[RAG Basics]]"`.
- **When:** `extract_typed_links("- requires:: [[RAG Basics]]")`.
- **Then:** returns `[Link(type="requires", target="RAG Basics")]`.

#### `test_extract_typed_links_bare_wikilink_defaults_to_related`
- **Use Case** SPEC success criterion #2 (as amended: default is `related`).
- **Given:** the string `"See [[Some Note]] for detail."`.
- **When:** `extract_typed_links("See [[Some Note]] for detail.")`.
- **Then:** returns `[Link(type="related", target="Some Note")]`.

#### `test_extract_typed_links_arbitrary_field_name_is_kept_verbatim`
- **Use Case** SPEC success criterion #3.
- **Given:** the string `"custom-field:: [[X]]"`.
- **When:** `extract_typed_links("custom-field:: [[X]]")`.
- **Then:** returns `[Link(type="custom-field", target="X")]`. The type is not normalised, lowercased, or rejected.

#### `test_extract_typed_links_field_types_only_first_following_wikilink`
- **Use Case** scope decision: a standalone field types exactly one target.
- **Given:** the string `"requires:: [[A]] and [[B]]"`.
- **When:** `extract_typed_links("requires:: [[A]] and [[B]]")`.
- **Then:** returns `[Link(type="requires", target="A"), Link(type="related", target="B")]`, in that order.

#### `test_extract_typed_links_strips_header_and_alias_from_target`
- **Use Case** existing wikilink semantics must survive typing.
- **Given:** the string `"requires:: [[RAG Basics#Section|Alias]]"`.
- **When:** `extract_typed_links("requires:: [[RAG Basics#Section|Alias]]")`.
- **Then:** returns `[Link(type="requires", target="RAG Basics")]`.

#### `test_extract_typed_links_ignores_external_markdown_link_target`
- **Use Case** SPEC out-of-scope item: non-wikilink targets are not extracted.
- **Given:** the string `"implements:: [ColBERT](https://x.com)"`.
- **When:** `extract_typed_links("implements:: [ColBERT](https://x.com)")`.
- **Then:** returns `[]`.

#### `test_extract_typed_links_ignores_double_colon_in_prose`
- **Use Case** false-positive guard (edge case, verified in Implementation Research).
- **Given:** two strings: `"see the C++ std::vector [[docs]]"` and `"prose not::a field but [[Y]]"`.
- **When:** `extract_typed_links(...)` on each.
- **Then:** returns `[Link(type="related", target="docs")]` and `[Link(type="related", target="Y")]` respectively. Neither `vector` nor `not` appears as a type.

#### `test_extract_typed_links_preserves_source_order`
- **Use Case** ordering contract that replaces the current `set()`-based behavior.
- **Given:** the string `"[[Zulu]] then requires:: [[Alpha]] then [[Mike]]"`.
- **When:** `extract_typed_links(...)`.
- **Then:** returns exactly `[Link(type="related", target="Zulu"), Link(type="requires", target="Alpha"), Link(type="related", target="Mike")]`. This is a list equality assertion, not a set comparison.

#### `test_extract_typed_links_empty_content_returns_empty_list`
- **Use Case** edge case, mirrors existing `test_extract_wikilinks_empty_content`.
- **Given:** the string `""`.
- **When:** `extract_typed_links("")`.
- **Then:** returns `[]`.

#### `test_extract_typed_links_empty_wikilink_is_skipped`
- **Use Case** edge case, mirrors the existing `("[[]]", [])` parametrized case.
- **Given:** the string `"requires:: [[]]"`.
- **When:** `extract_typed_links("requires:: [[]]")`.
- **Then:** returns `[]`.

#### `test_dedupe_links_removes_exact_duplicate_pairs`
- **Use Case** dedup decision, identical-pair case.
- **Given:** `[Link(type="requires", target="A"), Link(type="requires", target="A")]`.
- **When:** `dedupe_links(...)`.
- **Then:** returns `[Link(type="requires", target="A")]`.

#### `test_dedupe_links_explicit_type_absorbs_default_to_same_target`
- **Use Case** dedup decision, precedence case. This is the concrete-fixture case that must not be written vacuously.
- **Given:** `[Link(type="related", target="A"), Link(type="requires", target="A"), Link(type="related", target="B")]` — note `A` appears both bare and explicitly typed, `B` only bare.
- **When:** `dedupe_links(...)`.
- **Then:** returns exactly `[Link(type="requires", target="A"), Link(type="related", target="B")]`. The `related`-to-`A` entry is dropped; the `related`-to-`B` entry survives because `B` carries no explicit type.

#### `test_dedupe_links_keeps_two_distinct_explicit_types_to_same_target`
- **Use Case** dedup decision, multi-edge case.
- **Given:** `[Link(type="requires", target="A"), Link(type="refines", target="A")]`.
- **When:** `dedupe_links(...)`.
- **Then:** returns both, in source order: `[Link(type="requires", target="A"), Link(type="refines", target="A")]`.

#### `test_dedupe_links_preserves_first_occurrence_order`
- **Use Case** dedup decision, ordering case.
- **Given:** `[Link(type="related", target="Z"), Link(type="requires", target="A"), Link(type="related", target="Z")]`.
- **When:** `dedupe_links(...)`.
- **Then:** returns `[Link(type="related", target="Z"), Link(type="requires", target="A")]` — `Z` keeps its original first position.

#### `test_chunk_rejects_misaligned_link_arrays`
- **Use Case** storage-shape invariant (parallel arrays must stay aligned).
- **Given:** the valid payload from `tests/test_chunk.py:18-36`, modified so `links=["A", "B"]` and `link_types=["requires"]`.
- **When:** `Chunk.model_validate(payload)`.
- **Then:** raises `pydantic.ValidationError`.

#### `test_chunk_typed_links_zips_parallel_arrays`
- **Use Case** read path for the two aligned columns.
- **Given:** a `Chunk` built with `links=["A", "B"]`, `link_types=["requires", "related"]`, all other fields as in `tests/test_chunk.py:18-36`.
- **When:** accessing `chunk.typed_links`.
- **Then:** returns `[Link(type="requires", target="A"), Link(type="related", target="B")]`.

#### `test_chunk_defaults_link_arrays_to_empty`
- **Use Case** backward-compatible construction for the many test fixtures that specify no links.
- **Given:** the payload from `tests/test_chunk.py:18-36` with neither `links` nor `link_types` supplied.
- **When:** `Chunk.model_validate(payload)`.
- **Then:** succeeds; `chunk.links == []`, `chunk.link_types == []`, `chunk.typed_links == []`.

#### `test_create_chunk_stores_typed_links_from_chunk_content`
- **Use Case** SPEC criteria #1-#3 at the chunk-construction boundary.
- **Given:** the `document` fixture described above.
- **When:** `create_chunk(document, "Body with [[Alpha]] and requires:: [[Beta]].", 0)`.
- **Then:** the returned chunk has `links == ["Alpha", "Beta"]` and `link_types == ["related", "requires"]`.

#### `test_create_summary_chunk_extracts_typed_links_from_whole_document`
- **Use Case** summary chunks scan the whole document, not the summary string (`document_processing.py:77`). This replaces `tests/test_document_processing.py:399-424`.
- **Given:** a `Document` whose `content` contains `"requires:: [[Global Link]]"` and `"[[Second Link]]"`, and a summary string containing neither.
- **When:** `create_chunk(document, summary_text, SUMMARY_CHUNK_POSITION)`.
- **Then:** `chunk.typed_links == [Link(type="requires", target="Global Link"), Link(type="related", target="Second Link")]`.

#### `test_search_neighbor_merging_unions_tags_and_links` (rewrite of `tests/test_search.py:282-307`)
- **Use Case** `_merge_window` must union typed links pairwise, applying the same dedup rule.
- **Given:** `center = make_chunk("doc", content="center", relevance_score=0.8, position=1, tags=["center"], links=[Link(type="requires", target="Shared")])` and `neighbor = make_chunk("doc", content="neighbor", position=2, tags=["neighbor", "center"], links=[Link(type="related", target="Shared"), Link(type="related", target="Other")])`.
- **When:** `await engine.search(SearchQuery(text="query", tags=[]))` with `neighbor_offset=1` and a vector store returning `center` as the search result and `[center, neighbor]` as fetchable chunks.
- **Then:** `result[0].tags == ["center", "neighbor"]` (unchanged behavior) and `result[0].typed_links == [Link(type="requires", target="Shared"), Link(type="related", target="Other")]` — the neighbor's bare `related`-to-`Shared` edge is absorbed by the center's explicit `requires`-to-`Shared` edge, and `Other` survives.

**Also update, without adding new assertions** (these are mechanical fixture edits required for the suite to run at all — every `Chunk(...)` construction that passes `outgoing_links=`):
- `tests/test_lancedb_store.py:34-113` `sample_chunks` — replace `outgoing_links=["artificial_intelligence"]` with `links=["artificial_intelligence"], link_types=["related"]`; `["neural_networks"]` → `links=["neural_networks"], link_types=["related"]`; `["language_models"]` → `links=["language_models"], link_types=["related"]`; `["python", "data_science"]` → `links=["python", "data_science"], link_types=["related", "related"]`.
- `tests/test_lancedb_store.py:476-494` `make_chunk` — replace `outgoing_links=[]` with `links=[], link_types=[]`.
- `tests/test_vault_summary_chunks.py:160-178` — replace the assertion `set(summary_chunk.outgoing_links) == {"Whole Link"}` with `summary_chunk.typed_links == [Link(type="related", target="Whole Link")]`. The `document` fixture at `:104-118` keeps its content `"# Note\n\nBody with [[Whole Link]] and #whole-tag."`.
- `tests/test_document_processing.py:839-865` and `:1005-1016` — delete `test_extract_wikilinks_various_patterns`, `test_extract_wikilinks_duplicates_removed`, `test_extract_wikilinks_empty_content`, and `test_extract_wikilinks_parametrized`. `extract_wikilinks` no longer exists after this phase; the target-shape cases they covered (`[[Simple]]`, `[[Link|Display]]`, `[[Multi Word Link]]`, two links in one string, no links, `[[]]`, `[[Link with | pipe]]`) are re-covered by the `extract_typed_links` cases above. Add one parametrized case to `test_extract_typed_links_*` for `("[[Link with | pipe]]", [Link(type="related", target="Link with ")])` to preserve that exact existing behavior.

→ **EXPECTED: FAIL** — `Link`, `dedupe_links`, `extract_typed_links`, `Chunk.links`, `Chunk.link_types`, and `Chunk.typed_links` do not exist; imports of `Link` from `mcps.rag.interfaces` raise `ImportError`.

---

### CONFIRM_RED
Run:
1. `test.sh 'tests/test_document_processing.py'`
2. `test.sh 'tests/test_chunk.py'`
3. `test.sh 'tests/test_search.py'`

Confirm: failures in all three, and record the exact error messages. Every new test must fail for a *missing-symbol* reason (`ImportError`, `AttributeError`, `NameError`) or a genuine assertion failure — a test that passes because it asserts nothing about the new shape is vacuous. In particular `test_dedupe_links_explicit_type_absorbs_default_to_same_target` and `test_extract_typed_links_preserves_source_order` must fail on their assertions once the symbols exist, not merely on import. If any new test passes → STOP, report, await direction.

Request user to validate. Do not move to GREEN implementation phase before user approval.

---

### GREEN — `src/mcps/rag/interfaces.py`, `src/mcps/rag/document_processing.py`, `src/mcps/rag/search.py`, `src/mcps/tools/obsidian_vault.py`

**Types to create — `src/mcps/rag/interfaces.py`, inserted after `Metadata` (which ends at line 27) and before `Document` (which begins at line 30):**
- `Link(BaseModel)` — fields: `type: str`, `target: str`. `model_config = ConfigDict(frozen=True)` so instances are hashable and usable in sets and dict keys.

**Constants to create — `src/mcps/rag/interfaces.py`, immediately above `Link`:**
- `DEFAULT_LINK_TYPE = "related"` — the type assigned to a wikilink carrying no inline-field prefix.

**Functions to create — `src/mcps/rag/interfaces.py`, immediately after `Link`:**
- `dedupe_links(links: Iterable[Link]) -> list[Link]`
  Behavior: returns links in order of first occurrence, with two rules applied. First, drop any `(type, target)` pair already seen. Second, drop any surviving link whose `type` equals `DEFAULT_LINK_TYPE` when some other link in the same input has the same `target` and a different type. Links with two distinct non-default types to one target both survive. Requires `from collections.abc import Iterable`.

**Types to modify — `src/mcps/rag/interfaces.py:43-68` `Chunk`:**
- Remove field `outgoing_links: list[str]` (line 52).
- Add field `links: list[str] = Field(default_factory=list)` — link targets, positionally aligned with `link_types`.
- Add field `link_types: list[str] = Field(default_factory=list)` — link types, positionally aligned with `links`.
- Add `@model_validator(mode="after")` method `_link_arrays_aligned(self) -> "Chunk"` — raises `ValueError` when `len(self.links) != len(self.link_types)`, naming both lengths in the message. This is the single enforcement point of the parallel-array invariant; it fires on `model_validate` of DB rows, on reconstruction in `_merge_window`, and on hand-built test chunks alike. Requires `from pydantic import model_validator` added to the existing pydantic import.
- Add read-only `@property` `typed_links(self) -> list[Link]` — returns `[Link(type=t, target=n) for t, n in zip(self.link_types, self.links, strict=True)]`.
- `__hash__`/`__eq__` at lines 62-68 are unchanged.

**Functions to modify — `src/mcps/rag/document_processing.py`:**
- Delete `extract_wikilinks` (lines 34-52) including its module-level `wikilink_pattern` local at line 48.
- Add module-level compiled patterns near line 23, beside `SUMMARY_CHUNK_POSITION`:
  - the wikilink pattern, unchanged in content from the deleted line 48: `r'!?\[\[((?:[^\[\]]|\[[^\[\]]*\])*?)(?:[#|][^\]]*?)?\]\]'`
  - the inline-field anchor pattern: `r'(?:^[ \t]*(?:[-*+]\s+)?|[(\[])([a-zA-Z][a-zA-Z0-9_-]*)[ \t]*::'`, compiled with `re.MULTILINE`.
- Create `extract_typed_links(content: str) -> list[Link]` at the position vacated by `extract_wikilinks` (line 34).
  Behavior: scan `content` once for both patterns and produce links in ascending order of the wikilink's position in the source string. For each wikilink match whose captured interior is non-blank after stripping: determine its type by checking whether the closest preceding inline-field match ends at or before the wikilink's start position **and** no other wikilink match lies between that field and this one — if so the type is that field's captured name, otherwise the type is `DEFAULT_LINK_TYPE`. A field therefore types exactly the first wikilink that follows it. Blank interiors (`[[]]`) are skipped entirely and do not consume a preceding field. Return `dedupe_links(...)` over the resulting list, so duplicate pairs collapse and explicit types absorb default ones. Requires importing `Link`, `DEFAULT_LINK_TYPE`, and `dedupe_links` from `.interfaces`.
- Modify `create_chunk` (lines 63-100): replace line 80 `outgoing_links = extract_wikilinks(metadata_source_text)` with a call to `extract_typed_links(metadata_source_text)`, and replace the `outgoing_links=outgoing_links` argument at line 92 with `links=[link.target for link in typed_links]` and `link_types=[link.type for link in typed_links]`. Lines 70-77 and 83-84 are unchanged; summary chunks continue to scan `document.content`.

**Functions to modify — `src/mcps/rag/search.py:273-323` `_merge_window`:**
- Replace lines 295-297. Instead of a sorted set of strings, build `merged = dedupe_links(link for chunk in window_chunks for link in chunk.typed_links)`, iterating `window_chunks` in their existing order so first-occurrence ordering follows chunk position.
- Replace line 312 `outgoing_links=list(outgoing_links)` with `links=[link.target for link in merged]` and `link_types=[link.type for link in merged]`.
- Lines 306-311 and 313-322 (including the `_relevance_score` restoration) are unchanged.
- Add `dedupe_links` to the existing `from .interfaces import (...)` statement.

**Types to modify — `src/mcps/tools/obsidian_vault.py:145-154` `SearchResultFullItem`:**
- Change field at line 150 from `outgoing_links: list[str]` to `outgoing_links: list[Link] = Field(default_factory=list)`. The field name is deliberately kept; only its element type changes. Add `Link` to the existing `from mcps.rag.interfaces import ...` statement.
- At `src/mcps/tools/obsidian_vault.py:375`, replace `outgoing_links=c.outgoing_links` with `outgoing_links=c.typed_links`. No other change to `search()` in this phase — `backlinks` arrives in Phase 3.

→ **EXPECTED: PASS** — all tests from the RED phase.

---

### VERIFY_GREEN

Run, in order:
1. `test.sh 'tests/test_document_processing.py'`
2. `test.sh 'tests/test_chunk.py'`
3. `test.sh 'tests/test_search.py'`
4. `test.sh 'tests/test_vault_summary_chunks.py'`
5. `test.sh 'tests/test_vault.py'`
6. `compile.sh 'src/mcps/rag/interfaces.py' 'src/mcps/rag/document_processing.py' 'src/mcps/rag/search.py' 'src/mcps/tools/obsidian_vault.py'`
7. `lint.sh 'src/mcps/rag/interfaces.py' 'src/mcps/rag/document_processing.py' 'src/mcps/rag/search.py' 'src/mcps/tools/obsidian_vault.py'`

Confirm: all pass. `tests/test_lancedb_store.py` is **expected to fail** at this point only if its fixtures were not updated as instructed — if it fails for any reason other than a pre-existing LanceDB table on disk, STOP and report. Any other regression → STOP, report, await direction.

**Manual check:** none — no runtime surface changes until Phase 3.
