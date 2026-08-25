# Phase 0: Baseline correction [REFACTOR]

## Deviations recorded during implementation

1. **CONFIRM_GREEN found a second pre-existing failure the plan's research missed.** In addition to the predicted `test_obsidian_vault_tools.py` failure, 4 neighbor-merge tests in `tests/test_search.py` failed on a clean tree: `test_search_neighbor_offset_one_fetches_adjacent_chunks`, `test_search_overlapping_neighbors_merge_into_single_window`, `test_search_non_overlapping_neighbors_remain_separate`, `test_search_neighbor_boundary_clamps_to_zero`. Root cause: commit `6b42cca` introduced tests expecting merged-chunk ids in `{document_id}_{min_position}_{max_position}` format while the implementation at `search.py:_merge_window` produces `id=f"{document_id}_{min_position}"` — these tests never passed. **User decision: fix the tests** (option 2), codifying current runtime behavior. The 4 assertions now expect `{document_id}_{min_position}` (`tests/test_search.py:208` `"doc_0"`, `:233` `"doc_2"`, `:260` `{"doc_1", "doc_5"}`, `:275` `"doc_0"`). Baseline is then exactly as the plan predicted: search/vault fully green, `test_obsidian_vault_tools.py` failing only on `obsidian_rename_move`.
2. **`SearchResult` also removed from the `search.py` import.** The plan said to leave every other name intact, but `SearchResult` was used only by the deleted `CompactResultFormatter`; keeping it would violate the phase's own VERIFY_GREEN requirement of "no unused-import warning". No other usage of `SearchResult` exists outside `interfaces.py`.
3. **Lint is not fully clean on the baseline and this phase does not change that.** Pre-existing violations unrelated to the deleted code remain in both the baseline and the result: E501 line-length in `search.py`/`vault.py`, F841 `max_position` in `_merge_window` (unused variable, pre-existing), F401 `datetime` imports in `vault.py` and both test files, B009/RUF005 in `tests/test_search.py`. Phase 0 removes the two F401 unused-import warnings tied to the deleted symbols and introduces zero new violations. Fixing the rest is out of scope for this plan.

Removes dead code and one stale test datum so that every later phase starts from a fully green suite. No externally observable behavior changes: the deleted classes have no production call site and the corrected test datum names a tool that is not registered.

### CONFIRM_GREEN — `tests/test_search.py`, `tests/test_vault.py`, `tests/test_obsidian_vault_tools.py`

**Source under test:** `src/mcps/rag/search.py:326`, `src/mcps/rag/interfaces.py:233`, `src/mcps/rag/vault.py:222`
**Functions/methods under test:** `SemanticSearchEngine.search()`, `Vault.update_index()`, `register_tools()`

**Fixtures:** all existing, unchanged — `tests/test_search.py:10-37` `make_chunk`, `tests/test_search.py:40-50` `make_vector_store`, `tests/test_vault.py:42-75` `FakeVectorStore`, `tests/test_obsidian_vault_tools.py:30-73` `FakeVault`.

**Test cases:** the existing suites in these three files, unmodified. `tests/test_search.py` and `tests/test_vault.py` must be fully green. `tests/test_obsidian_vault_tools.py` has one known pre-existing failure, recorded below.

→ **EXPECTED:** `test.sh 'tests/test_search.py'` all pass; `test.sh 'tests/test_vault.py'` all pass; `test.sh 'tests/test_obsidian_vault_tools.py'` → **1 failed, 12 passed**, the failure being `test_obsidian_tools_are_registered_when_vault_dir_is_configured` at `tests/test_obsidian_vault_tools.py:88` with `Extra items in the left set: 'obsidian_rename_move'`.

If `tests/test_search.py` or `tests/test_vault.py` shows any failure, or if `tests/test_obsidian_vault_tools.py` fails in any way other than exactly that one test with exactly that message → STOP, report, await direction.

Request user to validate this baseline. Do not proceed to MAINTAIN_GREEN before user approval.

---

### MAINTAIN_GREEN

**Deletions — `src/mcps/rag/search.py`:**
- Delete `MarkdownResultFormatter` in its entirety (lines 326-367).
- Delete `CompactResultFormatter` in its entirety (lines 370-410). Note it calls `MarkdownResultFormatter._format_score` at line 394, so the two must be deleted together.
- Remove `IResultFormatter` from the import at line 13. Leave every other name in that import statement intact.

**Deletions — `src/mcps/rag/interfaces.py`:**
- Delete the `IResultFormatter(ABC)` class in its entirety (lines 233-239).

**Deletions — `src/mcps/rag/vault.py`:**
- Remove `IResultFormatter` from the import at line 38.
- Remove `MarkdownResultFormatter` from the multi-name import at lines 47-51 (it is line 49). `HypotheticalDocumentGenerator` and `SemanticSearchEngine` remain.
- Delete the stale docstring line 243, `result_formatter: Service for formatting search results`, inside `Vault.__init__`'s docstring. It documents a parameter that has never existed in the signature at lines 222-231.

**Test-data correction — `tests/test_obsidian_vault_tools.py:17-22`:**
- Remove the `"obsidian_rename_move"` entry from `OBSIDIAN_TOOL_NAMES`. Its registration is commented out at `src/mcps/tools/obsidian_vault.py:227-231`; the set must describe tools that are actually registered. Do not uncomment the registration — `rename_move_note` (`obsidian_vault.py:294-353`) is out of scope for this plan.

**No other file is touched in this phase.** Do not delete `rename_move_note` itself.

→ **EXPECTED: PASS** — same tests as the baseline, plus the previously failing registration test now passing.

---

### VERIFY_GREEN

Run, in order:
1. `test.sh 'tests/test_search.py'`
2. `test.sh 'tests/test_vault.py'`
3. `test.sh 'tests/test_obsidian_vault_tools.py'`
4. `compile.sh 'src/mcps/rag/search.py' 'src/mcps/rag/interfaces.py' 'src/mcps/rag/vault.py'`
5. `lint.sh 'src/mcps/rag/search.py' 'src/mcps/rag/interfaces.py' 'src/mcps/rag/vault.py' 'tests/test_obsidian_vault_tools.py'`

Confirm: all three test files fully pass — `tests/test_obsidian_vault_tools.py` must now report **13 passed, 0 failed**. Compilation clean (no dangling `IResultFormatter` or `MarkdownResultFormatter` reference anywhere). Lint clean, in particular no unused-import warning.

Any regression → STOP, report, await direction.

**Manual check:** none. Behavior is identical; only unreachable code was removed.
