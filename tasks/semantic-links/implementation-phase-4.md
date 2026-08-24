# Phase 4: `obsidian_traverse_relations` tool [NEW_FEATURE]

Adds the graph-walk tool. Traversal always goes both directions; relation-type filtering is OR-combined; depth and node count are bounded.

### RED — `tests/test_obsidian_vault_tools.py` (new tests appended after the Phase 3 tests)

**Source under test:** `src/mcps/tools/obsidian_vault.py:209`, `src/mcps/rag/vault.py:443`
**Functions/methods under test:** `obsidian_vault.traverse_relations()`, `Vault.traverse_relations()`, `register_tools()`

**Fixtures:**
- `FakeVault` → existing, extended again with `async def traverse_relations(self, wikilink_name: str, depth: int = 1, relation_types: list[str] | None = None) -> TraversalResult`, returning a `self.traversal_result` attribute set per test, and recording its arguments in `self.traverse_calls: list[tuple[str, int, list[str] | None]]`. The MCP-layer tests exercise the tool wrapper against this fake.
- `graph_store` → `IVectorStore` — an `AsyncMock` configured for the `Vault.traverse_relations` unit tests, with `get_notes_with_links` and `get_notes_linking_to` side effects driven by this fixed graph:

  | Note | Outgoing links |
  |---|---|
  | `Alpha` | `requires→Beta`, `related→Gamma` |
  | `Beta` | `refines→Delta` |
  | `Gamma` | `requires→Delta` |
  | `Zulu` | `requires→Alpha` |
  | `Delta` | *(none)* |

  `get_notes_with_links(names)` returns a `NoteLinks` per requested name present in the table above, with `title=f"{name} Note"` and `description=f"About {name.lower()}"`. `get_notes_linking_to(targets, relation_types)` returns, per source note holding a matching edge, a `NoteLinks` containing only the matching edges. `Nonexistent` is in neither.

**Test cases:**

#### `test_traverse_relations_depth_one_returns_both_directions`
- **Use Case** SPEC success criterion #6, depth 1.
- **Given:** `graph_store`; origin `Alpha`.
- **When:** `await vault.traverse_relations("Alpha", depth=1)`.
- **Then:** `result.nodes` contains exactly three entries keyed by `note`: `Beta` (`depth=1`, `direction="outgoing"`, `relation="requires"`, `via="Alpha"`), `Gamma` (`depth=1`, `direction="outgoing"`, `relation="related"`, `via="Alpha"`), and `Zulu` (`depth=1`, `direction="incoming"`, `relation="requires"`, `via="Alpha"`). `Alpha` itself is not in `nodes`. `result.truncated is False`.

#### `test_traverse_relations_depth_two_reaches_second_hop`
- **Use Case** SPEC success criterion #6, depth > 1.
- **Given:** `graph_store`; origin `Alpha`.
- **When:** `await vault.traverse_relations("Alpha", depth=2)`.
- **Then:** `result.nodes` includes `Delta` at `depth=2`. `Delta` appears exactly once, with `via` naming whichever of `Beta`/`Gamma` was expanded first at level 2, and `relation` matching that edge (`refines` from `Beta`, `requires` from `Gamma`).

#### `test_traverse_relations_note_appears_once_at_shallowest_depth`
- **Use Case** traversal-dedup decision.
- **Given:** `graph_store`; origin `Zulu`, which reaches `Alpha` at depth 1 and, through `Alpha`'s neighbours, would reach it again on a later level.
- **When:** `await vault.traverse_relations("Zulu", depth=3)`.
- **Then:** `[n.note for n in result.nodes].count("Alpha") == 1`, and that entry has `depth == 1`. `Zulu` (the origin) never appears in `nodes`.

#### `test_traverse_relations_filters_to_requested_relation_types`
- **Use Case** SPEC success criterion #6, type filtering.
- **Given:** `graph_store`; origin `Alpha`.
- **When:** `await vault.traverse_relations("Alpha", depth=1, relation_types=["requires"])`.
- **Then:** `result.nodes` contains `Beta` (outgoing `requires`) and `Zulu` (incoming `requires`) and **not** `Gamma`, whose only edge from `Alpha` is `related`.

#### `test_traverse_relations_or_combines_multiple_relation_types`
- **Use Case** SPEC: types are OR-combined.
- **Given:** `graph_store`; origin `Alpha`.
- **When:** `await vault.traverse_relations("Alpha", depth=1, relation_types=["related", "refines"])`.
- **Then:** `result.nodes` contains `Gamma` only — `Beta` and `Zulu` are reached solely by `requires` edges.

#### `test_traverse_relations_excludes_dangling_targets`
- **Use Case** dangling-link decision: only notes present in the index are returned.
- **Given:** `graph_store` amended so `Alpha` also carries `requires→Ghost`, with `Ghost` absent from the index.
- **When:** `await vault.traverse_relations("Alpha", depth=1)`.
- **Then:** no node has `note == "Ghost"`. No exception is raised.

#### `test_traverse_relations_aggregates_multiple_chunks_of_one_note`
- **Use Case** note-level aggregation (Implementation Research, "Traversal aggregation unit").
- **Given:** an `AsyncMock` store whose `get_notes_with_links` is driven by a `side_effect` keyed on its argument: `["Alpha"]` returns a single `NoteLinks` for `Alpha` whose `links` were aggregated from two chunk rows — `[Link(type="requires", target="Beta"), Link(type="related", target="Gamma")]` — and `["Beta", "Gamma"]` returns a `NoteLinks` for each, with `links=[]`, `title=f"{name} Note"`, `description=f"About {name.lower()}"`. `get_notes_linking_to` returns `[]` for every argument. The second stub is required: candidate resolution (GREEN step 6) drops any candidate the store does not return, so a mock stubbed only for `["Alpha"]` would discard `Beta` and `Gamma` as dangling and the test would fail for the wrong reason.
- **When:** `await vault.traverse_relations("Alpha", depth=1)`.
- **Then:** both `Beta` and `Gamma` appear as depth-1 nodes, with `title` values `"Beta Note"` and `"Gamma Note"`. `get_notes_with_links` was awaited exactly twice — once with `["Alpha"]` (origin validation, which doubles as level 1's forward data) and once with `["Beta", "Gamma"]` (candidate resolution). One call per level plus one, never one call per node — the store, not the vault, performs chunk-row grouping.

#### `test_traverse_relations_truncates_at_node_cap`
- **Use Case** traversal-bounds decision (100 distinct notes).
- **Given:** an `AsyncMock` store where `Hub` links `related→N{i}` for `i in range(150)` and every `N{i}` exists.
- **When:** `await vault.traverse_relations("Hub", depth=2)`.
- **Then:** `len(result.nodes) == 100`, `result.truncated is True`, and `result.warning` is a non-empty string naming both the cap (`100`) and the depth at which truncation occurred.

#### `test_traverse_relations_unknown_origin_raises`
- **Use Case** error path for a caller-supplied note that does not exist.
- **Given:** `graph_store`; origin `Nonexistent`.
- **When:** `await vault.traverse_relations("Nonexistent", depth=1)`.
- **Then:** raises `FileNotFoundError` naming the note.

#### `test_traverse_relations_batches_queries_per_level`
- **Use Case** performance contract: per-level batching, not per-node queries.
- **Given:** `graph_store`; origin `Alpha`.
- **When:** `await vault.traverse_relations("Alpha", depth=2)`.
- **Then:** `graph_store.get_notes_with_links` was awaited exactly **3** times (`depth + 1`: one origin validation, then one candidate resolution per level, each resolution doubling as the next level's forward data) and `graph_store.get_notes_linking_to` exactly **2** times (`depth`: one per level). Every call after the first received the whole frontier or candidate set as a single list argument — assert on the argument lists, not only on the counts, so a per-node implementation that happens to hit the same totals still fails.

#### `test_obsidian_traverse_relations_tool_is_registered`
- **Use Case** the tool must be reachable through the server.
- **Given:** `ServerConfig(vault_dir=tmp_path)` and `create_server(config)`, mirroring `tests/test_obsidian_vault_tools.py:76-88`.
- **When:** `await server.mcp.list_tools()`.
- **Then:** `"obsidian_traverse_relations"` is in the returned tool names. Add it to `OBSIDIAN_TOOL_NAMES` at `tests/test_obsidian_vault_tools.py:17-22` so the existing registration test covers it.

#### `test_traverse_relations_tool_rejects_out_of_range_depth`
- **Use Case** bounds enforcement at the tool boundary.
- **Given:** the registered `obsidian_traverse_relations` tool.
- **When:** invoking it through `server.mcp` with `depth=4`, and separately with `depth=0`.
- **Then:** both are rejected by parameter validation before the tool body runs; the fake vault's `traverse_calls` remains empty.

#### `test_traverse_relations_tool_raises_tool_error_for_unknown_note`
- **Use Case** MCP-layer error translation, mirroring `get_file_content` at `obsidian_vault.py:271-291`.
- **Given:** a `FakeVault` whose `traverse_relations` raises `FileNotFoundError("Nonexistent")`.
- **When:** `await obsidian_vault.traverse_relations("Nonexistent", FakeContext(vault))`.
- **Then:** raises `ToolError` whose message names the note.

→ **EXPECTED: FAIL** — `TraversalNode`, `TraversalResult`, `Vault.traverse_relations`, and the tool function do not exist.

---

### CONFIRM_RED
Run: `test.sh 'tests/test_obsidian_vault_tools.py'`

Confirm: the new tests fail on missing symbols; all Phase 0-3 tests in the file still pass. Verify that `test_traverse_relations_truncates_at_node_cap` and `test_traverse_relations_note_appears_once_at_shallowest_depth` fail on assertions once the symbols exist rather than passing on an empty node list — an implementation returning `[]` would satisfy neither. If any new test passes → STOP, report, await direction.

Request user to validate. Do not move to GREEN implementation phase before user approval.

---

### GREEN — `src/mcps/rag/interfaces.py`, `src/mcps/rag/vault.py`, `src/mcps/tools/obsidian_vault.py`

**Types to create — `src/mcps/rag/interfaces.py`, inserted immediately after `NoteLinks`:**
- `TraversalNode(BaseModel)` — fields: `note: str` (the reached note's `wikilink_name`), `title: str | None = None`, `description: str | None = None`, `depth: int` (hops from the origin, ≥ 1), `direction: str` (`"outgoing"` when the edge points from `via` to `note`, `"incoming"` when it points from `note` to `via`), `relation: str` (the edge's type), `via: str` (the note from which this node was reached).
- `TraversalResult(BaseModel)` — fields: `origin: str`, `nodes: list[TraversalNode] = Field(default_factory=list)`, `truncated: bool = False`, `warning: str | None = None`.

**Constants to create — `src/mcps/rag/vault.py`, beside the module's other module-level constants:**
- `MAX_TRAVERSAL_NODES = 100`.

**Types to modify — `src/mcps/rag/interfaces.py` `IVault(ABC)`, one new abstract method appended after `get_backlinks`:**
- `async def traverse_relations(self, wikilink_name: str, depth: int = 1, relation_types: list[str] | None = None) -> TraversalResult`
  Contract: breadth-first walk of the typed link graph in both directions from `wikilink_name`, up to `depth` hops, keeping only edges whose type is in `relation_types` when that argument is given. Raises `FileNotFoundError` when the origin is not in the index.

**Imports to add:**
- `src/mcps/rag/vault.py` — add `TraversalNode` and `TraversalResult` to the existing `from .interfaces import ...` statement.
- `src/mcps/tools/obsidian_vault.py` — add `TraversalResult` to the existing `from mcps.rag.interfaces import ...` statement.
- `tests/test_obsidian_vault_tools.py` — add `NoteLinks`, `TraversalNode`, and `TraversalResult` to the existing `from mcps.rag.interfaces import ...` statement.

**Methods to create — `src/mcps/rag/vault.py`, appended after `get_backlinks`:**
- `async def traverse_relations(self, wikilink_name: str, depth: int = 1, relation_types: list[str] | None = None) -> TraversalResult`
  Behavior:
  1. Verify the origin exists by calling `await self.vector_store.get_notes_with_links([wikilink_name])`; if it returns nothing, raise `FileNotFoundError` naming the note. Retain that result as level 1's forward data so the origin is not queried twice.
  2. Maintain `visited: set[str]` seeded with the origin, and `nodes: list[TraversalNode]`.
  3. For each level from 1 to `depth`: the frontier's **outgoing** data is the `NoteLinks` list already in hand — from step 1 for level 1, from step 6 of the previous level thereafter — and is never re-fetched. Issue exactly **one** batched query per level, `get_notes_linking_to(frontier, relation_types)`, for the incoming edges of the whole frontier. Never query per node. Over a full traversal this costs `depth` calls to `get_notes_linking_to` and `depth + 1` calls to `get_notes_with_links` (one in step 1, one per level in step 6).
  4. Outgoing edges: for each frontier note's `NoteLinks`, each link whose type passes the `relation_types` filter proposes a node `(note=link.target, direction="outgoing", relation=link.type, via=<frontier note>)`. Incoming edges: each returned source `NoteLinks` proposes, per matching edge, a node `(note=<source note>, direction="incoming", relation=<edge type>, via=<edge target>)`.
  5. Drop any proposal whose `note` is already in `visited` — this enforces "each note appears once, at the shallowest depth, with the provenance of the first path that reached it". Add survivors to `visited`.
  6. Resolve `title`/`description` and **existence** for all surviving proposals — outgoing and incoming alike — with one `get_notes_with_links(candidate_names)` call. Drop any candidate the store does not return: these are dangling targets, excluded silently. Retain this same result as the next level's forward data (step 3), so no note's outgoing links are ever fetched twice. This call is issued at every level including the last, where it serves only as the existence and metadata check.
  7. Append the surviving nodes with `depth=<current level>`. If `len(nodes)` would exceed `MAX_TRAVERSAL_NODES`, truncate to exactly that many, set `truncated=True`, set `warning` to a message naming the cap, the depth at which truncation happened, and the remedy (narrow `relation_types` or reduce `depth`), and stop the walk immediately.
  8. The next frontier is the surviving node names of the level just completed. Stop early when the frontier is empty.
  9. Return `TraversalResult(origin=wikilink_name, nodes=nodes, truncated=..., warning=...)`.

**Parameter aliases to create — `src/mcps/tools/obsidian_vault.py`, appended after `ReadLimit` (which ends at line 139), matching the existing `Annotated[..., Field(...)]` style:**
- `TraversalDepth = Annotated[int, Field(default=1, ge=1, le=3, description=...)]` — number of hops to walk from the origin note; 1 means direct neighbours only. The `ge`/`le` bounds are enforced by FastMCP before the tool body runs.
- `RelationTypes = Annotated[list[str] | None, Field(default=None, description=...)]` — optional list of relation type strings; an edge is followed when its type matches any of them (OR). Omit to follow every relation type. Types are free-form strings taken verbatim from note text; no taxonomy is enforced.
- `note: WikilinkName` (reused alias).

**Functions to create — `src/mcps/tools/obsidian_vault.py`, defined at module level after `search()` (which ends at line 392):**
- `async def traverse_relations(note: WikilinkName, ctx: Context, depth: TraversalDepth = 1, relation_types: RelationTypes = None) -> TraversalResult`
  Behavior: delegate to `await _vault_from_context(ctx).traverse_relations(note, depth, relation_types)` and return its result unchanged. Catch `FileNotFoundError` and re-raise as `ToolError` naming the note, exactly as `get_file_content` does at `obsidian_vault.py:271-291`. Returning a `BaseModel` yields both `content` and `structuredContent` in the MCP response.

**Registration — `src/mcps/tools/obsidian_vault.py:209-249` `register_tools()`:**
- Add a `mcp.tool(traverse_relations, name="obsidian_traverse_relations", description=...)` call after the `obsidian_search` registration at lines 232-247, following the identical call shape. The description must state: walks typed relation links forward and backward from a note; `depth` is 1-3; `relation_types` OR-combines free-form type strings and defaults to all types; results are capped at 100 distinct notes with a warning when truncated; each node reports the relation type, the direction, and the note it was reached through.

→ **EXPECTED: PASS** — all tests from the RED phase.

---

### VERIFY_GREEN

Run, in order:
1. `test.sh 'tests/test_obsidian_vault_tools.py'`
2. `test.sh 'tests/test_vault.py'`
3. `test.sh 'tests/test_server.py'`
4. `compile.sh 'src/mcps/rag/interfaces.py' 'src/mcps/rag/vault.py' 'src/mcps/tools/obsidian_vault.py'`
5. `lint.sh 'src/mcps/rag/interfaces.py' 'src/mcps/rag/vault.py' 'src/mcps/tools/obsidian_vault.py' 'tests/test_obsidian_vault_tools.py'`

Confirm: all pass. Any regression → STOP, report, await direction.

**Manual check:** `uv run mcps` against a re-indexed vault; call `obsidian_traverse_relations` with a note known to carry `requires::` links at `depth=1` and `depth=2`, then again with `relation_types=["requires"]`, and confirm the node sets narrow as expected and that every returned note exists in the vault.
