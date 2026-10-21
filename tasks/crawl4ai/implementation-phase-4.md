<!-- Context: read Goal, Specification, Out of Scope, Research Findings, Implementation Research Findings and Conventions in implementation-plan.md before starting this phase. -->

## Phase 4: Query propagation and removal of url_context recovery [NEW_FEATURE]

### RED — `tests/test_deep_research.py` (append class `TestWebResearchFetch`)
**Source under test:** `src/mcps/research/deep_research.py:387-507`
**Functions under test:** `ResearchAgent.web_research()`, `ResearchAgent.clean_result()`
**Fixtures:**
- `fetch_calls: list[tuple[str, str | None]]`; `fake_fetch(url, query)` records and returns per-URL results from a dict.
- `fake_search(query) -> [SearchResult(url="https://a.example", title="A", snippet="s"), SearchResult(url="https://b.example", title="B", snippet="s"), SearchResult(url="https://c.example", title="C", snippet="s")]`.
- `fast = FakeMessagesListChatModel(responses=[AIMessage("CLEANED")])` (`langchain_core.language_models.fake_chat_models`), same for `small`.
- `agent = ResearchAgent(ResearchConfig(fast=fast, small=small, search=fake_search, fetch=fake_fetch))`.

#### `test_initial_branch_uses_original_question`
- **When:** `await agent.web_research({"original_question": "How does quantum routing work?", "search_query": "quantum routing", "id": 0}, {})`
- **Then:** every `fetch_calls` query == `"How does quantum routing work?"`

#### `test_follow_up_uses_knowledge_gap`
- **When:** state adds `"knowledge_gap": "What are the latency limits?"`; repeat with gap `"N/A"` and `"  "`
- **Then:** queries `"What are the latency limits?"`, then original question twice

#### `test_direct_url_passes_query`
- **When:** `search_query="https://source.example/page"`
- **Then:** `fetch_calls == [("https://source.example/page", "How does quantum routing work?")]`

#### `test_failed_and_empty_fetches_are_not_evidence`
- **Given:** a → `"content A"`, b → `"ERROR: http code 403"`, c → `""`; `fast` wrapped to record input messages
- **When:** `web_research` search branch
- **Then:** exactly one `fast` call; its human message contains `https://a.example` and not `b.example` / `c.example`; result contains "CLEANED"; all fetches failing/empty → result contains `NO_RELEVANT_EVIDENCE` and `fast` not called

→ **EXPECTED: FAIL** — queries are `None`; blank result counted; Gemini branch invokes `bind_tools`.

### CONFIRM_RED
Run `test.sh "$(pwd)/tests/test_deep_research.py"`. Get approval.

### GREEN — `src/mcps/research/deep_research.py`
- Add `_fetch_query(state: WebSearchState) -> str` — `knowledge_gap` stripped if nonblank and not `"N/A"`, else `original_question`.
- `:396` and `:413` — pass `_fetch_query(state)` as second argument.
- `clean_result` `:431-438` — success requires `fr.strip()` and not `fr.startswith("ERROR")`; failed list only used for logging count. Delete `:464-505` recovery block and `_FAILURE_RECOVERY_PROMPT` `:92-109`. Signature unchanged.
→ **EXPECTED: PASS**.

### VERIFY_GREEN
Run `tests/test_deep_research.py`; lint/compile.
