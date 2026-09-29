<!-- Context: read Goal, Specification, Out of Scope, Research Findings, Implementation Research Findings and Conventions in implementation-plan.md before starting this phase. -->

## Phase 3: Browser startup and tool availability [NEW_FEATURE]

### RED — `tests/test_browser_runtime.py:1` (NEW)
**Source under test:** `src/mcps/research/tools/browser.py` (append), `src/mcps/research/lifespan.py:13`
**Functions under test:** `browser_endpoint()`, `build_research_lifespan()`
**Fixtures:**
- `probe_results: dict[str, bool]` and `fake_probe(cdp_url: str) -> bool` async, records calls.
- `FakeProcess` — `returncode: int | None = None`, `terminate()` / `kill()` record calls and set returncode, `async wait()`.
- monkeypatch `shutil.which` → `"/test-bin/obscura"` or `None`; monkeypatch `asyncio.create_subprocess_exec` recording argv and returning `FakeProcess`.
- For lifespan test: FastMCP in-memory `Client(server.mcp)` from `create_server(config)` with `ROUTER_API_BASE`/`ROUTER_API_KEY` env as `tests/test_deep_research.py:253`; monkeypatch `mcps.research.lifespan.browser_endpoint` to an async context manager yielding `None` or `"ws://127.0.0.1:9222"`.

#### `test_configured_endpoint_is_used`
- **Given:** `probe_results={"ws://remote:9222": True}`
- **When:** `async with browser_endpoint("ws://remote:9222", probe=fake_probe) as url`
- **Then:** `url == "ws://remote:9222"`; no subprocess spawned

#### `test_falls_back_to_local_obscura`
- **Given:** remote probe False (and separately cdp_url `""`), local probe True
- **When:** enter/exit `browser_endpoint(...)`
- **Then:** yields `"ws://127.0.0.1:9222"`; argv `("/test-bin/obscura", "serve", "--stealth", "--allow-private-network")`; on exit `terminate()` called

#### `test_no_browser_yields_none`
- **Given:** remote False; case A `which → None`; case B process `returncode=1` immediately; case C local probe always False with patched `LOCAL_STARTUP_SECONDS = 0.1`
- **Then:** yields `None`; spawned process (B, C) terminated/killed

#### `test_startup_failure_hides_only_web_research`
- **Given:** lifespan `browser_endpoint` patched to yield `None`
- **When:** `async with Client(server.mcp) as c: tools = await c.list_tools()`
- **Then:** `"web_research"` not in names. With patch yielding a URL → `"web_research"` present.

Existing `tests/test_deep_research.py:247-260` `test_tool_is_registered` runs the lifespan through the in-memory client; with no Obscura on the test PATH it would now see the tool disabled. Update it in this phase to monkeypatch `mcps.research.lifespan.browser_endpoint` to yield `"ws://127.0.0.1:9222"` (same helper as above).

→ **EXPECTED: FAIL** — `browser_endpoint` missing; lifespan does not disable.

### CONFIRM_RED
Run `test.sh "$(pwd)/tests/test_browser_runtime.py"`. Get approval.

### GREEN
- `browser.py` — `LOCAL_CDP_URL = "ws://127.0.0.1:9222"`, `REMOTE_CONNECT_SECONDS = 10`, `LOCAL_STARTUP_SECONDS = 30`.
- `async probe_cdp(cdp_url: str) -> bool` — `async with _default_crawler_factory(cdp_url)(): return True`; any `Exception` → `False`.
- `@asynccontextmanager async browser_endpoint(cdp_url: str, *, probe: Callable[[str], Awaitable[bool]] = probe_cdp) -> AsyncIterator[str | None]` — if `cdp_url` and `probe` within `asyncio.timeout(REMOTE_CONNECT_SECONDS)` → yield it. Else `shutil.which("obscura")`; None → log warning, yield None. Spawn `asyncio.create_subprocess_exec(path, "serve", "--stealth", "--allow-private-network")`; poll `probe(LOCAL_CDP_URL)` every 0.5 s until success, process exit, or `LOCAL_STARTUP_SECONDS`. Yield URL or None. `finally`: if process running → `terminate()`, `wait()` within 5 s, else `kill()` + `wait()`.
- `lifespan.py:15-26` — inside the httpx client block: `async with browser_endpoint(config.browser_cdp_url) as cdp_url:`; None → `server.disable(names={"web_research"})`, yield `{"researcher": None, "http_client": http_client}`; else `server.enable(names={"web_research"})`, `build_research_config(dataclasses.replace(config, browser_cdp_url=cdp_url), http_client=http_client)` and yield as today.
- `tests/web_research_evaluation.py:43-45` — wrap in `async with browser_endpoint(server_config.browser_cdp_url) as cdp_url:`; exit with error message if None; pass `dataclasses.replace(server_config, browser_cdp_url=cdp_url)`. Keep `:52`.
→ **EXPECTED: PASS** — 4/4.

### VERIFY_GREEN
Run Phase 3 tests plus `tests/test_server.py`, `tests/test_deep_research.py`; lint/compile. **Manual check:** with Obscura on PATH, `uv run mcps` lists `web_research`; without it and without `BROWSER_CDP_URL`, it is absent and vault tools remain.
