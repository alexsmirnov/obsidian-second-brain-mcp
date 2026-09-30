# crawl4ai + Obscura fetch implementation plan

## Goal

Fetch generic web pages through crawl4ai over an Obscura CDP browser, return only query-relevant Markdown with links, keep GitHub/arXiv specialized fetchers without fallback, drop restricted domains, fall back to one commercial provider on blocked pages, limit parallel browser load, disable `web_research` when no browser is reachable, and measure the result against the saved baseline cases. Personal-use tool: occasional failures are acceptable; prefer the smallest change over defensive completeness.

## Specification

1. When the research lifespan starts, the server shall connect to `BROWSER_CDP_URL` (10 s budget) if set.
2. If `BROWSER_CDP_URL` is unset or unreachable, the server shall start `obscura serve --stealth --allow-private-network` from `PATH` and connect to `ws://127.0.0.1:9222` (30 s budget).
3. If neither succeeds, the server shall disable only the `web_research` tool; other tools stay available. On shutdown the server shall terminate only the Obscura process it started.
4. The fetch callable shall be `async fetch(url: str, query: str | None) -> str`; errors remain `ERROR:` strings.
5. When the requested hostname equals or is a subdomain of an entry in `FETCH_RESTRICTED_DOMAINS` (comma-separated, default empty), fetch shall return `""` without I/O.
6. When the URL is GitHub (repo root or blob) or arXiv, fetch shall use the existing specialized fetcher and never fall back.
7. When the URL path ends with `.pdf`, fetch shall use the existing httpx + PyMuPDF extractor.
8. For all other URLs, fetch shall render the page in the CDP browser via crawl4ai.
9. When the query is nonblank, every successful source shall be filtered by crawl4ai `LLMContentFilter` if `FETCH_MODEL` is set, else `BM25ContentFilter`; blank/None query returns unfiltered Markdown. Links are kept inline and resolved to absolute URLs. Output is truncated to 15,000 chars after filtering. Every source reaches the filter as HTML and is converted HTML→Markdown exactly once (by the filter): HTML pages (browser, httpx, arXiv, providers) pass through unconverted; only genuine Markdown/plain-text sources (GitHub raw files, `text/plain`/`text/markdown`, browser pages whose body is a single `<pre>`) are converted Markdown→HTML once; PDF text becomes escaped `<p>` blocks with `<br>` line breaks. GitHub repository README relative links resolve against `https://github.com/<owner>/<repo>/blob/HEAD/`.
10. If the browser or PDF result is 401/403/429, empty, timeout, or browser-unavailable and `SCRAPER_PROVIDER` is configured, fetch shall return the provider's Markdown filtered by the query; provider unavailability keeps the original error.
11. Concurrent browser fetches across all research branches shall be limited to `FETCH_CONCURRENCY` (default 2).
12. The research agent shall pass the knowledge gap as fetch query, or the original question when the gap is absent/blank/`N/A`. Gemini `url_context` recovery of failed fetches is removed; the summary cleanup LLM call stays. Blank fetch results are not evidence.
13. Compose runs one Obscura container and one MCP container sharing a network namespace; CDP listens on loopback only.
14. `tests/web_fetch_evaluation.py [--cases PATH] --output PATH` replays a JSONL case file (default: tracked smoke set in `tests/evaluation/data/`) through the production fetch and prints failure count and mean size versus baseline.

## Out of Scope

- `OBSCURA_CDP_TOKEN` / CDP bearer auth: replaced by Compose shared network namespace with a loopback-only CDP (user decision). Remote `BROWSER_CDP_URL` must be unauthenticated.
- Wikipedia and Reddit specialized fetchers: deleted; both use the generic browser path.
- PDFs at URLs without `.pdf` suffix, crawl4ai PDF strategies.
- Structured outcome types, per-chunk LLM failure diagnostics, Markdown-aware truncation, per-provider pools, configurable timeouts, independent per-provider diagnostic runs, Compose YAML tests.
- Obscura restart after startup; commercial provider chains; CDP connection isolation beyond crawl4ai's `create_isolated_context`.
- Obsidian RAG.

## Research Findings

Files:
- `src/mcps/research/tools/common.py:28` — `Fetch = Callable[[str], Awaitable[str]]`; error constants `:60-65`; `_ESCALATABLE_STATUS_CODES` `:69`; `is_escalatable(result: str) -> bool` `:83` (true for 401/403/429 and `ERROR: empty response`); `format_source_output(url, content, max_chars) -> str` `:97` logs `Fetch N chars` and truncates with `\n\n[Content truncated]`; `extract_hostname` `:105` uses netloc.
- `src/mcps/research/tools/fetch.py:35` — `_SITE_ROUTES` (arxiv, wikipedia, reddit, github blob, github repo); `_fetch_direct` `:51` maps exceptions to error strings; `create_fetch(*, http_client=None, max_chars=15000, fallbacks=()) -> Fetch` `:65`, escalation loop `:79-88` skips `ERROR_FETCHER_UNAVAILABLE` candidates.
- `src/mcps/research/tools/browser.py:21-28` — optional crawl4ai import + `CRAWL4AI_AVAILABLE`; `_default_crawler_factory(cdp_url)` `:35` (`browser_mode="custom"`, `cache_cdp_connection=True`, `create_isolated_context=True`, `verbose=False`); `_to_fetch_result` `:48` maps status ≥400 → `http_status_error`, `success=False` → `ERROR_FETCHER_UNAVAILABLE`, blank `raw_markdown` → `ERROR_EMPTY_RESPONSE`; `create_browser_fetch(cdp_url, *, max_chars=15000, crawler_factory=None) -> Fetch` `:61`.
- `src/mcps/research/tools/default.py:38` — `fetch_default(url, *, http_client, max_chars) -> str`, html2text/PyMuPDF via `extract.py`.
- `src/mcps/research/tools/github.py:39,44,69,93` — `is_github_blob_url`, `is_github_repo_url`, `fetch_github_blob`, `fetch_github_repo` (same keyword signature, truncate via `format_source_output`).
- `src/mcps/research/tools/arxiv.py:22,45` — `is_arxiv_url`, `fetch_arxiv` (html → pdf → abs).
- `src/mcps/research/tools/wikipedia.py:1`, `src/mcps/research/tools/reddit.py:1` — to delete.
- `src/mcps/research/tools/scrape_do.py:49` `create_scrape_do_fetch(token, *, http_client, max_chars=15000, api_url=...) -> Fetch`; `src/mcps/research/tools/bright_data.py:46` `create_bright_data_fetch(api_key, zone, *, http_client, max_chars=15000, api_url=...) -> Fetch`. Both return Markdown or error strings.
- `src/mcps/research/config.py:44` `ResearchConfig(fast, small, search, fetch: Callable[[str], Awaitable[str]])`; `_create_browser_fallback` `:103`; `_create_provider_fallback(config, http_client) -> Fetch | None` `:115` (SCRAPER_PROVIDER match); `create_fetch_fallbacks` `:146`; `create_fetch_tool(*, config, http_client)` `:157`; `build_research_config(config, *, http_client)` `:167`, fetch wired `:196`.
- `src/mcps/config.py:14` `@dataclass ServerConfig`; fetch fields `:57-64`; env reads `:129-133`; `router_api_base`/`router_api_key` `:21-22`.
- `src/mcps/research/lifespan.py:13-28` — `build_research_lifespan(config)`; `research_lifespan(server: FastMCP)` owns `httpx.AsyncClient`, yields `{"researcher", "http_client"}`.
- `src/mcps/server.py:96-105` lifespan composition; `:121-129` `web_research` tool reads `ctx.lifespan_context["researcher"]`.
- `src/mcps/research/deep_research.py:92-109` `_FAILURE_RECOVERY_PROMPT`; `:247` `WebSearchState`; `:387` `web_research(state, config)`, fetch calls `:396` (direct URL) and `:413` (search results); `:424` `clean_result(results, fetch_results, question, knowledge_gap) -> str`, success filter `:431-435`, failed list `:436-438`, Gemini recovery `:464-505`; `:585` initial Send has no `knowledge_gap`; `:608-613` follow-up Send passes `knowledge_gap`.
- `tests/web_research_evaluation.py:39-55` — builds `build_research_config` directly (`:45`); user edit `load_draco_questions()[:3]` at `:52` must be preserved.
- `Dockerfile:28-44` — non-root `mcps` user (uid 999, home created), `ENV VAULT=/vault` `:36`, CMD http on 8000.
- `env.example:55-66` — BROWSER_CDP_URL / SCRAPER_PROVIDER / provider tokens.
- `pyproject.toml:14` `markdown>=3.10.0`; `:23` `lxml>=5.0.0`; `:25` `pymupdf`; `:28` `crawl4ai>=0.7` (installed 0.9.4).
- Installed crawl4ai 0.9.4 (`.venv/lib/python3.13/site-packages/crawl4ai/`): `BM25ContentFilter(user_query=..., bm25_threshold=1.0)` `content_filter_strategy.py:404`; `LLMContentFilter(llm_config, instruction, ..., verbose=False, ignore_cache=True)` `:849`, calls `perform_completion_with_backoff` (imported into that module, `:13`, called `:1020`) in a 4-worker thread pool, swallows per-chunk errors `:1079`, response parsed from `<content>` XML `:1068`; `DefaultMarkdownGenerator(content_filter=None, options=None)` `markdown_generation_strategy.py:74`; `generate_markdown(input_html, base_url="", ...)` is synchronous, resolves links via `CustomHTML2Text(baseurl=base_url)` `:180`, fit filter errors become `"Error generating fit markdown: ..."` `:241`; `AsyncWebCrawler.arun` calls `generate_markdown` synchronously on the event loop `async_webcrawler.py:873`; `_verify_cdp_ready` skips HTTP precheck for `ws://` URLs `browser_manager.py:1054`; `LLMConfig(provider, api_token, base_url)` `async_configs.py:2349`; all exported from `crawl4ai` `__init__.py:4-182`.
- FastMCP 4.0.3: `FastMCP` inherits `AggregateProvider` → `Provider.enable(*, names=...)` / `disable(*, names=...)` `fastmcp/server/providers/base.py:578,622` (later call wins).

Test files:
- `tests/test_research_tools.py:18` `client` fixture; `:93-190` default/wikipedia/github/arxiv/error fetch tests using `create_fetch(http_client=client)`; `:219-315` escalation tests using `fallbacks=`.
- `tests/test_research_fallback_fetchers.py:35-64` `FakeCrawler`/`crawl_result` fakes; `:67-107` browser tests; `:115-218` provider request contracts (unchanged); `:226-281` `create_fetch_fallbacks`/`create_fetch_tool` assembly.
- `tests/test_deep_research.py:53-86` build_research_config contract; `:247-260` `web_research` registered.
- `tests/test_config.py:1` env config tests.

Missing coverage: query filtering, restricted domains, fetch concurrency limit, browser startup/Obscura spawn, tool disabling, query propagation in agent.

Baseline data (current location, gitignored via `.gitignore:23` `/tasks/`): `tasks/crawl4ai/fetch-cases-baseline.jsonl:1` (279 cases, 266 KB), `tasks/crawl4ai/fetch-cases-smoke.jsonl:1` (8 cases). Row fields: `case_id`, `url`, `query`, `task_id`, `research_round`, `baseline.observations[]` each `{line, status: "success"|"retrieval_error", error, pre_truncation_chars, inferred_returned_chars}`. Phase 5 moves both into the tracked evaluation package `tests/evaluation/` (`__init__.py` present; siblings `dataset.py`, `report.py`); `pyproject.toml:63-64` `testpaths = ["tests"]`, `pythonpath = ["src", "tests"]`.

## Implementation Deviations

- **Phase 1 — BM25 keyword guarantee (approved by user).** The plan expected plain `BM25ContentFilter(user_query=query)` to keep the `TOPIC_HTML` section whose heading *and* body contain the query `"quantum optimization"`. On that 4-block fixture both query terms occur in exactly half the candidates, so `BM25Okapi` IDF is `log(2.5/2.5) = 0` and every score is `0` — below the default threshold `1.0`. The user requires that a section containing all query keywords is always kept, so `filtering.py` implements `_KeywordGuaranteedBM25`, a `BM25ContentFilter` subclass that unions the thresholded selection with blocks containing every (stemmed) query term, in document order. This makes `tests/test_fetch_filtering.py::test_bm25_keeps_relevant_block_and_absolute_link` meaningful without weakening the fixture.

- **Phase 2 — test expectation corrected.** `test_specialized_sources_filter_and_never_escalate` first asserted the GitHub README's `/paper` link resolved to `https://source.example/paper`. For a specialized source the base URL is the page's own URL, so it correctly resolves to `https://github.com/paper`; the assertion (not the implementation) was wrong and was fixed.

- **Phase 2 — pre-existing suite failures (not regressions).** On clean `HEAD` the offline suite already fails 16 tests: `tests/test_llm_reranker.py` (`RRFReranker` has no `_table_to_documents`) and `TestServerConfigContract::test_create_config_reads_router_env_vars` (container has `ROUTER_DOCKER_API_BASE` set, so `/.dockerenv` selects the docker URL). `tests/test_embedding_service.py::...embedding_service0` additionally fails only in a full-suite run due to a pytest-asyncio event-loop ordering flake. None are touched by this work.

- **Phase 3 — manual Obscura check pending.** `tests/test_browser_runtime.py` (8 tests) covers `browser_endpoint` and the lifespan enable/disable of `web_research` through an in-memory `Client`; `tests/test_server.py` and the updated `tests/test_deep_research.py` pass. The manual `uv run mcps` check with/without Obscura on PATH has not been run (no Obscura in this container) and awaits user confirmation.
- **Phase 3 — pre-existing lint.** `tests/test_deep_research.py:195,211` exceed 88 columns; both predate this work (outside the edited region) and were left untouched.

- **Phase 4 — `WebSearchState.knowledge_gap` made `NotRequired`.** The initial `continue_to_web_research` `Send` omits `knowledge_gap`, so the required-`str` annotation was inaccurate. Marking it `NotRequired[str]` matches runtime and lets the new tests construct the initial-branch state directly without `type: ignore`.

- **Phase 5 — `rows` type-safe size parsing.** `summarize` reads `response_size` with an `isinstance(raw_size, int)` guard instead of `int(row.get(...))` to satisfy pyright on the `dict[str, object]` signature.

- **Phase 4 — non-blocking rubber-duck note.** `clean_result`'s `failed_fetches` log count excludes blank results (blank is neither success nor error); left as-is because the list is only used for a log line.

- **Phase 6 — `docs/tests_coverage.md` also updated.** Beyond the files listed in the phase, `tests_coverage.md` gained entries for `tests/test_web_fetch_evaluation.py` and `tests/web_fetch_evaluation.py` plus the `tests/evaluation/data/` case files, keeping the test/docs mind map synchronized.

- **Phase 6 — Docker manual check pending.** The container has no `docker` binary, so `docker compose config --quiet` / `docker compose up` cannot run here. `docker-compose.yaml` was validated as parseable YAML; the Compose verification awaits the user on a Docker host.

- **Phase 5/6 — rubber-duck hardening.** Review verified `summarize()` and the Compose loopback/CDP isolation; it noted `load_cases` could keep a non-string `case_id` and then fail the `by_id.get(str(...))` lookup. All shipped case files use string ids, but `case_id` is now normalized with `str(...)`; tests, lint, and compile re-run clean.

- **Phase 7 — manual smoke run pending.** `uv run tests/web_fetch_evaluation.py --output tmp/fetch-smoke-p7.jsonl` needs user permission and a reachable browser; not run. Automated: `tests/test_research_tools.py` (34), `tests/test_research_fallback_fetchers.py` (23), `tests/test_fetch_filtering.py` (6), `tests/test_web_fetch_evaluation.py` (3) pass; `rg html2text src pyproject.toml` is empty.
- **Phase 7 — pre-existing pyright error.** `tests/test_research_tools.py` `asyncio.create_task(fetch(...))` in `test_browser_concurrency_is_limited` (`Awaitable[str]` vs `Coroutine`) predates this phase and was left untouched.
- **Phase 7 — `text_page_to_html` on provider output.** Provider results pass through `text_page_to_html` before filtering (as the plan specifies), so a provider returning a sole `<pre>` page is treated as plain text.

## Implementation Research Findings

- **Filtering location** — considered filter inside `arun` vs one post-retrieval filter; chosen: browser returns `cleaned_html`, every source goes through one `filter_page` that runs `DefaultMarkdownGenerator(content_filter=...).generate_markdown` in `asyncio.to_thread`; why: `arun` runs `generate_markdown` on the event loop (`async_webcrawler.py:873`), so an LLM filter there would stall the MCP server; one path serves browser, GitHub, arXiv, PDF, provider.
- **Markdown sources** — considered passing Markdown/plain text directly to the filter vs converting to HTML; chosen: convert with the already-installed `markdown` package (`extensions=["extra"]`) before filtering; why: `BM25ContentFilter.filter_content` (`content_filter_strategy.py:441-476`) chunks by HTML block tags and link resolution runs only on `<a href>` via html2text (`markdown_generation_strategy.py:180`). Verified on 0.9.4 with the Phase 1 Markdown fixture and query "quantum optimization": raw Markdown → `fit_markdown == "\n"`; `<body><pre>md</pre></body>` → `"\n"`; `markdown_to_html(md)` → 1177 chars, relevant text and absolute link kept.
- **Single conversion (Phase 7)** — considered: HTML-first pipeline; a second Markdown-native filter (`rank_bm25` + own link resolution); GitHub API rendered HTML (`Accept: application/vnd.github.html+json`, 60 req/h unauthenticated, link rewriting undocumented); chosen: HTML-first (user decision); why: crawl4ai `BM25ContentFilter.filter_content` requires HTML blocks (`content_filter_strategy.py:459-473`) and `generate_markdown` always runs html2text on filter output (`markdown_generation_strategy.py:235-239`), so one filter, one conversion. Verified on 0.9.4: double conversion of `<p>Use &lt;style&gt;alpha&lt;/style&gt;</p>` yields `Use  carefully`, single keeps `<style>alpha</style>`; `extract.py:22` `ignore_images=True` drops images. Scrape.do without `output` returns raw rendered HTML; Bright Data `format=json` without `data_format` returns raw HTML in `body` (scrape.do/documentation, docs.brightdata.com/api-reference/rest-api/unlocker/unlock-website). Code wrapped in `<pre>`/`<pre><code>` is dropped by BM25 for any query (verified), so GitHub blobs stay on `markdown_to_html`.
- **Plain-text/Markdown/JSON URLs on the browser path** — crawl4ai `cleaned_html` for such pages is exactly `<html><body><pre>…</pre></body></html>` (verified with `LXMLWebScrapingStrategy.scrap`); chosen: unwrap a sole `<pre>` child of `<body>` with `lxml.html` and pass its text through `markdown_to_html`; other HTML untouched.
- **LLM filter** — `LLMConfig(provider="openai/" + FETCH_MODEL (no double prefix), api_token=router_api_key, base_url=router_api_base)`, `verbose=False`, `ignore_cache=True`; credentials only inside `LLMConfig`. Partial chunk failures are accepted silently.
- **CDP auth** — dropped (user decision). Compose `network_mode: service:browser` keeps CDP on loopback. crawl4ai skips its unauthenticated HTTP precheck for `ws://` URLs.
- **Connection reuse** — keep existing `cache_cdp_connection=True`, `create_isolated_context=True`.
- **Tool disabling** — `server.disable(names={"web_research"})` on failure, `server.enable(...)` on success, inside the research lifespan.
- **Concurrency** — one `asyncio.Semaphore(FETCH_CONCURRENCY)` created inside `create_fetch`, around the browser call only; one fetch per lifespan → shared across branches.
- **Commercial fallback** — reuse `SCRAPER_PROVIDER` selection `config.py:115`; extend `is_escalatable` with timeout and fetcher-unavailable.
- **Obscura** — `h4ckf0r0day/obscura:0.2.3@sha256:475def3ddf1ec513b3d1bc36e8ad15f0d192538cb15f814c77215aa70c418ca2`, ENTRYPOINT `/obscura`, default bind `127.0.0.1:9222`.

## Conventions

1. Phases in order; each is `[NEW_FEATURE]`. RED → CONFIRM_RED → user approval → GREEN → VERIFY_GREEN. **AUTO-STOP** when an EXPECTED result differs from actual.
2. Tests: `test.sh "$(pwd)/tests/<file>.py"`; lint/compile: `lint.sh "$(pwd)/<file>.py"`, `compile.sh "$(pwd)/<file>.py"`.
3. Mock only external boundaries: HTTP via `pytest_httpx` `HTTPXMock`, crawler via `crawler_factory` (`FakeCrawler` pattern `tests/test_research_fallback_fetchers.py:35`), LLM via monkeypatching `crawl4ai.content_filter_strategy.perform_completion_with_backoff`, subprocess via monkeypatching `asyncio.create_subprocess_exec` and `shutil.which`. Run real crawl4ai BM25/Markdown code.
4. Update or delete existing tests for removed behavior in the same phase that removes it. No skip markers.
5. Preserve `tests/web_research_evaluation.py:52` `[:3]`.
6. Shared test HTML fixture `TOPIC_HTML` (used in Phases 1–2): `<html><body><h2>Quantum optimization</h2><p>` + 30× `"quantum optimization improves routing "` + `<a href="/paper">paper</a></p><h2>Recipes</h2><p>` + 30× `"bread flour baking kitchen "` + `<a href="/recipes">recipes</a></p></body></html>`, base URL `https://source.example/articles/page`.

## Phases

Each phase file holds the full RED / CONFIRM_RED / GREEN / VERIFY_GREEN instructions. Execute in order.

1. [x] [Phase 1: Query content filter](implementation-phase-1.md) — `filtering.py`: one post-retrieval `PageFilter` (BM25 or LLM by `FETCH_MODEL`), Markdown/plain-text to HTML conversion, absolute links, `ERROR_FILTERING`.
2. [x] [Phase 2: Fetch routing, restrictions, fallback, concurrency](implementation-phase-2.md) — new `create_fetch` signature `(url, query)`, restricted domains, GitHub/arXiv/PDF routing, browser via crawl4ai returning cleaned HTML, provider fallback, concurrency semaphore, removal of Reddit/Wikipedia fetchers, new `FETCH_*` config.
3. [x] [Phase 3: Browser startup and tool availability](implementation-phase-3.md) — `browser_endpoint` (configured CDP, else local Obscura spawn), lifespan enables/disables only `web_research`, evaluation script adapted.
4. [x] [Phase 4: Query propagation and removal of url_context recovery](implementation-phase-4.md) — research agent passes knowledge gap / original question as fetch query, blank results are not evidence, Gemini recovery removed.
5. [x] [Phase 5: Fetch evaluation script](implementation-phase-5.md) — `tests/web_fetch_evaluation.py` replaying JSONL cases, baseline data moved into `tests/evaluation/data/`.
6. [x] [Phase 6: Compose deployment and docs](implementation-phase-6.md) — Compose with shared network namespace Obscura container, `env.example`, documentation updates.
7. [x] [Phase 7: Single HTML-to-Markdown conversion per source](implementation-phase-7.md) — extractors return HTML, providers request HTML, `_filter_markdown` removed, PDF `<p>`/`<br>` blocks, GitHub repo link base, `html2text` dependency removed.

## Final Verification

1. Full offline suite: `test.sh "$(pwd)/tests"` passes with no unrelated regressions.
2. `lint.sh` and `compile.sh` clean on every changed file.
3. Manual (Docker host): `docker compose config --quiet`, `docker compose up -d --build`, MCP `list_tools` includes `web_research`; `docker compose stop browser` then restart `mcps` shows `web_research` absent while vault tools remain.
4. Manual (user permission): `uv run tests/web_fetch_evaluation.py --output tmp/fetch-smoke.jsonl`; full run adds `--cases tests/evaluation/data/fetch-cases-baseline.jsonl`.

## Acceptance

1. Phases 1-5 and 7 tests pass; no unrelated test regressions.
2. Smoke replay run by user shows fewer errors than baseline on matched cases and smaller mean response size; reported numbers only, no target percentage.
