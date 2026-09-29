<!-- Context: read Goal, Specification, Out of Scope, Research Findings, Implementation Research Findings and Conventions in implementation-plan.md before starting this phase. -->

## Phase 2: Fetch routing, restrictions, fallback, concurrency [NEW_FEATURE]

### RED — `tests/test_research_tools.py:93` (rewrite fetch section), `tests/test_research_fallback_fetchers.py:67` (browser section), `tests/test_config.py` (additions)
**Source under test:** `fetch.py:65`, `browser.py:61`, `common.py:83`, `config.py:103-164` (research), `src/mcps/config.py:57-64,129-133`
**Functions under test:** `create_fetch()`, `create_browser_fetch()`, `create_fetch_tool()`, `create_config()`
**Fixtures:**
- `client` existing `tests/test_research_tools.py:18`.
- `bm25 = create_page_filter(fetch_model="", router_url="", router_key="")`.
- `FakeBrowser` — async callable `(url) -> str` returning a configured HTML or error string, records URLs, tracks current/maximum in-flight using an `asyncio.Event` gate.
- `FakeProvider` — async callable `(url) -> str` returning configured Markdown/error, records URLs.
- Existing `FakeCrawler`/`crawl_result` at `tests/test_research_fallback_fetchers.py:35-64`, extend `crawl_result` with `cleaned_html: str = ""`.

Test cases (`create_fetch(http_client=client, browser=FakeBrowser(...), provider=..., page_filter=bm25, restricted_domains=(...), concurrency=2)`):

#### `test_restricted_domain_returns_empty_without_io`
- **Given:** `restricted_domains=("blocked.example",)`; no HTTPXMock responses registered
- **When:** `await fetch(u, "q")` for `https://blocked.example/a`, `https://Sub.Blocked.Example/a`; and `https://notblocked.example/a`
- **Then:** first two `""`, browser not called; third reaches browser

#### `test_generic_url_uses_browser_and_filters`
- **Given:** browser returns `TOPIC_HTML`
- **When:** `await fetch("https://source.example/articles/page", "quantum optimization")`
- **Then:** contains `https://source.example/paper`, not "bread flour"; no httpx request

#### `test_plain_text_url_via_browser_is_filtered`
- **Given:** browser returns `<html><body><pre>` + `html.escape(md)` + `</pre></body></html>` (`md` from Phase 1 `test_markdown_source_is_filterable`)
- **When:** `await fetch("https://source.example/notes.md", "quantum optimization")`
- **Then:** contains `https://source.example/paper`, not "bread flour"; provider not called

#### `test_reddit_and_wikipedia_use_browser`
- **When:** fetch `https://www.reddit.com/r/x/comments/1/t/` and `https://en.wikipedia.org/wiki/Physics`
- **Then:** both URLs recorded by `FakeBrowser`; no `.json` / `action=raw` request

#### `test_specialized_sources_filter_and_never_escalate`
- **Given:** HTTPXMock `https://raw.githubusercontent.com/org/project/main/README.md` → Markdown fixture of Phase 1 `test_markdown_source_is_filterable`; `https://arxiv.org/html/2406.02530` 403, `https://arxiv.org/pdf/2406.02530` 403, `https://arxiv.org/abs/2406.02530` 403; provider configured
- **When:** fetch `https://github.com/org/project` with "quantum optimization"; fetch `https://arxiv.org/abs/2406.02530`
- **Then:** GitHub result filtered (no "bread flour"); arXiv returns `"ERROR: http code 403"`; provider and browser URL lists empty

#### `test_pdf_uses_http_extractor`
- **Given:** HTTPXMock `https://source.example/paper.pdf` → PDF bytes generated in-test with PyMuPDF containing "Quantum routing results", `Content-Type: application/pdf`
- **When:** `await fetch("https://source.example/paper.pdf", None)`
- **Then:** contains "Quantum routing results"; browser not called

#### `test_blocked_page_escalates_to_provider_filtered`
- **Given:** browser returns `"ERROR: http code 403"`; provider returns the Phase 1 Markdown fixture
- **When:** `await fetch(generic, "quantum optimization")`
- **Then:** filtered provider content; parametrize browser result over `"ERROR: empty response"`, `"ERROR: request timeout"`, `"ERROR: fetcher unavailable"` → provider called

#### `test_non_blocking_errors_do_not_escalate`
- **Given:** browser returns `"ERROR: http code 404"`; or provider=None with 403; or provider returns `"ERROR: fetcher unavailable"` after 403
- **Then:** results `"ERROR: http code 404"` (provider untouched), `"ERROR: http code 403"`, `"ERROR: http code 403"`

#### `test_output_truncated_after_filter`
- **Given:** browser HTML of 40,000 chars of relevant "quantum" paragraphs, query None
- **Then:** `len(result) <= 15000 + len("\n\n[Content truncated]")`, ends with `[Content truncated]`

#### `test_browser_concurrency_is_limited`
- **Given:** `concurrency=2`, `FakeBrowser` blocks on gate
- **When:** start 5 `fetch` tasks, yield until 2 in flight, release gate
- **Then:** max in-flight == 2; all 5 complete. Cancel/await tasks in `finally`.

Browser fetcher (`tests/test_research_fallback_fetchers.py:67-107`, replace):
#### `test_browser_fetch_returns_cleaned_html`
- **Given:** `FakeCrawler(result=crawl_result(markdown="# Rendered", cleaned_html="<h1>Rendered</h1>"))`
- **When:** `await create_browser_fetch("ws://cdp", crawler_factory=lambda: crawler)(TARGET)`
- **Then:** `== "<h1>Rendered</h1>"`
Keep `test_browser_fetch_maps_crawl_outcomes` and `test_browser_fetch_connection_failure_is_unavailable` unchanged; delete `test_browser_fetch_truncates_to_max_chars`.

Assembly (`tests/test_research_fallback_fetchers.py:226-281`, replace): `test_create_fetch_fallbacks_from_config` removed; `test_fetch_tool_escalates_blocked_page_to_configured_provider` rewritten: `ServerConfig(scraper_provider="scrape_do", scrape_do_token="t", browser_cdp_url="")` → `create_fetch_tool(config=config, http_client=client)`; with no browser configured generic URLs use `fetch_default` (httpx); 403 → Scrape.do `# Unblocked`; `await fetch(TARGET, None) == "# Unblocked"`.

Config (`tests/test_config.py`):
#### `test_fetch_settings_from_env`
- **Given:** env `FETCH_MODEL=m`, `FETCH_RESTRICTED_DOMAINS=" A.com, b.org ,"`, `FETCH_CONCURRENCY=3`
- **Then:** `fetch_model == "m"`, `fetch_restricted_domains == ("a.com", "b.org")`, `fetch_concurrency == 3`; unset → `""`, `()`, `2`

Delete: `tests/test_research_tools.py:119` wikipedia test, `:219-315` escalation-chain tests (replaced above). Update remaining `create_fetch(http_client=client)` calls at `:100,:140,:156,:167,:176,:186` to the new signature with `browser=None` and call `fetch(url, None)`; delete `:108` truncation test (covered above).

→ **EXPECTED: FAIL** — new signature/behavior absent.

### CONFIRM_RED
Run `test.sh` on `tests/test_research_tools.py`, `tests/test_research_fallback_fetchers.py`, `tests/test_config.py`. Get approval.

### GREEN
- `common.py:28` — `Fetch = Callable[[str, str | None], Awaitable[str]]`; add `Retrieve = Callable[[str], Awaitable[str]]` to `__all__` (browser/provider single-URL callables). `:69-80` escalatable set also contains `ERROR_REQUEST_TIMEOUT`, `ERROR_FETCHER_UNAVAILABLE`.
- `browser.py` — remove optional import/`CRAWL4AI_AVAILABLE` (`:21-28`, docstring `:1-5`); `create_browser_fetch(cdp_url: str, *, crawler_factory: CrawlerFactory | None = None) -> Retrieve`; success returns `crawl.cleaned_html` (keep `_to_fetch_result` status/empty mapping, drop `max_chars`).
- `fetch.py` — delete Reddit/Wikipedia imports and routes (`:25-26,:37-38`); delete files `reddit.py`, `wikipedia.py`. New `create_fetch(*, http_client: httpx.AsyncClient | None, browser: Retrieve | None, provider: Retrieve | None, page_filter: PageFilter, restricted_domains: tuple[str, ...] = (), concurrency: int = 2, max_chars: int = 15000) -> Fetch`. Flow: restricted hostname (`urlparse(url).hostname`, `== d or endswith("." + d)`) → `""`. arXiv/GitHub → existing fetcher with `max_chars=_NO_TRUNCATION` (`10**9`), errors returned as-is, success → `markdown_to_html` → `page_filter` → `format_source_output`. `.pdf` path or `browser is None` → `fetch_default` (same markdown path). Otherwise `async with semaphore: html = await browser(url)`; success → `text_page_to_html(html)` → `page_filter`. Escalatable result + provider → `provider(url)`; `ERROR_FETCHER_UNAVAILABLE` from provider keeps the previous error; provider Markdown → filter. Filtered `""` returned as `""`.
- `research/config.py` — delete `_create_browser_fallback`, `create_fetch_fallbacks` (and `__all__` entry `:36`). Provider factories built with `max_chars=10**9`. `create_fetch_tool(*, config, http_client) -> Fetch` builds `create_fetch(http_client=..., browser=create_browser_fetch(config.browser_cdp_url) if config.browser_cdp_url else None, provider=_create_provider_fallback(...), page_filter=create_page_filter(fetch_model=config.fetch_model, router_url=config.router_api_base, router_key=config.router_api_key), restricted_domains=config.fetch_restricted_domains, concurrency=config.fetch_concurrency)`. `ResearchConfig.fetch: Fetch` (`:50`).
- `src/mcps/config.py:64` — add `fetch_model: str = ""`, `fetch_restricted_domains: tuple[str, ...] = ()`, `fetch_concurrency: int = 2`; env reads at `:133` (`FETCH_CONCURRENCY` via `int(os.environ.get(...) or 2)`, domains split/strip/lower/drop empty).
- `tools/__init__.py` — drop removed exports; fix docstring.
- `deep_research.py:396,413` — temporarily call `self.config.fetch(url, None)` (Phase 4 replaces).
→ **EXPECTED: PASS**.

### VERIFY_GREEN
Run the three test files plus `tests/test_deep_research.py`, `tests/test_fetch_filtering.py`; lint/compile changed files.
