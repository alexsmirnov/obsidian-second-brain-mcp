# Research: Web Fetch Tools for Web Research (crawl4ai + Obscura)

## Summary

The spec targets the web-fetch layer under `src/mcps/research/tools/`. Today a
`Fetch` callable takes a single URL (`Callable[[str], Awaitable[str]]`,
`common.py:28`), routes to site-specific fetchers when the hostname matches, and
otherwise uses a generic `httpx` extractor. When the direct result is
blocked/empty, an ordered fallback chain (crawl4ai-over-CDP browser, then one
commercial provider) is tried. Fallbacks are assembled from `ServerConfig` in
`research/config.py`; the crawl4ai browser fallback only runs when
`BROWSER_CDP_URL` is set (`config.py:103-112`).

No code exists today for: a `query` parameter on fetch, query-relevance
filtering (BM25 or LLM), a restricted-domain list, `FETCH_MODEL`, keeping links
as structured output, starting/authenticating an Obscura browser, limiting
parallel fetches via config, conditional disabling of the `web_research` tool,
or a `docker compose` fleet. `crawl4ai>=0.7` is listed as a mandatory
dependency (`pyproject.toml:28`; installed `0.9.4`), although `browser.py:3-4`
and `env.example:57-58` describe it as an optional extra (`uv sync --extra
browser`); `pyproject.toml` has no `[project.optional-dependencies]` browser
extra, so crawl4ai is always installed.

Related prior notes in the Obsidian vault (context, not code): "Crawl4AI:
Open-Source LLM Friendly Web Crawler and Scrapper" and "Install Obscura headless
browser in (part-of:: [[Ai SWE assistant]]) docker container".

Baseline failure evidence from `tmp/evaluation-20260928_120233.log`: 197
`Fetch N chars` successes+attempts, 91 fetch failures = 81 × `ERROR: http code
403`, 8 × `ERROR: empty response`, 1 × 404, 1 × 405. The log has zero
`Escalating` and zero `Browser fetch` lines, i.e. the baseline run had no
fallbacks configured. Failures cluster on publisher domains
(`sciencedirect.com` 18, `reddit.com` 11, `link.aps.org` 7, `pubmed...` 5,
`papers.ssrn.com` 5, `academic.oup.com` 5, …).

## Research Findings

Files:
- `src/mcps/research/tools/fetch.py:1` — fetch router + escalation chain; entry point for all requests.
- `src/mcps/research/tools/common.py:1` — `Fetch` type alias, error strings, escalation predicate, truncation, shared `request_get`.
- `src/mcps/research/tools/default.py:1` — generic `httpx` fetcher, content-type extraction.
- `src/mcps/research/tools/browser.py:1` — crawl4ai-over-CDP fallback fetcher; imports crawl4ai here.
- `src/mcps/research/tools/arxiv.py:1` — site fetcher, tries html→pdf→abs.
- `src/mcps/research/tools/github.py:1` — blob→raw and repo→README fetchers.
- `src/mcps/research/tools/reddit.py:1` — `.json` endpoint fetcher.
- `src/mcps/research/tools/wikipedia.py:1` — `?action=raw` fetcher.
- `src/mcps/research/tools/scrape_do.py:1` — Scrape.do commercial fallback.
- `src/mcps/research/tools/bright_data.py:1` — Bright Data Web Unlocker fallback.
- `src/mcps/research/tools/extract.py:1` — HTML/PDF/plain content-type extractors.
- `src/mcps/research/config.py:1` — builds `ResearchConfig`, models, search, and fetch (fallback assembly).
- `src/mcps/config.py:1` — `ServerConfig` fields + env loading + `validate_config`.
- `src/mcps/research/tools/__init__.py:1` — re-exports `Fetch`, `SearchResult`, `create_fetch`, search factories.
- `pyproject.toml:28` — `crawl4ai>=0.7` in `[project].dependencies`; no browser extra defined.
- `src/mcps/research/lifespan.py:1` — owns `httpx.AsyncClient`, builds config + researcher.
- `src/mcps/research/deep_research.py:1` — LangGraph agent; calls `config.fetch`.
- `src/mcps/server.py:1` — FastMCP server; registers `web_research` tool.
- `Dockerfile:1` — single-image Python build; no compose/browser service.
- `env.example:1` — documented env template (no `FETCH_MODEL`, restricted list, or parallel limit).
- `tests/test_research_tools.py:1`, `tests/test_research_fallback_fetchers.py:1` — fetch/fallback contract tests.
- `tests/test_config.py:1`, `tests/test_server.py:1` — config and server CLI tests.
- `tests/web_research_evaluation.py:1` + `tests/evaluation/` — DRACO evaluation harness writing logs/reports to `tmp/`.
- `docs/config_environment.md:213-237` — documents fetch fallbacks, `BROWSER_CDP_URL`, `SCRAPER_PROVIDER`, provider tokens.
- `docs/dependencies_libraries.md:82-83` — documents crawl4ai as optional extra `browser`.
- `docs/packages_modules.md:122-135` — documents research tools modules and `create_fetch` behavior.
- `docs/architecture_overview.md:155` — states `web_research` is always registered.
- `docs/deployment_infrastructure.md:154` — documents tool registration step.

Functions/Methods under change:
- `src/mcps/research/tools/common.py:28` `Fetch = Callable[[str], Awaitable[str]]` — current fetch contract; one positional URL, returns `str`.
- `src/mcps/research/tools/fetch.py:65` `create_fetch(*, http_client=None, max_chars=15000, fallbacks=()) -> Fetch` — returns inner `fetch(url) -> str` (`fetch.py:79`); routes direct, then escalates over fallbacks. Escalation is applied to every direct result, including site-routed arxiv/wikipedia/reddit/github results — no known-domain exemption exists today.
- `src/mcps/research/tools/fetch.py:44` `_select_fetcher(url) -> SiteFetcher` — first matching predicate wins; falls through to `fetch_default`.
- `src/mcps/research/tools/fetch.py:35-41` `_SITE_ROUTES` — arxiv, wikipedia, reddit, github blob, github repo.
- `src/mcps/research/tools/fetch.py:51` `_fetch_direct(url, *, http_client, max_chars) -> str` — catches `httpx.HTTPError`/`Exception` into error strings.
- `src/mcps/research/tools/common.py:83` `is_escalatable(result: str) -> bool` — true only for 401/403/429 strings or `ERROR: empty response`.
- `src/mcps/research/tools/common.py:97` `format_source_output(url, content, max_chars) -> str` — logs size, truncates with `[Content truncated]`.
- `src/mcps/research/tools/browser.py:61` `create_browser_fetch(cdp_url, *, max_chars=15000, crawler_factory=None) -> Fetch` — builds `AsyncWebCrawler` per call, returns `markdown.raw_markdown`.
- `src/mcps/research/tools/browser.py:35` `_default_crawler_factory(cdp_url)` — `BrowserConfig(browser_mode="custom", cdp_url=..., cache_cdp_connection=True, create_isolated_context=True)`.
- `src/mcps/research/config.py:103` `_create_browser_fallback(config) -> Fetch | None` — `None` if no `BROWSER_CDP_URL` or crawl4ai missing.
- `src/mcps/research/config.py:146` `create_fetch_fallbacks(*, config, http_client) -> list[Fetch]` — order: browser, then provider.
- `src/mcps/research/config.py:157` `create_fetch_tool(*, config, http_client) -> Callable[[str], Awaitable[str]]` — wraps `create_fetch` with fallbacks.
- `src/mcps/research/config.py:167` `build_research_config(config, *, http_client) -> ResearchConfig` — sets `fetch=create_fetch_tool(...)`.
- `src/mcps/research/lifespan.py:13` `build_research_lifespan(config) -> Lifespan` — yields `{"researcher", "http_client"}` (`lifespan.py:26`).
- `src/mcps/research/deep_research.py:387` `web_research(state, config)` — direct-URL path calls `self.config.fetch(query)` (`:396`); search path calls `asyncio.gather(*(self.config.fetch(result.url) ...))` (`:413-414`).
- `src/mcps/research/deep_research.py:285` `_is_valid_url(url)` — decides direct-fetch vs search.
- `src/mcps/research/deep_research.py:424` `clean_result(results, fetch_results, question, knowledge_gap)` — separates `"ERROR"`-prefixed results; formats successful content and LLM-filters it (`_CLEAN_RESULT_PROMPT`, `:74-90`).
- `src/mcps/server.py:121` `web_research(query, ctx)` — FastMCP tool; reads `ctx.lifespan_context["researcher"]`, invokes it with a progress reporter.

Types/Classes:
- `src/mcps/research/config.py:44` `ResearchConfig` — fields `fast: BaseChatModel`, `small: BaseChatModel`, `search: Callable[[str], Awaitable[list[SearchResult]]]`, `fetch: Callable[[str], Awaitable[str]]`.
- `src/mcps/config.py:14` `ServerConfig` — relevant fields: `research_fast_model: str` (`:54`), `research_infer_model: str` (`:56`), `browser_cdp_url: str` (`:60`), `scraper_provider: str` (`:61`), `scrape_do_token: str` (`:62`), `bright_data_api_key: str` (`:63`), `bright_data_zone: str` (`:64`).
- `src/mcps/research/tools/models.py:10` `SearchResult` — `url: str`, `title: str`, `snippet: str`.
- `src/mcps/research/deep_research.py:247` `WebSearchState` — per-branch payload: `original_question`, `knowledge_gap`, `search_query`, `id` (query text available here, not passed to fetch today).

Integration points:
- `src/mcps/research/lifespan.py:19` — `build_research_config` invoked in FastMCP lifespan.
- `src/mcps/research/config.py:196` — `fetch=create_fetch_tool(config=config, http_client=http_client)`.
- `src/mcps/server.py:96` — `build_research_lifespan(self.config)` passed to `FastMCP(... lifespan=...)` (`:101-105`).
- `src/mcps/server.py:123-129` — tool resolves `researcher` from lifespan context; no enable/disable flag.
- `src/mcps/research/deep_research.py:396,413` — the only `config.fetch` call sites.
- `src/mcps/research/deep_research.py:455-472` — `clean_result` already LLM-filters fetched text post-hoc; `bind_tools([{"url_context": {}}])` Gemini fallback path (`:467`).
- `env.example:55-64` — documents `BROWSER_CDP_URL`, `SCRAPER_PROVIDER`, `SCRAPE_DO_TOKEN`, `BRIGHT_DATA_*`.
- `Dockerfile:44` — container runs `mcps --transport http --host 0.0.0.0 --port 8000`; no browser sidecar.
- `docs/config_environment.md:213-237`, `docs/dependencies_libraries.md:82-83`, `docs/packages_modules.md:122-135`, `docs/architecture_overview.md:155` — document the fetch fallback/env/tool behavior that the spec changes.

Test files:
- `tests/test_research_tools.py:93` — default HTML→markdown, truncation, Wikipedia/GitHub/arXiv routing, timeout/unsupported/HTTP-code error mapping.
- `tests/test_research_tools.py:210-315` — escalation chain: escalation triggers, non-escalation, ordering, unavailable-fallback skip, last-error retention.
- `tests/test_research_fallback_fetchers.py:35-107` — browser fetcher contract with a `FakeCrawler` (markdown, truncation, status mapping, connection failure).
- `tests/test_research_fallback_fetchers.py:115-218` — Scrape.do and Bright Data request shape + status mapping.
- `tests/test_research_fallback_fetchers.py:226-281` — `create_fetch_fallbacks` assembly from `ServerConfig`; end-to-end escalation to provider.
- `tests/test_config.py:1` — config env precedence, model defaults, `validate_config` warnings.
- `tests/test_server.py:72` — server-mode `create_server`/`start` invocation.
- `tests/web_research_evaluation.py:39` — DRACO eval entry; builds research config, runs agent, writes `tmp/evaluation-*.log` + HTML (`report.py:148` `REPORT_DIR = Path("tmp")`).

Missing coverage:
- No test passes a `query` argument to fetch or asserts query-relevant filtering / link preservation.
- No test for a restricted-domain list returning empty.
- No test for `FETCH_MODEL` selection (LLM vs BM25).
- No test for browser auto-start/`obscura serve`, CDP bearer auth, or tool disabling when the browser is unavailable.
- No test for limiting parallel fetches.
- No test for a `docker compose` deployment.
- No test that `web_research` registration is conditional.

## Implementation Research Findings

- **crawl4ai CDP bearer auth** — considered: (a) `BrowserConfig` public option, (b) URL-embedded token (`?token=`), (c) forward headers into Playwright `connect_over_cdp`. Findings: crawl4ai `0.9.4` `BrowserConfig` has `cdp_url`, `headers`, `extra_args`, but no CDP-handshake header field; `BrowserConfig.headers` is applied to the browser context via `set_extra_http_headers` (`crawl4ai/browser_manager.py:570`), not the CDP socket. crawl4ai calls `playwright.chromium.connect_over_cdp(cdp_url)` with no `headers` in both the cached pool (`browser_manager.py:632`) and managed path (`browser_manager.py:920`); HTTP `cdp_url` pre-check GETs `/json/version` without auth (`browser_manager.py:1050+`), while `ws://` URLs skip that check. Playwright's `connect_over_cdp` does accept `headers=` (`playwright` installed). Obscura officially documents token delivery as the `Authorization: Bearer <token>` CDP client header (`docs.obscura.sh/reference/environment-variables`). Chosen: forward `Authorization: Bearer` into the Playwright CDP connect (the capability exists beneath crawl4ai); why: spec mandates crawl4ai as the fetch engine and Obscura's documented auth is a header. `?token=` for Obscura CDP is `[UNVERIFIED: not documented by Obscura; only asserted by a third-party summary]`.
- **Obscura browser runtime** — considered: managed/remote CDP vs embedding. Chosen: connect to an external Obscura CDP endpoint; fall back to spawning `obscura serve --stealth --allow-private-network`. Findings (docs.obscura.sh CLI + env refs): `obscura serve` default endpoint `ws://127.0.0.1:9222`; `--host` default `127.0.0.1`; `--stealth`, `--proxy`, `--allow-private-network` are global flags; token env `OBSCURA_CDP_TOKEN`, optional on loopback, required (≥32 bytes) for non-loopback binds; Docker image `h4ckf0r0day/obscura` runs as uid/gid 65532 with default command `serve`; container bind must be `--host 0.0.0.0` and requires `OBSCURA_CDP_TOKEN`.
- **Query filtering (BM25)** — considered: `crawl4ai.content_filter_strategy.BM25ContentFilter` vs the already-present `rank_bm25.BM25Okapi`. Chosen: crawl4ai `BM25ContentFilter` wired through `DefaultMarkdownGenerator(content_filter=...)` in `CrawlerRunConfig.markdown_generator`; why: filtering happens during crawl and exposes `result.markdown.fit_markdown` alongside `raw_markdown` (`docs.crawl4ai.com/core/fit-markdown`). `rank_bm25` is a declared dependency (`pyproject.toml:15`) but currently unused anywhere in `src/`.
- **Query filtering (LLM)** — considered: crawl4ai `LLMContentFilter` vs the existing post-hoc LLM cleanup in `deep_research.clean_result`. Chosen: crawl4ai `LLMContentFilter(llm_config=...)` via `DefaultMarkdownGenerator` when `FETCH_MODEL` is set; why: filtering is co-located with the crawl and yields `fit_markdown` (`crawl4ai` 0.5.0 release docs). `LLMConfig` uses an OpenAI-compatible provider/token, so the existing router (`router_api_base`/`router_api_key`) is the credential source.
- **Link preservation** — considered: crawl4ai `result.links` (`internal`/`external`) and `CrawlerRunConfig` link options (`exclude_external_links`, `exclude_domains`) vs markdown-only output. Findings: crawl4ai returns structured links in `CrawlResult.links`; `raw_markdown` retains inline links; `DefaultMarkdownGenerator` docs note `fit_markdown`/`fit_html` are populated only when a filter is used. Chosen: read `CrawlResult.links` and keep links; why: spec requires links preserved on fetch.
- **Restricted-domain list** — considered: new `ServerConfig` field populated from an env var (comma-separated). Findings: no existing deny/allow/restricted-domain config anywhere in `src/`; `crawl4ai` itself has `exclude_domains` but only for content/link exclusion, not whole-fetch short-circuit. Chosen: configurable list on `ServerConfig` (new field, empty default) checked before any fetch; why: spec says configurable restricted list → empty result.
- **Parallel-load limit** — considered: `asyncio.Semaphore` around `config.fetch` in `deep_research.web_research` vs crawl4ai's `CrawlerRunConfig.semaphore_count` (default 5) vs a new config field. Findings: current fan-out uses unbounded `asyncio.gather` over search results (`deep_research.py:413-414`); a `asyncio.Semaphore` limiter pattern already exists in the repo (`tests/evaluation/pipeline.py:25` `JUDGE_CONCURRENCY = 10`; semaphore constructed at `:35`). Chosen: a new `ServerConfig` parallel-limit field applied with an `asyncio.Semaphore` to the fetch fan-out (and/or crawl4ai `semaphore_count`); why: spec requires a configured parallel-load limit independent of crawl4ai internals.
- **Tool enable/disable** — considered: conditional FastMCP tool registration based on browser availability detected during lifespan. Findings: `web_research` is registered unconditionally in `DevAutomationServer.register()` (`server.py:121`); lifespan already owns startup (`lifespan.py:13-26`). Chosen: resolve browser availability during the research lifespan and register the tool only when the researcher/fetch backend is available; why: spec says disable the tool if no browser can be reached or started.
- **Commercial fallback after crawl4ai** — considered: existing `scrape_do`/`bright_data` fetchers. Findings: both exist as `Fetch` callables and are assembled by `create_fetch_fallbacks` (`config.py:146-154`) after the browser; `is_escalatable` gates them (`common.py:83`). Chosen: reuse the existing provider fallbacks, appending query-relevant filtering; why: spec says fall back to the commercial provider when crawl4ai fails.
- **Baseline measurement** — considered: `tests/web_research_evaluation.py` DRACO harness. Findings: eval writes `tmp/evaluation-<ts>.log` and HTML (`report.py:148,318-321`); baseline log shows 91/197 failed fetches with no fallback escalation configured. Chosen: re-run this same harness against the new fetch layer to compare failure counts; why: spec's success is reducing logged failures.
- **Deployment fleet** — considered: single Dockerfile (current) vs `docker compose` with an Obscura service + MCP service. Findings: only `Dockerfile` and `.dockerignore` exist; no compose file. Chosen: two-service compose (Obscura: `h4ckf0r0day/obscura serve --host 0.0.0.0 --stealth --allow-private-network` with `OBSCURA_CDP_TOKEN`; MCP: existing image with `BROWSER_CDP_URL`/`OBSCURA_CDP_TOKEN`); why: spec in-scope deployment is a docker compose fleet of one browser + one MCP container.

## External References

- https://docs.obscura.sh/reference/environment-variables — `OBSCURA_CDP_TOKEN` bearer header; private-network defaults.
- https://docs.obscura.sh/reference/cli-reference — `obscura serve` flags (`--host`, `--stealth`, `--allow-private-network`).
- https://docs.obscura.sh/guides/run-in-production-at-scale — Docker image `h4ckf0r0day/obscura`, uid 65532, container `--host 0.0.0.0` + token.
- https://docs.obscura.sh/quickstart/connect-puppeteer-or-playwright — Playwright `connectOverCDP` usage.
- https://docs.crawl4ai.com/core/browser-crawler-config/ — `BrowserConfig` fields (`cdp_url`, `headers`, `browser_mode`).
- https://docs.crawl4ai.com/core/fit-markdown/ — `BM25ContentFilter` + `DefaultMarkdownGenerator`, `fit_markdown`.
- https://github.com/unclecode/crawl4ai/blob/main/docs/md_v2/blog/releases/0.5.0.md — `LLMContentFilter` + `LLMConfig`.
- https://playwright.dev/python/docs/api/class-browsertype — `connect_over_cdp(..., headers=...)`.
- https://pypi.org/project/rank-bm25/ — `BM25Okapi` API (already a dependency).
