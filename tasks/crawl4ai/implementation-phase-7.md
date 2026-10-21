<!-- Context: read Goal, Specification, Out of Scope, Research Findings, Implementation Research Findings and Conventions in implementation-plan.md before starting this phase. -->

## Phase 7: Single HTML-to-Markdown conversion per source [NEW_FEATURE]

Every source reaches `page_filter` (`filtering.py:136`) as HTML; the filter's `generate_markdown` is the only HTML→Markdown step. Markdown/plain-text sources are converted Markdown→HTML exactly once. GitHub relative links resolve inside the repository.

### RED
**Source under test:** `src/mcps/research/tools/extract.py:20-97`, `default.py:38`, `github.py:69-114`, `fetch.py:99-143`, `scrape_do.py:62-69`, `bright_data.py:57-58`
**Functions under test:** `create_fetch()`, `create_scrape_do_fetch()`, `create_bright_data_fetch()`, `create_fetch_tool()`
**Fixtures:** existing `client`, `bm25`, `FakeBrowser`, `FakeProvider`, `_pdf_bytes` (`tests/test_research_tools.py:37-92`); `create_fetch(http_client=client, browser=None, provider=None, page_filter=bm25)` unless stated.

`tests/test_research_tools.py` (append to "Direct (no browser) fetch behavior", after `:466`):

#### `test_fetch_default_html_is_converted_once`
- **Use Case:** Spec 9 (single conversion)
- **Given:** HTTPXMock `https://site.example/page` → html `<html><body><h1>Tags</h1><p>Use &lt;style&gt;alpha&lt;/style&gt; carefully <img src="/fig.png" alt="Fig"></p></body></html>`
- **When:** `await fetch("https://site.example/page", None)`
- **Then:** contains `Use <style>alpha</style> carefully` and `![Fig](https://site.example/fig.png)`
- → today: `Use  carefully`, no image (html2text drops images, second pass eats `<style>`)

#### `test_arxiv_html_is_converted_once`
- **Given:** HTTPXMock `https://arxiv.org/html/2401.00002` → same HTML as above
- **When:** `await fetch("https://arxiv.org/abs/2401.00002", None)`
- **Then:** contains `Use <style>alpha</style> carefully` and `![Fig](https://arxiv.org/fig.png)`

#### `test_pdf_keeps_line_breaks`
- **Given:** monkeypatch `mcps.research.tools.extract._convert_pdf_to_text` to return `"Quantum routing results\ncol_a   col_b\n1       2\n\nbread flour baking"`; HTTPXMock `https://source.example/paper.pdf` → `content=b"%PDF-1.4"`, `content-type: application/pdf`
- **When:** `await fetch(url, None)` and `await fetch(url, "quantum routing")`
- **Then:** unfiltered contains `"Quantum routing results  \ncol_a col_b  \n1 2"` and `"bread flour baking"`; filtered contains `"col_a col_b"` and not `"bread flour"` (values verified with crawl4ai 0.9.4)

#### `test_blank_html_is_empty_response`
- **Given:** HTTPXMock → html `<html><body>  </body></html>`
- **Then:** `await fetch("https://site.example/blank", None) == "ERROR: empty response"` (guard; passes today)

#### `test_github_repo_relative_link_resolves_in_repository`
- **Given:** HTTPXMock `https://raw.githubusercontent.com/org/project/main/README.md` → text `"See [guide](docs/guide.md) and [root](/paper)."`
- **When:** `await fetch("https://github.com/org/project", None)`
- **Then:** contains `https://github.com/org/project/blob/HEAD/docs/guide.md` and `https://github.com/paper`
- → today: `https://github.com/org/docs/guide.md`

#### `test_github_blob_relative_link_resolves_in_directory`
- **Given:** HTTPXMock `https://raw.githubusercontent.com/o/r/main/docs/README.md` → text `"See [guide](guide.md)."`
- **When:** `await fetch("https://github.com/o/r/blob/main/docs/README.md", None)`
- **Then:** contains `https://github.com/o/r/blob/main/docs/guide.md` (guard; passes today)

#### `test_provider_html_is_filtered_once`
- **Given:** `FakeBrowser("ERROR: http code 403")`; `FakeProvider("<html><body><h2>Quantum optimization</h2><p>" + "quantum optimization improves routing " * 30 + '<a href="/paper">paper</a></p><h2>Recipes</h2><p>' + "bread flour baking kitchen " * 30 + "</p></body></html>")` (= `TOPIC_HTML` without the recipes link)
- **When:** `await fetch(GENERIC, "quantum optimization")` with `http_client=None, browser=..., provider=..., page_filter=bm25`
- **Then:** contains `https://source.example/paper`, not "bread flour"

Update existing tests (same file):
- `:283-294` `test_blocked_page_escalates_to_provider_filtered` and `:195-207` provider — `FakeProvider` returns `TOPIC_HTML` instead of `MARKDOWN_PAGE`; assertions unchanged.
- `FakeProvider` docstring `:64` → "HTML/error".
- `:245-248` unchanged (root-relative `/paper` still → `https://github.com/paper`).

`tests/test_research_fallback_fetchers.py`:
- `:125-143` rename to `test_scrape_do_requests_rendered_html_with_unblocking`; expected params drop `"output": "markdown"`; response `text="<h1>Article</h1>"`, `result == "<h1>Article</h1>"`.
- `:184-205` rename to `test_bright_data_posts_zone_request_for_html`; expected body drops `"data_format"`; payload body `"<h1>Article</h1>"`, `result == "<h1>Article</h1>"`.
- `:240-252` drop `"output": "markdown"` from the mocked Scrape.do params; response `text="<html><body><h1>Unblocked</h1></body></html>"`; `await fetch(TARGET, None) == "# Unblocked"`.

→ **EXPECTED: FAIL** — `test_fetch_default_html_is_converted_once`, `test_arxiv_html_is_converted_once`, `test_pdf_keeps_line_breaks` (no `_convert_pdf_to_text`), `test_github_repo_relative_link_resolves_in_repository`, both provider request-shape tests, the Scrape.do assembly test. Guards `test_blank_html_is_empty_response`, `test_github_blob_relative_link_resolves_in_directory`, `test_provider_html_is_filtered_once` pass.

### CONFIRM_RED
Run `test.sh` on `tests/test_research_tools.py` and `tests/test_research_fallback_fetchers.py`; failures match the list above. Get approval.

### GREEN
- `extract.py` — extractors return HTML:
  - Delete `_convert_html_to_markdown` `:20-24` and `import html2text` `:7`.
  - `_extract_html_response` `:55` → `response.text` if `lxml.html.document_fromstring(response.text).text_content().strip()` is nonblank (parse errors → `None`), else `None`. `# ponytail: <script>/<style> text counts as content; browser path handles JS shells`.
  - Rename `_convert_pdf_to_markdown` `:27` → `_convert_pdf_to_text` (body unchanged). `_extract_pdf_response` `:59` → split text on `"\n\n"`, drop blank blocks, join `f"<p>{html.escape(block).replace(chr(10), '<br>')}</p>"`; `None` if no blocks.
  - `_extract_plain_text_response` `:65` → `markdown_to_html(content)` (import from `mcps.research.tools.filtering`; no cycle: `filtering.py` imports only `common`).
  - Module docstring `:1` → "Response body to HTML extractors keyed by content type."
- `default.py:38` — docstring: returns HTML. Code unchanged.
- `github.py` — `fetch_github_blob` `:90` and `fetch_github_repo` `:112` return `format_source_output(url, markdown_to_html(content), max_chars)`. Comment on the blob path: `# ponytail: source code becomes paragraphs (indentation lost); <pre> keeps layout but BM25 drops <pre> blocks for any query (verified 0.9.4) — per-language handling if code fidelity matters`. Add public `github_link_base(url: str) -> str`: `is_github_repo_url(url)` → `f"{url.rstrip('/')}/blob/HEAD/"`, else `url`; add to `__all__`.
- `fetch.py` — delete `_filter_markdown` `:102-103` and the `markdown_to_html` import `:25`. `_filter(html, url, query, base_url=url)` passes `base_url` to `page_filter`. Specialized route `:118` → `_filter(result, url, query, github_link_base(url))`. `fetch_default` success `:126` → `_filter(result, url, query)`. Provider success `:142` → `_filter(text_page_to_html(candidate), url, query)`. `format_source_output` keeps `url` (logging) not `base_url`.
- `scrape_do.py:68` — delete `"output": "markdown"`; docstring `:1` → "(rendered HTML output)".
- `bright_data.py:58` — delete `"data_format": "markdown"`; docstring `:1` → "(HTML output)".
- `uv remove html2text` (only user was `extract.py:7`; crawl4ai vendors its own `crawl4ai/html2text`).
- Docs: `docs/dependencies_libraries.md:60-62` delete html2text section; `docs/deployment_infrastructure.md:67` delete line; `docs/packages_modules.md:123` drop html2text from "Uses", `:131` → "HTML/PDF/plain-text to HTML by content type".
→ **EXPECTED: PASS** — all tests in both files.

### VERIFY_GREEN
Run `tests/test_research_tools.py`, `tests/test_research_fallback_fetchers.py`, `tests/test_fetch_filtering.py`, `tests/test_web_fetch_evaluation.py`; lint/compile changed files. `rg -n "html2text" src pyproject.toml` returns nothing. **Manual check (user permission):** `uv run tests/web_fetch_evaluation.py --output tmp/fetch-smoke-p7.jsonl`; compare summary with the Phase 5 smoke run.
