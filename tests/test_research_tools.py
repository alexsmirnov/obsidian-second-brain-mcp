"""Contract tests for research search and fetch callables (HTTP mocked)."""

from __future__ import annotations

import asyncio
import html
from collections.abc import AsyncIterator

import httpx
import pymupdf
import pytest
from pytest_httpx import HTTPXMock

from mcps.research.tools import (
    create_duckduckgo_search,
    create_fetch,
    create_google_search,
)
from mcps.research.tools.arxiv import ArxivFetch
from mcps.research.tools.default import HttpFetch
from mcps.research.tools.filtering import Bm25RelevanceFilter
from mcps.research.tools.github import GitHubBlobFetch, GitHubRepoFetch
from mcps.research.tools.models import FetchResult, FetchStatus

_ERROR_SCRIPTS = {
    "ERROR: http code 403": (FetchStatus.HTTP_ERROR, 403),
    "ERROR: http code 404": (FetchStatus.HTTP_ERROR, 404),
    "ERROR: empty response": (FetchStatus.EMPTY, None),
    "ERROR: request timeout": (FetchStatus.TIMEOUT, None),
    "ERROR: fetcher unavailable": (FetchStatus.UNAVAILABLE, None),
}


def scripted_result(url: str, script: str) -> FetchResult:
    """Turn a scripted HTML body or ``ERROR: ...`` marker into a FetchResult."""
    if script in _ERROR_SCRIPTS:
        status, http_status = _ERROR_SCRIPTS[script]
        return FetchResult(url, status, "", http_status=http_status)
    return FetchResult(url, FetchStatus.OK, "text/html", script)

TOPIC_HTML = (
    "<html><body><h2>Quantum optimization</h2><p>"
    + "quantum optimization improves routing " * 30
    + '<a href="/paper">paper</a></p><h2>Recipes</h2><p>'
    + "bread flour baking kitchen " * 30
    + '<a href="/recipes">recipes</a></p></body></html>'
)
GENERIC = "https://source.example/articles/page"
MARKDOWN_PAGE = (
    "## Quantum\n\n"
    + "quantum optimization improves routing " * 30
    + "[paper](/paper)\n\n## Recipes\n\n"
    + "bread flour baking kitchen " * 30
)


class FakeBrowser:
    """Async browser stub returning HTML/error per URL and tracking concurrency."""

    def __init__(
        self, result: str | dict[str, str], *, gate: asyncio.Event | None = None
    ):
        self.result = result
        self.gate = gate
        self.calls: list[str] = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def __call__(self, url: str, query: str | None = None, /) -> FetchResult:
        self.calls.append(url)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self.gate is not None:
                await self.gate.wait()
            if isinstance(self.result, dict):
                return scripted_result(
                    url, self.result.get(url, "ERROR: empty response")
                )
            return scripted_result(url, self.result)
        finally:
            self.in_flight -= 1


class FakeProvider:
    """Async provider stub returning HTML/error and recording calls."""

    def __init__(self, result: str) -> None:
        self.result = result
        self.calls: list[str] = []

    async def __call__(self, url: str, query: str | None = None, /) -> FetchResult:
        self.calls.append(url)
        return scripted_result(url, self.result)


class MarkdownBrowser:
    """Browser stub returning a declared Markdown result."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[str] = []

    async def __call__(self, url: str, query: str | None = None, /) -> FetchResult:
        self.calls.append(url)
        return FetchResult(url, FetchStatus.OK, "text/markdown", self.content)


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(follow_redirects=True) as http_client:
        yield http_client


@pytest.fixture
def bm25():
    return Bm25RelevanceFilter()


def _pdf_bytes(text: str) -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    data = document.tobytes()
    document.close()
    return data


def _pdf_paragraphs(paragraphs: list[str]) -> bytes:
    """One text box per page; pymupdf joins pages with a blank line."""
    document = pymupdf.open()
    for text in paragraphs:
        page = document.new_page()
        page.insert_textbox(pymupdf.Rect(72, 72, 520, 700), text, fontsize=11)
    data = document.tobytes()
    document.close()
    return data


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


async def test_google_search_parses_items_and_skips_missing_links(
    httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    httpx_mock.add_response(
        json={
            "items": [
                {"link": "https://a.example", "title": "A", "snippet": "sa"},
                {"title": "no link"},
            ]
        }
    )
    search = create_google_search("key", "cx", http_client=client)

    results = await search("query")

    assert [(r.url, r.title, r.snippet) for r in results] == [
        ("https://a.example", "A", "sa")
    ]


async def test_google_search_without_credentials_returns_empty(
    client: httpx.AsyncClient,
):
    search = create_google_search("", "", http_client=client)

    assert await search("query") == []


async def test_duckduckgo_search_parses_results_and_skips_ads(
    httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    httpx_mock.add_response(
        html=(
            "<html><body>"
            "<div class='result body'><h2>Title One</h2>"
            "<a href='https://one.example'>Snippet one</a></div>"
            "<div class='result body'><h2>Ad</h2>"
            "<a href='https://duckduckgo.com/y.js?ad=1'>ad</a></div>"
            "</body></html>"
        )
    )
    search = create_duckduckgo_search(http_client=client)

    results = await search("query")

    assert [(r.url, r.title, r.snippet) for r in results] == [
        ("https://one.example", "Title One", "Snippet one")
    ]


async def test_duckduckgo_search_http_error_returns_empty(
    httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    httpx_mock.add_response(status_code=500)
    search = create_duckduckgo_search(http_client=client)

    assert await search("query") == []


# ---------------------------------------------------------------------------
# Fetch routing, restrictions, fallback, concurrency
# ---------------------------------------------------------------------------


async def test_restricted_domain_is_restricted_without_io(bm25):
    browser = FakeBrowser(TOPIC_HTML)
    provider = FakeProvider(TOPIC_HTML)
    fetch = create_fetch(
        http_client=None,
        browser=browser,
        provider=provider,
        page_filter=bm25,
        restricted_domains=("blocked.example",),
    )

    exact = await fetch("https://blocked.example/a", "q")
    subdomain = await fetch("https://Sub.Blocked.Example/a", "q")
    allowed = await fetch("https://notblocked.example/a", "q")

    assert (exact.status, exact.content) == (FetchStatus.RESTRICTED, "")
    assert subdomain.status is FetchStatus.RESTRICTED
    assert allowed.ok
    assert browser.calls == ["https://notblocked.example/a"]
    assert provider.calls == []


async def test_generic_url_uses_browser_and_filters(bm25, httpx_mock: HTTPXMock):
    browser = FakeBrowser(TOPIC_HTML)
    fetch = create_fetch(
        http_client=None, browser=browser, provider=None, page_filter=bm25
    )

    result = await fetch(GENERIC, "quantum optimization")

    assert "https://source.example/paper" in result.content
    assert "bread flour" not in result.content
    assert browser.calls == [GENERIC]
    assert httpx_mock.get_requests() == []


async def test_declared_markdown_via_browser_is_filtered(bm25):
    browser = MarkdownBrowser(MARKDOWN_PAGE)
    provider = FakeProvider(TOPIC_HTML)
    fetch = create_fetch(
        http_client=None, browser=browser, provider=provider, page_filter=bm25
    )

    result = await fetch("https://source.example/notes.md", "quantum optimization")

    assert "https://source.example/paper" in result.content
    assert "bread flour" not in result.content
    assert provider.calls == []


async def test_browser_html_pre_code_survives_rendering(bm25):
    code = "def quantum():\n    return 1"
    page = (
        "<html><body><h2>Quantum</h2><pre><code>"
        + html.escape(code)
        + "</code></pre></body></html>"
    )
    browser = FakeBrowser(page)
    fetch = create_fetch(
        http_client=None, browser=browser, provider=None, page_filter=bm25
    )

    result = await fetch(GENERIC, "quantum")

    assert "def quantum():" in result.content
    assert "    return 1" in result.content


async def test_reddit_and_wikipedia_use_browser(bm25, httpx_mock: HTTPXMock):
    browser = FakeBrowser(TOPIC_HTML)
    fetch = create_fetch(
        http_client=None, browser=browser, provider=None, page_filter=bm25
    )

    await fetch("https://www.reddit.com/r/x/comments/1/t/", None)
    await fetch("https://en.wikipedia.org/wiki/Physics", None)

    assert browser.calls == [
        "https://www.reddit.com/r/x/comments/1/t/",
        "https://en.wikipedia.org/wiki/Physics",
    ]
    assert httpx_mock.get_requests() == []


async def test_specialized_sources_filter_and_never_escalate(
    bm25, httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    httpx_mock.add_response(
        url="https://raw.githubusercontent.com/org/project/main/README.md",
        text=MARKDOWN_PAGE,
    )
    for arxiv_url in (
        "https://arxiv.org/html/2406.02530",
        "https://arxiv.org/pdf/2406.02530",
        "https://arxiv.org/abs/2406.02530",
    ):
        httpx_mock.add_response(url=arxiv_url, status_code=403)
    browser = FakeBrowser(TOPIC_HTML)
    provider = FakeProvider(MARKDOWN_PAGE)
    fetch = create_fetch(
        http_client=client, browser=browser, provider=provider, page_filter=bm25
    )

    github = await fetch("https://github.com/org/project", "quantum optimization")
    arxiv = await fetch("https://arxiv.org/abs/2406.02530", "quantum optimization")

    assert "https://github.com/paper" in github.content
    assert "bread flour" not in github.content
    assert (arxiv.status, arxiv.http_status) == (FetchStatus.HTTP_ERROR, 403)
    assert browser.calls == []
    assert provider.calls == []


async def test_pdf_uses_http_extractor(
    bm25, httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    httpx_mock.add_response(
        url="https://source.example/paper.pdf",
        content=_pdf_bytes("Quantum routing results"),
        headers={"content-type": "application/pdf"},
    )
    browser = FakeBrowser(TOPIC_HTML)
    fetch = create_fetch(
        http_client=client, browser=browser, provider=None, page_filter=bm25
    )

    result = await fetch("https://source.example/paper.pdf", None)

    assert "Quantum routing results" in result.content
    assert browser.calls == []


@pytest.mark.parametrize(
    "browser_error",
    [
        "ERROR: http code 403",
        "ERROR: empty response",
        "ERROR: request timeout",
        "ERROR: fetcher unavailable",
    ],
)
async def test_blocked_page_escalates_to_provider_filtered(bm25, browser_error: str):
    browser = FakeBrowser(browser_error)
    provider = FakeProvider(TOPIC_HTML)
    fetch = create_fetch(
        http_client=None, browser=browser, provider=provider, page_filter=bm25
    )

    result = await fetch(GENERIC, "quantum optimization")

    assert "https://source.example/paper" in result.content
    assert "bread flour" not in result.content
    assert provider.calls == [GENERIC]


async def test_non_blocking_errors_do_not_escalate(bm25):
    provider = FakeProvider("provider content")
    doomed = create_fetch(
        http_client=None,
        browser=FakeBrowser("ERROR: http code 404"),
        provider=provider,
        page_filter=bm25,
    )
    doomed_result = await doomed(GENERIC, None)
    assert (doomed_result.status, doomed_result.http_status) == (
        FetchStatus.HTTP_ERROR,
        404,
    )
    assert provider.calls == []

    no_provider = create_fetch(
        http_client=None,
        browser=FakeBrowser("ERROR: http code 403"),
        provider=None,
        page_filter=bm25,
    )
    no_provider_result = await no_provider(GENERIC, None)
    assert (no_provider_result.status, no_provider_result.http_status) == (
        FetchStatus.HTTP_ERROR,
        403,
    )

    unavailable = FakeProvider("ERROR: fetcher unavailable")
    keeps_error = create_fetch(
        http_client=None,
        browser=FakeBrowser("ERROR: http code 403"),
        provider=unavailable,
        page_filter=bm25,
    )
    kept = await keeps_error(GENERIC, None)
    assert (kept.status, kept.http_status) == (FetchStatus.HTTP_ERROR, 403)
    assert unavailable.calls == [GENERIC]


async def test_output_truncated_after_filter(bm25):
    long_html = "<html><body><p>" + "quantum " * 8000 + "</p></body></html>"
    fetch = create_fetch(
        http_client=None,
        browser=FakeBrowser(long_html),
        provider=None,
        page_filter=bm25,
    )

    result = await fetch(GENERIC, None)

    assert len(result.content) <= 15000 + len("\n\n[Content truncated]")
    assert result.content.endswith("[Content truncated]")


async def test_browser_concurrency_is_limited(bm25):
    gate = asyncio.Event()
    browser = FakeBrowser(TOPIC_HTML, gate=gate)
    fetch = create_fetch(
        http_client=None,
        browser=browser,
        provider=None,
        page_filter=bm25,
        concurrency=2,
    )
    tasks = [
        asyncio.create_task(fetch(f"https://source.example/{i}", None))
        for i in range(5)
    ]
    try:
        for _ in range(200):
            if browser.in_flight == 2:
                break
            await asyncio.sleep(0.01)
        assert browser.in_flight == 2
        gate.set()
        results = await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    assert browser.max_in_flight == 2
    assert len(results) == 5


# ---------------------------------------------------------------------------
# Direct (no browser) fetch behavior
# ---------------------------------------------------------------------------


async def test_fetch_default_html_returns_markdown(
    bm25, httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    httpx_mock.add_response(
        url="https://site.example/page",
        html="<html><body><h1>Hello</h1><p>World</p></body></html>",
    )
    fetch = create_fetch(
        http_client=client, browser=None, provider=None, page_filter=bm25
    )

    result = await fetch("https://site.example/page", None)

    assert "# Hello" in result.content
    assert "World" in result.content


async def test_fetch_github_blob_uses_raw_url(
    bm25, httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    httpx_mock.add_response(
        url="https://raw.githubusercontent.com/o/r/main/src/a.py",
        text="print(1)",
    )
    fetch = create_fetch(
        http_client=client, browser=None, provider=None, page_filter=bm25
    )

    result = await fetch("https://github.com/o/r/blob/main/src/a.py", None)

    assert result.content == "print(1)"


async def test_fetch_arxiv_falls_back_from_html_to_pdf_to_abs(
    bm25, httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    httpx_mock.add_response(url="https://arxiv.org/html/2401.00001", status_code=404)
    httpx_mock.add_response(url="https://arxiv.org/pdf/2401.00001", status_code=404)
    httpx_mock.add_response(
        url="https://arxiv.org/abs/2401.00001",
        html="<html><body><p>Abstract text</p></body></html>",
    )
    fetch = create_fetch(
        http_client=client, browser=None, provider=None, page_filter=bm25
    )

    result = await fetch("https://arxiv.org/abs/2401.00001", None)

    assert result.content == "Abstract text"


async def test_fetch_timeout_returns_timeout_error(
    bm25, httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    httpx_mock.add_exception(httpx.ReadTimeout("slow"))
    fetch = create_fetch(
        http_client=client, browser=None, provider=None, page_filter=bm25
    )

    result = await fetch("https://site.example/", None)

    assert result.status is FetchStatus.TIMEOUT


async def test_fetch_unknown_content_type_is_unsupported(
    bm25, httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    httpx_mock.add_response(content=b"\x00", headers={"content-type": "image/png"})
    fetch = create_fetch(
        http_client=client, browser=None, provider=None, page_filter=bm25
    )

    result = await fetch("https://site.example/a.png", None)

    assert result.status is FetchStatus.UNSUPPORTED


@pytest.mark.parametrize("status_code", [401, 403, 404, 429, 500])
async def test_fetch_http_error_reports_exact_status(
    bm25, httpx_mock: HTTPXMock, client: httpx.AsyncClient, status_code: int
):
    httpx_mock.add_response(status_code=status_code)
    fetch = create_fetch(
        http_client=client, browser=None, provider=None, page_filter=bm25
    )

    result = await fetch("https://site.example/", None)

    assert (result.status, result.http_status) == (
        FetchStatus.HTTP_ERROR,
        status_code,
    )


PAGE_WITH_STYLE_TEXT = (
    "<html><body><h1>Tags</h1><p>Use &lt;style&gt;alpha&lt;/style&gt; carefully "
    '<img src="/fig.png" alt="Fig"></p></body></html>'
)


async def test_fetch_default_html_is_converted_once(
    bm25, httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    httpx_mock.add_response(url="https://site.example/page", html=PAGE_WITH_STYLE_TEXT)
    fetch = create_fetch(
        http_client=client, browser=None, provider=None, page_filter=bm25
    )

    result = await fetch("https://site.example/page", None)

    assert "Use <style>alpha</style> carefully" in result.content
    assert "![Fig](https://site.example/fig.png)" in result.content


async def test_arxiv_html_is_converted_once(
    bm25, httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    httpx_mock.add_response(
        url="https://arxiv.org/html/2401.00002", html=PAGE_WITH_STYLE_TEXT
    )
    fetch = create_fetch(
        http_client=client, browser=None, provider=None, page_filter=bm25
    )

    result = await fetch("https://arxiv.org/abs/2401.00002", None)

    assert "Use <style>alpha</style> carefully" in result.content
    assert "![Fig](https://arxiv.org/fig.png)" in result.content


async def test_pdf_keeps_extracted_text(
    bm25,
    httpx_mock: HTTPXMock,
    client: httpx.AsyncClient,
):
    url = "https://source.example/paper.pdf"
    pdf = _pdf_paragraphs(
        ["Quantum routing results\ncol_a   col_b\n1       2", "bread flour baking"]
    )
    for _ in range(2):
        httpx_mock.add_response(
            url=url, content=pdf, headers={"content-type": "application/pdf"}
        )
    fetch = create_fetch(
        http_client=client, browser=None, provider=None, page_filter=bm25
    )

    unfiltered = await fetch(url, None)
    filtered = await fetch(url, "quantum routing")

    assert unfiltered.mime == "text/plain"
    assert unfiltered.content == (
        "Quantum routing results\ncol_a   col_b\n1       2\n\nbread flour baking"
    )
    assert "col_a   col_b" in filtered.content
    assert filtered.mime == "text/plain"
    assert "bread flour" not in filtered.content


async def test_blank_html_is_empty_response(
    bm25, httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    httpx_mock.add_response(
        url="https://site.example/blank", html="<html><body>  </body></html>"
    )
    fetch = create_fetch(
        http_client=client, browser=None, provider=None, page_filter=bm25
    )

    result = await fetch("https://site.example/blank", None)

    assert result.status is FetchStatus.EMPTY


async def test_github_repo_links_resolve_against_requested_url(
    bm25, httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    httpx_mock.add_response(
        url="https://raw.githubusercontent.com/org/project/main/README.md",
        text="See [guide](docs/guide.md) and [root](/paper).",
    )
    fetch = create_fetch(
        http_client=client, browser=None, provider=None, page_filter=bm25
    )

    result = await fetch("https://github.com/org/project", None)

    assert result.url == "https://github.com/org/project"
    assert "https://github.com/org/docs/guide.md" in result.content
    assert "https://github.com/paper" in result.content


async def test_github_blob_relative_link_resolves_in_directory(
    bm25, httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    httpx_mock.add_response(
        url="https://raw.githubusercontent.com/o/r/main/docs/README.md",
        text="See [guide](guide.md).",
    )
    fetch = create_fetch(
        http_client=client, browser=None, provider=None, page_filter=bm25
    )

    result = await fetch("https://github.com/o/r/blob/main/docs/README.md", None)

    assert "https://github.com/o/r/blob/main/docs/guide.md" in result.content


async def test_filter_yielding_nothing_stays_ok_with_empty_content(bm25):
    fetch = create_fetch(
        http_client=None,
        browser=FakeBrowser(TOPIC_HTML),
        provider=None,
        page_filter=bm25,
    )

    result = await fetch(GENERIC, "zzzz unrelated")

    assert (result.status, result.content) == (FetchStatus.OK, "")


async def test_blocked_pdf_skips_browser_and_escalates_to_provider(
    bm25, httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    pdf_url = "https://source.example/paper.pdf"
    httpx_mock.add_response(url=pdf_url, status_code=403)
    browser = FakeBrowser(TOPIC_HTML)
    provider = FakeProvider(TOPIC_HTML)
    fetch = create_fetch(
        http_client=client, browser=browser, provider=provider, page_filter=bm25
    )

    result = await fetch(pdf_url, "quantum optimization")

    assert result.ok
    assert "https://source.example/paper" in result.content
    assert browser.calls == []
    assert provider.calls == [pdf_url]


async def test_restricted_url_never_reaches_provider(bm25):
    provider = FakeProvider(TOPIC_HTML)
    fetch = create_fetch(
        http_client=None,
        browser=FakeBrowser("ERROR: http code 403"),
        provider=provider,
        page_filter=bm25,
        restricted_domains=("source.example",),
    )

    result = await fetch(GENERIC, None)

    assert result.status is FetchStatus.RESTRICTED
    assert provider.calls == []


async def test_provider_html_is_filtered_once(bm25):
    provider_html = (
        "<html><body><h2>Quantum optimization</h2><p>"
        + "quantum optimization improves routing " * 30
        + '<a href="/paper">paper</a></p><h2>Recipes</h2><p>'
        + "bread flour baking kitchen " * 30
        + "</p></body></html>"
    )
    fetch = create_fetch(
        http_client=None,
        browser=FakeBrowser("ERROR: http code 403"),
        provider=FakeProvider(provider_html),
        page_filter=bm25,
    )

    result = await fetch(GENERIC, "quantum optimization")

    assert "https://source.example/paper" in result.content
    assert "bread flour" not in result.content


class ExplodingClient:
    """Client whose requests raise a non-httpx error (e.g. a malformed URL)."""

    async def get(self, *_args: object, **_kwargs: object) -> httpx.Response:
        raise ValueError("boom")


@pytest.mark.parametrize(
    ("build", "url"),
    [
        (lambda c: HttpFetch(c), "https://site.example/"),
        (lambda c: GitHubBlobFetch(HttpFetch(c)), "https://github.com/o/r/blob/main/a.py"),
        (lambda c: GitHubRepoFetch(HttpFetch(c)), "https://github.com/o/r"),
        (lambda c: ArxivFetch(HttpFetch(c)), "https://arxiv.org/abs/2401.00001"),
        (lambda c: ArxivFetch(HttpFetch(c)), "https://arxiv.org/list/cs/new"),
    ],
    ids=["http", "github-blob", "github-repo", "arxiv", "arxiv-no-id"],
)
async def test_unexpected_exception_becomes_unsupported_result(build, url: str):
    fetch = build(ExplodingClient())

    result = await fetch(url)

    assert result.status is FetchStatus.UNSUPPORTED
    assert result.url == url


# ---------------------------------------------------------------------------
# Source mime contract: each source declares what it returns
# ---------------------------------------------------------------------------

HTML_BODY = "<html><body><h1>Hello</h1><p>World</p></body></html>"


@pytest.mark.parametrize(
    ("url", "response", "expected_mime", "expected_content"),
    [
        pytest.param(
            "https://site.example/page",
            {"html": HTML_BODY},
            "text/html",
            HTML_BODY,
            id="html",
        ),
        pytest.param(
            "https://site.example/notes.txt",
            {"text": "# Title", "headers": {"content-type": "text/plain"}},
            "text/plain",
            "# Title",
            id="plain-text",
        ),
        pytest.param(
            "https://site.example/notes.md",
            {"text": "# Title", "headers": {"content-type": "text/markdown"}},
            "text/markdown",
            "# Title",
            id="markdown",
        ),
    ],
)
async def test_http_source_declares_mime_and_native_content(
    httpx_mock: HTTPXMock,
    client: httpx.AsyncClient,
    url: str,
    response: dict,
    expected_mime: str,
    expected_content: str,
):
    httpx_mock.add_response(**response)

    result = await HttpFetch(client)(url)

    assert result.ok
    assert (result.mime, result.content) == (expected_mime, expected_content)
    assert result.url == url


async def test_github_blob_returns_markdown_at_requested_url(
    httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    httpx_mock.add_response(
        url="https://raw.githubusercontent.com/o/r/main/a.md", text="# Doc"
    )
    url = "https://github.com/o/r/blob/main/a.md"

    result = await GitHubBlobFetch(HttpFetch(client))(url)

    assert (result.mime, result.content) == ("text/markdown", "# Doc")
    assert result.url == url


async def test_github_repo_readme_is_markdown_at_requested_url(
    httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    httpx_mock.add_response(
        url="https://raw.githubusercontent.com/org/project/main/README.md",
        text="# Readme",
    )
    url = "https://github.com/org/project"

    result = await GitHubRepoFetch(HttpFetch(client))(url)

    assert (result.mime, result.content) == ("text/markdown", "# Readme")
    assert result.url == url


async def test_github_blob_strips_surrounding_whitespace(
    httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    httpx_mock.add_response(
        url="https://raw.githubusercontent.com/o/r/main/a.md",
        text="  \n# Doc\n\n",
        headers={"content-type": "text/plain"},
    )

    result = await GitHubBlobFetch(HttpFetch(client))(
        "https://github.com/o/r/blob/main/a.md"
    )

    assert (result.status, result.content) == (FetchStatus.OK, "# Doc")


async def test_github_blob_blank_body_is_empty(
    httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    httpx_mock.add_response(
        url="https://raw.githubusercontent.com/o/r/main/a.md",
        text="   \n",
        headers={"content-type": "text/plain"},
    )

    result = await GitHubBlobFetch(HttpFetch(client))(
        "https://github.com/o/r/blob/main/a.md"
    )

    assert result.status is FetchStatus.EMPTY


async def test_github_blob_invalid_url_is_unsupported_without_request(
    httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    result = await GitHubBlobFetch(HttpFetch(client))("https://github.com/o/r")

    assert result.status is FetchStatus.UNSUPPORTED
    assert httpx_mock.get_requests() == []


async def test_github_blob_http_error_reports_requested_url(
    httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    httpx_mock.add_response(
        url="https://raw.githubusercontent.com/o/r/main/a.py", status_code=403
    )
    url = "https://github.com/o/r/blob/main/a.py"

    result = await GitHubBlobFetch(HttpFetch(client))(url)

    assert (result.status, result.http_status, result.url) == (
        FetchStatus.HTTP_ERROR,
        403,
        url,
    )


async def test_github_blob_timeout_reports_requested_url(
    httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    httpx_mock.add_exception(
        httpx.ReadTimeout("slow"),
        url="https://raw.githubusercontent.com/o/r/main/a.py",
    )
    url = "https://github.com/o/r/blob/main/a.py"

    result = await GitHubBlobFetch(HttpFetch(client))(url)

    assert (result.status, result.url) == (FetchStatus.TIMEOUT, url)


async def test_github_blob_html_body_keeps_html_mime(
    httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    body = "<html><body><h1>Doc</h1></body></html>"
    httpx_mock.add_response(
        url="https://raw.githubusercontent.com/o/r/main/page.html", html=body
    )
    url = "https://github.com/o/r/blob/main/page.html"

    result = await GitHubBlobFetch(HttpFetch(client))(url)

    assert (result.status, result.mime) == (FetchStatus.OK, "text/html")
    assert result.url == url


async def test_github_blob_json_is_supported_text(
    httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    httpx_mock.add_response(
        url="https://raw.githubusercontent.com/o/r/main/data.json",
        content=b"{}",
        headers={"content-type": "application/json"},
    )
    url = "https://github.com/o/r/blob/main/data.json"

    result = await GitHubBlobFetch(HttpFetch(client))(url)

    assert (result.status, result.mime, result.content) == (
        FetchStatus.OK,
        "text/plain",
        "{}",
    )
    assert result.url == url


async def test_github_repo_skips_missing_and_blank_readmes(
    httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    httpx_mock.add_response(
        url="https://raw.githubusercontent.com/org/project/main/README.md",
        status_code=404,
    )
    httpx_mock.add_response(
        url="https://raw.githubusercontent.com/org/project/main/README.rst",
        text="   ",
        headers={"content-type": "text/plain"},
    )
    httpx_mock.add_response(
        url="https://raw.githubusercontent.com/org/project/main/README.txt",
        text="# Readme",
        headers={"content-type": "text/plain"},
    )

    result = await GitHubRepoFetch(HttpFetch(client))("https://github.com/org/project")

    assert (result.status, result.content) == (FetchStatus.OK, "# Readme")
    assert result.url == "https://github.com/org/project"


async def test_github_repo_all_candidates_missing_is_empty(
    httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    for branch in ("main", "master"):
        for file_name in ("README.md", "README.rst", "README.txt", "README"):
            httpx_mock.add_response(
                url=(
                    "https://raw.githubusercontent.com/org/project/"
                    f"{branch}/{file_name}"
                ),
                status_code=404,
            )

    result = await GitHubRepoFetch(HttpFetch(client))("https://github.com/org/project")

    assert (result.status, result.url) == (
        FetchStatus.EMPTY,
        "https://github.com/org/project",
    )


async def test_github_repo_non_404_error_stops_lookup(
    httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    httpx_mock.add_response(
        url="https://raw.githubusercontent.com/org/project/main/README.md",
        status_code=403,
    )

    result = await GitHubRepoFetch(HttpFetch(client))("https://github.com/org/project")

    assert (result.status, result.http_status) == (FetchStatus.HTTP_ERROR, 403)
    assert result.url == "https://github.com/org/project"
    assert [request.url for request in httpx_mock.get_requests()] == [
        "https://raw.githubusercontent.com/org/project/main/README.md"
    ]


async def test_pdf_source_is_plain_text(
    httpx_mock: HTTPXMock,
    client: httpx.AsyncClient,
):
    httpx_mock.add_response(
        content=_pdf_paragraphs(["Alpha", "Beta"]),
        headers={"content-type": "application/pdf"},
    )

    result = await HttpFetch(client)("https://site.example/a.pdf")

    assert result.mime == "text/plain"
    assert result.content == "Alpha\n\nBeta"


async def test_http_source_does_not_truncate(
    httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    body = "<html><body><p>" + "word " * 8000 + "</p></body></html>"
    httpx_mock.add_response(html=body)

    result = await HttpFetch(client)("https://site.example/long")

    assert result.content == body


# ---------------------------------------------------------------------------
# Phase 2: native textual source types and composed filtering
# ---------------------------------------------------------------------------

TEXT_BODY = (
    "  # Quantum routing\n\n    value = 2 < 3\n    keep   spaces\n\nBread flour.\n"
)

TEXT_CASES = [
    "text/plain; charset=utf-8",
    "TEXT/CSV",
    "text/x-python",
    "application/json",
    "application/problem+json",
    "application/xml",
    "application/atom+xml",
]


def _large_html() -> str:
    parts = ["<html><body><h1>Handbook</h1>"]
    for number in range(300):
        parts.append(f"<h2>Topic {number}</h2>")
        if number == 0:
            parts.append("<p>Quantum routing evidence START.</p>")
        elif number == 150:
            parts.append("<p>Quantum routing evidence MIDDLE.</p>")
        elif number == 299:
            parts.append("<p>Quantum routing evidence END.</p>")
        else:
            parts.append("<p>" + "bread flour kitchen dough " * 30 + "</p>")
    parts.append("</body></html>")
    return "".join(parts)


@pytest.mark.parametrize("content_type", TEXT_CASES)
async def test_http_textual_types_preserve_plain_content(
    httpx_mock: HTTPXMock, client: httpx.AsyncClient, content_type: str
):
    httpx_mock.add_response(
        url="https://source.example/content",
        text=TEXT_BODY,
        headers={"content-type": content_type},
    )

    result = await HttpFetch(client)("https://source.example/content")

    assert result.ok
    assert result.mime == "text/plain"
    assert result.content == TEXT_BODY
    assert result.url == "https://source.example/content"


async def test_fetch_plain_query_does_not_interpret_markdown_or_html(
    bm25, httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    body = "Quantum # routing <tag>literal</tag>   spacing.\n\nBread flour."
    httpx_mock.add_response(
        url="https://source.example/content",
        text=body,
        headers={"content-type": "text/plain"},
    )
    fetch = create_fetch(
        http_client=client, browser=None, provider=None, page_filter=bm25
    )

    result = await fetch("https://source.example/content", "quantum")

    assert result.mime == "text/plain"
    assert result.content == "Quantum # routing <tag>literal</tag>   spacing."


async def test_pdf_source_preserves_extracted_text(
    httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    url = "https://source.example/paper.pdf"
    httpx_mock.add_response(
        url=url,
        content=_pdf_paragraphs(
            ["Quantum routing results\ncol_a   col_b\n1       2", "Bread flour baking"]
        ),
        headers={"content-type": "application/pdf"},
    )

    result = await HttpFetch(client)(url)

    assert result.mime == "text/plain"
    assert result.content == (
        "Quantum routing results\ncol_a   col_b\n1       2\n\nBread flour baking"
    )


async def test_pdf_filtered_output_preserves_extracted_text(
    bm25, httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    url = "https://source.example/paper.pdf"
    httpx_mock.add_response(
        url=url,
        content=_pdf_paragraphs(
            ["Quantum routing results\ncol_a   col_b\n1       2", "Bread flour baking"]
        ),
        headers={"content-type": "application/pdf"},
    )
    fetch = create_fetch(
        http_client=client, browser=None, provider=None, page_filter=bm25
    )

    result = await fetch(url, "quantum routing")

    assert result.mime == "text/plain"
    assert "col_a   col_b" in result.content
    assert "Bread flour" not in result.content


async def test_large_html_keeps_start_middle_end_evidence(
    bm25, httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    httpx_mock.add_response(url="https://source.example/large", html=_large_html())
    fetch = create_fetch(
        http_client=client,
        browser=None,
        provider=None,
        page_filter=bm25,
        max_chars=15000,
    )

    result = await fetch("https://source.example/large", "quantum routing")

    content = result.content
    assert content.index("START") < content.index("MIDDLE") < content.index("END")
    assert "bread flour" not in content
    assert result.mime == "text/markdown"


async def test_large_pdf_keeps_start_middle_end_evidence(
    bm25, httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    pages = ["Bread flour kitchen dough. " * 12 for _ in range(60)]
    pages[0] = "Quantum routing evidence START."
    pages[30] = "Quantum routing evidence MIDDLE."
    pages[59] = "Quantum routing evidence END."
    url = "https://source.example/big.pdf"
    httpx_mock.add_response(
        url=url,
        content=_pdf_paragraphs(pages),
        headers={"content-type": "application/pdf"},
    )
    fetch = create_fetch(
        http_client=client,
        browser=None,
        provider=None,
        page_filter=bm25,
        max_chars=15000,
    )

    result = await fetch(url, "quantum routing")

    content = result.content
    assert content.index("START") < content.index("MIDDLE") < content.index("END")
    assert "bread" not in content
    assert "dough" not in content
    assert result.mime == "text/plain"


async def test_truncated_markdown_does_not_leak_discarded_links(
    bm25, httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    relevant = (
        "## Quantum\n\n"
        + (
            "Quantum routing improves latency. "
            + "[paper](https://source.example/paper)\n\n"
        )
        * 300
    )
    discarded = (
        "## Baking\n\nBread flour. See "
        "[recipe](https://source.example/recipe).\n\n"
    )
    httpx_mock.add_response(
        url="https://source.example/big.md",
        text=relevant + discarded,
        headers={"content-type": "text/markdown"},
    )
    fetch = create_fetch(
        http_client=client, browser=None, provider=None, page_filter=bm25
    )

    result = await fetch("https://source.example/big.md", "quantum routing")

    assert result.content.endswith("[Content truncated]")
    assert "https://source.example/recipe" not in result.content
    assert "Baking" not in result.content
