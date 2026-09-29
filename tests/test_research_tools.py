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
from mcps.research.tools.filtering import create_page_filter

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

    async def __call__(self, url: str) -> str:
        self.calls.append(url)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self.gate is not None:
                await self.gate.wait()
            if isinstance(self.result, dict):
                return self.result.get(url, "ERROR: empty response")
            return self.result
        finally:
            self.in_flight -= 1


class FakeProvider:
    """Async provider stub returning Markdown/error and recording calls."""

    def __init__(self, result: str) -> None:
        self.result = result
        self.calls: list[str] = []

    async def __call__(self, url: str) -> str:
        self.calls.append(url)
        return self.result


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(follow_redirects=True) as http_client:
        yield http_client


@pytest.fixture
def bm25():
    return create_page_filter(fetch_model="", router_url="", router_key="")


def _pdf_bytes(text: str) -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
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


async def test_restricted_domain_returns_empty_without_io(bm25):
    browser = FakeBrowser(TOPIC_HTML)
    fetch = create_fetch(
        http_client=None,
        browser=browser,
        provider=None,
        page_filter=bm25,
        restricted_domains=("blocked.example",),
    )

    assert await fetch("https://blocked.example/a", "q") == ""
    assert await fetch("https://Sub.Blocked.Example/a", "q") == ""
    await fetch("https://notblocked.example/a", "q")

    assert browser.calls == ["https://notblocked.example/a"]


async def test_generic_url_uses_browser_and_filters(bm25, httpx_mock: HTTPXMock):
    browser = FakeBrowser(TOPIC_HTML)
    fetch = create_fetch(
        http_client=None, browser=browser, provider=None, page_filter=bm25
    )

    result = await fetch(GENERIC, "quantum optimization")

    assert "https://source.example/paper" in result
    assert "bread flour" not in result
    assert browser.calls == [GENERIC]
    assert httpx_mock.get_requests() == []


async def test_plain_text_url_via_browser_is_filtered(bm25):
    page = f"<html><body><pre>{html.escape(MARKDOWN_PAGE)}</pre></body></html>"
    browser = FakeBrowser(page)
    provider = FakeProvider(MARKDOWN_PAGE)
    fetch = create_fetch(
        http_client=None, browser=browser, provider=provider, page_filter=bm25
    )

    result = await fetch("https://source.example/notes.md", "quantum optimization")

    assert "https://source.example/paper" in result
    assert "bread flour" not in result
    assert provider.calls == []


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

    assert "https://github.com/paper" in github
    assert "bread flour" not in github
    assert arxiv == "ERROR: http code 403"
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

    assert "Quantum routing results" in result
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
    provider = FakeProvider(MARKDOWN_PAGE)
    fetch = create_fetch(
        http_client=None, browser=browser, provider=provider, page_filter=bm25
    )

    result = await fetch(GENERIC, "quantum optimization")

    assert "https://source.example/paper" in result
    assert "bread flour" not in result
    assert provider.calls == [GENERIC]


async def test_non_blocking_errors_do_not_escalate(bm25):
    provider = FakeProvider("provider content")
    doomed = create_fetch(
        http_client=None,
        browser=FakeBrowser("ERROR: http code 404"),
        provider=provider,
        page_filter=bm25,
    )
    assert await doomed(GENERIC, None) == "ERROR: http code 404"
    assert provider.calls == []

    no_provider = create_fetch(
        http_client=None,
        browser=FakeBrowser("ERROR: http code 403"),
        provider=None,
        page_filter=bm25,
    )
    assert await no_provider(GENERIC, None) == "ERROR: http code 403"

    unavailable = FakeProvider("ERROR: fetcher unavailable")
    keeps_error = create_fetch(
        http_client=None,
        browser=FakeBrowser("ERROR: http code 403"),
        provider=unavailable,
        page_filter=bm25,
    )
    assert await keeps_error(GENERIC, None) == "ERROR: http code 403"
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

    assert len(result) <= 15000 + len("\n\n[Content truncated]")
    assert result.endswith("[Content truncated]")


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

    assert "# Hello" in result
    assert "World" in result


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

    assert result == "print(1)"


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

    assert result == "Abstract text"


async def test_fetch_timeout_returns_timeout_error(
    bm25, httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    httpx_mock.add_exception(httpx.ReadTimeout("slow"))
    fetch = create_fetch(
        http_client=client, browser=None, provider=None, page_filter=bm25
    )

    assert await fetch("https://site.example/", None) == "ERROR: request timeout"


async def test_fetch_unknown_content_type_is_unsupported(
    bm25, httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    httpx_mock.add_response(content=b"\x00", headers={"content-type": "image/png"})
    fetch = create_fetch(
        http_client=client, browser=None, provider=None, page_filter=bm25
    )

    assert (
        await fetch("https://site.example/a.png", None)
        == "ERROR: unsupported content"
    )


@pytest.mark.parametrize("status_code", [401, 403, 404, 429, 500])
async def test_fetch_http_error_reports_exact_status(
    bm25, httpx_mock: HTTPXMock, client: httpx.AsyncClient, status_code: int
):
    httpx_mock.add_response(status_code=status_code)
    fetch = create_fetch(
        http_client=client, browser=None, provider=None, page_filter=bm25
    )

    result = await fetch("https://site.example/", None)

    assert result == f"ERROR: http code {status_code}"
