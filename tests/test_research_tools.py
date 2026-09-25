"""Contract tests for research search and fetch callables (HTTP mocked)."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest
from pytest_httpx import HTTPXMock

from mcps.research.tools import (
    create_duckduckgo_search,
    create_fetch,
    create_google_search,
)


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(follow_redirects=True) as http_client:
        yield http_client


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
# Fetch routing
# ---------------------------------------------------------------------------


async def test_fetch_default_html_returns_markdown(
    httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    httpx_mock.add_response(
        url="https://site.example/page",
        html="<html><body><h1>Hello</h1><p>World</p></body></html>",
    )
    fetch = create_fetch(http_client=client)

    result = await fetch("https://site.example/page")

    assert "# Hello" in result
    assert "World" in result


async def test_fetch_truncates_to_max_chars(
    httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    httpx_mock.add_response(text="x" * 50, headers={"content-type": "text/plain"})
    fetch = create_fetch(http_client=client, max_chars=10)

    result = await fetch("https://site.example/long.txt")

    assert result == "x" * 10 + "\n\n[Content truncated]"


async def test_fetch_wikipedia_uses_raw_action(
    httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    httpx_mock.add_response(
        url="https://en.wikipedia.org/wiki/Physics?action=raw",
        text="== History ==\n'''Physics''' is [[science|a science]]",
    )
    fetch = create_fetch(http_client=client)

    result = await fetch("https://en.wikipedia.org/wiki/Physics")

    assert result == "# History\n**Physics** is a science"


async def test_fetch_github_blob_uses_raw_url(
    httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    httpx_mock.add_response(
        url="https://raw.githubusercontent.com/o/r/main/src/a.py",
        text="print(1)",
    )
    fetch = create_fetch(http_client=client)

    result = await fetch("https://github.com/o/r/blob/main/src/a.py")

    assert result == "print(1)"


async def test_fetch_arxiv_falls_back_from_html_to_pdf_to_abs(
    httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    httpx_mock.add_response(url="https://arxiv.org/html/2401.00001", status_code=404)
    httpx_mock.add_response(url="https://arxiv.org/pdf/2401.00001", status_code=404)
    httpx_mock.add_response(
        url="https://arxiv.org/abs/2401.00001",
        html="<html><body><p>Abstract text</p></body></html>",
    )
    fetch = create_fetch(http_client=client)

    result = await fetch("https://arxiv.org/abs/2401.00001")

    assert result == "Abstract text"


async def test_fetch_timeout_returns_timeout_error(
    httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    httpx_mock.add_exception(httpx.ReadTimeout("slow"))
    fetch = create_fetch(http_client=client)

    assert await fetch("https://site.example/") == "ERROR: request timeout"


async def test_fetch_unknown_content_type_is_unsupported(
    httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    httpx_mock.add_response(content=b"\x00", headers={"content-type": "image/png"})
    fetch = create_fetch(http_client=client)

    assert await fetch("https://site.example/a.png") == "ERROR: unsupported content"


@pytest.mark.parametrize("status_code", [401, 403, 404, 429, 500])
async def test_fetch_http_error_reports_exact_status(
    httpx_mock: HTTPXMock, client: httpx.AsyncClient, status_code: int
):
    httpx_mock.add_response(status_code=status_code)
    fetch = create_fetch(http_client=client)

    result = await fetch("https://site.example/")

    assert result == f"ERROR: http code {status_code}"


# ---------------------------------------------------------------------------
# Fallback chain
# ---------------------------------------------------------------------------


class RecordingFallback:
    """Fake fallback fetcher returning a fixed result and recording calls."""

    def __init__(self, result: str) -> None:
        self.result = result
        self.calls: list[str] = []

    async def __call__(self, url: str) -> str:
        self.calls.append(url)
        return self.result


@pytest.mark.parametrize(
    ("response_kwargs", "expected_first_error"),
    [
        ({"status_code": 401}, "ERROR: http code 401"),
        ({"status_code": 403}, "ERROR: http code 403"),
        ({"status_code": 429}, "ERROR: http code 429"),
        ({"html": "<html><body></body></html>"}, "ERROR: empty response"),
    ],
)
async def test_fetch_escalates_blocked_or_empty_to_fallback(
    httpx_mock: HTTPXMock,
    client: httpx.AsyncClient,
    response_kwargs: dict,
    expected_first_error: str,
):
    httpx_mock.add_response(**response_kwargs)
    fallback = RecordingFallback("rendered content")
    fetch = create_fetch(http_client=client, fallbacks=[fallback])

    result = await fetch("https://site.example/")

    assert result == "rendered content", expected_first_error
    assert fallback.calls == ["https://site.example/"]


@pytest.mark.parametrize(
    "response_kwargs",
    [{"status_code": 404}, {"status_code": 500}, {"html": "<p>ok</p>"}],
)
async def test_fetch_does_not_escalate_success_or_other_errors(
    httpx_mock: HTTPXMock, client: httpx.AsyncClient, response_kwargs: dict
):
    httpx_mock.add_response(**response_kwargs)
    fallback = RecordingFallback("rendered content")
    fetch = create_fetch(http_client=client, fallbacks=[fallback])

    await fetch("https://site.example/")

    assert fallback.calls == []


async def test_fetch_does_not_escalate_timeout(
    httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    httpx_mock.add_exception(httpx.ReadTimeout("slow"))
    fallback = RecordingFallback("rendered content")
    fetch = create_fetch(http_client=client, fallbacks=[fallback])

    assert await fetch("https://site.example/") == "ERROR: request timeout"
    assert fallback.calls == []


async def test_fetch_tries_fallbacks_in_order_until_one_succeeds(
    httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    httpx_mock.add_response(status_code=403)
    browser = RecordingFallback("ERROR: http code 403")
    provider = RecordingFallback("provider content")
    unused = RecordingFallback("never")
    fetch = create_fetch(http_client=client, fallbacks=[browser, provider, unused])

    result = await fetch("https://site.example/")

    assert result == "provider content"
    assert (browser.calls, provider.calls, unused.calls) == (
        ["https://site.example/"],
        ["https://site.example/"],
        [],
    )


async def test_fetch_returns_last_error_when_all_fallbacks_fail(
    httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    httpx_mock.add_response(status_code=403)
    browser = RecordingFallback("ERROR: empty response")
    provider = RecordingFallback("ERROR: http code 429")
    fetch = create_fetch(http_client=client, fallbacks=[browser, provider])

    assert await fetch("https://site.example/") == "ERROR: http code 429"


async def test_fetch_skips_unavailable_fallback_and_keeps_target_error(
    httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    httpx_mock.add_response(status_code=403)
    browser = RecordingFallback("ERROR: fetcher unavailable")
    provider = RecordingFallback("ERROR: fetcher unavailable")
    fetch = create_fetch(http_client=client, fallbacks=[browser, provider])

    result = await fetch("https://site.example/")

    assert result == "ERROR: http code 403"
    assert provider.calls == ["https://site.example/"]


async def test_fetch_stops_chain_on_non_escalatable_fallback_error(
    httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    httpx_mock.add_response(status_code=403)
    browser = RecordingFallback("ERROR: http code 404")
    provider = RecordingFallback("provider content")
    fetch = create_fetch(http_client=client, fallbacks=[browser, provider])

    assert await fetch("https://site.example/") == "ERROR: http code 404"
    assert provider.calls == []
