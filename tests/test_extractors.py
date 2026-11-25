"""Contract tests for the binary document extractor registry."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pymupdf
import pytest
from pytest_httpx import HTTPXMock

from mcps.research.tools import extractors
from mcps.research.tools.bm25_ranker import Bm25Ranker
from mcps.research.tools.extractors import (
    BINARY_EXTRACTORS,
    BinaryExtractor,
    extractor_for_content,
    extractor_for_mime,
    extractor_for_url,
)
from mcps.research.tools.fetch import create_fetch
from mcps.research.tools.filtering import ChunkedRelevanceFilter
from mcps.research.tools.markdown_chunker import MarkdownChunker
from mcps.research.tools.models import FetchResult, FetchStatus
from tests.research_fakes import StaticFetch

FAKE_MIME = "application/x-fake"
FAKE_TEXT = "text extracted by the fake format"
FAKE_EXTRACTOR = BinaryExtractor(FAKE_MIME, b"FAKE", (".fake",), lambda _: FAKE_TEXT)


def _pdf_bytes(text: str) -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    data = document.tobytes()
    document.close()
    return data


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(follow_redirects=True) as http_client:
        yield http_client


@pytest.fixture
def with_fake_format(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        extractors, "BINARY_EXTRACTORS", (*BINARY_EXTRACTORS, FAKE_EXTRACTOR)
    )


def test_extractor_for_mime_pdf_with_parameters_and_case_is_found():
    found = extractor_for_mime("Application/PDF; qs=0.001")

    assert found is not None
    assert found.mime == "application/pdf"


@pytest.mark.parametrize("mime", ["image/png", "text/html", ""])
def test_extractor_for_mime_unregistered_type_is_none(mime: str):
    assert extractor_for_mime(mime) is None


def test_extractor_for_content_pdf_signature_is_found():
    found = extractor_for_content(_pdf_bytes("hello"))

    assert found is not None
    assert found.mime == "application/pdf"


@pytest.mark.parametrize("data", [b"<html><body>", b"", b"plain text"])
def test_extractor_for_content_unknown_signature_is_none(data: bytes):
    assert extractor_for_content(data) is None


@pytest.mark.parametrize(
    "url", ["https://x.example/a/Paper.PDF?x=1", "https://x.example/p.pdf#page=2"]
)
def test_extractor_for_url_registered_extension_is_found(url: str):
    found = extractor_for_url(url)

    assert found is not None
    assert found.mime == "application/pdf"


@pytest.mark.parametrize(
    "url", ["https://x.example/paper/abc", "https://x.example/a.pdf/view"]
)
def test_extractor_for_url_without_registered_extension_is_none(url: str):
    assert extractor_for_url(url) is None


def test_registry_entries_are_well_formed():
    mimes = [entry.mime for entry in BINARY_EXTRACTORS]

    assert len(mimes) == len(set(mimes))
    for entry in BINARY_EXTRACTORS:
        assert entry.mime == entry.mime.lower()
        assert entry.signature
        assert all(ext.startswith(".") for ext in entry.extensions)


def test_pdf_extractor_returns_page_text():
    pdf = extractor_for_mime("application/pdf")
    assert pdf is not None

    text = pdf.to_text(_pdf_bytes("Quantum routing results"))

    assert "Quantum routing results" in text


@pytest.mark.usefixtures("with_fake_format")
async def test_new_registry_entry_serves_browser_handoff_without_other_changes(
    httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    url = "https://source.example/doc/abc"
    httpx_mock.add_response(
        url=url, content=b"FAKE-bytes", headers={"content-type": FAKE_MIME}
    )
    browser = StaticFetch(FetchResult(url, FetchStatus.UNSUPPORTED, FAKE_MIME))
    fetch = create_fetch(
        http_client=client,
        browser=browser,
        page_filter=ChunkedRelevanceFilter(MarkdownChunker(), Bm25Ranker()),
    )

    result = await fetch(url, None)

    assert (result.status, result.content) == (FetchStatus.OK, FAKE_TEXT)
    assert browser.calls == [url]


@pytest.mark.usefixtures("with_fake_format")
async def test_new_registry_extension_skips_browser(
    httpx_mock: HTTPXMock, client: httpx.AsyncClient
):
    url = "https://source.example/files/report.fake"
    httpx_mock.add_response(
        url=url, content=b"FAKE-bytes", headers={"content-type": FAKE_MIME}
    )
    browser = StaticFetch(FetchResult(url, FetchStatus.UNAVAILABLE, ""))
    fetch = create_fetch(
        http_client=client,
        browser=browser,
        page_filter=ChunkedRelevanceFilter(MarkdownChunker(), Bm25Ranker()),
    )

    result = await fetch(url, None)

    assert (result.status, result.content) == (FetchStatus.OK, FAKE_TEXT)
    assert browser.calls == []
