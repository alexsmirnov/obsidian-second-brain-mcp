"""Generic page fetcher: one HTTP GET per page plus content-type extraction."""

from __future__ import annotations

import logging

import httpx
import pymupdf
from lxml import html as lxml_html

from mcps.research.tools.common import (
    MIME_HTML,
    MIME_PLAIN,
    failure,
    normalize_mime,
    textual_mime,
)
from mcps.research.tools.models import FetchResult, FetchStatus

__all__ = ["HttpFetch"]

logger = logging.getLogger(__name__)

# Collected from actual Chrome request headers
_CHROME_HEADERS = {
    "accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,image/apng,*/*;"
        "q=0.8,application/signed-exchange;v=b3;q=0.7"
    ),
    "accept-encoding": "gzip, deflate",
    "accept-language": "en-US,en;q=0.9",
    "cache-control": "no-cache",
    "dnt": "1",
    "pragma": "no-cache",
    "priority": "u=0, i",
    "sec-ch-ua": '"Google Chrome";v="141", "Not?A_Brand";v="8", "Chromium";v="141"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "same-origin",
    "sec-fetch-user": "?1",
    "upgrade-insecure-requests": "1",
    "user-agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/141.0.0.0 Safari/537.36"
    ),
}

_PDF_ANNOT_SCREEN = 21  # pymupdf.PDF_ANNOT_SCREEN

# (content, mime); None when the body holds no content.
Extracted = tuple[str, str] | None


def _convert_pdf_to_text(pdf_bytes: bytes) -> str:
    page_text: list[str] = []
    # Open, clean, and reload the PDF to fix annotation errors
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
        # clean=True fixes appearance streams; deflate recompresses objects
        cleaned_bytes = document.tobytes(clean=True, deflate=True)

    with pymupdf.open(stream=cleaned_bytes, filetype="pdf") as document:
        for page in document:
            # Strip out Screen annotations entirely if errors persist
            for annot in page.annots():
                if annot.type[0] == _PDF_ANNOT_SCREEN:
                    page.delete_annot(annot)

            raw_text = page.get_text("text")
            if not isinstance(raw_text, str):
                continue
            extracted = raw_text.strip()
            if extracted:
                page_text.append(extracted)
    return "\n\n".join(page_text).strip()


def _extract_html_response(response: httpx.Response) -> Extracted:
    # ponytail: <script>/<style> text counts as content; browser path handles JS shells
    try:
        text = lxml_html.document_fromstring(response.text).text_content()
    except Exception:
        return None
    return (response.text, MIME_HTML) if text.strip() else None


def _extract_pdf_response(response: httpx.Response) -> Extracted:
    if not response.content:
        return None
    extracted = _convert_pdf_to_text(response.content)
    if not extracted.strip():
        return None
    return extracted, MIME_PLAIN


def _extract_plain_text_response(
    response: httpx.Response, mime: str
) -> Extracted:
    content = response.text
    if not content or not content.strip():
        return None
    return content, mime


def _looks_like_pdf(content: bytes) -> bool:
    return content.startswith(b"%PDF-")


def _looks_like_html(content: str) -> bool:
    lowered = content.lstrip().lower()
    html_prefixes = ("<!doctype html", "<html", "<head", "<body")
    return lowered.startswith(html_prefixes)


def _extract_without_content_type(response: httpx.Response) -> Extracted:
    if response.content and _looks_like_pdf(response.content):
        return _extract_pdf_response(response)
    if response.text and _looks_like_html(response.text):
        return _extract_html_response(response)
    return _extract_plain_text_response(response, MIME_PLAIN)


def _extract(response: httpx.Response) -> Extracted:
    """Return extracted text, or raise LookupError for unsupported types."""
    content_type = normalize_mime(response.headers.get("content-type", ""))
    if not content_type:
        return _extract_without_content_type(response)
    if content_type == "application/pdf":
        return _extract_pdf_response(response)
    classified = textual_mime(content_type)
    if classified is None:
        raise LookupError(content_type)
    if classified == MIME_HTML:
        return _extract_html_response(response)
    return _extract_plain_text_response(response, classified)


def _error_result(url: str, error: Exception) -> FetchResult:
    """Map an httpx exception to a failed result."""
    if isinstance(error, httpx.TimeoutException):
        return failure(url, FetchStatus.TIMEOUT)
    if isinstance(error, httpx.HTTPStatusError):
        return failure(url, FetchStatus.HTTP_ERROR, error.response.status_code)
    return failure(url, FetchStatus.UNSUPPORTED)


class HttpFetch:
    """Fetch ``url`` with browser-like headers and return its extracted content."""

    def __init__(self, http_client: httpx.AsyncClient | None) -> None:
        self._http_client = http_client

    async def __call__(self, url: str, query: str | None = None, /) -> FetchResult:
        try:
            return await self._fetch(url)
        except httpx.HTTPError as error:
            result = _error_result(url, error)
        except Exception:
            result = failure(url, FetchStatus.UNSUPPORTED)
        logger.warning("Web fetch failed for %s: %s", url, result.status)
        return result

    async def _fetch(self, url: str) -> FetchResult:
        response = await self._request_get(url)
        try:
            extracted = _extract(response)
        except Exception:
            logger.warning("Web fetch failed for %s: unsupported content", url)
            return failure(url, FetchStatus.UNSUPPORTED)
        if extracted is None:
            logger.warning("Web fetch failed for %s: empty response", url)
            return failure(url, FetchStatus.EMPTY)
        content, mime = extracted
        return FetchResult(url=url, status=FetchStatus.OK, mime=mime, content=content)

    async def _request_get(self, url: str) -> httpx.Response:
        """GET ``url`` with the shared client (or an owned one) and raise on 4xx/5xx."""
        if self._http_client is not None:
            response = await self._http_client.get(url, headers=_CHROME_HEADERS)
            response.raise_for_status()
            return response
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as owned:
            response = await owned.get(url, headers=_CHROME_HEADERS)
            response.raise_for_status()
            return response
