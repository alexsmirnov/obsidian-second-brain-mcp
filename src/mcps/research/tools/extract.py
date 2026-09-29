"""Response body to HTML extractors keyed by content type."""

from __future__ import annotations

import html
from collections.abc import Callable

import httpx
import pymupdf
from lxml import html as lxml_html

from mcps.research.tools.filtering import markdown_to_html

__all__ = [
    "CONTENT_TYPE_EXTRACTORS",
    "extract_without_content_type",
    "normalize_content_type",
]

_PDF_ANNOT_SCREEN = 21  # pymupdf.PDF_ANNOT_SCREEN


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


def normalize_content_type(header_value: str) -> str:
    """Return the lowercase media type without parameters."""
    return header_value.split(";", maxsplit=1)[0].strip().lower()


def _extract_html_response(response: httpx.Response) -> str | None:
    # ponytail: <script>/<style> text counts as content; browser path handles JS shells
    try:
        text = lxml_html.document_fromstring(response.text).text_content()
    except Exception:
        return None
    return response.text if text.strip() else None


def _extract_pdf_response(response: httpx.Response) -> str | None:
    if not response.content:
        return None
    blocks = [
        block
        for block in _convert_pdf_to_text(response.content).split("\n\n")
        if block.strip()
    ]
    if not blocks:
        return None
    return "".join(
        f"<p>{html.escape(block).replace(chr(10), '<br>')}</p>" for block in blocks
    )


def _extract_plain_text_response(response: httpx.Response) -> str | None:
    content = response.text
    if not content or not content.strip():
        return None
    return markdown_to_html(content)


def _looks_like_pdf(content: bytes) -> bool:
    return content.startswith(b"%PDF-")


def _looks_like_html(content: str) -> bool:
    lowered = content.lstrip().lower()
    html_prefixes = ("<!doctype html", "<html", "<head", "<body")
    return lowered.startswith(html_prefixes)


def extract_without_content_type(response: httpx.Response) -> str | None:
    """Sniff PDF/HTML/plain text when the server omits content-type."""
    if response.content and _looks_like_pdf(response.content):
        return _extract_pdf_response(response)
    if response.text and _looks_like_html(response.text):
        return _extract_html_response(response)
    return _extract_plain_text_response(response)


CONTENT_TYPE_EXTRACTORS: dict[str, Callable[[httpx.Response], str | None]] = {
    "text/html": _extract_html_response,
    "application/xhtml+xml": _extract_html_response,
    "application/pdf": _extract_pdf_response,
    "text/plain": _extract_plain_text_response,
    "text/markdown": _extract_plain_text_response,
}
