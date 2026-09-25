"""Response body to markdown/text extractors keyed by content type."""

from __future__ import annotations

from collections.abc import Callable

import html2text
import httpx
import pymupdf

__all__ = [
    "CONTENT_TYPE_EXTRACTORS",
    "extract_without_content_type",
    "normalize_content_type",
]

_PDF_ANNOT_SCREEN = 21  # pymupdf.PDF_ANNOT_SCREEN


def _convert_html_to_markdown(html: str) -> str:
    converter = html2text.HTML2Text()
    converter.ignore_images = True
    converter.body_width = 0
    return converter.handle(html).strip()


def _convert_pdf_to_markdown(pdf_bytes: bytes) -> str:
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
    return _convert_html_to_markdown(response.text) or None


def _extract_pdf_response(response: httpx.Response) -> str | None:
    if not response.content:
        return None
    return _convert_pdf_to_markdown(response.content) or None


def _extract_plain_text_response(response: httpx.Response) -> str | None:
    content = response.text
    if not content or not content.strip():
        return None
    return content


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
