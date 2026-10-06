"""Mime constants and result builders shared by fetchers."""

from __future__ import annotations

from urllib.parse import urlparse

from mcps.research.tools.models import FetchResult, FetchStatus

__all__ = [
    "MIME_HTML",
    "MIME_MARKDOWN",
    "MIME_PLAIN",
    "TRUNCATION_MARKER",
    "extract_hostname",
    "failure",
    "normalize_mime",
    "textual_mime",
]

MIME_HTML = "text/html"
MIME_MARKDOWN = "text/markdown"
MIME_PLAIN = "text/plain"

TRUNCATION_MARKER = "\n\n[Content truncated]"

_MARKDOWN_MIMES = frozenset([MIME_MARKDOWN, "text/x-markdown"])
_HTML_MIMES = frozenset([MIME_HTML, "application/xhtml+xml"])
_JSON_XML_MIMES = frozenset(["application/json", "application/xml"])


def normalize_mime(value: str) -> str:
    """Return the lowercase media type without parameters."""
    return value.split(";", maxsplit=1)[0].strip().lower()


def textual_mime(value: str) -> str | None:
    """Classify a media type as decoded textual content, or ``None``.

    HTML/XHTML stay HTML so callers can render markup; Markdown stays
    Markdown so source structure is preserved; every other ``text/*`` and
    approved JSON/XML type is treated as plain text. PDF is handled by HTTP
    extraction and is not classified here.
    """
    normalized = normalize_mime(value)
    if normalized in _HTML_MIMES:
        return MIME_HTML
    if normalized in _MARKDOWN_MIMES:
        return MIME_MARKDOWN
    if (
        normalized.startswith("text/")
        or normalized in _JSON_XML_MIMES
        or normalized.endswith(("+json", "+xml"))
    ):
        return MIME_PLAIN
    return None


def failure(
    url: str, status: FetchStatus, http_status: int | None = None
) -> FetchResult:
    """Build an unsuccessful result; error results carry no mime."""
    return FetchResult(url=url, status=status, mime="", http_status=http_status)


def extract_hostname(url: str) -> str:
    """Return the lowercase network location of ``url``."""
    return urlparse(url).netloc.lower()
