"""Mime constants and result builders shared by fetchers."""

from __future__ import annotations

from urllib.parse import urlparse

from mcps.research.tools.models import FetchResult, FetchStatus

__all__ = [
    "MIME_HTML",
    "MIME_MARKDOWN",
    "extract_hostname",
    "failure",
]

MIME_HTML = "text/html"
MIME_MARKDOWN = "text/markdown"


def failure(
    url: str, status: FetchStatus, http_status: int | None = None
) -> FetchResult:
    """Build an unsuccessful result; error results carry no mime."""
    return FetchResult(url=url, status=status, mime="", http_status=http_status)


def extract_hostname(url: str) -> str:
    """Return the lowercase network location of ``url``."""
    return urlparse(url).netloc.lower()
