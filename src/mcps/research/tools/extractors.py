"""Binary document extractors: the registry that drives non-HTML page fetching.

Every non-textual page format the fetcher can read is one :class:`BinaryExtractor`
entry in :data:`BINARY_EXTRACTORS`. The registry is the single source of truth for
four decisions: which response media types ``HttpFetch`` converts to text, which
byte signatures identify a body that has no content type, which URL extensions skip
the browser, and which browser results are handed to ``HttpFetch``.

To support a new format (for example DOCX), write ``to_text(data: bytes) -> str``
and append one entry::

    BinaryExtractor(
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        b"PK\\x03\\x04",
        (".docx",),
        docx_to_text,
    )

Nothing else changes: not the browser, not the routing, not ``HttpFetch``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlparse

import pymupdf

from mcps.research.tools.common import MIME_PDF, normalize_mime

__all__ = [
    "BINARY_EXTRACTORS",
    "BinaryExtractor",
    "extractor_for_content",
    "extractor_for_mime",
    "extractor_for_url",
    "pdf_to_text",
]

_PDF_ANNOT_SCREEN = 21  # pymupdf.PDF_ANNOT_SCREEN


@dataclass(frozen=True, slots=True)
class BinaryExtractor:
    """One readable binary format.

    ``signature`` is the byte prefix of a body in this format, and ``to_text``
    returns the document text (blank when there is none). A signature is used
    only when a response has no content type, so it must be specific enough to
    tell this format from other formats and from plain text. A shared prefix,
    such as ZIP's ``PK``, would claim unrelated files.
    """

    mime: str
    signature: bytes
    extensions: tuple[str, ...]
    to_text: Callable[[bytes], str]


def pdf_to_text(pdf_bytes: bytes) -> str:
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


BINARY_EXTRACTORS: tuple[BinaryExtractor, ...] = (
    BinaryExtractor(MIME_PDF, b"%PDF-", (".pdf",), pdf_to_text),
)


def extractor_for_mime(mime: str) -> BinaryExtractor | None:
    """Extractor registered for a media type; parameters and case are ignored."""
    normalized = normalize_mime(mime)
    return next((e for e in BINARY_EXTRACTORS if e.mime == normalized), None)


def extractor_for_content(data: bytes) -> BinaryExtractor | None:
    """Extractor whose signature starts ``data``."""
    return next((e for e in BINARY_EXTRACTORS if data.startswith(e.signature)), None)


def extractor_for_url(url: str) -> BinaryExtractor | None:
    """Extractor registered for the URL path's extension; query is ignored."""
    path = urlparse(url).path.lower()
    return next((e for e in BINARY_EXTRACTORS if path.endswith(e.extensions)), None)
