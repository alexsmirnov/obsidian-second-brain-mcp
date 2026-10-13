"""Default document extractor: prepare a fetch result into a mapped document."""

from __future__ import annotations

from mcps.research.tools.filtering import FilterLimits
from mcps.research.tools.filtering_content import ContentDocument, prepare_document
from mcps.research.tools.models import FetchResult

__all__ = ["DefaultExtractor"]


class DefaultExtractor:
    """Extractor rendering and parsing a fetch result into a content document.

    HTML is rendered once, Markdown keeps its source structure, and other
    textual types stay plain. ``limits`` owns the input and window budgets.
    """

    def __init__(self, limits: FilterLimits | None = None) -> None:
        self._limits = limits or FilterLimits()

    def __call__(self, result: FetchResult) -> ContentDocument:
        limits = self._limits
        return prepare_document(
            result,
            max_input_chars=limits.max_input_chars,
            window_chars=limits.window_chars,
        )
