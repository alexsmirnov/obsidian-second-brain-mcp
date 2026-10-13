"""Semantic chunker: regroup a document into selection units and windows."""

from __future__ import annotations

from mcps.research.tools.filtering import FilterLimits
from mcps.research.tools.filtering_content import (
    ContentDocument,
    ScoringWindow,
    build_scoring_windows,
    build_selection_units,
)

__all__ = ["SemanticChunker"]


class SemanticChunker:
    """Chunker grouping sections into units, then windowing each unit.

    Units follow the RAG ``SemanticChunker`` rules; scoring windows carry the
    ``block_id`` values of the returned document.
    """

    def __init__(self, limits: FilterLimits | None = None) -> None:
        self._limits = limits or FilterLimits()

    def __call__(
        self, document: ContentDocument
    ) -> tuple[ContentDocument, tuple[ScoringWindow, ...]]:
        limits = self._limits
        units = build_selection_units(
            document,
            unit_chars=limits.unit_chars,
            min_unit_chars=limits.min_unit_chars,
        )
        windows = build_scoring_windows(
            units,
            window_chars=limits.window_chars,
            overlap_chars=limits.overlap_chars,
            max_windows=limits.max_windows,
        )
        return units, windows
