"""Markdown chunker: splits text into paragraph fragments for ranking."""

from __future__ import annotations

import re
from collections.abc import Sequence

from mcps.research.tools.filtering import Fragment

__all__ = ["MarkdownChunker"]

_HEADING_RE = re.compile(r"#{1,3}\s+\S")
_FENCE_RE = re.compile(r"\s{0,3}(```|~~~)")

Span = tuple[int, int]


def _line_spans(text: str) -> list[Span]:
    """Offsets of every line in ``text``, without its line terminator."""
    spans: list[Span] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        spans.append((offset, offset + len(line.splitlines()[0])))
        offset += len(line)
    return spans


def _trimmed(text: str, start: int, end: int) -> Span:
    """Drop trailing whitespace from the span."""
    return start, start + len(text[start:end].rstrip())


def _sections(text: str, lines: list[Span]) -> list[list[Span]]:
    """Group lines into sections that start at a heading outside a code fence."""
    sections: list[list[Span]] = [[]]
    in_fence = False
    for span in lines:
        line = text[span[0] : span[1]]
        if _FENCE_RE.match(line):
            in_fence = not in_fence
        elif not in_fence and _HEADING_RE.match(line):
            sections.append([])
        sections[-1].append(span)
    return sections


def _blocks(text: str, lines: list[Span]) -> list[Span]:
    """Split lines at blank lines, keeping code fences whole."""
    blocks: list[list[Span]] = [[]]
    in_fence = False
    for span in lines:
        line = text[span[0] : span[1]]
        if _FENCE_RE.match(line):
            in_fence = not in_fence
        if not in_fence and not line.strip():
            blocks.append([])
        else:
            blocks[-1].append(span)
    spans = [_trimmed(text, block[0][0], block[-1][1]) for block in blocks if block]
    return [(start, end) for start, end in spans if start < end]


def _hard_split(text: str, span: Span, limit: int) -> list[Span]:
    """Cut at the last space within ``limit``, or at ``limit`` when there is none."""
    start, end = span
    pieces: list[Span] = []
    while end - start > limit:
        cut = text.rfind(" ", start, start + limit)
        cut = cut if cut > start else start + limit
        pieces.append((start, cut))
        start = cut
        while start < end and text[start] == " ":
            start += 1
    pieces.append((start, end))
    return pieces


def _pack(units: list[Span], limit: int) -> list[Span]:
    """Join consecutive units while the source between them stays within ``limit``."""
    packed: list[Span] = []
    for start, end in units:
        if packed and end - packed[-1][0] <= limit:
            packed[-1] = (packed[-1][0], end)
        else:
            packed.append((start, end))
    return packed


def _split_block(text: str, block: Span, limit: int) -> list[Span]:
    start, end = block
    if end - start <= limit:
        return [block]
    units = [
        piece
        for line_start, line_end in _line_spans(text[start:end])
        for piece in _hard_split(text, (start + line_start, start + line_end), limit)
        if text[piece[0] : piece[1]].strip()
    ]
    return _pack(units, limit)


class MarkdownChunker:
    """Split Markdown or plain text into paragraph fragments.

    A section runs from an H1-H3 heading to the next one; headings inside code
    fences do not count. Each paragraph, fence or table becomes a fragment that
    starts with its section's own heading. A paragraph longer than ``max_chars``
    is split at lines, then spaces. A section without a body yields nothing.

    A fragment's text is the source slices of its heading span and body span,
    joined by a blank line.
    """

    def __init__(self, max_chars: int = 4_000) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        self._max_chars = max_chars

    def split(self, text: str) -> list[Fragment]:
        return [
            fragment
            for lines in _sections(text, _line_spans(text))
            for fragment in self._section_fragments(text, lines)
        ]

    def merge(self, text: str, fragments: Sequence[Fragment]) -> str:
        """Join the union of the fragments' spans as source runs, in source order.

        Overlapping, touching and whitespace-separated spans form one run, so
        shared text and shared headings appear once.
        """
        spans = sorted(span for fragment in fragments for span in fragment.spans)
        if any(end > len(text) for _, end in spans):
            raise ValueError("fragment span ends after the source text")
        runs: list[Span] = []
        for start, end in spans:
            if runs and (start <= runs[-1][1] or text[runs[-1][1] : start].isspace()):
                runs[-1] = (runs[-1][0], max(runs[-1][1], end))
            else:
                runs.append((start, end))
        return "\n\n".join(text[start:end] for start, end in runs)

    def _section_fragments(self, text: str, lines: list[Span]) -> list[Fragment]:
        heading: Span | None = None
        body_lines = lines
        if lines and _HEADING_RE.match(text[lines[0][0] : lines[0][1]]):
            heading = _trimmed(text, *lines[0])
            body_lines = lines[1:]
        prefix_len = 0 if heading is None else heading[1] - heading[0] + 2
        limit = max(self._max_chars - prefix_len, self._max_chars // 2)
        bodies = [
            piece
            for block in _blocks(text, body_lines)
            for piece in _split_block(text, block, limit)
        ]
        return [
            self._fragment(text, (body,) if heading is None else (heading, body))
            for body in bodies
        ]

    @staticmethod
    def _fragment(text: str, spans: tuple[Span, ...]) -> Fragment:
        return Fragment("\n\n".join(text[start:end] for start, end in spans), spans)
