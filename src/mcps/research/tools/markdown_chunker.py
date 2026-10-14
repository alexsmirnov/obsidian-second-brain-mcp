"""Markdown chunker: splits text into paragraph fragments for ranking."""

from __future__ import annotations

import re

__all__ = ["MarkdownChunker"]

_HEADING_RE = re.compile(r"#{1,3}\s+\S")
_FENCE_RE = re.compile(r"\s{0,3}(```|~~~)")


def _sections(text: str) -> list[list[str]]:
    """Group lines into sections that start at a heading outside a code fence."""
    sections: list[list[str]] = [[]]
    in_fence = False
    for line in text.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
        elif not in_fence and _HEADING_RE.match(line):
            sections.append([])
        sections[-1].append(line)
    return sections


def _blocks(lines: list[str]) -> list[str]:
    """Split lines at blank lines, keeping code fences whole."""
    blocks: list[list[str]] = [[]]
    in_fence = False
    for line in lines:
        if _FENCE_RE.match(line):
            in_fence = not in_fence
        if not in_fence and not line.strip():
            blocks.append([])
        else:
            blocks[-1].append(line)
    return ["\n".join(block).strip() for block in blocks if any(block)]


def _hard_split(text: str, limit: int) -> list[str]:
    """Cut at the last space within ``limit``, or at ``limit`` when there is none."""
    pieces: list[str] = []
    while len(text) > limit:
        cut = text.rfind(" ", 0, limit)
        cut = cut if cut > 0 else limit
        pieces.append(text[:cut])
        text = text[cut:].lstrip(" ")
    pieces.append(text)
    return pieces


def _pack(units: list[str], limit: int, separator: str) -> list[str]:
    """Join consecutive units while the result stays within ``limit``."""
    packed: list[str] = []
    for unit in units:
        if packed and len(packed[-1]) + len(separator) + len(unit) <= limit:
            packed[-1] = f"{packed[-1]}{separator}{unit}"
        else:
            packed.append(unit)
    return packed


def _split_block(block: str, limit: int) -> list[str]:
    if len(block) <= limit:
        return [block]
    lines = [piece for line in block.splitlines() for piece in _hard_split(line, limit)]
    return _pack(lines, limit, "\n")


class MarkdownChunker:
    """Split Markdown or plain text into paragraph fragments.

    A section runs from an H1-H3 heading to the next one; headings inside code
    fences do not count. Each paragraph, fence or table becomes a fragment that
    starts with its section's own heading. A paragraph longer than ``max_chars``
    is split at lines, then spaces. A section without a body yields nothing.
    """

    def __init__(self, max_chars: int = 2_000) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        self._max_chars = max_chars

    def __call__(self, text: str) -> list[str]:
        return [
            fragment
            for lines in _sections(text)
            for fragment in self._section_fragments(lines)
        ]

    def _section_fragments(self, lines: list[str]) -> list[str]:
        has_heading = bool(lines) and bool(_HEADING_RE.match(lines[0]))
        prefix = f"{lines[0].strip()}\n\n" if has_heading else ""
        limit = max(self._max_chars - len(prefix), self._max_chars // 2)
        blocks = [
            piece
            for block in _blocks(lines[1:] if has_heading else lines)
            for piece in _split_block(block, limit)
        ]
        return [prefix + piece for piece in blocks]
