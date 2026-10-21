"""Markdown chunker: splits text into overlapping windows with source offsets."""

from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from collections.abc import Sequence

from mcps.research.tools.filtering import Fragment

__all__ = ["MarkdownChunker"]

_HEADING_RE = re.compile(r" {0,3}#{1,6}(?:[ \t]|$)")


def _boundary_offsets(text: str) -> list[int]:
    """Offsets where a window can start: a heading line or a line after a blank line."""
    offsets: list[int] = []
    position = 0
    previous_blank = False
    for line in text.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        blank = not content.strip()
        if previous_blank and not blank:
            offsets.append(position)
        if _HEADING_RE.match(content):
            offsets.append(position)
        position += len(line)
        previous_blank = blank
    return sorted(set(offsets))


def _snap(target: int, start: int, overlap: int, delta: int, offsets: list[int]) -> int:
    """Nearest boundary to ``target`` within ``delta``, or ``target`` itself.

    Boundaries at or before ``start + overlap`` are ignored because they would
    stop the next window from advancing.
    """
    left = bisect_left(offsets, max(target - delta, start + overlap + 1))
    right = bisect_right(offsets, target + delta)
    nearby = offsets[left:right]
    return min(nearby, key=lambda offset: abs(offset - target), default=target)


class MarkdownChunker:
    """Split text into windows that target ``max_chars`` and keep source offsets.

    A cut moves up to ``max_chars // 10`` from its target to the nearest heading
    line or line after a blank line. Every fragment is the exact source slice
    ``text[start:end]``: nothing is trimmed, repeated or synthesized. Merging
    all fragments restores the text.
    """

    def __init__(self, max_chars: int = 4_000, overlap_chars: int = 0) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        if not 0 <= overlap_chars < max_chars:
            raise ValueError("overlap_chars must be in the range [0, max_chars)")
        self._max_chars = max_chars
        self._overlap_chars = overlap_chars
        self._delta = max_chars // 10

    def split(self, text: str) -> list[Fragment]:
        offsets = _boundary_offsets(text)
        fragments: list[Fragment] = []
        start = 0
        while start < len(text):
            if len(text) - start <= self._max_chars:
                fragments.append(Fragment(text[start:], start, len(text)))
                break
            end = _snap(
                start + self._max_chars,
                start,
                self._overlap_chars,
                self._delta,
                offsets,
            )
            fragments.append(Fragment(text[start:end], start, end))
            start = end - self._overlap_chars
        return fragments

    def merge(self, fragments: Sequence[Fragment]) -> str:
        pieces: list[str] = []
        cursor: int | None = None
        for fragment in sorted(fragments, key=lambda item: (item.start, item.end)):
            if fragment.end <= fragment.start:
                continue
            if cursor is None:
                pieces.append(fragment.text)
            elif fragment.start > cursor:
                pieces.append("\n\n")
                pieces.append(fragment.text)
            elif fragment.end > cursor:
                pieces.append(fragment.text[cursor - fragment.start :])
            else:
                continue
            cursor = fragment.end
        return "".join(pieces)
