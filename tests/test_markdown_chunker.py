"""Contract tests for the offset-based Markdown window chunker."""

from __future__ import annotations

from itertools import pairwise

import pytest

from mcps.research.tools.filtering import Fragment
from mcps.research.tools.markdown_chunker import MarkdownChunker


def ranges(chunker: MarkdownChunker, text: str) -> list[tuple[int, int]]:
    return [(fragment.start, fragment.end) for fragment in chunker.split(text)]


ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def test_chunker_empty_text_returns_no_fragments():
    assert MarkdownChunker().split("") == []


def test_chunker_whitespace_only_text_is_one_fragment():
    assert MarkdownChunker().split("  \n\n ") == [Fragment("  \n\n ", 0, 5)]


def test_chunker_short_text_is_one_fragment():
    text = "# Only a heading\n\nshort body"

    assert MarkdownChunker().split(text) == [Fragment(text, 0, len(text))]


def test_chunker_windows_overlap_by_the_configured_amount():
    assert ranges(MarkdownChunker(max_chars=10, overlap_chars=3), ALPHABET) == [
        (0, 10),
        (7, 17),
        (14, 24),
        (21, 26),
    ]


def test_chunker_zero_overlap_makes_windows_touch():
    assert ranges(MarkdownChunker(max_chars=10), ALPHABET) == [
        (0, 10),
        (10, 20),
        (20, 26),
    ]


def test_chunker_windows_overlap_across_headings():
    text = "# A\n\n" + "a" * 20 + "\n\n# B\n\n" + "b" * 20
    fragments = MarkdownChunker(max_chars=10, overlap_chars=2).split(text)

    assert len(fragments) > 2
    assert all(later.start == earlier.end - 2 for earlier, later in pairwise(fragments))


def test_chunker_cut_moves_back_to_a_blank_line_within_delta():
    text = "a" * 17 + "\n\n" + "b" * 20

    assert ranges(MarkdownChunker(max_chars=20), text) == [(0, 19), (19, 39)]


def test_chunker_cut_moves_forward_to_a_heading_within_delta():
    text = "a" * 17 + "\n# H\n" + "b" * 16

    assert ranges(MarkdownChunker(max_chars=20), text) == [(0, 18), (18, 38)]


def test_chunker_nearest_boundary_wins():
    text = "a" * 95 + "\n# H\n\n" + "b" * 40

    assert ranges(MarkdownChunker(max_chars=100), text) == [(0, 101), (101, 141)]


def test_chunker_tie_prefers_the_earlier_boundary():
    text = "a" * 99 + "\n# H\n# I\n" + "b" * 40

    assert ranges(MarkdownChunker(max_chars=102), text) == [(0, 100), (100, 148)]


def test_chunker_ignores_boundaries_outside_delta():
    text = "a" * 10 + "\n\n" + "b" * 40

    assert ranges(MarkdownChunker(max_chars=20), text)[0] == (0, 20)


def test_chunker_skips_boundary_that_would_block_progress():
    text = "a" * 7 + "\n\n# H\n" + "b" * 20

    assert ranges(MarkdownChunker(max_chars=10, overlap_chars=9), text)[0] == (0, 10)


def test_chunker_repeated_boundary_overlap_still_advances():
    text = "abcdef\n\nghijkl\n\nmnopqr"
    chunker = MarkdownChunker(max_chars=9, overlap_chars=8)
    fragments = chunker.split(text)

    assert len(fragments) > 1
    assert all(later.start > earlier.start for earlier, later in pairwise(fragments))
    assert fragments[-1].end == len(text)
    assert chunker.merge(fragments) == text


def test_chunker_text_without_boundaries_cuts_at_target():
    assert ranges(MarkdownChunker(max_chars=300), "w" * 1000) == [
        (0, 300),
        (300, 600),
        (600, 900),
        (900, 1000),
    ]


def test_chunker_fragments_are_exact_source_slices():
    text = "# Title\r\n\r\npara one\r\n\r\n- item \u00fc\r\n"
    chunker = MarkdownChunker(max_chars=12, overlap_chars=4)
    fragments = chunker.split(text)

    assert len(fragments) > 1
    assert all(
        fragment.text == text[fragment.start : fragment.end]
        for fragment in fragments
    )
    assert chunker.merge(fragments) == text


def test_chunker_all_fragments_restore_the_document():
    text = "# A\n\n" + "a" * 300 + "\n\n## B\n\n" + "b" * 300 + "\n"

    assert MarkdownChunker(max_chars=100, overlap_chars=20).merge(
        MarkdownChunker(max_chars=100, overlap_chars=20).split(text)
    ) == text


@pytest.mark.parametrize(
    ("max_chars", "overlap_chars"), [(0, 0), (-1, 0), (10, -1), (10, 10), (10, 11)]
)
def test_chunker_rejects_invalid_sizes(max_chars, overlap_chars):
    with pytest.raises(ValueError):
        MarkdownChunker(max_chars=max_chars, overlap_chars=overlap_chars)


def test_fragment_rejects_negative_start():
    with pytest.raises(ValueError):
        Fragment("x", -1, 0)


def test_fragment_rejects_reversed_range():
    with pytest.raises(ValueError):
        Fragment("x", 3, 1)


def test_fragment_rejects_text_length_mismatch():
    with pytest.raises(ValueError):
        Fragment("abc", 0, 2)


def test_merge_joins_overlapping_fragments_once():
    fragments = [Fragment("abc def", 0, 7), Fragment("def ghi", 4, 11)]

    assert MarkdownChunker().merge(fragments) == "abc def ghi"


def test_merge_touching_fragments_have_no_separator():
    fragments = [Fragment("ab", 0, 2), Fragment("cd", 2, 4)]

    assert MarkdownChunker().merge(fragments) == "abcd"


def test_merge_gap_gets_a_blank_line():
    fragments = [Fragment("ab", 0, 2), Fragment("xy", 8, 10)]

    assert MarkdownChunker().merge(fragments) == "ab\n\nxy"


def test_merge_keeps_whitespace_inside_gap():
    fragments = [Fragment("ab\n\n", 0, 4), Fragment("xy", 8, 10)]

    assert MarkdownChunker().merge(fragments) == "ab\n\n\n\nxy"


def test_merge_ignores_order_duplicates_and_contained_fragments():
    fragments = [
        Fragment("abc", 4, 7),
        Fragment("ab", 0, 2),
        Fragment("abc", 4, 7),
        Fragment("b", 1, 2),
    ]

    assert MarkdownChunker().merge(fragments) == "ab\n\nabc"


def test_merge_keeps_repeated_text_at_distinct_offsets():
    fragments = [Fragment("abc", 0, 3), Fragment("abc", 6, 9)]

    assert MarkdownChunker().merge(fragments) == "abc\n\nabc"


def test_merge_zero_length_fragment_adds_nothing():
    assert MarkdownChunker().merge([Fragment("", 2, 2), Fragment("ab", 0, 2)]) == "ab"


def test_merge_nothing_is_empty():
    assert MarkdownChunker().merge([]) == ""
