"""Contract tests for the Markdown fragment chunker."""

from __future__ import annotations

import pytest

from mcps.research.tools.filtering import Fragment
from mcps.research.tools.markdown_chunker import MarkdownChunker


def texts(chunker: MarkdownChunker, text: str) -> list[str]:
    return [fragment.text for fragment in chunker.split(text)]


def test_chunker_empty_text_returns_no_fragments():
    assert texts(MarkdownChunker(), "  \n\n ") == []


def test_chunker_each_paragraph_becomes_a_fragment_with_its_own_heading():
    text = "# One\n\n" + "a" * 60 + "\n\n" + "b" * 60 + "\n\n# Two\n\n" + "c" * 60

    fragments = texts(MarkdownChunker(max_chars=200), text)

    assert fragments == [
        "# One\n\n" + "a" * 60,
        "# One\n\n" + "b" * 60,
        "# Two\n\n" + "c" * 60,
    ]


def test_chunker_short_paragraph_is_not_merged_with_unrelated_neighbour():
    text = "# One\n\nshort\n\n" + "b" * 80

    assert texts(MarkdownChunker(), text) == [
        "# One\n\nshort",
        "# One\n\n" + "b" * 80,
    ]


def test_chunker_headings_inside_code_fence_do_not_split():
    text = "# Real\n\n```sh\n# comment, not a heading\n\necho hi\n```\n"

    fragments = texts(MarkdownChunker(), text)

    assert fragments == [text.strip()]


def test_chunker_table_stays_one_fragment():
    table = "| a | b |\n| --- | --- |\n| 1 | 2 |"

    assert texts(MarkdownChunker(), f"# T\n\n{table}") == [f"# T\n\n{table}"]


def test_chunker_plain_text_splits_on_paragraphs():
    text = "\n\n".join("word " * 20 + str(index) for index in range(3))

    fragments = texts(MarkdownChunker(max_chars=250), text)

    assert fragments == text.split("\n\n")


def test_chunker_section_without_body_yields_nothing():
    assert texts(MarkdownChunker(), "# Only a heading") == []


def test_chunker_oversized_paragraph_splits_and_repeats_only_its_own_heading():
    long_paragraph = "\n".join(f"line {index} " + "x" * 80 for index in range(6))
    text = f"# Parent\n\nintro text that is long enough to stay alone here\n\n## Child\n\n{long_paragraph}"

    fragments = texts(MarkdownChunker(max_chars=250), text)

    child = [f for f in fragments if "line" in f]
    assert len(child) > 1
    assert all(f.startswith("## Child") and len(f) <= 250 for f in child)
    assert not any("# Parent" in f for f in child)


def test_chunker_line_longer_than_limit_is_hard_split_without_loss():
    text = "w" * 1000

    fragments = texts(MarkdownChunker(max_chars=300), text)

    assert all(len(fragment) <= 300 for fragment in fragments)
    assert "".join(fragments) == text


def test_chunker_keeps_document_order():
    text = "\n\n".join(f"# S{index}\n\n" + "y" * 100 for index in range(5))

    fragments = texts(MarkdownChunker(max_chars=150), text)

    assert [fragment.splitlines()[0] for fragment in fragments] == [
        f"# S{index}" for index in range(5)
    ]


LINES = [f"line {index} ".ljust(90, "x") for index in range(6)]
MIXED = "\n\n".join(
    [
        "intro paragraph",
        "# A",
        "   lead line",
        "```sh\necho one\n\necho two\n```",
        "| a | b |\n| --- | --- |\n| 1 | 2 |",
        "## B",
        "\n".join(LINES),
        "## C",
        "w" * 700,
    ]
)
MIXED_EXPECTED = [
    "intro paragraph",
    "# A\n\n   lead line",
    "# A\n\n```sh\necho one\n\necho two\n```",
    "# A\n\n| a | b |\n| --- | --- |\n| 1 | 2 |",
    "## B\n\n" + "\n".join(LINES[0:2]),
    "## B\n\n" + "\n".join(LINES[2:4]),
    "## B\n\n" + "\n".join(LINES[4:6]),
    "## C\n\n" + "w" * 244,
    "## C\n\n" + "w" * 244,
    "## C\n\n" + "w" * 212,
]


def test_split_spans_point_at_heading_and_body():
    fragments = MarkdownChunker().split("# One\n\naaa\n\nbbb")

    assert fragments == [
        Fragment("# One\n\naaa", ((0, 5), (7, 10))),
        Fragment("# One\n\nbbb", ((0, 5), (12, 15))),
    ]


def test_split_preamble_without_heading_has_body_span_only():
    fragments = MarkdownChunker().split("intro\n\n# H\n\nbody")

    assert fragments[0] == Fragment("intro", ((0, 5),))


def test_split_text_equals_joined_span_slices():
    fragments = MarkdownChunker(max_chars=250).split(MIXED)

    assert [fragment.text for fragment in fragments] == MIXED_EXPECTED
    assert all(
        fragment.text == "\n\n".join(MIXED[start:end] for start, end in fragment.spans)
        for fragment in fragments
    )


def test_split_hard_split_keeps_source_space():
    assert texts(MarkdownChunker(max_chars=5), "ab cd efghijk") == [
        "ab cd",
        "efghi",
        "jk",
    ]


def test_split_oversized_fence_with_blank_line_has_no_empty_fragment():
    text = "# H\n\n```\n" + "x" * 60 + "\n\n" + "y" * 60 + "\n```"

    fragments = MarkdownChunker(max_chars=50).split(text)

    assert fragments
    assert all(fragment.text != "# H" for fragment in fragments)
    assert all(
        start < end and text[start:end].strip()
        for fragment in fragments
        for start, end in fragment.spans[-1:]
    )


def test_fragment_rejects_reversed_span():
    with pytest.raises(ValueError):
        Fragment("x", ((3, 1),))


def test_merge_adjacent_paragraphs_share_one_heading():
    chunker = MarkdownChunker()
    text = "# One\n\naaa\n\nbbb"

    assert chunker.merge(text, chunker.split(text)) == "# One\n\naaa\n\nbbb"


def test_merge_skipped_paragraph_leaves_a_gap():
    chunker = MarkdownChunker()
    text = "# One\n\naaa\n\nbbb\n\nccc"
    fragments = chunker.split(text)

    merged = chunker.merge(text, [fragments[0], fragments[2]])

    assert merged == "# One\n\naaa\n\nccc"


def test_merge_drops_unselected_empty_heading():
    chunker = MarkdownChunker()
    text = "# A\n\naaa\n\n## Empty\n\n## B\n\nbbb"
    fragments = chunker.split(text)
    assert len(fragments) == 2

    assert chunker.merge(text, fragments) == "# A\n\naaa\n\n## B\n\nbbb"


def test_merge_overlapping_spans_emit_shared_text_once():
    fragments = [
        Fragment("abc def", ((0, 7),)),
        Fragment("def ghi", ((4, 11),)),
    ]

    assert MarkdownChunker().merge("abc def ghi", fragments) == "abc def ghi"


def test_merge_ignores_fragment_order_and_duplicates():
    chunker = MarkdownChunker()
    text = "# One\n\naaa\n\nbbb\n\nccc"
    fragments = chunker.split(text)

    shuffled = chunker.merge(text, [fragments[2], fragments[0], fragments[2]])

    assert shuffled == chunker.merge(text, [fragments[0], fragments[2]])


def test_merge_nothing_is_empty():
    assert MarkdownChunker().merge("# One\n\naaa", []) == ""


def test_merge_hard_split_pieces_rejoin_the_source():
    chunker = MarkdownChunker(max_chars=100)
    text = " ".join(f"word{index}" for index in range(200))
    fragments = chunker.split(text)

    assert len(fragments) > 1
    assert chunker.merge(text, fragments) == text


def test_merge_all_fragments_restores_the_document():
    chunker = MarkdownChunker(max_chars=250)

    assert chunker.merge(MIXED, chunker.split(MIXED)) == MIXED.strip()


def test_merge_span_past_text_end_raises():
    with pytest.raises(ValueError):
        MarkdownChunker().merge("abc", [Fragment("abc", ((0, 9),))])
