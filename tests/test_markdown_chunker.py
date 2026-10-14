"""Contract tests for the Markdown fragment chunker."""

from __future__ import annotations

from mcps.research.tools.markdown_chunker import MarkdownChunker


def test_chunker_empty_text_returns_no_fragments():
    assert MarkdownChunker()("  \n\n ") == []


def test_chunker_each_paragraph_becomes_a_fragment_with_its_own_heading():
    text = "# One\n\n" + "a" * 60 + "\n\n" + "b" * 60 + "\n\n# Two\n\n" + "c" * 60

    fragments = MarkdownChunker(max_chars=200)(text)

    assert fragments == [
        "# One\n\n" + "a" * 60,
        "# One\n\n" + "b" * 60,
        "# Two\n\n" + "c" * 60,
    ]


def test_chunker_short_paragraph_is_not_merged_with_unrelated_neighbour():
    text = "# One\n\nshort\n\n" + "b" * 80

    assert MarkdownChunker()(text) == ["# One\n\nshort", "# One\n\n" + "b" * 80]


def test_chunker_headings_inside_code_fence_do_not_split():
    text = "# Real\n\n```sh\n# comment, not a heading\n\necho hi\n```\n"

    fragments = MarkdownChunker()(text)

    assert fragments == [text.strip()]


def test_chunker_table_stays_one_fragment():
    table = "| a | b |\n| --- | --- |\n| 1 | 2 |"

    assert MarkdownChunker()(f"# T\n\n{table}") == [f"# T\n\n{table}"]


def test_chunker_plain_text_splits_on_paragraphs():
    text = "\n\n".join("word " * 20 + str(index) for index in range(3))

    fragments = MarkdownChunker(max_chars=250)(text)

    assert fragments == text.split("\n\n")


def test_chunker_section_without_body_yields_nothing():
    assert MarkdownChunker()("# Only a heading") == []


def test_chunker_oversized_paragraph_splits_and_repeats_only_its_own_heading():
    long_paragraph = "\n".join(f"line {index} " + "x" * 80 for index in range(6))
    text = f"# Parent\n\nintro text that is long enough to stay alone here\n\n## Child\n\n{long_paragraph}"

    fragments = MarkdownChunker(max_chars=250)(text)

    child = [f for f in fragments if "line" in f]
    assert len(child) > 1
    assert all(f.startswith("## Child") and len(f) <= 250 for f in child)
    assert not any("# Parent" in f for f in child)


def test_chunker_line_longer_than_limit_is_hard_split_without_loss():
    text = "w" * 1000

    fragments = MarkdownChunker(max_chars=300)(text)

    assert all(len(fragment) <= 300 for fragment in fragments)
    assert "".join(fragments) == text


def test_chunker_keeps_document_order():
    text = "\n\n".join(f"# S{index}\n\n" + "y" * 100 for index in range(5))

    fragments = MarkdownChunker(max_chars=150)(text)

    assert [fragment.splitlines()[0] for fragment in fragments] == [
        f"# S{index}" for index in range(5)
    ]
