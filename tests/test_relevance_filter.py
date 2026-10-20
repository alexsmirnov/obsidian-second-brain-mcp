"""Contract tests for RelevanceFilter with injected chunker and ranker."""

from __future__ import annotations

from collections.abc import Sequence

from mcps.research.tools.filtering import Fragment, RelevanceFilter
from mcps.research.tools.models import FetchResult, FetchStatus


def page(content: str, mime: str = "text/markdown", status=FetchStatus.OK):
    return FetchResult(
        url="https://example.org/p", status=status, mime=mime, content=content
    )


class ParagraphChunker:
    """Splits on blank lines with real offsets and records every merge call."""

    def __init__(self) -> None:
        self.merges: list[tuple[str, list[Fragment]]] = []

    def split(self, text: str) -> list[Fragment]:
        fragments: list[Fragment] = []
        offset = 0
        for part in text.split("\n\n"):
            fragments.append(Fragment(part, ((offset, offset + len(part)),)))
            offset += len(part) + 2
        return fragments

    def merge(self, text: str, fragments: Sequence[Fragment]) -> str:
        self.merges.append((text, list(fragments)))
        return "\n\n".join(fragment.text for fragment in fragments)


class BrokenSplitChunker(ParagraphChunker):
    def split(self, text: str) -> list[Fragment]:
        raise RuntimeError("boom")


class BrokenMergeChunker(ParagraphChunker):
    def merge(self, text: str, fragments: Sequence[Fragment]) -> str:
        raise RuntimeError("boom")


class FixedMergeChunker(ParagraphChunker):
    def merge(self, text: str, fragments: Sequence[Fragment]) -> str:
        return "MERGED"


class EmptySplitChunker(ParagraphChunker):
    def split(self, text: str) -> list[Fragment]:
        return []


class PickRanker:
    def __init__(self, indices: list[int]) -> None:
        self.indices = indices
        self.calls: list[tuple[list[str], str]] = []

    async def __call__(self, fragments: list[str], query: str) -> list[int]:
        self.calls.append((fragments, query))
        return self.indices


class BrokenRanker:
    async def __call__(self, fragments: list[str], query: str) -> list[int]:
        raise RuntimeError("boom")


async def test_relevance_filter_merges_selected_fragments_in_document_order():
    chunker = ParagraphChunker()
    ranker = PickRanker([2, 0, 2, 9])

    result = await RelevanceFilter(chunker, ranker)(page("a\n\nb\n\nc"), "q")

    assert result.status is FetchStatus.OK
    assert result.content == "a\n\nc"
    assert result.mime == "text/markdown"
    assert ranker.calls == [(["a", "b", "c"], "q")]
    assert chunker.merges == [
        ("a\n\nb\n\nc", [Fragment("a", ((0, 1),)), Fragment("c", ((6, 7),))])
    ]


async def test_relevance_filter_content_is_what_merge_returns():
    result = await RelevanceFilter(FixedMergeChunker(), PickRanker([0]))(
        page("a\n\nb"), "q"
    )

    assert result.content == "MERGED"


async def test_relevance_filter_duplicate_indices_select_once():
    result = await RelevanceFilter(ParagraphChunker(), PickRanker([1, 1]))(
        page("a\n\nb"), "q"
    )

    assert result.content == "b"


async def test_relevance_filter_blank_query_returns_result_unchanged():
    ranker = PickRanker([0])
    source = page("a\n\nb")

    result = await RelevanceFilter(ParagraphChunker(), ranker)(source, "   ")

    assert result == source
    assert ranker.calls == []


async def test_relevance_filter_empty_selection_is_ok_with_empty_content():
    result = await RelevanceFilter(ParagraphChunker(), PickRanker([]))(
        page("a\n\nb"), "q"
    )

    assert result.status is FetchStatus.OK
    assert result.content == ""


async def test_relevance_filter_ignores_out_of_range_indices():
    result = await RelevanceFilter(ParagraphChunker(), PickRanker([5, 0]))(
        page("a\n\nb"), "q"
    )

    assert result.content == "a"


async def test_relevance_filter_passes_failed_results_through():
    failed = page("", status=FetchStatus.TIMEOUT)

    filtered = await RelevanceFilter(ParagraphChunker(), PickRanker([0]))(failed, "q")

    assert filtered is failed


async def test_relevance_filter_unsupported_mime_returns_unsupported():
    result = await RelevanceFilter(ParagraphChunker(), PickRanker([0]))(
        page("\x00", mime="image/png"), "q"
    )

    assert result.status is FetchStatus.UNSUPPORTED


async def test_relevance_filter_overlong_query_returns_filter_failed():
    ranker = PickRanker([0])

    result = await RelevanceFilter(ParagraphChunker(), ranker, max_query_chars=5)(
        page("a"), "x" * 6
    )

    assert result.status is FetchStatus.FILTER_FAILED
    assert ranker.calls == []


async def test_relevance_filter_overlong_input_returns_filter_failed():
    result = await RelevanceFilter(
        ParagraphChunker(), PickRanker([0]), max_input_chars=3
    )(page("abcd"), "q")

    assert result.status is FetchStatus.FILTER_FAILED


async def test_relevance_filter_ranker_error_returns_filter_failed():
    result = await RelevanceFilter(ParagraphChunker(), BrokenRanker())(page("a"), "q")

    assert result.status is FetchStatus.FILTER_FAILED


async def test_relevance_filter_chunker_error_returns_filter_failed():
    ranker = PickRanker([0])

    result = await RelevanceFilter(BrokenSplitChunker(), ranker)(page("a"), "q")

    assert result.status is FetchStatus.FILTER_FAILED
    assert ranker.calls == []


async def test_relevance_filter_merge_error_returns_filter_failed():
    ranker = PickRanker([0])

    result = await RelevanceFilter(BrokenMergeChunker(), ranker)(page("a"), "q")

    assert result.status is FetchStatus.FILTER_FAILED
    assert ranker.calls == [(["a"], "q")]


async def test_relevance_filter_no_fragments_skips_the_ranker():
    chunker = EmptySplitChunker()
    ranker = PickRanker([0])

    result = await RelevanceFilter(chunker, ranker)(page("   "), "q")

    assert result.status is FetchStatus.OK
    assert result.content == ""
    assert ranker.calls == []
    assert chunker.merges == [("   ", [])]
