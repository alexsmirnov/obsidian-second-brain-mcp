"""Contract tests for RelevanceFilter with injected chunker and ranker."""

from __future__ import annotations

from mcps.research.tools.filtering import RelevanceFilter
from mcps.research.tools.models import FetchResult, FetchStatus


def page(content: str, mime: str = "text/markdown", status=FetchStatus.OK):
    return FetchResult(
        url="https://example.org/p", status=status, mime=mime, content=content
    )


def by_blank_line(text: str) -> list[str]:
    return text.split("\n\n")


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


async def test_relevance_filter_joins_selected_fragments_in_document_order():
    ranker = PickRanker([2, 0])

    result = await RelevanceFilter(by_blank_line, ranker)(page("a\n\nb\n\nc"), "q")

    assert result.status is FetchStatus.OK
    assert result.content == "a\n\nc"
    assert result.mime == "text/markdown"
    assert ranker.calls == [(["a", "b", "c"], "q")]


async def test_relevance_filter_duplicate_indices_select_once():
    result = await RelevanceFilter(by_blank_line, PickRanker([1, 1]))(
        page("a\n\nb"), "q"
    )

    assert result.content == "b"


async def test_relevance_filter_blank_query_returns_result_unchanged():
    ranker = PickRanker([0])
    source = page("a\n\nb")

    result = await RelevanceFilter(by_blank_line, ranker)(source, "   ")

    assert result == source
    assert ranker.calls == []


async def test_relevance_filter_empty_selection_is_ok_with_empty_content():
    result = await RelevanceFilter(by_blank_line, PickRanker([]))(page("a\n\nb"), "q")

    assert result.status is FetchStatus.OK
    assert result.content == ""


async def test_relevance_filter_ignores_out_of_range_indices():
    result = await RelevanceFilter(by_blank_line, PickRanker([5, 0]))(
        page("a\n\nb"), "q"
    )

    assert result.content == "a"


async def test_relevance_filter_passes_failed_results_through():
    failed = page("", status=FetchStatus.TIMEOUT)

    assert await RelevanceFilter(by_blank_line, PickRanker([0]))(failed, "q") is failed


async def test_relevance_filter_unsupported_mime_returns_unsupported():
    result = await RelevanceFilter(by_blank_line, PickRanker([0]))(
        page("\x00", mime="image/png"), "q"
    )

    assert result.status is FetchStatus.UNSUPPORTED


async def test_relevance_filter_overlong_query_returns_filter_failed():
    ranker = PickRanker([0])

    result = await RelevanceFilter(by_blank_line, ranker, max_query_chars=5)(
        page("a"), "x" * 6
    )

    assert result.status is FetchStatus.FILTER_FAILED
    assert ranker.calls == []


async def test_relevance_filter_overlong_input_returns_filter_failed():
    result = await RelevanceFilter(by_blank_line, PickRanker([0]), max_input_chars=3)(
        page("abcd"), "q"
    )

    assert result.status is FetchStatus.FILTER_FAILED


async def test_relevance_filter_ranker_error_returns_filter_failed():
    result = await RelevanceFilter(by_blank_line, BrokenRanker())(page("a"), "q")

    assert result.status is FetchStatus.FILTER_FAILED


async def test_relevance_filter_chunker_error_returns_filter_failed():
    def broken(text: str) -> list[str]:
        raise RuntimeError("boom")

    result = await RelevanceFilter(broken, PickRanker([0]))(page("a"), "q")

    assert result.status is FetchStatus.FILTER_FAILED


async def test_relevance_filter_no_fragments_skips_the_ranker():
    ranker = PickRanker([0])

    result = await RelevanceFilter(lambda text: [], ranker)(page("   "), "q")

    assert result.status is FetchStatus.OK
    assert result.content == ""
    assert ranker.calls == []
