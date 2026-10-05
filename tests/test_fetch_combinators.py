"""Contract tests for FetchResult and the Fetch/Filter combinators."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from mcps.research.tools.combinators import (
    Blocked,
    Fallback,
    FilterChain,
    Filtered,
    FilterSelector,
    Throttled,
    Truncate,
    UrlSelector,
)
from mcps.research.tools.models import FetchResult, FetchStatus

URL = "https://source.example/page"


def ok(content: str = "body", mime: str = "text/html") -> FetchResult:
    return FetchResult(url=URL, status=FetchStatus.OK, mime=mime, content=content)


def failed(status: FetchStatus, http_status: int | None = None) -> FetchResult:
    return FetchResult(url=URL, status=status, mime="", http_status=http_status)


class StubFetch:
    """Fetch returning a fixed result and recording calls."""

    def __init__(self, result: FetchResult) -> None:
        self.result = result
        self.calls: list[tuple[str, str | None]] = []

    async def __call__(self, url: str, query: str | None = None, /) -> FetchResult:
        self.calls.append((url, query))
        return self.result


class RecordingFilter:
    """Filter appending a tag to the content and recording calls."""

    def __init__(self, tag: str = "", result: FetchResult | None = None) -> None:
        self.tag = tag
        self.result = result
        self.calls: list[tuple[FetchResult, str | None]] = []

    async def __call__(
        self, result: FetchResult, query: str | None = None, /
    ) -> FetchResult:
        self.calls.append((result, query))
        if self.result is not None:
            return self.result
        return replace(result, content=result.content + self.tag)


# ---------------------------------------------------------------------------
# FetchResult
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "http_status", "expected"),
    [
        (FetchStatus.OK, None, False),
        (FetchStatus.RESTRICTED, None, False),
        (FetchStatus.UNSUPPORTED, None, False),
        (FetchStatus.FILTER_FAILED, None, False),
        (FetchStatus.TIMEOUT, None, True),
        (FetchStatus.EMPTY, None, True),
        (FetchStatus.UNAVAILABLE, None, True),
        (FetchStatus.HTTP_ERROR, 401, True),
        (FetchStatus.HTTP_ERROR, 403, True),
        (FetchStatus.HTTP_ERROR, 429, True),
        (FetchStatus.HTTP_ERROR, 404, False),
        (FetchStatus.HTTP_ERROR, 500, False),
    ],
)
def test_is_retryable_covers_only_blocking_failures(
    status: FetchStatus, http_status: int | None, expected: bool
):
    assert failed(status, http_status).is_retryable() is expected


def test_ok_is_true_only_for_ok_status():
    assert ok().ok is True
    assert failed(FetchStatus.EMPTY).ok is False


# ---------------------------------------------------------------------------
# Filtered
# ---------------------------------------------------------------------------


async def test_filtered_skips_filter_for_error_results():
    error = failed(FetchStatus.HTTP_ERROR, 403)
    page_filter = RecordingFilter("!")
    fetch = Filtered(StubFetch(error), page_filter)

    result = await fetch(URL, "q")

    assert result is error
    assert page_filter.calls == []


async def test_filtered_passes_result_and_query_to_filter():
    source = ok("text")
    page_filter = RecordingFilter("!")
    fetch = Filtered(StubFetch(source), page_filter)

    result = await fetch(URL, "q")

    assert page_filter.calls == [(source, "q")]
    assert (result.status, result.content) == (FetchStatus.OK, "text!")


async def test_filtered_forwards_url_and_query_to_fetch():
    inner = StubFetch(ok())
    fetch = Filtered(inner, RecordingFilter())

    await fetch(URL, "q")

    assert inner.calls == [(URL, "q")]


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------


async def test_fallback_returns_secondary_result_after_retryable_failure():
    primary = StubFetch(failed(FetchStatus.HTTP_ERROR, 403))
    secondary = StubFetch(ok("unblocked"))

    result = await Fallback(primary, secondary)(URL, "q")

    assert secondary.calls == [(URL, "q")]
    assert result.content == "unblocked"


async def test_fallback_keeps_non_retryable_failure():
    error = failed(FetchStatus.HTTP_ERROR, 404)
    secondary = StubFetch(ok())

    result = await Fallback(StubFetch(error), secondary)(URL)

    assert result is error
    assert secondary.calls == []


async def test_fallback_does_not_call_secondary_after_success():
    secondary = StubFetch(ok("second"))

    result = await Fallback(StubFetch(ok("first")), secondary)(URL)

    assert result.content == "first"
    assert secondary.calls == []


@pytest.mark.parametrize(
    "primary_error",
    [
        failed(FetchStatus.HTTP_ERROR, 403),
        failed(FetchStatus.UNAVAILABLE),
    ],
)
async def test_fallback_keeps_primary_when_secondary_unavailable(
    primary_error: FetchResult,
):
    secondary = StubFetch(failed(FetchStatus.UNAVAILABLE))

    result = await Fallback(StubFetch(primary_error), secondary)(URL)

    assert result is primary_error
    assert len(secondary.calls) == 1


async def test_fallback_returns_secondary_error_other_than_unavailable():
    secondary_error = failed(FetchStatus.HTTP_ERROR, 404)

    result = await Fallback(
        StubFetch(failed(FetchStatus.TIMEOUT)), StubFetch(secondary_error)
    )(URL)

    assert result is secondary_error


async def test_fallback_uses_custom_predicate():
    secondary = StubFetch(ok("second"))
    fetch = Fallback(
        StubFetch(ok("first")), secondary, when=lambda result: result.ok
    )

    result = await fetch(URL)

    assert result.content == "second"


# ---------------------------------------------------------------------------
# UrlSelector / Blocked
# ---------------------------------------------------------------------------


async def test_url_selector_first_matching_route_wins():
    first, second, default = (StubFetch(ok(name)) for name in ("a", "b", "d"))
    selector = UrlSelector(
        [(lambda url: "source" in url, first), (lambda url: True, second)],
        default=default,
    )

    result = await selector(URL, "q")

    assert result.content == "a"
    assert first.calls == [(URL, "q")]
    assert second.calls == default.calls == []


async def test_url_selector_uses_default_without_match():
    route, default = StubFetch(ok("r")), StubFetch(ok("d"))
    selector = UrlSelector([(lambda url: False, route)], default=default)

    result = await selector(URL)

    assert result.content == "d"
    assert route.calls == []


async def test_blocked_returns_restricted_without_content():
    result = await Blocked()(URL, "q")

    assert (result.status, result.content, result.url) == (
        FetchStatus.RESTRICTED,
        "",
        URL,
    )


# ---------------------------------------------------------------------------
# FilterSelector / FilterChain
# ---------------------------------------------------------------------------


async def test_filter_selector_routes_by_result():
    html_filter, md_filter = RecordingFilter("h"), RecordingFilter("m")
    selector = FilterSelector(
        [
            (lambda r: r.mime == "text/html", html_filter),
            (lambda r: r.mime == "text/markdown", md_filter),
        ]
    )

    result = await selector(ok("x", "text/markdown"), "q")

    assert result.content == "xm"
    assert html_filter.calls == []
    assert md_filter.calls[0][1] == "q"


async def test_filter_selector_without_match_returns_same_object():
    source = ok("x", "application/json")
    selector = FilterSelector([(lambda r: r.mime == "text/html", RecordingFilter())])

    assert await selector(source) is source


async def test_filter_selector_uses_default_when_no_match():
    default = RecordingFilter("d")
    selector = FilterSelector([(lambda r: False, RecordingFilter())], default)

    assert (await selector(ok("x"))).content == "xd"


async def test_filter_chain_applies_filters_in_order():
    chain = FilterChain(RecordingFilter("1"), RecordingFilter("2"))

    assert (await chain(ok("x"))).content == "x12"


async def test_filter_chain_stops_at_first_failure():
    failure = failed(FetchStatus.FILTER_FAILED)
    later = RecordingFilter("2")
    chain = FilterChain(RecordingFilter(result=failure), later)

    result = await chain(ok())

    assert result is failure
    assert later.calls == []


# ---------------------------------------------------------------------------
# Throttled
# ---------------------------------------------------------------------------


class GatedFetch:
    """Fetch that blocks on a gate and tracks peak concurrency."""

    def __init__(self, gate: asyncio.Event) -> None:
        self.gate = gate
        self.in_flight = 0
        self.max_in_flight = 0

    async def __call__(self, url: str, query: str | None = None, /) -> FetchResult:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await self.gate.wait()
            return ok()
        finally:
            self.in_flight -= 1


async def test_throttled_caps_concurrency_at_limit():
    gate = asyncio.Event()
    inner = GatedFetch(gate)
    fetch = Throttled(inner, 2)

    tasks = [asyncio.create_task(fetch(URL)) for _ in range(5)]
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    peak_before_release = inner.max_in_flight
    gate.set()
    results = await asyncio.gather(*tasks)

    assert peak_before_release == 2
    assert inner.max_in_flight == 2
    assert all(result.ok for result in results)


# ---------------------------------------------------------------------------
# Truncate
# ---------------------------------------------------------------------------

MARKER = "\n\n[Content truncated]"


async def test_truncate_keeps_content_within_limit():
    source = ok("x" * 10)

    result = await Truncate(10)(source)

    assert result.content == "x" * 10


async def test_truncate_cuts_content_over_limit_and_marks_it():
    result = await Truncate(10)(ok("x" * 11, "text/plain"))

    assert result.content == "x" * 10 + MARKER
    assert result.status is FetchStatus.OK


# ---------------------------------------------------------------------------
# Phase 4: syntax-safe final truncation
# ---------------------------------------------------------------------------

PREFIX = "Intro."
FENCE = "```python\nprint('quantum')\n```"
MARKDOWN = PREFIX + "\n\n" + FENCE + "\n\nTail."
TABLE = "| a | b |\n| --- | --- |\n| 1 | 2 |"
LINKED = "See [paper](https://source.example/paper)."
REFERENCED = (
    "Read [paper][p].\n\nUnrelated tail.\n\n"
    "[p]: https://source.example/paper\n"
)


async def test_truncate_markdown_stops_before_partial_fence():
    cap = len(PREFIX) + 2 + 8

    result = await Truncate(cap)(ok(MARKDOWN, "text/markdown"))

    assert result.content == PREFIX + MARKER
    assert result.status is FetchStatus.OK


@pytest.mark.parametrize(
    "source", [FENCE, TABLE, LINKED], ids=["fence", "table", "linked"]
)
async def test_truncate_markdown_first_oversized_atomic_block_returns_marker_only(
    source: str,
):
    result = await Truncate(5)(ok(source, "text/markdown"))

    assert result.content == MARKER


async def test_truncate_markdown_counts_required_reference_definition():
    retained = "Read [paper][p].\n\n[p]: https://source.example/paper"
    cap = len(retained)

    result = await Truncate(cap)(ok(REFERENCED, "text/markdown"))

    assert result.content == retained + MARKER


async def test_truncate_markdown_omits_link_if_definition_cannot_fit():
    cap = len("Read [paper][p].")

    result = await Truncate(cap)(ok(REFERENCED, "text/markdown"))

    assert result.content == MARKER


async def test_truncate_markdown_does_not_leave_orphan_heading():
    source = "## Quantum\n\n" + FENCE
    cap = len("## Quantum") + 2 + 4

    result = await Truncate(cap)(ok(source, "text/markdown"))

    assert result.content == MARKER


async def test_truncate_plain_text_prefers_complete_word_boundary():
    result = await Truncate(12)(ok("alpha beta gamma", "text/plain"))

    assert result.content == "alpha beta" + MARKER


def test_truncate_negative_limit_is_invalid():
    with pytest.raises(ValueError):
        Truncate(-1)


async def test_truncate_exact_limit_is_unchanged():
    result = await Truncate(4)(ok("abcd", "text/plain"))

    assert result.content == "abcd"


async def test_truncate_empty_content_is_unchanged():
    result = await Truncate(0)(ok("", "text/plain"))

    assert result.content == ""


async def test_truncate_zero_limit_marks_nonempty_content():
    result = await Truncate(0)(ok("body", "text/plain"))

    assert result.content == MARKER


async def test_truncate_keeps_whole_table_that_fits():
    source = TABLE + "\n\nTail."

    result = await Truncate(len(TABLE))(
        ok(source, "text/markdown")
    )

    assert result.content == TABLE + MARKER


async def test_truncate_keeps_whole_fence_that_fits():
    source = PREFIX + "\n\n" + FENCE

    result = await Truncate(len(source))(
        ok(source, "text/markdown")
    )

    assert result.content == source


async def test_truncate_plain_prose_paragraph_is_trimmable():
    source = "alpha beta gamma delta"

    result = await Truncate(12)(ok(source, "text/plain"))

    assert result.content == "alpha beta" + MARKER


async def test_truncate_heading_only_markdown_is_kept():
    source = "## Quantum"

    result = await Truncate(50)(ok(source, "text/markdown"))

    assert result.content == source
