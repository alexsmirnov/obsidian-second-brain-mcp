"""Contract tests for the HTML to Markdown filter."""

from __future__ import annotations

from typing import Any

from mcps.research.tools.combinators import FilterSelector
from mcps.research.tools.html_markdown import (
    HtmlToMarkdown,
    create_markdown_generator,
    is_html,
)
from mcps.research.tools.models import FetchResult, FetchStatus

HTML = (
    "<html><body><h1>Title</h1>"
    '<p>See <a href="https://example.org/paper">the paper</a> for details.</p>'
    "</body></html>"
)


def page(content: str, mime: str) -> FetchResult:
    return FetchResult(
        url="https://example.org/page",
        status=FetchStatus.OK,
        mime=mime,
        content=content,
    )


async def test_html_to_markdown_converts_html_and_keeps_links():
    result = await HtmlToMarkdown(create_markdown_generator())(
        page(HTML, "text/html")
    )

    assert result.status is FetchStatus.OK
    assert result.mime == "text/markdown"
    assert "Title" in result.content
    assert "https://example.org/paper" in result.content
    assert "<p>" not in result.content


def test_is_html_matches_only_html_results():
    assert is_html(page(HTML, "text/html"))
    assert not is_html(page("# x", "text/markdown"))
    assert not is_html(page("x", "text/plain"))


async def test_selector_leaves_non_html_results_untouched():
    source = page("# Already markdown", "text/markdown")
    selector = FilterSelector([(is_html, HtmlToMarkdown(create_markdown_generator()))])

    assert await selector(source) == source


async def test_html_to_markdown_generator_error_returns_filter_failed():
    class Broken:
        def generate_markdown(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("boom")

    result = await HtmlToMarkdown(Broken())(page(HTML, "text/html"))  # type: ignore[arg-type]

    assert result.status is FetchStatus.FILTER_FAILED
