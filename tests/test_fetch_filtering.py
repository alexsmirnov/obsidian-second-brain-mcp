"""Contract tests for the post-retrieval page filter (crawl4ai BM25/LLM)."""

from __future__ import annotations

import html
from types import SimpleNamespace
from typing import Any

import pytest

from mcps.research.tools.filtering import (
    MarkdownToHtml,
    PreTextToHtml,
    RelevanceFilter,
)
from mcps.research.tools.models import FetchResult, FetchStatus

TOPIC_HTML = (
    "<html><body><h2>Quantum optimization</h2><p>"
    + "quantum optimization improves routing " * 30
    + '<a href="/paper">paper</a></p><h2>Recipes</h2><p>'
    + "bread flour baking kitchen " * 30
    + '<a href="/recipes">recipes</a></p></body></html>'
)
BASE = "https://source.example/articles/page"


def page(content: str, mime: str = "text/html", **fields: Any) -> FetchResult:
    return FetchResult(
        url=BASE, status=FetchStatus.OK, mime=mime, content=content, **fields
    )


def bm25() -> RelevanceFilter:
    return RelevanceFilter(fetch_model="", router_url="", router_key="")

MARKDOWN_PAGE = (
    "## Quantum\n\n"
    + "quantum optimization improves routing " * 30
    + "[paper](/paper)\n\n## Recipes\n\n"
    + "bread flour baking kitchen " * 30
)


def _llm_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content=f"<content>{text}</content>"))
        ],
        usage=SimpleNamespace(
            completion_tokens=5,
            prompt_tokens=10,
            total_tokens=15,
            completion_tokens_details=None,
            prompt_tokens_details=None,
        ),
    )


@pytest.fixture
def llm_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_completion_with_backoff(
        provider: str, _prompt: str, api_token: str, **kwargs: Any
    ) -> SimpleNamespace:
        calls.append(
            {
                "provider": provider,
                "api_token": api_token,
                "base_url": kwargs.get("base_url"),
            }
        )
        return _llm_response(
            "Quantum kept [paper](https://source.example/paper)"
        )

    monkeypatch.setattr(
        "crawl4ai.content_filter_strategy.perform_completion_with_backoff",
        fake_completion_with_backoff,
    )
    return calls


async def test_blank_query_returns_unfiltered_markdown(
    llm_calls: list[dict[str, Any]],
):
    page_filter = RelevanceFilter(
        fetch_model="fetch-test",
        router_url="https://router.example/v1",
        router_key="k",
    )

    for query in (None, "", "  "):
        result = await page_filter(page(TOPIC_HTML), query)

        assert result.ok
        assert "quantum optimization" in result.content
        assert "bread flour" in result.content
    assert llm_calls == []


async def test_bm25_keeps_relevant_block_and_absolute_link():
    result = await bm25()(page(TOPIC_HTML), "quantum optimization")

    assert result.mime == "text/markdown"
    assert "quantum optimization" in result.content
    assert "https://source.example/paper" in result.content
    assert "bread flour" not in result.content


async def test_links_resolve_against_base_url_when_set():
    source = page(TOPIC_HTML, base_url="https://other.example/dir/")

    result = await bm25()(source, "quantum optimization")

    assert "https://other.example/paper" in result.content


async def test_llm_filter_uses_router_model(llm_calls: list[dict[str, Any]]):
    page_filter = RelevanceFilter(
        fetch_model="fetch-test",
        router_url="https://router.example/v1",
        router_key="k",
    )

    result = await page_filter(page(TOPIC_HTML), "quantum")

    assert "Quantum kept" in result.content
    assert any(
        call["provider"] == "openai/fetch-test"
        and call["api_token"] == "k"
        and call["base_url"] == "https://router.example/v1"
        for call in llm_calls
    )

    prefixed = RelevanceFilter(
        fetch_model="openai/fetch-test",
        router_url="https://router.example/v1",
        router_key="k",
    )
    await prefixed(page(TOPIC_HTML), "quantum")

    assert any(call["provider"] == "openai/fetch-test" for call in llm_calls)


async def test_no_match_is_ok_with_empty_content():
    result = await bm25()(page(TOPIC_HTML), "zzzz unrelated")

    assert (result.status, result.content) == (FetchStatus.OK, "")


async def test_filter_engine_error_is_filter_failed(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "crawl4ai.DefaultMarkdownGenerator.generate_markdown",
        lambda *_args, **_kwargs: SimpleNamespace(
            raw_markdown="", fit_markdown="Error generating fit markdown: boom"
        ),
    )

    result = await bm25()(page(TOPIC_HTML), "quantum")

    assert result.status is FetchStatus.FILTER_FAILED
    assert result.url == BASE


async def test_markdown_to_html_makes_markdown_filterable():
    source = page(MARKDOWN_PAGE, mime="text/markdown")

    converted = await MarkdownToHtml()(source)
    result = await bm25()(converted, "quantum optimization")

    assert converted.mime == "text/html"
    assert "<h2>Quantum</h2>" in converted.content
    assert "https://source.example/paper" in result.content
    assert "bread flour" not in result.content


async def test_pre_text_page_is_converted_to_filterable_html():
    pre_page = f"<html><body><pre>{html.escape(MARKDOWN_PAGE)}</pre></body></html>"

    converted = await PreTextToHtml()(page(pre_page))
    result = await bm25()(converted, "quantum optimization")

    assert "https://source.example/paper" in result.content
    assert "bread flour" not in result.content


@pytest.mark.parametrize(
    "content",
    [TOPIC_HTML, "<html><body><h2>x</h2><pre>code</pre></body></html>"],
    ids=["html", "pre-with-sibling"],
)
async def test_pre_text_leaves_regular_html_unchanged(content: str):
    result = await PreTextToHtml()(page(content))

    assert result.content == content
