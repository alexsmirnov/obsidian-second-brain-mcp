"""Contract tests for the post-retrieval page filter (crawl4ai BM25/LLM)."""

from __future__ import annotations

import html
from types import SimpleNamespace
from typing import Any

import pytest

from mcps.research.tools.filtering import (
    create_page_filter,
    markdown_to_html,
    text_page_to_html,
)

TOPIC_HTML = (
    "<html><body><h2>Quantum optimization</h2><p>"
    + "quantum optimization improves routing " * 30
    + '<a href="/paper">paper</a></p><h2>Recipes</h2><p>'
    + "bread flour baking kitchen " * 30
    + '<a href="/recipes">recipes</a></p></body></html>'
)
BASE = "https://source.example/articles/page"

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
    page_filter = create_page_filter(
        fetch_model="fetch-test",
        router_url="https://router.example/v1",
        router_key="k",
    )

    for query in (None, "", "  "):
        result = await page_filter(TOPIC_HTML, BASE, query)

        assert "quantum optimization" in result
        assert "bread flour" in result
    assert llm_calls == []


async def test_bm25_keeps_relevant_block_and_absolute_link():
    page_filter = create_page_filter(fetch_model="", router_url="", router_key="")

    result = await page_filter(TOPIC_HTML, BASE, "quantum optimization")

    assert "quantum optimization" in result
    assert "https://source.example/paper" in result
    assert "bread flour" not in result


async def test_llm_filter_uses_router_model(llm_calls: list[dict[str, Any]]):
    page_filter = create_page_filter(
        fetch_model="fetch-test",
        router_url="https://router.example/v1",
        router_key="k",
    )

    result = await page_filter(TOPIC_HTML, BASE, "quantum")

    assert "Quantum kept" in result
    assert any(
        call["provider"] == "openai/fetch-test"
        and call["api_token"] == "k"
        and call["base_url"] == "https://router.example/v1"
        for call in llm_calls
    )

    prefixed = create_page_filter(
        fetch_model="openai/fetch-test",
        router_url="https://router.example/v1",
        router_key="k",
    )
    await prefixed(TOPIC_HTML, BASE, "quantum")

    assert any(call["provider"] == "openai/fetch-test" for call in llm_calls)


async def test_no_match_returns_empty_string():
    page_filter = create_page_filter(fetch_model="", router_url="", router_key="")

    result = await page_filter(TOPIC_HTML, BASE, "zzzz unrelated")

    assert result == ""


async def test_markdown_source_is_filterable():
    page_filter = create_page_filter(fetch_model="", router_url="", router_key="")

    result = await page_filter(
        markdown_to_html(MARKDOWN_PAGE), BASE, "quantum optimization"
    )

    assert "https://source.example/paper" in result
    assert "bread flour" not in result


async def test_plain_text_page_is_filterable():
    page_filter = create_page_filter(fetch_model="", router_url="", router_key="")
    page = f"<html><body><pre>{html.escape(MARKDOWN_PAGE)}</pre></body></html>"

    result = await page_filter(text_page_to_html(page), BASE, "quantum optimization")

    assert "https://source.example/paper" in result
    assert "bread flour" not in result
    assert text_page_to_html(TOPIC_HTML) == TOPIC_HTML
    assert (
        text_page_to_html("<html><body><h2>x</h2><pre>code</pre></body></html>")
        == "<html><body><h2>x</h2><pre>code</pre></body></html>"
    )

