"""Relevance filtering of real fetched pages, compared across ranker implementations.

Pages and queries come from the fetch smoke dataset. ``expectations.json`` lists,
per case, passages the filter must keep and boilerplate it must drop.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator, Callable

import httpx
import pytest
from dotenv import find_dotenv, load_dotenv
from web_fetch_evaluation import CASES_DIR, load_cases

from mcps.config import ServerConfig, create_config
from mcps.research.tools.bm25_ranker import Bm25Ranker
from mcps.research.tools.filtering import RelevanceFilter
from mcps.research.tools.filtering_models import RouterPassageModels, valid_router_url
from mcps.research.tools.llm_ranker import LlmRanker
from mcps.research.tools.markdown_chunker import MarkdownChunker
from mcps.research.tools.models import FetchResult, FetchStatus

logger = logging.getLogger(__name__)

PAGES_DIR = CASES_DIR / "filtering"
EXPECTATIONS: dict[str, dict] = json.loads(
    (PAGES_DIR / "expectations.json").read_text(encoding="utf-8")
)
CASES = {
    case.case_id: case for case in load_cases(CASES_DIR / "fetch-cases-smoke.jsonl")
}

FilterBuilder = Callable[[ServerConfig, httpx.AsyncClient], RelevanceFilter]


def bm25_filter(
    _config: ServerConfig, _http_client: httpx.AsyncClient
) -> RelevanceFilter:
    return RelevanceFilter(MarkdownChunker(), Bm25Ranker())


def llm_filter(config: ServerConfig, http_client: httpx.AsyncClient) -> RelevanceFilter:
    if not (
        config.fetch_model
        and config.rag_embedding_model
        and config.router_api_key
        and valid_router_url(config.router_api_base)
    ):
        pytest.skip("FETCH_MODEL, RAG_EMBEDDING_MODEL or router settings not set")
    models = RouterPassageModels(
        http_client,
        router_url=config.router_api_base,
        router_key=config.router_api_key,
        fetch_model=config.fetch_model,
        embedding_model=config.rag_embedding_model,
        embedding_dimensions=config.rag_embedding_dimensions,
    )
    return RelevanceFilter(MarkdownChunker(), LlmRanker(models))


# Append an entry to compare another filter implementation on the same pages.
FILTERS: dict[str, FilterBuilder] = {"bm25": bm25_filter, "llm": llm_filter}


def page_content(case_id: str) -> str:
    return (PAGES_DIR / EXPECTATIONS[case_id]["file"]).read_text(encoding="utf-8")


@pytest.fixture
async def http_client() -> AsyncGenerator[httpx.AsyncClient]:
    async with httpx.AsyncClient(timeout=120.0) as client:
        yield client


@pytest.fixture(scope="module")
def config() -> ServerConfig:
    load_dotenv(find_dotenv(usecwd=True))
    return create_config()


@pytest.fixture(params=list(FILTERS))
def page_filter(
    request: pytest.FixtureRequest,
    config: ServerConfig,
    http_client: httpx.AsyncClient,
) -> RelevanceFilter:
    return FILTERS[request.param](config, http_client)


@pytest.mark.parametrize("case_id", sorted(EXPECTATIONS))
def test_expected_substrings_occur_once_in_page(case_id: str) -> None:
    content = page_content(case_id)
    expected = EXPECTATIONS[case_id]

    counts = {
        text: content.count(text)
        for text in expected["relevant"] + expected["irrelevant"]
    }

    assert expected["relevant"]
    assert expected["irrelevant"]
    assert {text: count for text, count in counts.items() if count != 1} == {}


@pytest.mark.parametrize("case_id", sorted(EXPECTATIONS))
async def test_filter_keeps_relevant_and_drops_irrelevant_passages(
    page_filter: RelevanceFilter, case_id: str
) -> None:
    case = CASES[case_id]
    expected = EXPECTATIONS[case_id]
    content = page_content(case_id)
    page = FetchResult(
        url=case.url, status=FetchStatus.OK, mime=expected["mime"], content=content
    )

    result = await page_filter(page, case.query)

    kept = result.content
    logger.info("%s kept %d of %d chars", case_id, len(kept), len(content))
    assert result.status is FetchStatus.OK
    assert {
        "missing": [text for text in expected["relevant"] if text not in kept],
        "leaked": [text for text in expected["irrelevant"] if text in kept],
    } == {"missing": [], "leaked": []}
