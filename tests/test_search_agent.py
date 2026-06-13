from datetime import datetime

import pytest
from unittest.mock import AsyncMock, MagicMock

from mcps.rag.search_agent import SearchAgent
from mcps.rag.interfaces import SearchQuery, Chunk


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _make_chunk(id_: str, title: str | None = None) -> Chunk:
    """Return a minimal valid Chunk instance for tests."""
    return Chunk(
        id=id_,
        content=f"content-{id_}",
        title=title,
        description=None,
        outgoing_links=[],
        tags=[],
        source_path=f"path/{id_}.md",
        modified_at=datetime.utcnow(),
        position=0,
    )


@pytest.fixture()
def vector_store_mock():
    """Return a mock implementing the IVectorStore interface."""
    vector_store = MagicMock()
    vector_store.search = AsyncMock(return_value=[])
    return vector_store


@pytest.fixture()
def llm_mock():
    """Return a mock implementing the ILLM interface."""
    llm = MagicMock()
    llm.generate = AsyncMock(return_value="")
    return llm


@pytest.fixture()
def agent(vector_store_mock, llm_mock):
    """Return a fresh SearchAgent instance for every test."""
    return SearchAgent(vector_store=vector_store_mock, llm=llm_mock)


# ---------------------------------------------------------------------------
# _rewrite_query
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rewrite_query_success(agent, llm_mock):
    """LLM happy path – return the rewritten text."""
    llm_mock.generate.return_value = "expanded query"
    query = SearchQuery(text="original", tags=[])

    result = await agent._rewrite_query(query)

    llm_mock.generate.assert_awaited_once()
    assert result == "expanded query"


@pytest.mark.asyncio
async def test_rewrite_query_llm_error_returns_original(agent, llm_mock):
    """If LLM fails, the original query text should be returned."""
    llm_mock.generate.side_effect = RuntimeError("boom")
    query = SearchQuery(text="original", tags=[])

    result = await agent._rewrite_query(query)

    llm_mock.generate.assert_awaited_once()
    assert result == "original"


# ---------------------------------------------------------------------------
# _infer_tags
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_infer_tags_merges_and_deduplicates(agent, llm_mock):
    """Should merge LLM-inferred and user-provided tags without duplicates."""
    llm_mock.generate.return_value = "tag2, tag3"
    query = SearchQuery(text="query", tags=["tag1", "tag2"])

    result = await agent._infer_tags(query)

    llm_mock.generate.assert_awaited_once()
    # Order is not important – convert to set for comparison
    assert set(result) == {"tag1", "tag2", "tag3"}


@pytest.mark.asyncio
async def test_infer_tags_llm_error_returns_query_tags(agent, llm_mock):
    """If LLM fails, return the tags provided in the query as-is."""
    llm_mock.generate.side_effect = RuntimeError("boom")
    query = SearchQuery(text="query", tags=["tag1", "tag2"])

    result = await agent._infer_tags(query)

    llm_mock.generate.assert_awaited_once()
    assert result == ["tag1", "tag2"]


# ---------------------------------------------------------------------------
# _initial_search
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_initial_search_removes_duplicates(agent, vector_store_mock):
    """Should eliminate duplicate chunks based on their id."""
    dup_chunk = _make_chunk("1")
    vector_store_mock.search.return_value = [dup_chunk, dup_chunk]

    result = await agent._initial_search("full query", ["tag"])

    vector_store_mock.search.assert_awaited_once()
    assert result == [dup_chunk]


@pytest.mark.asyncio
async def test_initial_search_error_returns_empty(agent, vector_store_mock):
    """If the vector store fails, an empty list should be returned."""
    vector_store_mock.search.side_effect = RuntimeError("boom")

    result = await agent._initial_search("full query", [])

    vector_store_mock.search.assert_awaited_once()
    assert result == []


# ---------------------------------------------------------------------------
# _collect_linked_notes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_collect_linked_notes_empty_links_returns_empty(agent):
    """No outgoing links – nothing to collect."""
    query = SearchQuery(text="q", tags=[])
    result = await agent._collect_linked_notes(query, set())
    assert result == []


@pytest.mark.asyncio
async def test_collect_linked_notes_error_returns_empty(agent, vector_store_mock):
    vector_store_mock.search.side_effect = RuntimeError("boom")
    query = SearchQuery(text="q", tags=[])
    links = {"Note1.md"}

    result = await agent._collect_linked_notes(query, links)

    vector_store_mock.search.assert_awaited()
    assert result == []


# ---------------------------------------------------------------------------
# _collect_incoming_notes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_collect_incoming_notes_error_returns_empty(agent, vector_store_mock):
    vector_store_mock.search.side_effect = RuntimeError("boom")
    query = SearchQuery(text="q", tags=[])
    chunks = [_make_chunk("1")]

    result = await agent._collect_incoming_notes(query, chunks)

    vector_store_mock.search.assert_awaited()
    assert result == []


# ---------------------------------------------------------------------------
# _generate_answer
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_answer_success(agent, llm_mock):
    llm_mock.generate.return_value = "final answer"
    query = SearchQuery(text="q", tags=[])
    chunks = {_make_chunk("1"), _make_chunk("2")}

    result = await agent._generate_answer(query, chunks)

    llm_mock.generate.assert_awaited_once()
    assert result == "final answer"


@pytest.mark.asyncio
async def test_generate_answer_error_returns_fallback(agent, llm_mock):
    llm_mock.generate.side_effect = RuntimeError("boom")
    query = SearchQuery(text="q", tags=[])

    result = await agent._generate_answer(query, set())

    llm_mock.generate.assert_awaited_once()
    assert "Unable to generate" in result  # Expect some graceful degradation


# ---------------------------------------------------------------------------
# search – high-level orchestration
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_calls_all_steps(monkeypatch):
    """Ensure the orchestration method delegates to all internal helpers."""
    agent = SearchAgent(vector_store=MagicMock(), llm=MagicMock())
    query = SearchQuery(text="hi", tags=[])

    # Patch all helper methods to controlled AsyncMocks
    steps_return_values = {
        "_rewrite_query": "rewritten",
        "_infer_tags": ["t1"],
        "_initial_search": [_make_chunk("1")],
        "_collect_linked_notes": [],
        "_collect_incoming_notes": [],
        "_generate_answer": "ANSWER",
    }

    for name, retval in steps_return_values.items():
        async_mock = AsyncMock(return_value=retval)
        monkeypatch.setattr(agent, name, async_mock)

    # Act
    result = await agent.search(query)

    # Assert result propagated
    assert result == "ANSWER"

    # Verify each helper was awaited exactly once with expected args (order unimportant)
    agent._rewrite_query.assert_awaited_once_with(query)
    agent._infer_tags.assert_awaited_once_with(query)
    agent._initial_search.assert_awaited_once_with("rewritten", ["t1"])
    agent._collect_linked_notes.assert_awaited_once()
    agent._collect_incoming_notes.assert_awaited_once()
    agent._generate_answer.assert_awaited_once() 