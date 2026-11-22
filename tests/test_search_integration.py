"""Live integration test for the free web search fallback.

Hits the real Yahoo search page. Skipped by default; run with
``uv run pytest -m integration tests/test_search_integration.py``.
"""

from __future__ import annotations

import pytest

from mcps.research.tools import create_yahoo_search

pytestmark = pytest.mark.integration


async def test_yahoo_search_returns_real_unwrapped_results():
    search = create_yahoo_search()

    results = await search("python asyncio documentation")

    assert results, "Yahoo returned no results: blocked IP or changed markup"
    assert all(r.url.startswith("http") for r in results)
    assert not any("r.search.yahoo.com" in r.url for r in results)
    assert all(r.title for r in results)
