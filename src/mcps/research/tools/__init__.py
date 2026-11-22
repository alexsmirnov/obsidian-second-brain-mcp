"""Async research callables for web search and content fetching.

Each factory returns a callable performing I/O with an optionally injected
``httpx.AsyncClient`` shared from the FastMCP lifespan.
"""

from mcps.research.tools.fetch import create_fetch
from mcps.research.tools.google import create_google_search
from mcps.research.tools.models import (
    Fetch,
    FetchResult,
    FetchStatus,
    Filter,
    Search,
    SearchResult,
)
from mcps.research.tools.yahoo import create_yahoo_search

__all__ = [
    "Fetch",
    "FetchResult",
    "FetchStatus",
    "Filter",
    "Search",
    "SearchResult",
    "create_fetch",
    "create_google_search",
    "create_yahoo_search",
]
