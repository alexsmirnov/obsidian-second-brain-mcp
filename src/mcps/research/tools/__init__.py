"""Async research callables for web search and content fetching.

Each factory returns a callable performing I/O with an optionally injected
``httpx.AsyncClient`` shared from the FastMCP lifespan.
"""

from mcps.research.tools.duckduckgo import create_duckduckgo_search
from mcps.research.tools.fetch import create_fetch
from mcps.research.tools.google import create_google_search
from mcps.research.tools.models import SearchResult
from mcps.research.tools.result import Fetch, FetchResult, FetchStatus, Filter, Search

__all__ = [
    "Fetch",
    "FetchResult",
    "FetchStatus",
    "Filter",
    "Search",
    "SearchResult",
    "create_duckduckgo_search",
    "create_fetch",
    "create_google_search",
]
