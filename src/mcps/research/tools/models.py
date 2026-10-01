"""Data models shared by research search callables."""

from __future__ import annotations

from dataclasses import dataclass
from pydantic import BaseModel, Field
from enum import StrEnum
from typing import Protocol, runtime_checkable


class SearchResult(BaseModel):
    """Single search result item returned by search callables."""

    url: str = Field(description="Result URL")
    title: str = Field(description="Result title")
    snippet: str = Field(description="Result summary text")


# Blocked, auth-walled or rate-limited targets: a browser or unblocking
# provider may succeed where the previous fetch did not.
_RETRYABLE_HTTP_STATUSES = frozenset([401, 403, 429])


class FetchStatus(StrEnum):
    OK = "ok"
    RESTRICTED = "restricted"
    HTTP_ERROR = "http_error"
    TIMEOUT = "timeout"
    EMPTY = "empty"
    UNSUPPORTED = "unsupported"
    # A fetcher could not run (browser down, provider out of credits). Says
    # nothing about the target page.
    UNAVAILABLE = "unavailable"
    # The post-retrieval filter failed, not the target page.
    FILTER_FAILED = "filter_failed"


@dataclass(frozen=True, slots=True)
class FetchResult:
    """Outcome of fetching ``url``; ``url`` is always the requested URL."""

    url: str
    status: FetchStatus
    mime: str
    content: str = ""
    base_url: str | None = None
    http_status: int | None = None

    @property
    def ok(self) -> bool:
        return self.status is FetchStatus.OK

    def is_retryable(self) -> bool:
        """True when trying another fetcher may succeed."""
        match self.status:
            case FetchStatus.TIMEOUT | FetchStatus.EMPTY | FetchStatus.UNAVAILABLE:
                return True
            case FetchStatus.HTTP_ERROR:
                return self.http_status in _RETRYABLE_HTTP_STATUSES
            case _:
                return False


@runtime_checkable
class Fetch(Protocol):
    """Retrieve ``url``; never raises for expected failures."""

    async def __call__(self, url: str, query: str | None = None, /) -> FetchResult: ...


@runtime_checkable
class Filter(Protocol):
    """Transform a fetched result; receives only successful results from chains."""

    async def __call__(
        self, result: FetchResult, query: str | None = None, /
    ) -> FetchResult: ...


@runtime_checkable
class Search(Protocol):
    async def __call__(self, query: str, /) -> list[SearchResult]: ...
