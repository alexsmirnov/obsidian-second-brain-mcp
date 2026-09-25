"""Data models shared by research search callables."""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = ["SearchResult"]


class SearchResult(BaseModel):
    """Single search result item returned by search callables."""

    url: str = Field(description="Result URL")
    title: str = Field(description="Result title")
    snippet: str = Field(description="Result summary text")
