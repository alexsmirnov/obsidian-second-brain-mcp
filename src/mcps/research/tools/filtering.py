"""Relevance filtering: chunk the content, rank the fragments, keep the best."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from mcps.research.tools.common import failure, textual_mime
from mcps.research.tools.models import FetchResult, FetchStatus

__all__ = ["Chunker", "FilterModelError", "Fragment", "Ranker", "RelevanceFilter", "ChunkedRelevanceFilter"]

logger = logging.getLogger(__name__)


class FilterModelError(ValueError):
    """A model-backed ranker failed or returned unusable output."""


@dataclass(frozen=True, slots=True)
class Fragment:
    """One contiguous slice of a source text, ranked by its ``text``.

    ``start`` and ``end`` are half-open character offsets into the source, so
    ``text`` equals ``source[start:end]``. Fragments of one source may overlap
    and may arrive in any order.
    """

    text: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid fragment range: [{self.start}, {self.end})")
        if len(self.text) != self.end - self.start:
            raise ValueError(f"fragment text does not match [{self.start}, {self.end})")


class Chunker(Protocol):
    """Split content into fragments and merge the selected ones back into text.

    Fragment offsets address the text given to ``split``. Fragments may overlap
    and arrive in any order. ``merge`` rebuilds the text from the fragments
    alone, so it needs no copy of the source.
    """

    def split(self, text: str) -> list[Fragment]: ...

    def merge(self, fragments: Sequence[Fragment]) -> str: ...


class Ranker(Protocol):
    """Return the indices of the fragments relevant to the query."""

    async def __call__(self, fragments: list[str], query: str) -> list[int]: ...

class RelevanceFilter(Protocol):
    """Keep the query-relevant fragments of a textual result.  """
    async def __call__(
        self, result: FetchResult, query: str | None = None, /
    ) -> FetchResult: ...

class ChunkedRelevanceFilter:
    """Keep the query-relevant fragments of a textual result, merged in document order.

    Non-textual media give ``UNSUPPORTED`` and oversized input or query give
    ``FILTER_FAILED``, even for a blank query. Otherwise a blank query returns
    the result unchanged, and any chunker or ranker error gives
    ``FILTER_FAILED``. Output size is bounded by the downstream truncation.
    """

    def __init__(
        self,
        chunker: Chunker,
        ranker: Ranker,
        *,
        max_input_chars: int = 2_000_000,
        max_query_chars: int = 8_192,
    ) -> None:
        self._chunker = chunker
        self._ranker = ranker
        self._max_input_chars = max_input_chars
        self._max_query_chars = max_query_chars

    async def __call__(
        self, result: FetchResult, query: str | None = None, /
    ) -> FetchResult:
        if not result.ok:
            return result
        if textual_mime(result.mime) is None:
            return failure(result.url, FetchStatus.UNSUPPORTED)
        query = (query or "").strip()
        if (
            len(query) > self._max_query_chars
            or len(result.content) > self._max_input_chars
        ):
            return failure(result.url, FetchStatus.FILTER_FAILED)
        if not query:
            return result
        try:
            fragments = await asyncio.to_thread(self._chunker.split, result.content)
            selected = (
                await self._ranker([fragment.text for fragment in fragments], query)
                if fragments
                else []
            )
            kept = sorted({index for index in selected if 0 <= index < len(fragments)})
            content = self._chunker.merge([fragments[index] for index in kept])
        except Exception as error:
            logger.warning(
                "Relevance filtering failed for %s: %r", result.url, error
            )
            return failure(result.url, FetchStatus.FILTER_FAILED)
        return replace(result, content=content)

_FILTER_PROMPT = """You copy parts of a web page. You never write your own text.

You get a QUERY and a DOCUMENT. Copy the parts of the DOCUMENT that help answer the QUERY.

Rules:
1. Copy text exactly, character for character. Do not reword, shorten, fix, or summarize.
2. Copy whole paragraphs, lists, tables, and code blocks. Keep their markdown.
3. Keep a paragraph if it talks about the topic of the QUERY, even if it gives only part of the answer.
4. Keep the section heading above each copied paragraph.
5. Drop menus, cookie notices, ads, sign-up forms, author lists, funding notes. Drop repeated links.
6. If the QUERY is empty, copy the main article text and drop everything from rule 5.
7. Output only copied text. No introduction, no comments, no headings of your own.

Example:
QUERY: battery life of the X200 laptop
DOCUMENT:
Home | Laptops | Deals
## Battery
The X200 ran for 11 hours in our video test. Charging to full takes 90 minutes.
## Display
The screen is 14 inches with a matte finish.
We use cookies to improve your experience.

Output:
## Battery
The X200 ran for 11 hours in our video test. Charging to full takes 90 minutes.
"""

_FILTER_REMINDER = (
    "Now copy the parts of the DOCUMENT that help answer the QUERY. "
    "Copy them exactly. Output only copied text."
)


def _filter_message(query: str | None, document: str) -> str:
    """Wrap the query and page, repeating the task after the page for small models."""
    return (
        f"QUERY: {(query or '').strip()}\n\n"
        f"<DOCUMENT>\n{document}\n</DOCUMENT>\n\n"
        f"{_FILTER_REMINDER}"
    )


class LlmRelevanceFilter:

    def __init__(self, model: BaseChatModel) -> None:
        self.model = model

    async def __call__(
        self, result: FetchResult, query: str | None = None, 
    ) -> FetchResult:
        extract_message = _filter_message(query, result.content)
        try:
            response = await self.model.ainvoke(input=[SystemMessage(_FILTER_PROMPT),HumanMessage(extract_message)])
            text = response.text
            logger.info("llm filter returned %d chars",len(text))
            return replace(result, content=text)
        except Exception as error:
            logger.warning(
                "LLM filtering failed for %s: %r", result.url, error
            )
            return failure(result.url, FetchStatus.FILTER_FAILED)
        pass