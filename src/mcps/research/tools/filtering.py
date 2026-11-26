"""Relevance filtering: chunk the content, rank the fragments, keep the best."""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from mcps.common.ai_models import strip_thinking, tokens_for_chars
from mcps.research.tools.common import failure, textual_mime
from mcps.research.tools.models import FetchResult, FetchStatus

__all__ = [
    "ChunkedRelevanceFilter",
    "Chunker",
    "FilterModelError",
    "Fragment",
    "LlmRelevanceFilter",
    "Ranker",
    "RelevanceFilter",
    "llm_filter_model_limits",
]

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

# Page and output sizes the LLM filter is sized for.
LLM_FILTER_MAX_INPUT_CHARS = 200_000
LLM_FILTER_MAX_OUTPUT_CHARS = 25_000
# Room for the system prompt, reminder and a long research query.
_LLM_FILTER_OVERHEAD_CHARS = 12_000
# Output tokens a thinking model spends on reasoning before the extract;
# local Gemma used about 1.5k on research pages.
LLM_FILTER_REASONING_TOKENS = 2048
# Round context windows up to this many tokens.
_CONTEXT_STEP_TOKENS = 1024

_FILTER_PROMPT = """You copy parts of a web page. You never write your own text.

You get a QUERY and a DOCUMENT. Copy the parts of the DOCUMENT that help answer the QUERY, and those that preface, explain, or extend selected.

Rules:
1. Copy text exactly, character for character. Do not reword, shorten, fix, or summarize.
2. Copy whole sentences, lists, tables, and code blocks. When unsure whether a part is relevant, copy it: related background, methods, and numbers from similar setups are useful too.
3. Do not worry about formatting, result will be used by machine, not human.
4. Drop menus, cookie notices, ads, sign-up forms, author lists, funding notes. Drop repeated links.
5. If the QUERY is empty, copy the main article text and drop everything from rule 5.
6. Output only copied text. No introduction, no comments, no headings of your own.

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
The X200 ran for 11 hours in our video test. Charging to full takes 90 minutes.
"""

_FILTER_REMINDER = (
    "Now copy the parts of the DOCUMENT that help answer the QUERY and connected text. "
    "Copy them exactly. Output only copied text."
)


def _filter_message(query: str | None, document: str) -> str:
    """Wrap the query and page, repeating the task after the page for small models."""
    return (
        f"QUERY: {(query or '').strip()}\n\n"
        f"<DOCUMENT>\n{document}\n</DOCUMENT>\n\n"
        f"{_FILTER_REMINDER}"
    )


def llm_filter_model_limits(
    max_input_chars: int = LLM_FILTER_MAX_INPUT_CHARS,
    max_output_chars: int = LLM_FILTER_MAX_OUTPUT_CHARS,
    reasoning_tokens: int = LLM_FILTER_REASONING_TOKENS,
) -> dict[str, int]:
    """``create_chat_model`` token limits that fit a full page and its extract.

    Local Ollama models default to a 4096-token context, far below a page.
    The output limit counts reasoning tokens too, so they are reserved on top
    of the extract.
    """
    output_tokens = tokens_for_chars(max_output_chars) + reasoning_tokens
    input_tokens = tokens_for_chars(max_input_chars + _LLM_FILTER_OVERHEAD_CHARS)
    context_window = (
        math.ceil((input_tokens + output_tokens) / _CONTEXT_STEP_TOKENS)
        * _CONTEXT_STEP_TOKENS
    )
    return {"context_window": context_window, "max_output_tokens": output_tokens}


class LlmRelevanceFilter:
    """Ask a chat model to copy the query-relevant parts of a page.

    Pages longer than ``max_input_chars`` lose their end, so the model never
    has to drop the start of the prompt, where its instructions are.
    """

    def __init__(
        self,
        model: BaseChatModel,
        *,
        max_input_chars: int = LLM_FILTER_MAX_INPUT_CHARS,
    ) -> None:
        self.model = model
        self._max_input_chars = max_input_chars

    async def __call__(
        self, result: FetchResult, query: str | None = None, 
    ) -> FetchResult:
        document = result.content[: self._max_input_chars]
        if len(document) < len(result.content):
            logger.info(
                "llm filter cut %s from %d to %d chars",
                result.url,
                len(result.content),
                len(document),
            )
        extract_message = _filter_message(query, document)
        try:
            response = await self.model.ainvoke(input=[SystemMessage(_FILTER_PROMPT),HumanMessage(extract_message)])
            text = strip_thinking(response.text)
            logger.info(
                "llm filter returned %d chars for %s: usage=%s finish=%s",
                len(text),
                result.url,
                getattr(response, "usage_metadata", None),
                response.response_metadata.get("finish_reason"),
            )
            return replace(result, content=text)
        except Exception as error:
            logger.warning(
                "LLM filtering failed for %s: %r", result.url, error
            )
            return failure(result.url, FetchStatus.FILTER_FAILED)
        pass