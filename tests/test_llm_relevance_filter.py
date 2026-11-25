"""Contract tests for LlmRelevanceFilter with a recording fake chat model."""

from __future__ import annotations

from typing import cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage

from mcps.common.ai_models import tokens_for_chars
from mcps.research.tools.filtering import LlmRelevanceFilter, llm_filter_model_limits
from mcps.research.tools.models import FetchResult, FetchStatus


class RecordingModel:
    """Chat model boundary fake: records the prompt and returns a fixed reply."""

    def __init__(self, reply: str = "kept text") -> None:
        self.reply = reply
        self.prompts: list[str] = []

    async def ainvoke(self, input: list[BaseMessage]) -> AIMessage:
        self.prompts.append("\n".join(str(message.content) for message in input))
        return AIMessage(self.reply)


def page(content: str) -> FetchResult:
    return FetchResult(
        url="https://example.org/p",
        status=FetchStatus.OK,
        mime="text/markdown",
        content=content,
    )


async def test_returns_model_reply_as_content():
    model = RecordingModel("kept text")
    page_filter = LlmRelevanceFilter(cast(BaseChatModel, model))

    result = await page_filter(page("body"), "query")

    assert result.content == "kept text"


async def test_sends_whole_document_within_limit():
    model = RecordingModel()
    page_filter = LlmRelevanceFilter(cast(BaseChatModel, model), max_input_chars=20)

    await page_filter(page("a" * 19 + "Z"), "query")

    assert "a" * 19 + "Z" in model.prompts[0]


async def test_cuts_document_end_beyond_limit():
    model = RecordingModel()
    page_filter = LlmRelevanceFilter(cast(BaseChatModel, model), max_input_chars=20)

    await page_filter(page("a" * 20 + "TAIL"), "query")

    assert "a" * 20 in model.prompts[0]
    assert "TAIL" not in model.prompts[0]


def test_model_limits_fit_full_input_and_output_with_reasoning():
    limits = llm_filter_model_limits(
        max_input_chars=200_000, max_output_chars=20_000, reasoning_tokens=2048
    )

    assert limits["max_output_tokens"] == tokens_for_chars(20_000) + 2048
    assert limits["context_window"] >= (
        tokens_for_chars(200_000) + limits["max_output_tokens"]
    )


def test_model_limits_reserve_reasoning_tokens_by_default():
    limits = llm_filter_model_limits(max_input_chars=200_000, max_output_chars=20_000)

    assert limits["max_output_tokens"] > tokens_for_chars(20_000)


async def test_removes_model_thinking_from_content():
    model = RecordingModel("<|channel>thought\nwhich parts?<channel|>kept text")
    page_filter = LlmRelevanceFilter(cast(BaseChatModel, model))

    result = await page_filter(page("body"), "query")

    assert result.content == "kept text"
