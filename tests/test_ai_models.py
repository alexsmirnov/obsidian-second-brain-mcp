"""Contract tests for the router-backed model factory and structured invocation."""

from typing import Any, cast

import httpx
import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel

from mcps.common.ai_models import (
    DEFAULT_MAX_RETRIES,
    StructuredOutputError,
    create_chat_model,
    create_embeddings_model,
    invoke_structured,
    invoke_structured_with_raw,
    strip_thinking,
    tokens_for_chars,
)
from mcps.config import RouterSettings

ROUTER = RouterSettings(api_base="http://router:4000", api_key="sk-test")


class TestClientSelection:
    def test_gemini_prefix_uses_google_client(self):
        model = create_chat_model("gemini-3-flash", ROUTER)

        assert isinstance(model, ChatGoogleGenerativeAI)
        assert model.model.endswith("gemini-3-flash")

    def test_gpt_prefix_uses_chat_completions(self):
        model = create_chat_model("gpt-5-mini", ROUTER)

        assert isinstance(model, ChatOpenAI)
        assert model.use_responses_api is False
        assert model.openai_api_base == ROUTER.api_base

    @pytest.mark.parametrize("name", ["local-gemma", "deepseek-flash", "my-gemini-3"])
    def test_other_names_use_chat_completions(self, name):
        model = create_chat_model(name, ROUTER)

        assert isinstance(model, ChatOpenAI)
        assert model.use_responses_api is False


class TestGeminiOptions:
    def test_gemini_3_maps_effort_to_thinking_level(self):
        model = create_chat_model("gemini-3-pro", ROUTER, reasoning_effort="high")

        assert model.reasoning_effort == "high"

    def test_gemini_3_maps_no_effort_to_minimal_level(self):
        model = create_chat_model("gemini-3-flash", ROUTER, reasoning_effort="none")

        assert model.reasoning_effort == "minimal"

    def test_gemini_3_ignores_temperature(self):
        model = create_chat_model("gemini-3-flash", ROUTER, temperature=0.2)

        assert model.temperature is None

    def test_gemini_25_maps_effort_to_thinking_budget(self):
        model = create_chat_model("gemini-2.5-flash", ROUTER, reasoning_effort="low")

        assert model.thinking_budget == 1024
        assert model.reasoning_effort is None

    def test_gemini_25_flash_disables_thinking_for_no_effort(self):
        model = create_chat_model("gemini-2.5-flash", ROUTER, reasoning_effort="none")

        assert model.thinking_budget == 0

    def test_gemini_25_pro_uses_minimum_budget_for_no_effort(self):
        model = create_chat_model("gemini-2.5-pro", ROUTER, reasoning_effort="none")

        assert model.thinking_budget == 128

    def test_gemini_25_keeps_temperature(self):
        model = create_chat_model("gemini-2.5-flash", ROUTER, temperature=0.2)

        assert model.temperature == 0.2

    def test_older_gemini_ignores_effort(self):
        model = create_chat_model("gemini-2.0-flash", ROUTER, reasoning_effort="high")

        assert model.thinking_budget is None
        assert model.reasoning_effort is None


class TestGptOptions:
    def test_gpt_5_sends_effort_as_reasoning_effort(self):
        model = create_chat_model("gpt-5-mini", ROUTER, reasoning_effort="low")

        assert model.reasoning_effort == "low"
        assert model.reasoning is None

    def test_gpt_4_ignores_effort(self):
        model = create_chat_model("gpt-4o", ROUTER, reasoning_effort="low")

        assert model.reasoning_effort is None

    def test_gpt_keeps_temperature(self):
        model = create_chat_model("gpt-4o", ROUTER, temperature=0.3)

        assert model.temperature == 0.3

    def test_gpt_ignores_context_window(self):
        model = create_chat_model("gpt-5-mini", ROUTER, context_window=8192)

        assert not model.extra_body


class TestCompletionsOptions:
    def test_sends_effort_as_reasoning_effort(self):
        model = create_chat_model("local-qwen", ROUTER, reasoning_effort="medium")

        assert model.reasoning_effort == "medium"
        assert model.reasoning is None

    def test_sends_context_window_as_num_ctx(self):
        model = create_chat_model("local-qwen", ROUTER, context_window=16384)

        assert model.extra_body == {"num_ctx": 16384}

    def test_merges_context_window_into_given_extra_body(self):
        model = create_chat_model(
            "local-qwen", ROUTER, context_window=8192, extra_body={"keep_alive": "5m"}
        )

        assert model.extra_body == {"keep_alive": "5m", "num_ctx": 8192}

    def test_keeps_temperature(self):
        model = create_chat_model("local-qwen", ROUTER, temperature=0.0)

        assert model.temperature == 0.0


class TestMaxOutputTokens:
    def test_gemini_sets_max_output_tokens(self):
        model = create_chat_model("gemini-3-flash", ROUTER, max_output_tokens=6000)

        assert model.max_output_tokens == 6000

    def test_gpt_sets_max_tokens(self):
        model = create_chat_model("gpt-5-mini", ROUTER, max_output_tokens=6000)

        assert model.max_tokens == 6000

    def test_completions_sets_max_tokens(self):
        model = create_chat_model("local-qwen", ROUTER, max_output_tokens=6000)

        assert model.max_tokens == 6000


@pytest.mark.parametrize(("chars", "tokens"), [(0, 0), (3000, 1000), (3001, 1001)])
def test_tokens_for_chars_rounds_up(chars, tokens):
    assert tokens_for_chars(chars) == tokens


def test_passes_other_kwargs_to_constructor():
    model = create_chat_model("local-qwen", ROUTER, max_retries=7)

    assert model.max_retries == 7


class TestStripThinking:
    @pytest.mark.parametrize(
        "text",
        [
            "<|channel>thought\nplan the answer<channel|>## Kept\nbody",
            "<think>\nplan the answer\n</think>\n## Kept\nbody",
        ],
    )
    def test_removes_closed_thinking_block(self, text):
        assert strip_thinking(text) == "## Kept\nbody"

    def test_removes_unclosed_thinking_block_cut_by_output_limit(self):
        assert strip_thinking("<|channel>thought\nplan the ans") == ""

    def test_keeps_text_without_thinking(self):
        assert strip_thinking("## Kept\nbody") == "## Kept\nbody"


class TestEmbeddings:
    def test_uses_router_and_shared_client(self):
        client = httpx.AsyncClient()

        embeddings = create_embeddings_model("embed-m", 768, ROUTER, client)

        assert isinstance(embeddings, OpenAIEmbeddings)
        assert embeddings.model == "embed-m"
        assert embeddings.dimensions == 768
        assert embeddings.openai_api_base == ROUTER.api_base
        assert embeddings.http_async_client is client
        assert embeddings.check_embedding_ctx_length is False


class TestDefaultRetries:
    @pytest.mark.parametrize(
        ("name", "retries"),
        [("gpt-5-mini", 2), ("local-qwen", 2), ("gemini-3-flash", 3)],
    )
    def test_sets_two_retries(self, name, retries):
        # Gemini counts total attempts, so two retries means three attempts.
        assert DEFAULT_MAX_RETRIES == 2

        model = create_chat_model(name, ROUTER)

        assert model.max_retries == retries

    def test_keeps_explicit_max_retries_for_gemini(self):
        model = create_chat_model("gemini-3-flash", ROUTER, max_retries=7)

        assert model.max_retries == 7


class Answer(BaseModel):
    value: int


class ScriptedChatModel:
    """Stands in for a chat model: replays scripted replies, raising exceptions."""

    def __init__(self, replies: list[object]) -> None:
        self.replies = list(replies)
        self.calls: list[list[Any]] = []
        self.schema: Any = None
        self.options: dict[str, Any] = {}

    def with_structured_output(
        self, schema: Any, **options: Any
    ) -> "ScriptedChatModel":
        self.schema = schema
        self.options = options
        return self

    async def ainvoke(self, messages: list[Any]) -> Any:
        self.calls.append(messages)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def scripted(*replies: object) -> ScriptedChatModel:
    return ScriptedChatModel(list(replies))


def as_chat_model(model: ScriptedChatModel) -> BaseChatModel:
    return cast(BaseChatModel, model)


def parse_error() -> OutputParserException:
    return OutputParserException("bad json", llm_output='{"value": "x"}')


QUESTION: list[BaseMessage] = [HumanMessage("q")]


class TestInvokeStructured:
    async def test_returns_parsed_reply_and_passes_output_options(self):
        model = scripted(Answer(value=1))

        result = await invoke_structured(
            as_chat_model(model), Answer, QUESTION, label="t", strict=True
        )

        assert result == Answer(value=1)
        assert model.schema is Answer
        assert model.options == {"strict": True}
        assert len(model.calls) == 1

    async def test_reprompts_with_failed_reply_and_error(self):
        model = scripted(parse_error(), Answer(value=2))

        result = await invoke_structured(
            as_chat_model(model), Answer, QUESTION, attempts=2, label="t"
        )

        assert result == Answer(value=2)
        retry = model.calls[1]
        assert retry[:2] == [HumanMessage("q"), AIMessage('{"value": "x"}')]
        assert "bad json" in retry[2].content

    async def test_raises_structured_output_error_after_all_attempts(self):
        model = scripted(parse_error(), parse_error())

        with pytest.raises(StructuredOutputError):
            await invoke_structured(
                as_chat_model(model), Answer, QUESTION, attempts=2, label="t"
            )

        assert len(model.calls) == 2

    async def test_does_not_retry_other_errors(self):
        model = scripted(RuntimeError("down"))

        with pytest.raises(RuntimeError):
            await invoke_structured(as_chat_model(model), Answer, QUESTION, label="t")

        assert len(model.calls) == 1


def raw_reply(parsed: Answer | None, raw: str, error: Exception | None = None) -> dict:
    return {"parsed": parsed, "raw": AIMessage(raw), "parsing_error": error}


class TestInvokeStructuredWithRaw:
    async def test_returns_parsed_and_raw(self):
        model = scripted(raw_reply(Answer(value=3), "r"))

        result = await invoke_structured_with_raw(
            as_chat_model(model), Answer, QUESTION, label="t"
        )

        assert result == (Answer(value=3), AIMessage("r"))
        assert model.options["include_raw"] is True

    async def test_reprompts_with_raw_reply_when_parsing_failed(self):
        model = scripted(
            raw_reply(None, "oops", parse_error()),
            raw_reply(Answer(value=4), "ok"),
        )

        result = await invoke_structured_with_raw(
            as_chat_model(model), Answer, QUESTION, attempts=2, label="t"
        )

        assert result == (Answer(value=4), AIMessage("ok"))
        assert model.calls[1][-2] == AIMessage("oops")
        assert "bad json" in model.calls[1][-1].content

    async def test_reprompts_when_nothing_parsed_and_no_error(self):
        model = scripted(raw_reply(None, ""), raw_reply(Answer(value=5), "ok"))

        result = await invoke_structured_with_raw(
            as_chat_model(model), Answer, QUESTION, attempts=2, label="t"
        )

        assert result == (Answer(value=5), AIMessage("ok"))
        assert len(model.calls) == 2

    async def test_raises_non_parse_parsing_error_as_is(self):
        model = scripted(raw_reply(None, "no", RuntimeError("refused")))

        with pytest.raises(RuntimeError, match="refused"):
            await invoke_structured_with_raw(
                as_chat_model(model), Answer, QUESTION, label="t"
            )

        assert len(model.calls) == 1
