"""Contract tests for the router-backed chat model factory."""

import pytest
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from mcps.common.ai_models import create_chat_model, strip_thinking, tokens_for_chars
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
