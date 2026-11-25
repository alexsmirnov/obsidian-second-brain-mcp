"""Router-backed chat model factory shared by research and vault tools.

The model name picks the client:

* ``gemini-*`` uses the Google GenAI client against the router.
* ``gpt-*`` uses OpenAI Chat Completions. Not the Responses API: the router's
  ChatGPT provider rejects system messages there and silently falls back to a
  local model.
* anything else uses OpenAI-compatible Chat Completions (local or other models).

``reasoning_effort``, ``context_window`` and ``temperature`` are converted to
each API's own settings; a setting the API cannot take is dropped with a debug
log instead of failing the request.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Literal

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from mcps.config import RouterSettings

__all__ = [
    "ReasoningEffort",
    "create_chat_model",
    "strip_thinking",
    "tokens_for_chars",
]

logger = logging.getLogger(__name__)

type ReasoningEffort = Literal["none", "minimal", "low", "medium", "high"]

# Gemini 2.5 thinking budgets in tokens. Flash models accept 0 (thinking off);
# Pro models cannot turn thinking off and accept 128 tokens at least.
GEMINI_THINKING_BUDGETS: dict[ReasoningEffort, int] = {
    "none": 0,
    "minimal": 512,
    "low": 1024,
    "medium": 8192,
    "high": 24576,
}
GEMINI_PRO_MIN_THINKING_BUDGET = 128

# Conservative characters per token: English prose runs near 4, markdown with
# links and code nearer 3. Overestimating tokens is safer: Ollama silently drops
# the start of a prompt (the system message first) that exceeds ``num_ctx``.
CHARS_PER_TOKEN = 3.0

# Reasoning some local models (Gemma, Qwen, DeepSeek) write into the reply.
# A block left open was cut by the output limit and runs to the end.
_THINKING_BLOCKS = re.compile(
    r"<\|channel>thought[\s\S]*?(?:<channel\|>|$)|<think>[\s\S]*?(?:</think>|$)"
)

_GEMINI_VERSION = re.compile(r"^gemini-(\d+)(?:\.(\d+))?")
_GPT_MAJOR_VERSION = re.compile(r"^gpt-(\d+)")


def create_chat_model(
    model_name: str,
    router: RouterSettings,
    http_client: httpx.AsyncClient | None = None,
    *,
    reasoning_effort: ReasoningEffort | None = None,
    context_window: int | None = None,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Instantiate a router-backed chat model.

    Args:
        model_name: Router model name; its prefix selects the client.
        router: Router URL and key.
        http_client: Shared async client for the OpenAI clients.
        reasoning_effort: Thinking depth; ``None`` keeps the model default.
        context_window: Context size in tokens; only Chat Completions models
            take it (sent as Ollama ``num_ctx``).
        temperature: Sampling temperature; ``None`` keeps the model default.
        max_output_tokens: Output limit in tokens, thinking tokens included;
            Ollama models get it as ``num_predict``.
        **kwargs: Passed unchanged to the client constructor.
    """
    if model_name.startswith("gemini-"):
        _ignore("context_window", context_window, model_name)
        return ChatGoogleGenerativeAI(
            model=model_name,
            base_url=router.api_base,
            google_api_key=SecretStr(router.api_key),
            **gemini_options(model_name, reasoning_effort, temperature),
            **_set("max_output_tokens", max_output_tokens),
            **kwargs,
        )
    if model_name.startswith("gpt-"):
        _ignore("context_window", context_window, model_name)
        return ChatOpenAI(
            model=model_name,
            base_url=router.api_base,
            api_key=SecretStr(router.api_key),
            http_async_client=http_client,
            use_responses_api=False,
            **gpt_options(model_name, reasoning_effort, temperature),
            **_set("max_tokens", max_output_tokens),
            **kwargs,
        )
    extra_body = kwargs.pop("extra_body", None)
    return ChatOpenAI(
        model=model_name,
        base_url=router.api_base,
        api_key=SecretStr(router.api_key),
        http_async_client=http_client,
        use_responses_api=False,
        **completions_options(
            reasoning_effort, context_window, temperature, extra_body
        ),
        **_set("max_tokens", max_output_tokens),
        **kwargs,
    )


def gemini_options(
    model_name: str,
    reasoning_effort: ReasoningEffort | None,
    temperature: float | None,
) -> dict[str, Any]:
    """Gemini 3+ takes a thinking level, Gemini 2.5 a token budget.

    Gemini 3 is tuned for its default temperature (1.0); Google warns that
    lower values can cause looping, so temperature is dropped for it.
    """
    major, minor = _gemini_version(model_name)
    options: dict[str, Any] = {}
    if major >= 3:
        _ignore("temperature", temperature, model_name)
        if reasoning_effort is not None:
            # Gemini 3 cannot turn thinking off; "minimal" is the lowest level.
            level = "minimal" if reasoning_effort == "none" else reasoning_effort
            options["thinking_level"] = level
        return options
    if temperature is not None:
        options["temperature"] = temperature
    if (major, minor) == (2, 5) and reasoning_effort is not None:
        budget = GEMINI_THINKING_BUDGETS[reasoning_effort]
        if "-pro" in model_name:
            budget = max(budget, GEMINI_PRO_MIN_THINKING_BUDGET)
        options["thinking_budget"] = budget
    else:
        _ignore("reasoning_effort", reasoning_effort, model_name)
    return options


def gpt_options(
    model_name: str,
    reasoning_effort: ReasoningEffort | None,
    temperature: float | None,
) -> dict[str, Any]:
    """GPT-5+ takes top-level ``reasoning_effort``; older GPT models do not reason.

    LangChain drops temperature for GPT-5 reasoning models unless the effort
    is ``"none"``, since the API rejects it there.
    """
    options: dict[str, Any] = {}
    if temperature is not None:
        options["temperature"] = temperature
    match = _GPT_MAJOR_VERSION.match(model_name)
    if match and int(match.group(1)) >= 5 and reasoning_effort is not None:
        options["reasoning_effort"] = reasoning_effort
    else:
        _ignore("reasoning_effort", reasoning_effort, model_name)
    return options


def completions_options(
    reasoning_effort: ReasoningEffort | None,
    context_window: int | None,
    temperature: float | None,
    extra_body: dict[str, Any] | None,
) -> dict[str, Any]:
    """Chat Completions takes top-level ``reasoning_effort``.

    The router forwards ``num_ctx`` in the request body to Ollama models.
    """
    options: dict[str, Any] = {}
    if temperature is not None:
        options["temperature"] = temperature
    if reasoning_effort is not None:
        options["reasoning_effort"] = reasoning_effort
    body = dict(extra_body or {})
    if context_window is not None:
        body["num_ctx"] = context_window
    if body:
        options["extra_body"] = body
    return options


def strip_thinking(text: str) -> str:
    """Remove reasoning blocks a model wrote into its reply."""
    return _THINKING_BLOCKS.sub("", text).strip()


def tokens_for_chars(chars: int) -> int:
    """Token count to reserve for ``chars`` characters of text."""
    return math.ceil(chars / CHARS_PER_TOKEN)


def _set(name: str, value: object) -> dict[str, Any]:
    return {} if value is None else {name: value}


def _gemini_version(model_name: str) -> tuple[int, int]:
    match = _GEMINI_VERSION.match(model_name)
    if match is None:
        return 0, 0
    return int(match.group(1)), int(match.group(2) or 0)


def _ignore(setting: str, value: object, model_name: str) -> None:
    if value is not None:
        logger.debug(
            "%s=%r is not supported by %s; ignored", setting, value, model_name
        )
