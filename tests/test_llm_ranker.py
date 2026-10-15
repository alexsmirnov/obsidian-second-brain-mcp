"""Contract tests for the router-backed hybrid ranker (embeddings + chat)."""

from __future__ import annotations

import asyncio
import json
import logging
import math
from typing import Any

import httpx
import pytest

from mcps.research.tools.filtering import RelevanceFilter
from mcps.research.tools.filtering_models import RouterPassageModels
from mcps.research.tools.llm_ranker import LlmRanker
from mcps.research.tools.models import FetchResult, FetchStatus


def page(content: str, mime: str = "text/plain") -> FetchResult:
    return FetchResult(
        url="https://source.example/page",
        status=FetchStatus.OK,
        mime=mime,
        content=content,
    )


def by_paragraph(text: str) -> list[str]:
    return text.split("\n\n")


ROUTER_BASE = "https://router.example/v1"
ROUTER_KEY = "test-key"
FETCH_MODEL = "fetch-test"
EMBED_MODEL = "embed-test"
EMBED_DIM = 3

HYBRID_QUERY = "How does quantum routing improve performance?"
HYBRID_TEXT = (
    "QUANTUM_LEXICAL: quantum routing reduces latency.\n\n"
    "SEMANTIC_ONLY: entangled processors accelerate path selection.\n\n"
    "DISTRACTOR: bread flour needs water."
)


def small_limits(**overrides: Any) -> FilterLimits:
    values: dict[str, Any] = {
        "window_chars": 120,
        "overlap_chars": 20,
        "embedding_batch_size": 2,
        "embedding_request_chars": 2000,
        "semantic_candidates": 2,
        "llm_batch_size": 2,
        "llm_request_chars": 8000,
        "max_llm_batches": 20,
        "model_concurrency": 2,
    }
    values.update(overrides)
    return FilterLimits(**values)


def _vector_for(text: str) -> list[float]:
    if text == HYBRID_QUERY:
        return [1.0, 0.0, 0.0]
    if "QUANTUM_LEXICAL" in text:
        return [0.8, 0.6, 0.0]
    if "SEMANTIC_ONLY" in text:
        return [1.0, 0.0, 0.0]
    if "DISTRACTOR" in text:
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]


def _embedding_json(vectors: list[list[float]]) -> dict[str, Any]:
    return {
        "object": "list",
        "model": EMBED_MODEL,
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
        "data": [
            {"object": "embedding", "index": index, "embedding": vector}
            for index, vector in enumerate(vectors)
        ],
    }


def _chat_json(
    content: str,
    *,
    finish_reason: str = "stop",
    message: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {"role": "assistant", "content": content}
    if message is not None:
        payload = message
    return {
        "choices": [
            {"index": 0, "message": payload, "finish_reason": finish_reason}
        ]
    }


class RouterHarness:
    """External HTTP boundary fake for the router embeddings/chat endpoints."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.embedding_inputs: list[list[str]] = []
        self.embedding_bodies: list[dict[str, Any]] = []
        self.chat_payloads: list[dict[str, Any]] = []
        self.vector_fn = lambda texts: [_vector_for(text) for text in texts]
        self.pending_embedding_failures: list[Any] = []
        self.embedding_status: int | None = None
        self.embedding_retry_after: str | None = "0"
        self.embedding_invalid_json = False
        self.pending_chat_failures: list[Any] = []
        self.chat_body_fn: Any = None
        self.embedding_redirect = False
        self.chat_redirect = False
        self.client: httpx.AsyncClient | None = None

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path.endswith("/embedding-target"):
            return self._embedding_target(request)
        if path.endswith("/embeddings"):
            return self._embedding(request)
        if path.endswith("/chat-target"):
            return self._chat(request)
        if path.endswith("/chat/completions"):
            return self._chat(request)
        return httpx.Response(404)

    def _embedding_target(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        texts = list(body["input"])
        self.embedding_inputs.append(texts)
        return httpx.Response(200, json=_embedding_json(self.vector_fn(texts)))

    def _embedding(self, request: httpx.Request) -> httpx.Response:
        if self.embedding_redirect:
            return httpx.Response(
                307,
                headers={
                    "location": "https://router.example/v1/embedding-target"
                },
            )
        if self.pending_embedding_failures:
            item = self.pending_embedding_failures.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        if self.embedding_status is not None:
            headers = {}
            if self.embedding_retry_after is not None:
                headers["retry-after"] = self.embedding_retry_after
            return httpx.Response(
                self.embedding_status,
                headers=headers,
                json={"error": "failure"},
            )
        if self.embedding_invalid_json:
            return httpx.Response(
                200,
                content=b"not json",
                headers={"content-type": "application/json"},
            )
        body = json.loads(request.content)
        self.embedding_bodies.append(body)
        texts = list(body["input"])
        self.embedding_inputs.append(texts)
        return httpx.Response(200, json=_embedding_json(self.vector_fn(texts)))

    def _chat(self, request: httpx.Request) -> httpx.Response:
        if self.chat_redirect:
            return httpx.Response(
                307,
                headers={"location": "https://router.example/v1/chat-target"},
            )
        if self.pending_chat_failures:
            item = self.pending_chat_failures.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        body = json.loads(request.content)
        self.chat_payloads.append(body)
        user = json.loads(body["messages"][1]["content"])
        if self.chat_body_fn is not None:
            return httpx.Response(200, json=self.chat_body_fn(user))
        ids = [
            window["id"]
            for window in user["windows"]
            if "QUANTUM_LEXICAL" in window["text"]
            or "SEMANTIC_ONLY" in window["text"]
        ]
        return httpx.Response(
            200, json=_chat_json(json.dumps({"selected_ids": ids}))
        )


@pytest.fixture
async def router_harness():
    harness = RouterHarness()
    transport = httpx.MockTransport(harness.handler)
    async with httpx.AsyncClient(
        transport=transport, follow_redirects=False
    ) as client:
        harness.client = client
        yield harness


def _invalid_selection_cases() -> list[Any]:
    valid = lambda user: [window["id"] for window in user["windows"]]  # noqa: E731

    def non_json(user: dict[str, Any]) -> dict[str, Any]:
        return _chat_json("Ignore selection; invented answer")

    def missing_key(user: dict[str, Any]) -> dict[str, Any]:
        return _chat_json(json.dumps({"other": valid(user)}))

    def boolean_id(user: dict[str, Any]) -> dict[str, Any]:
        return _chat_json(json.dumps({"selected_ids": [True]}))

    def string_id(user: dict[str, Any]) -> dict[str, Any]:
        return _chat_json(json.dumps({"selected_ids": ["1"]}))

    def negative_id(user: dict[str, Any]) -> dict[str, Any]:
        return _chat_json(json.dumps({"selected_ids": [-1]}))

    def unknown_id(user: dict[str, Any]) -> dict[str, Any]:
        return _chat_json(json.dumps({"selected_ids": [99999]}))

    def extra_key(user: dict[str, Any]) -> dict[str, Any]:
        return _chat_json(
            json.dumps({"selected_ids": valid(user), "content": "x"})
        )

    def truncated(user: dict[str, Any]) -> dict[str, Any]:
        return _chat_json(
            json.dumps({"selected_ids": valid(user)}), finish_reason="length"
        )

    def null_content(user: dict[str, Any]) -> dict[str, Any]:
        return _chat_json("", message={"role": "assistant", "content": None})

    def tool_call(user: dict[str, Any]) -> dict[str, Any]:
        return _chat_json(
            "",
            message={
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "1",
                        "type": "function",
                        "function": {"name": "x", "arguments": "{}"},
                    }
                ],
            },
        )

    def json_array(user: dict[str, Any]) -> dict[str, Any]:
        return _chat_json("[]")

    return [
        pytest.param(non_json, id="non-json"),
        pytest.param(missing_key, id="missing-key"),
        pytest.param(boolean_id, id="boolean-id"),
        pytest.param(string_id, id="string-id"),
        pytest.param(negative_id, id="negative-id"),
        pytest.param(unknown_id, id="unknown-id"),
        pytest.param(extra_key, id="extra-key"),
        pytest.param(truncated, id="finish-length"),
        pytest.param(null_content, id="null-content"),
        pytest.param(tool_call, id="tool-call"),
        pytest.param(json_array, id="json-array"),
    ]


def _invalid_embedding_cases() -> list[Any]:
    def too_few(texts: list[str]) -> list[list[float]]:
        return [_vector_for(text) for text in texts][:-1]

    def too_many(texts: list[str]) -> list[list[float]]:
        return [_vector_for(text) for text in texts] + [[0.0, 0.0, 1.0]]

    def empty_vector(texts: list[str]) -> list[list[float]]:
        return [[] for _ in texts]

    def string_coordinate(texts: list[str]) -> list[Any]:
        return [["a", 0.0, 0.0] for _ in texts]

    def nonfinite(texts: list[str]) -> list[list[float]]:
        return [[float("nan"), 0.0, 0.0] for _ in texts]

    def zero_norm(texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.0, 0.0] for _ in texts]

    def inconsistent_width(texts: list[str]) -> list[list[float]]:
        return [
            [1.0, 0.0] if "SEMANTIC_ONLY" in text else _vector_for(text)
            for text in texts
        ]

    def wrong_width(texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    return [
        pytest.param(too_few, id="too-few"),
        pytest.param(too_many, id="too-many"),
        pytest.param(empty_vector, id="empty-vector"),
        pytest.param(string_coordinate, id="string-coordinate"),
        pytest.param(nonfinite, id="nonfinite"),
        pytest.param(zero_norm, id="zero-norm"),
        pytest.param(inconsistent_width, id="inconsistent-width"),
        pytest.param(wrong_width, id="wrong-configured-width"),
    ]


class GatedTransport(httpx.AsyncBaseTransport):
    """HTTP boundary that can block requests and record peak concurrency."""

    def __init__(self, handler: Any, *, gate: bool = True) -> None:
        self._handler = handler
        self.gate = gate
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.active = 0
        self.max_active = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.entered.set()
        try:
            if self.gate:
                await self.release.wait()
            return self._handler(request)
        finally:
            self.active -= 1


def make_filter(
    harness: RouterHarness,
    *,
    concurrency: int = 2,
    **ranker_options: Any,
) -> RelevanceFilter:
    models = RouterPassageModels(
        ranker_options.pop("http_client", harness.client),
        router_url=ranker_options.pop("router_url", ROUTER_BASE),
        router_key=ranker_options.pop("router_key", ROUTER_KEY),
        fetch_model=ranker_options.pop("fetch_model", FETCH_MODEL),
        embedding_model=EMBED_MODEL,
        embedding_dimensions=ranker_options.pop("embedding_dimensions", EMBED_DIM),
        concurrency=concurrency,
    )
    options: dict[str, Any] = {
        "embedding_batch_size": 2,
        "embedding_request_chars": 2000,
        "semantic_candidates": 2,
        "llm_batch_size": 2,
        "llm_request_chars": 8000,
    }
    options.update(ranker_options)
    return RelevanceFilter(by_paragraph, LlmRanker(models, **options))


QUANTUM = "QUANTUM_LEXICAL: quantum routing reduces latency."
SEMANTIC = "SEMANTIC_ONLY: entangled processors accelerate path selection."


async def test_ranker_hybrid_keeps_semantic_only_fragment_and_excludes_distractor(
    router_harness: RouterHarness,
):
    result = await make_filter(router_harness)(page(HYBRID_TEXT), HYBRID_QUERY)

    assert result.status is FetchStatus.OK
    assert result.content == f"{QUANTUM}\n\n{SEMANTIC}"
    assert router_harness.embedding_bodies[0]["model"] == EMBED_MODEL
    assert router_harness.embedding_bodies[0].get("dimensions") == EMBED_DIM
    assert router_harness.chat_payloads[0]["model"] == FETCH_MODEL
    transmitted = [t for batch in router_harness.embedding_inputs for t in batch]
    assert HYBRID_QUERY in transmitted


async def test_ranker_empty_selection_returns_ok_empty(
    router_harness: RouterHarness,
):
    router_harness.chat_body_fn = lambda user: _chat_json(
        json.dumps({"selected_ids": []})
    )

    result = await make_filter(router_harness)(page(HYBRID_TEXT), HYBRID_QUERY)

    assert result.status is FetchStatus.OK
    assert result.content == ""


@pytest.mark.parametrize("body_fn", _invalid_selection_cases())
async def test_ranker_invalid_selection_returns_filter_failed(
    router_harness: RouterHarness, body_fn: Any
):
    router_harness.chat_body_fn = body_fn

    result = await make_filter(router_harness)(page(HYBRID_TEXT), HYBRID_QUERY)

    assert result.status is FetchStatus.FILTER_FAILED
    assert result.content == ""


@pytest.mark.parametrize("vector_fn", _invalid_embedding_cases())
async def test_ranker_invalid_embeddings_returns_filter_failed(
    router_harness: RouterHarness, vector_fn: Any
):
    router_harness.vector_fn = vector_fn

    result = await make_filter(router_harness)(page(HYBRID_TEXT), HYBRID_QUERY)

    assert result.status is FetchStatus.FILTER_FAILED
    assert router_harness.chat_payloads == []


async def test_ranker_inferred_dimension_accepts_consistent_width(
    router_harness: RouterHarness,
):
    router_harness.vector_fn = lambda texts: [
        [*_vector_for(text), 0.5] for text in texts
    ]

    result = await make_filter(router_harness, embedding_dimensions=0)(
        page(HYBRID_TEXT), HYBRID_QUERY
    )

    assert result.status is FetchStatus.OK
    assert "SEMANTIC_ONLY" in result.content


async def test_ranker_inconsistent_inferred_width_returns_filter_failed(
    router_harness: RouterHarness,
):
    router_harness.vector_fn = lambda texts: [
        [1.0, 0.0, 0.0, 0.0] if text == HYBRID_QUERY else [1.0, 0.0]
        for text in texts
    ]

    result = await make_filter(router_harness, embedding_dimensions=0)(
        page(HYBRID_TEXT), HYBRID_QUERY
    )

    assert result.status is FetchStatus.FILTER_FAILED


@pytest.mark.parametrize("status", [429, 500])
async def test_ranker_embedding_http_failure_returns_filter_failed(
    router_harness: RouterHarness, status: int
):
    router_harness.embedding_status = status

    result = await make_filter(router_harness)(page(HYBRID_TEXT), HYBRID_QUERY)

    assert result.status is FetchStatus.FILTER_FAILED
    assert router_harness.chat_payloads == []


async def test_ranker_embedding_malformed_json_returns_filter_failed(
    router_harness: RouterHarness,
):
    router_harness.embedding_invalid_json = True

    result = await make_filter(router_harness)(page(HYBRID_TEXT), HYBRID_QUERY)

    assert result.status is FetchStatus.FILTER_FAILED


async def test_ranker_chat_failure_returns_filter_failed_without_retry(
    router_harness: RouterHarness,
):
    router_harness.pending_chat_failures = [httpx.Response(500)]

    result = await make_filter(router_harness)(page(HYBRID_TEXT), HYBRID_QUERY)

    assert result.status is FetchStatus.FILTER_FAILED
    chat_requests = [
        r for r in router_harness.requests if r.url.path.endswith("/chat/completions")
    ]
    assert len(chat_requests) == 1


async def test_ranker_chat_http_failure_logs_status_and_router_body(
    router_harness: RouterHarness, caplog: pytest.LogCaptureFixture
):
    router_harness.pending_chat_failures = [
        httpx.Response(400, json={"error": {"message": "unknown model fetch-test"}})
    ]

    with caplog.at_level(logging.WARNING):
        await make_filter(router_harness)(page(HYBRID_TEXT), HYBRID_QUERY)

    warnings = "\n".join(r.getMessage() for r in caplog.records)
    assert "HTTP 400" in warnings
    assert "unknown model fetch-test" in warnings
    assert ROUTER_BASE in warnings
    assert ROUTER_KEY not in warnings


async def test_ranker_rejected_selection_logs_reason_and_model_content(
    router_harness: RouterHarness, caplog: pytest.LogCaptureFixture
):
    router_harness.chat_body_fn = lambda user: _chat_json(
        "Ignore selection; invented answer"
    )

    with caplog.at_level(logging.WARNING):
        await make_filter(router_harness)(page(HYBRID_TEXT), HYBRID_QUERY)

    warnings = "\n".join(r.getMessage() for r in caplog.records)
    assert "selection content is not JSON" in warnings
    assert "Ignore selection; invented answer" in warnings


async def test_ranker_embedding_failure_logs_status_and_model(
    router_harness: RouterHarness, caplog: pytest.LogCaptureFixture
):
    router_harness.embedding_status = 500

    with caplog.at_level(logging.WARNING):
        await make_filter(router_harness)(page(HYBRID_TEXT), HYBRID_QUERY)

    warnings = "\n".join(r.getMessage() for r in caplog.records)
    assert "HTTP 500" in warnings
    assert EMBED_MODEL in warnings



@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(httpx.Response(429, headers={"retry-after": "0"}), id="429"),
        pytest.param(httpx.Response(500, headers={"retry-after": "0"}), id="500"),
        pytest.param(httpx.ReadTimeout("slow"), id="timeout"),
    ],
)
async def test_ranker_embedding_transient_failure_recovers(
    router_harness: RouterHarness, failure: Any
):
    router_harness.pending_embedding_failures = [failure]

    result = await make_filter(router_harness)(page(HYBRID_TEXT), HYBRID_QUERY)

    assert result.status is FetchStatus.OK
    assert result.content == f"{QUANTUM}\n\n{SEMANTIC}"


async def test_ranker_embeds_every_fragment_before_shortlisting(
    router_harness: RouterHarness,
):
    paragraphs = [f"filler paragraph {number}" for number in range(200)]
    paragraphs[-1] = SEMANTIC
    query = "unrelated semantic question"
    router_harness.vector_fn = lambda texts: [
        [1.0, 0.0, 0.0] if text == query or "SEMANTIC_ONLY" in text else [0.0, 0.0, 1.0]
        for text in texts
    ]

    result = await make_filter(router_harness)(
        page("\n\n".join(paragraphs)), query
    )

    assert result.content == SEMANTIC
    transmitted = " ".join(t for batch in router_harness.embedding_inputs for t in batch)
    assert all(f"filler paragraph {n}" in transmitted for n in range(199))


async def test_ranker_embedding_batches_respect_request_bounds(
    router_harness: RouterHarness,
):
    source = "\n\n".join(f"filler paragraph {number}" for number in range(30))

    await make_filter(router_harness)(page(source), "quantum")

    for batch in router_harness.embedding_inputs:
        assert len(batch) <= 2
        assert sum(len(text.encode("utf-8")) for text in batch) <= 2000


async def test_ranker_selection_batch_budget_exhaustion_returns_filter_failed(
    router_harness: RouterHarness,
):
    result = await make_filter(router_harness, max_llm_batches=0)(
        page(HYBRID_TEXT), HYBRID_QUERY
    )

    assert result.status is FetchStatus.FILTER_FAILED


async def test_ranker_model_deadline_returns_filter_failed():
    harness = RouterHarness()
    transport = GatedTransport(harness.handler)
    async with httpx.AsyncClient(transport=transport) as client:
        harness.client = client
        page_filter = make_filter(harness, timeout_seconds=0.05)

        result = await asyncio.wait_for(
            page_filter(page(HYBRID_TEXT), HYBRID_QUERY), timeout=2
        )

    assert result.status is FetchStatus.FILTER_FAILED
    assert transport.active == 0


async def test_ranker_model_deadline_logs_stalled_stage(
    caplog: pytest.LogCaptureFixture,
):
    harness = RouterHarness()
    transport = GatedTransport(harness.handler)
    async with httpx.AsyncClient(transport=transport) as client:
        harness.client = client
        page_filter = make_filter(harness, timeout_seconds=0.05)

        with caplog.at_level(logging.WARNING):
            await asyncio.wait_for(
                page_filter(page(HYBRID_TEXT), HYBRID_QUERY), timeout=2
            )

    warnings = "\n".join(r.getMessage() for r in caplog.records)
    assert "model deadline during embedding" in warnings


async def test_ranker_cancellation_releases_model_work_without_closing_client():
    harness = RouterHarness()
    transport = GatedTransport(harness.handler)
    async with httpx.AsyncClient(transport=transport) as client:
        harness.client = client
        task = asyncio.create_task(
            make_filter(harness)(page(HYBRID_TEXT), HYBRID_QUERY)
        )
        await asyncio.wait_for(transport.entered.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        transport.gate = False
        transport.release.set()
        response = await client.get("https://router.example/v1/models")

        assert response.status_code == 404
        assert client.is_closed is False
        assert transport.active == 0


async def test_ranker_overlapping_calls_respect_shared_model_concurrency():
    harness = RouterHarness()
    transport = GatedTransport(harness.handler)
    async with httpx.AsyncClient(transport=transport) as client:
        harness.client = client
        page_filter = make_filter(harness, concurrency=2)
        tasks = [
            asyncio.create_task(page_filter(page(HYBRID_TEXT), HYBRID_QUERY))
            for _ in range(4)
        ]
        await asyncio.wait_for(transport.entered.wait(), timeout=2)
        await asyncio.sleep(0)
        transport.release.set()
        transport.gate = False
        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)

    assert transport.max_active <= 2
    assert all(result.status is FetchStatus.OK for result in results)


@pytest.mark.parametrize(
    ("configured", "sent"),
    [("openai/fetch-test", "fetch-test"), ("custom/fetch-test", "custom/fetch-test")],
)
async def test_ranker_strips_only_the_openai_model_prefix(
    router_harness: RouterHarness, configured: str, sent: str
):
    result = await make_filter(router_harness, fetch_model=configured)(
        page(HYBRID_TEXT), HYBRID_QUERY
    )

    assert result.status is FetchStatus.OK
    assert router_harness.chat_payloads[0]["model"] == sent


async def test_ranker_duplicate_selected_ids_deduplicate(
    router_harness: RouterHarness,
):
    def body(user: dict[str, Any]) -> dict[str, Any]:
        ids = [w["id"] for w in user["windows"] if "SEMANTIC_ONLY" in w["text"]]
        return _chat_json(json.dumps({"selected_ids": ids + ids}))

    router_harness.chat_body_fn = body

    result = await make_filter(router_harness)(page(HYBRID_TEXT), HYBRID_QUERY)

    assert result.content == SEMANTIC


async def test_ranker_empty_router_key_does_not_use_environment_key(monkeypatch):
    # An empty router key is a configuration failure, never an unauthenticated
    # request, and must not fall back to OPENAI_API_KEY.
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-used")
    harness = RouterHarness()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(harness.handler)
    ) as client:
        harness.client = client
        result = await make_filter(harness, router_key="")(
            page(HYBRID_TEXT), HYBRID_QUERY
        )

    assert result.status is FetchStatus.FILTER_FAILED
    assert all(
        "must-not-be-used" not in r.headers.get("authorization", "")
        for r in harness.requests
    )


async def test_ranker_embedding_redirect_follows_borrowed_client():
    harness = RouterHarness()
    harness.embedding_redirect = True
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(harness.handler), follow_redirects=True
    ) as client:
        harness.client = client
        result = await make_filter(harness)(page(HYBRID_TEXT), HYBRID_QUERY)

    assert result.status is FetchStatus.OK
    assert any(r.url.path.endswith("/embedding-target") for r in harness.requests)


async def test_ranker_chat_redirect_returns_filter_failed():
    harness = RouterHarness()
    harness.chat_redirect = True
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(harness.handler), follow_redirects=False
    ) as client:
        harness.client = client
        result = await make_filter(harness)(page(HYBRID_TEXT), HYBRID_QUERY)

    assert result.status is FetchStatus.FILTER_FAILED
    assert not any(r.url.path.endswith("/chat-target") for r in harness.requests)
