"""Contract tests for the router's TypeSafe System One decision client."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from mcps.common.decisions import (
    DecisionClient,
    DecisionError,
    Noul,
    decision_endpoint,
)

ROUTER = "http://router/v1"
RELEVANT = {"relevant": Noul(instructions="Is it relevant?")}
NOUL_BODY = {"answers": {"relevant": {"type": "noul", "noul": 0.2}}}


def make_client(
    handler: Callable[[httpx.Request], httpx.Response], *, key: str = "k"
) -> DecisionClient:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return DecisionClient(http, router_url=ROUTER, router_key=key, model="jev-latest")


def replying(
    body: Any = NOUL_BODY, status: int = 200
) -> tuple[list[httpx.Request], Callable[[httpx.Request], httpx.Response]]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if isinstance(body, bytes):
            return httpx.Response(status, content=body)
        return httpx.Response(status, json=body)

    return requests, handler


@pytest.mark.parametrize(
    ("router_url", "expected"),
    [
        ("http://h/v1", "http://h/v1/systemone"),
        ("http://h/v1/", "http://h/v1/systemone"),
        ("http://h:4000", "http://h:4000/v1/systemone"),
        ("http://h:4000/", "http://h:4000/v1/systemone"),
    ],
)
def test_decision_endpoint_strips_the_router_version_suffix(
    router_url: str, expected: str
):
    assert decision_endpoint(router_url) == expected


async def test_decide_posts_the_system_one_body_without_null_fields():
    requests, handler = replying()

    await make_client(handler).decide({"q": "x"}, RELEVANT)

    (request,) = requests
    assert request.method == "POST"
    assert request.url.path == "/v1/systemone"
    assert request.headers["authorization"] == "Bearer k"
    assert json.loads(request.content) == {
        "model": "jev-latest",
        "state": {"q": "x"},
        "questions": {
            "relevant": {"type": "noul", "instructions": "Is it relevant?"}
        },
    }


async def test_decide_omits_authorization_for_an_empty_key():
    requests, handler = replying()

    await make_client(handler, key="").decide("x", RELEVANT)

    assert "authorization" not in requests[0].headers


async def test_decide_returns_typed_noul_and_model():
    _, handler = replying(
        {
            "model": "jev-1.13.0",
            "answers": {"relevant": {"type": "noul", "noul": 0.73}},
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )

    response = await make_client(handler).decide("x", RELEVANT)

    assert response.noul("relevant") == 0.73
    assert response.model == "jev-1.13.0"


async def test_decide_accepts_a_response_without_model_and_usage():
    _, handler = replying()

    response = await make_client(handler).decide("x", RELEVANT)

    assert response.noul("relevant") == 0.2
    assert response.model is None


async def test_noul_rejects_an_unknown_question_id():
    _, handler = replying()
    response = await make_client(handler).decide("x", RELEVANT)

    with pytest.raises(DecisionError):
        response.noul("missing")


@pytest.mark.parametrize(
    ("body", "status"),
    [
        pytest.param(NOUL_BODY, 500, id="server-error"),
        pytest.param(NOUL_BODY, 429, id="rate-limited"),
        pytest.param(b"not json", 200, id="invalid-json"),
        pytest.param({}, 200, id="no-answers"),
        pytest.param(
            {"answers": {"relevant": {"type": "score", "score": 1.0}}},
            200,
            id="unsupported-answer-type",
        ),
        pytest.param(
            {"answers": {"relevant": {"type": "noul", "noul": 1.5}}},
            200,
            id="probability-out-of-range",
        ),
    ],
)
async def test_decide_wraps_bad_responses_in_decision_error(body: Any, status: int):
    _, handler = replying(body, status)

    with pytest.raises(DecisionError):
        await make_client(handler).decide("x", RELEVANT)


async def test_decide_wraps_transport_errors_in_decision_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    with pytest.raises(DecisionError):
        await make_client(handler).decide("x", RELEVANT)


async def test_decide_failure_logs_status_but_not_body_or_key(caplog):
    caplog.set_level(logging.WARNING)
    _, handler = replying("secret-body", 401)

    with pytest.raises(DecisionError):
        await make_client(handler, key="k-secret").decide("x", RELEVANT)

    messages = [record.getMessage() for record in caplog.records]
    assert any("401" in message for message in messages)
    assert not any("secret-body" in m or "k-secret" in m for m in messages)


@pytest.mark.parametrize("count", [0, 65])
async def test_decide_rejects_question_counts_the_api_does_not_accept(count: int):
    requests, handler = replying()
    questions = {f"q{i}": Noul(instructions="?") for i in range(count)}

    with pytest.raises(ValueError):
        await make_client(handler).decide("x", questions)

    assert requests == []
