"""Contract tests for DecisionRanker over a real DecisionClient and mock router."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Coroutine
from typing import Any

import httpx
import pytest

from mcps.common.decisions import DecisionClient, DecisionError
from mcps.research.tools.decision_ranker import DecisionRanker
from mcps.research.tools.filtering import FilterModelError

ROUTER = "http://router/v1"

Handler = Callable[[httpx.Request], Coroutine[Any, Any, httpx.Response]]


def noul_reply(value: float) -> httpx.Response:
    return httpx.Response(
        200, json={"answers": {"relevant": {"type": "noul", "noul": value}}}
    )


def passage_of(request: httpx.Request) -> str:
    return json.loads(request.content)["state"]["passage"]


def make_ranker(handler: Handler, **options: Any) -> DecisionRanker:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = DecisionClient(http, router_url=ROUTER, router_key="k", model="jev")
    return DecisionRanker(client, **options)


def scripted(
    nouls: dict[str, float],
    delay: dict[str, float] | None = None,
    requests: list[httpx.Request] | None = None,
) -> Handler:
    async def handler(request: httpx.Request) -> httpx.Response:
        if requests is not None:
            requests.append(request)
        passage = passage_of(request)
        await asyncio.sleep((delay or {}).get(passage, 0))
        return noul_reply(nouls[passage])

    return handler


async def test_each_fragment_is_sent_with_the_query_as_a_noul_question():
    requests: list[httpx.Request] = []
    ranker = make_ranker(scripted({"alpha": 0.9}, requests=requests))

    await ranker(["alpha"], "q")

    body = json.loads(requests[0].content)
    assert body["state"] == {"query": "q", "passage": "alpha"}
    assert body["questions"]["relevant"]["type"] == "noul"
    assert body["model"] == "jev"


async def test_keeps_fragments_at_or_above_the_threshold():
    nouls = {"a": 0.59, "b": 0.6, "c": 0.9, "d": 0.1}

    assert await make_ranker(scripted(nouls))(["a", "b", "c", "d"], "q") == [1, 2]


async def test_returns_document_order_when_responses_arrive_out_of_order():
    nouls = {"a": 0.9, "b": 0.9, "c": 0.9}
    ranker = make_ranker(scripted(nouls, delay={"a": 0.03, "b": 0.02}))

    assert await ranker(["a", "b", "c"], "q") == [0, 1, 2]


async def test_threshold_is_configurable():
    ranker = make_ranker(scripted({"a": 0.9, "b": 0.97}), threshold=0.95)

    assert await ranker(["a", "b"], "q") == [1]


async def test_in_flight_requests_never_exceed_the_concurrency_limit():
    in_flight = peak = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return noul_reply(0.9)

    fragments = [f"f{i}" for i in range(6)]

    result = await make_ranker(handler, concurrency=2)(fragments, "q")

    assert result == list(range(6))
    assert peak == 2


async def test_empty_fragments_make_no_request():
    requests: list[httpx.Request] = []

    assert await make_ranker(scripted({}, requests=requests))([], "q") == []
    assert requests == []


async def test_decision_failure_becomes_filter_model_error():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    with pytest.raises(FilterModelError) as raised:
        await make_ranker(handler)(["a", "b"], "q")

    assert isinstance(raised.value.__cause__, DecisionError)


async def test_first_failure_cancels_the_pending_requests():
    async def handler(request: httpx.Request) -> httpx.Response:
        if passage_of(request) == "a":
            return httpx.Response(500)
        await asyncio.sleep(1)
        return noul_reply(0.9)

    started = time.monotonic()
    with pytest.raises(FilterModelError):
        await make_ranker(handler)(["a", "b"], "q")

    assert time.monotonic() - started < 0.5


@pytest.mark.parametrize(
    "options", [{"threshold": 1.5}, {"threshold": -0.1}, {"concurrency": 0}]
)
def test_invalid_options_are_rejected(options: dict[str, Any]):
    async def handler(request: httpx.Request) -> httpx.Response:
        return noul_reply(0.9)

    with pytest.raises(ValueError):
        make_ranker(handler, **options)
