"""Unit tests for the DRACO criterion judge's structured call."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from evaluation.dataset import RubricCriterion
from evaluation.judge import CriterionJudgeResult, judge_criterion
from evaluation.scoring import CriterionVerdict
from langchain_core.exceptions import OutputParserException

_CRITERION = RubricCriterion(id="c1", weight=1, requirement="mentions X")


class _JudgeModel:
    """Fake chat model replaying scripted structured-output outcomes."""

    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[list[Any]] = []

    def with_structured_output(self, schema: Any, **_kwargs: Any) -> Any:
        async def ainvoke(messages: list[Any]) -> Any:
            self.calls.append(messages)
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        return SimpleNamespace(ainvoke=ainvoke)


def _bad_json() -> Exception:
    return OutputParserException("Invalid JSON")


async def _judge(model: _JudgeModel) -> CriterionVerdict:
    return await judge_criterion("query", "response", _CRITERION, model)  # type: ignore[arg-type]


async def test_valid_reply_gives_verdict_in_one_call():
    model = _JudgeModel([CriterionJudgeResult(met=True, justification="ok")])

    verdict = await _judge(model)

    assert verdict == CriterionVerdict("c1", True, "ok")
    assert len(model.calls) == 1


async def test_parse_error_is_retried_with_the_error_in_the_prompt():
    model = _JudgeModel(
        [_bad_json(), CriterionJudgeResult(met=False, justification="no X")]
    )

    verdict = await _judge(model)

    assert verdict == CriterionVerdict("c1", False, "no X")
    first, second = model.calls
    assert second[: len(first)] == first
    assert "Invalid JSON" in str(second[-1].content)


async def test_exhausted_retries_give_unmet_judge_error_verdict():
    model = _JudgeModel([_bad_json(), _bad_json(), _bad_json()])

    verdict = await _judge(model)

    assert verdict.criterion_id == "c1"
    assert verdict.met is False
    assert verdict.justification.startswith("Judge error")
    assert len(model.calls) == 3


async def test_non_parse_error_is_not_retried():
    model = _JudgeModel([RuntimeError("router down")])

    verdict = await _judge(model)

    assert verdict.met is False
    assert "router down" in verdict.justification
    assert len(model.calls) == 1


@pytest.mark.parametrize("met", [True, False])
async def test_verdict_keeps_judged_value(met: bool):
    model = _JudgeModel([CriterionJudgeResult(met=met, justification="j")])

    assert (await _judge(model)).met is met
