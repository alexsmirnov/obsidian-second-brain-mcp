"""LLM-as-judge for DRACO criterion-level evaluation."""

from __future__ import annotations

import logging
import os

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from mcps.config import ServerConfig
from mcps.research.config import _create_chat_model

from .dataset import RubricCriterion
from .scoring import CriterionVerdict

logger = logging.getLogger(__name__)


def create_judge_model(
    config: ServerConfig, *, http_client: httpx.AsyncClient
) -> BaseChatModel:
    """Build the judge model: RESEARCH_EVAL_MODEL, else RESEARCH_INFER_MODEL."""
    return _create_chat_model(
        model_name=os.getenv("RESEARCH_EVAL_MODEL") or config.research_infer_model,
        router_url=config.router_api_base,
        router_key=config.router_api_key,
        http_client=http_client,
    )

CRITERION_JUDGE_SYSTEM = """\
You are an evaluation judge. Given a research query and a system's response, \
determine whether the following criterion is MET or UNMET.

Rules:
- For positive-weight criteria: MET means the response satisfies \
the requirement.
- For negative-weight criteria: the requirement describes an error. \
MET means the error IS present in the response.
- Base your verdict strictly on what is written in the response.
- Be concise in justification (1-2 sentences).
"""

CRITERION_JUDGE_TEMPLATE = """\
<query>
{problem}
</query>

<response>
{response}
</response>

<criterion>
{requirement}
</criterion>
"""


class CriterionJudgeResult(BaseModel):
    """Structured output from criterion judge."""

    met: bool = Field(
        ..., description="True if criterion is MET, false if UNMET"
    )
    justification: str = Field(
        ..., description="Brief explanation of the verdict (1-2 sentences)"
    )


async def judge_criterion(
    problem: str,
    response: str,
    criterion: RubricCriterion,
    model: BaseChatModel,
) -> CriterionVerdict:
    """Judge a single criterion against the response.

    Uses async invocation for concurrent evaluation of multiple criteria.
    """
    prompt = CRITERION_JUDGE_TEMPLATE.format(
        problem=problem,
        response=response,
        requirement=criterion.requirement,
    )

    try:
        result = await model.with_structured_output(
            CriterionJudgeResult
        ).ainvoke([
            SystemMessage(content=CRITERION_JUDGE_SYSTEM),
            HumanMessage(content=prompt),
        ])
        return CriterionVerdict(
            criterion_id=criterion.id,
            met=result.met,  # type: ignore[union-attr]
            justification=result.justification,  # type: ignore[union-attr]
        )
    except Exception as exc:
        logger.warning(
            "Judge error on criterion %r: %s", criterion.id, exc
        )
        return CriterionVerdict(
            criterion_id=criterion.id,
            met=False,
            justification=f"Judge error: {exc}",
        )
