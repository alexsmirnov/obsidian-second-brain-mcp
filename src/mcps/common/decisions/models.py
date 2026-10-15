"""Wire models for the router's TypeSafe System One decision endpoint.

Only ``noul`` (yes/no probability) questions are modelled today. To add
``choice`` or ``score``:

1. Add the question and answer models with ``type: Literal["choice"]``.
2. Widen ``Question`` and ``Answer`` to discriminated unions on ``type``, e.g.
   ``Annotated[Noul | Choice, Field(discriminator="type")]``.
3. Add a typed accessor such as ``DecisionResponse.choice(question_id)``.

``DecisionClient`` itself does not change.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

__all__ = [
    "Answer",
    "DecisionError",
    "DecisionResponse",
    "Noul",
    "NoulAnswer",
    "NoulCriteria",
    "Question",
]


class DecisionError(Exception):
    """The decision endpoint failed or returned an unusable response."""


class NoulCriteria(BaseModel):
    """What the yes and no outcomes of a ``Noul`` question mean."""

    true: str | None = None
    false: str | None = None


class Noul(BaseModel):
    """Binary question answered with the probability of yes."""

    type: Literal["noul"] = "noul"
    instructions: str
    criteria: NoulCriteria | None = None


class NoulAnswer(BaseModel):
    """Probability, from 0 to 1, that a ``Noul`` question's answer is yes."""

    type: Literal["noul"]
    noul: float = Field(ge=0.0, le=1.0)


type Question = Noul
type Answer = NoulAnswer


class DecisionResponse(BaseModel):
    """Typed answers keyed by the question ids of the request."""

    answers: dict[str, Answer]
    model: str | None = None

    def noul(self, question_id: str) -> float:
        """Return the yes probability of a ``Noul`` question."""
        answer = self.answers.get(question_id)
        if not isinstance(answer, NoulAnswer):
            raise DecisionError(f"no noul answer for question {question_id!r}")
        return answer.noul
