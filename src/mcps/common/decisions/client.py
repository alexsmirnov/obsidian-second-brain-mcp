"""Borrowed-client boundary for the router's ``POST /v1/systemone`` endpoint."""

from __future__ import annotations

import logging
from collections.abc import Mapping

import httpx
from pydantic import JsonValue, SecretStr

from mcps.common.decisions.models import (
    DecisionError,
    DecisionResponse,
    Question,
)

__all__ = ["DecisionClient", "decision_endpoint"]

logger = logging.getLogger(__name__)

_MAX_QUESTIONS = 64


def decision_endpoint(router_url: str) -> str:
    """Return the System One URL for a router base with or without ``/v1``."""
    root = router_url.rstrip("/").removesuffix("/v1")
    return f"{root}/v1/systemone"


class DecisionClient:
    """Asks typed questions about a state through the router's decision model.

    The HTTP client is borrowed and never closed here.
    """

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        *,
        router_url: str,
        router_key: str,
        model: str,
    ) -> None:
        self._http = http_client
        self._endpoint = decision_endpoint(router_url)
        self._secret = SecretStr(router_key)
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    async def decide(
        self, state: JsonValue, questions: Mapping[str, Question]
    ) -> DecisionResponse:
        """Answer ``questions`` about ``state``; failures raise ``DecisionError``."""
        if not 1 <= len(questions) <= _MAX_QUESTIONS:
            raise ValueError(f"questions must hold 1 to {_MAX_QUESTIONS} entries")
        payload = {
            "model": self._model,
            "state": state,
            "questions": {
                question_id: question.model_dump(mode="json", exclude_none=True)
                for question_id, question in questions.items()
            },
        }
        try:
            response = await self._http.post(
                self._endpoint, json=payload, headers=self._headers()
            )
            response.raise_for_status()
            return DecisionResponse.model_validate(response.json())
        except httpx.HTTPStatusError as error:
            self._warn(f"HTTP {error.response.status_code}")
            raise DecisionError("decision request failed") from error
        except httpx.HTTPError as error:
            self._warn(type(error).__name__)
            raise DecisionError("decision request failed") from error
        except ValueError as error:
            self._warn(f"malformed response ({type(error).__name__})")
            raise DecisionError("malformed decision response") from error

    def _headers(self) -> dict[str, str]:
        key = self._secret.get_secret_value()
        return {"Authorization": f"Bearer {key}"} if key else {}

    def _warn(self, reason: str) -> None:
        logger.warning(
            "Decision request to %s with model %s failed: %s",
            self._endpoint,
            self._model,
            reason,
        )
