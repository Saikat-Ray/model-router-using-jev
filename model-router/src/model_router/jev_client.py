"""Thin client for Jev through OpenRouter's decisions endpoint.

Jev takes a `state` plus typed `questions` and returns typed answers with
probabilities. It does not generate text. The endpoint and model id are
configurable because OpenRouter's System One API is still moving
(alpha/decisions vs v1/systemone).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol

import httpx

DEFAULT_ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
DEFAULT_MODEL = "typesafe/jev-1.13"   # pin a version; "~typesafe/jev-latest" floats


class JevError(Exception):
    """Jev was unreachable or returned something we can't use."""


@dataclass
class ChoiceAnswer:
    choice: str
    confidence: float | None = None
    probabilities: dict[str, float] = field(default_factory=dict)


class Decider(Protocol):
    def choose(self, state: dict, instructions: str, options: dict[str, str]) -> ChoiceAnswer: ...


class JevClient:
    def __init__(self, api_key: str | None = None, endpoint: str | None = None,
                 model: str | None = None, timeout: float = 5.0,
                 client: httpx.Client | None = None):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.endpoint = endpoint or os.environ.get("JEV_ENDPOINT", DEFAULT_ENDPOINT)
        self.model = model or os.environ.get("JEV_MODEL", DEFAULT_MODEL)
        self.timeout = timeout
        self._client = client or httpx.Client()

    def choose(self, state: dict, instructions: str, options: dict[str, str],
               key: str = "model") -> ChoiceAnswer:
        body = {
            "model": self.model,
            "state": state,
            "questions": {
                key: {"type": "choice", "instructions": instructions, "criteria": options}
            },
        }
        try:
            resp = self._client.post(
                self.endpoint, json=body, timeout=self.timeout,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            resp.raise_for_status()
            answer = resp.json()["answers"][key]
        except (httpx.HTTPError, KeyError, ValueError) as e:
            raise JevError(f"decision call failed: {e}") from e

        choice = answer.get("choice")
        if answer.get("type") != "choice" or choice not in options:
            raise JevError(f"unexpected answer: {answer!r}")
        return ChoiceAnswer(
            choice=choice,
            confidence=answer.get("confidence"),
            probabilities=answer.get("probabilities") or {},
        )
