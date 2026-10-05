"""Calls the chosen model through its configured provider, with fallback."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import httpx

from .registry import ModelSpec
from .router import Request

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"


class ExecutionError(Exception):
    pass


@dataclass
class ExecResult:
    model: ModelSpec
    text: str
    usage: dict
    latency_ms: float
    failures: list[str] = field(default_factory=list)


class Executor:
    def __init__(self, api_key: str | None = None, timeout: float = 60.0,
                 client: httpx.Client | None = None):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.timeout = timeout
        self._client = client or httpx.Client()

    def run(self, request: Request, chain: list[ModelSpec]) -> ExecResult:
        failures: list[str] = []
        for model in chain:
            t0 = time.perf_counter()
            try:
                if model.provider == "ollama":
                    base_url = os.environ.get(
                        "OLLAMA_BASE_URL",
                        OLLAMA_CHAT_URL.removesuffix("/api/chat"),
                    )
                    resp = self._client.post(
                        f"{base_url.rstrip('/')}/api/chat",
                        timeout=self.timeout,
                        json={
                            "model": model.id,
                            "messages": request.messages(),
                            "stream": False,
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    text = data["message"]["content"]
                    usage = {
                        "prompt_tokens": data.get("prompt_eval_count", 0),
                        "completion_tokens": data.get("eval_count", 0),
                    }
                    return ExecResult(
                        model, text, usage,
                        (time.perf_counter() - t0) * 1000, failures,
                    )
                if model.provider != "openrouter":
                    raise ExecutionError(
                        f"unsupported provider {model.provider!r} for {model.name}"
                    )
                resp = self._client.post(
                    CHAT_URL, timeout=self.timeout,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": model.id, "messages": request.messages()},
                )
                resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
            except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
                failures.append(f"{model.name}: {e}")
                continue
            return ExecResult(model, text, data.get("usage", {}),
                              (time.perf_counter() - t0) * 1000, failures)
        raise ExecutionError("; ".join(failures))
