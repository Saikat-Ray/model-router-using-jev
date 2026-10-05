"""Every routing decision goes to JSONL. This is what makes the router improvable."""
from __future__ import annotations

import json
import time
from pathlib import Path

from .executor import ExecResult
from .router import Decision, Request


class DecisionLog:
    def __init__(self, path: str | Path = "logs/decisions.jsonl", preview_chars: int = 200):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.preview_chars = preview_chars   # set to 0 to avoid storing prompt text

    def write(self, request: Request, decision: Decision, result: ExecResult | None,
              error: str | None = None) -> None:
        row = {
            "ts": time.time(),
            "prompt_preview": request.prompt[: self.preview_chars],
            "prompt_chars": len(request.prompt),
            "picked": decision.model.name,
            "chain": [m.name for m in decision.chain],
            "reason": decision.reason,
            "confidence": decision.confidence,
            "probabilities": decision.probabilities,
            "route_ms": round(decision.route_ms, 1),
            "served_by": result.model.name if result else None,
            "exec_ms": round(result.latency_ms, 1) if result else None,
            "usage": result.usage if result else None,
            "failures": result.failures if result else None,
            "error": error,
        }
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")
