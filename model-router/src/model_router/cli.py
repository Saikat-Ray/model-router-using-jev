from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .executor import Executor
from .jev_client import JevClient
from .logger import DecisionLog
from .pipeline import Pipeline
from .registry import Registry
from .router import Request, Router


def build(config: str) -> tuple[Registry, Router]:
    registry = Registry.from_yaml(config)
    return registry, Router(registry, JevClient())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="model-router")
    p.add_argument("prompt", nargs="?", help="prompt text (reads stdin if omitted)")
    p.add_argument("--config", default=str(Path("config/models.yaml")))
    p.add_argument("--dry-run", action="store_true", help="route only, don't call the model")
    p.add_argument("--log", default="logs/decisions.jsonl")
    args = p.parse_args(argv)

    prompt = args.prompt or sys.stdin.read()
    request = Request(prompt=prompt)
    registry, router = build(args.config)

    if args.dry_run:
        d = router.route(request)
        print(f"pick={d.model.name} ({d.model.id}) reason={d.reason} "
              f"confidence={d.confidence} route_ms={d.route_ms:.0f}")
        print(f"probabilities={d.probabilities}")
        return 0

    pipeline = Pipeline(router, Executor(), DecisionLog(args.log))
    d, r = pipeline.handle(request)
    print(f"[routed to {r.model.name} | reason={d.reason} | confidence={d.confidence}]\n")
    print(r.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
