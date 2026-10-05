"""Score the router against labeled prompts. Extend eval_set.jsonl to ~100 rows.

Under-routing (picked weaker than expected) is a quality risk.
Over-routing (picked stronger) is wasted cost. Track both.
"""
import json
import sys
from collections import Counter
from pathlib import Path

from model_router.jev_client import JevClient
from model_router.registry import Registry
from model_router.router import Request, Router

root = Path(__file__).resolve().parent.parent
registry = Registry.from_yaml(root / "config/models.yaml")
router = Router(registry, JevClient())
rows = [json.loads(l) for l in (root / "evals/eval_set.jsonl").read_text().splitlines() if l.strip()]

exact, under, over, reasons = 0, 0, 0, Counter()
for row in rows:
    d = router.route(Request(row["prompt"]))
    got, want = d.model, registry.get(row["expected"])
    reasons[d.reason] += 1
    if got.rank == want.rank:
        exact += 1
    elif got.rank < want.rank:
        under += 1
    else:
        over += 1
    flag = "ok " if got.rank == want.rank else ("LOW" if got.rank < want.rank else "HI ")
    print(f"{flag} want={want.name:<7} got={got.name:<7} conf={d.confidence} {row['prompt'][:60]!r}")

n = len(rows)
print(f"\nexact {exact}/{n}  under-routed {under}  over-routed {over}  reasons {dict(reasons)}")
sys.exit(0)
