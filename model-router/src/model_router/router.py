"""Router: hard constraints in code, judgment in Jev, policy in a threshold."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .jev_client import ChoiceAnswer, Decider, JevError
from .registry import ModelSpec, Registry

MAX_STATE_CHARS = 4000

INSTRUCTIONS = (
    "Pick the cheapest model that can answer this request well. "
    "Choose a more capable model only when the request needs deep reasoning, "
    "multi-step code work, careful analysis, or the cost of a wrong answer is high."
)


class NoEligibleModel(Exception):
    pass


@dataclass
class Request:
    prompt: str
    system: str | None = None
    has_image: bool = False

    def messages(self) -> list[dict]:
        msgs = []
        if self.system:
            msgs.append({"role": "system", "content": self.system})
        msgs.append({"role": "user", "content": self.prompt})
        return msgs


@dataclass
class Decision:
    chain: list[ModelSpec]            # chain[0] is the pick; the rest are fallbacks
    reason: str                       # jev | low_confidence_escalated | only_eligible | jev_error
    confidence: float | None = None
    probabilities: dict[str, float] = field(default_factory=dict)
    route_ms: float = 0.0

    @property
    def model(self) -> ModelSpec:
        return self.chain[0]


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def build_chain(pick: ModelSpec, eligible: list[ModelSpec]) -> list[ModelSpec]:
    """Pick first, then stronger models (ascending), then weaker ones (descending)."""
    stronger = sorted((m for m in eligible if m.rank > pick.rank), key=lambda m: m.rank)
    weaker = sorted((m for m in eligible if m.rank < pick.rank), key=lambda m: -m.rank)
    return [pick, *stronger, *weaker]


class Router:
    def __init__(self, registry: Registry, decider: Decider):
        self.registry = registry
        self.decider = decider

    def route(self, request: Request) -> Decision:
        t0 = time.perf_counter()
        tokens = approx_tokens(request.prompt)
        eligible = self.registry.eligible(tokens, request.has_image)
        if not eligible:
            raise NoEligibleModel(f"no model fits ~{tokens} tokens (vision={request.has_image})")

        def done(pick, reason, answer: ChoiceAnswer | None = None, conf=None):
            return Decision(
                chain=build_chain(pick, eligible), reason=reason, confidence=conf,
                probabilities=answer.probabilities if answer else {},
                route_ms=(time.perf_counter() - t0) * 1000,
            )

        if len(eligible) == 1:
            return done(eligible[0], "only_eligible")

        options = {m.name: m.description for m in eligible}
        state = {"request": request.prompt[:MAX_STATE_CHARS], "approx_input_tokens": tokens}
        try:
            answer = self.decider.choose(state=state, instructions=INSTRUCTIONS, options=options)
        except JevError:
            names = {m.name: m for m in eligible}
            fallback = names.get(self.registry.default_name) or self.registry.strongest(eligible)
            return done(fallback, "jev_error")

        conf = answer.confidence
        if conf is None:
            conf = answer.probabilities.get(answer.choice)
        if conf is None or conf < self.registry.confidence_threshold:
            return done(self.registry.strongest(eligible), "low_confidence_escalated", answer, conf)
        return done(self.registry.get(answer.choice), "jev", answer, conf)
