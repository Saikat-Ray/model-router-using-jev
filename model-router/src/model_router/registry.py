"""Model registry: the table Jev routes over. Kept as data (YAML)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModelSpec:
    name: str                 # short key used as the Jev option label
    id: str                   # provider-specific model id
    description: str          # what Jev reads to decide
    rank: int                 # higher = more capable (drives escalation)
    context_window: int
    input_cost_per_m: float = 0.0
    output_cost_per_m: float = 0.0
    vision: bool = False
    provider: str = "openrouter"


class Registry:
    def __init__(self, models: list[ModelSpec], default: str | None = None,
                 confidence_threshold: float = 0.6):
        if not models:
            raise ValueError("registry needs at least one model")
        self.models = sorted(models, key=lambda m: m.rank)
        self._by_name = {m.name: m for m in self.models}
        if len(self._by_name) != len(self.models):
            raise ValueError("model names must be unique")
        self.default_name = default or self.models[0].name
        if self.default_name not in self._by_name:
            raise ValueError(f"default model {self.default_name!r} not in registry")
        self.confidence_threshold = confidence_threshold

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Registry":
        raw = yaml.safe_load(Path(path).read_text())
        models = [ModelSpec(**m) for m in raw["models"]]
        return cls(models, raw.get("default"), raw.get("confidence_threshold", 0.6))

    def get(self, name: str) -> ModelSpec:
        return self._by_name[name]

    def eligible(self, approx_input_tokens: int, needs_vision: bool = False) -> list[ModelSpec]:
        """Hard constraints live in code, not in Jev."""
        return [
            m for m in self.models
            if m.context_window >= approx_input_tokens and (m.vision or not needs_vision)
        ]

    @staticmethod
    def strongest(models: list[ModelSpec]) -> ModelSpec:
        return max(models, key=lambda m: m.rank)
