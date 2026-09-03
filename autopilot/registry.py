"""Model registry: ModelConfig + YAML loading + cost math.

Cost is computed in exactly one place (`ModelConfig.cost_for`) so that the
savings number the dashboard reports can never drift from what the router
used to make its decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from autopilot import config
from autopilot.schemas import QualityTier, Usage

_PER_TOKEN = 1_000_000.0


@dataclass(frozen=True)
class ModelConfig:
    key: str
    provider: str
    model_id: str
    display_name: str
    input_cost_per_1m: float
    output_cost_per_1m: float
    avg_latency_ms: float
    quality_tier: QualityTier
    context_window: int
    max_output_tokens: int
    pricing_verified: bool = True
    notes: str | None = None

    # ---- cost math ------------------------------------------------------
    @property
    def input_cost_per_token(self) -> float:
        return self.input_cost_per_1m / _PER_TOKEN

    @property
    def output_cost_per_token(self) -> float:
        return self.output_cost_per_1m / _PER_TOKEN

    def cost_for(self, usage: Usage) -> float:
        """USD cost of one call. The single source of truth for pricing."""
        return (
            usage.input_tokens * self.input_cost_per_token
            + usage.output_tokens * self.output_cost_per_token
        )

    @property
    def is_local(self) -> bool:
        return self.input_cost_per_1m == 0.0 and self.output_cost_per_1m == 0.0

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(
            key=d["key"],
            provider=d["provider"],
            model_id=d["model_id"],
            display_name=d.get("display_name", d["key"]),
            input_cost_per_1m=float(d["input_cost_per_1m"]),
            output_cost_per_1m=float(d["output_cost_per_1m"]),
            avg_latency_ms=float(d.get("avg_latency_ms", 0.0)),
            quality_tier=QualityTier(d.get("quality_tier", "medium")),
            context_window=int(d.get("context_window", 128000)),
            max_output_tokens=int(d.get("max_output_tokens", 4096)),
            pricing_verified=bool(d.get("pricing_verified", True)),
            notes=d.get("notes"),
        )


class ModelRegistry:
    """All known models, keyed by registry key."""

    def __init__(self, models: list[ModelConfig]):
        self._models = {m.key: m for m in models}
        if len(self._models) != len(models):
            raise ValueError("duplicate model keys in registry")

    # ---- construction ---------------------------------------------------
    @classmethod
    def load(
        cls,
        path: Path | None = None,
        latency_overrides: Path | None = None,
    ) -> "ModelRegistry":
        path = path or config.MODELS_YAML
        raw = yaml.safe_load(path.read_text())
        models = [ModelConfig.from_dict(d) for d in raw["models"]]

        # Layer measured latency (written by the Phase 1 baseline run) on top
        # of the seed values, so routing decisions use real numbers.
        overrides_path = latency_overrides or config.MEASURED_LATENCY_YAML
        if overrides_path.exists():
            measured = yaml.safe_load(overrides_path.read_text()) or {}
            by_key = measured.get("p50_latency_ms", {})
            models = [
                (
                    ModelConfig(**{**m.__dict__, "avg_latency_ms": float(by_key[m.key])})
                    if m.key in by_key
                    else m
                )
                for m in models
            ]
        return cls(models)

    # ---- access ---------------------------------------------------------
    def __getitem__(self, key: str) -> ModelConfig:
        try:
            return self._models[key]
        except KeyError:
            raise KeyError(
                f"unknown model key {key!r}; known keys: {sorted(self._models)}"
            ) from None

    def __contains__(self, key: str) -> bool:
        return key in self._models

    def __iter__(self):
        return iter(self._models.values())

    def __len__(self) -> int:
        return len(self._models)

    @property
    def keys(self) -> list[str]:
        return list(self._models)

    def by_provider(self, provider: str) -> list[ModelConfig]:
        return [m for m in self if m.provider == provider]

    def by_tier(self, tier: QualityTier | str) -> list[ModelConfig]:
        tier = QualityTier(tier)
        return [m for m in self if m.quality_tier is tier]

    def cheapest(self, models: list[ModelConfig] | None = None) -> ModelConfig:
        pool = models or list(self)
        return min(pool, key=lambda m: m.output_cost_per_1m + m.input_cost_per_1m)

    def available(self) -> list[ModelConfig]:
        """Models whose provider is actually usable right now (key present, etc.)."""
        from autopilot.providers import provider_available

        return [m for m in self if provider_available(m.provider)]
