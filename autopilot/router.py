"""Phase 2, step 4 -- the routing map: complexity tier -> model.

    decision = route("Summarize this in two sentences: ...", registry)
    resp = await send_request(prompt, decision.model)

Routing is deliberately two small, inspectable steps: classify, then look up
a model in config/routing.yaml. Nothing here calls a provider -- callers pass
the decision to autopilot.client.send_request themselves, and the async
verifier (Phase 3) decides whether to escalate after seeing the response.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from autopilot import config
from autopilot.classifier.model import get_classifier
from autopilot.registry import ModelConfig, ModelRegistry
from autopilot.schemas import ComplexityTier


@dataclass
class RoutingDecision:
    tier: ComplexityTier
    model: ModelConfig
    confidence: float
    classifier_tier: ComplexityTier  # what the classifier said, before any bump
    escalated_by_confidence: bool
    reason: str


class RoutingConfig:
    """Loaded config/routing.yaml -- the tier -> model map."""

    def __init__(self, raw: dict):
        self._raw = raw
        self.min_confidence: float = float(raw.get("min_confidence", 0.0))
        self.escalation_target: str = raw["escalation_target"]
        self.tiers: dict[int, dict] = {int(k): v for k, v in raw["tiers"].items()}

    @classmethod
    def load(cls, path: Path | None = None) -> "RoutingConfig":
        path = path or config.ROUTING_YAML
        return cls(yaml.safe_load(path.read_text()))

    def chain_for(self, tier: ComplexityTier) -> list[str]:
        """Model keys to try in order for a tier: primary, then its fallbacks."""
        spec = self.tiers[tier.value]
        return [spec["primary"], *spec.get("fallback", [])]


_singleton: RoutingConfig | None = None


def get_routing_config() -> RoutingConfig:
    global _singleton
    if _singleton is None:
        _singleton = RoutingConfig.load()
    return _singleton


def reload_routing_config() -> RoutingConfig:
    """Force a reload -- used by PUT /v1/routing-config (Phase 5)."""
    global _singleton
    _singleton = RoutingConfig.load()
    return _singleton


def first_available(keys: list[str], registry: ModelRegistry) -> ModelConfig | None:
    """First model in `keys` (registry keys, in priority order) that is usable."""
    for key in keys:
        if key not in registry:
            continue
        model = registry[key]
        from autopilot.providers import provider_available

        if provider_available(model.provider):
            return model
    return None


def route(
    prompt: str,
    registry: ModelRegistry,
    *,
    routing_config: RoutingConfig | None = None,
) -> RoutingDecision:
    """Classify a prompt and pick the model that should answer it."""
    rc = routing_config or get_routing_config()
    prediction = get_classifier().predict(prompt)

    tier = prediction.tier
    escalated = False
    reason = f"classifier: tier {tier.value} ({prediction.confidence:.0%} confidence)"

    if (
        not prediction.used_fallback
        and prediction.confidence < rc.min_confidence
        and tier.value < ComplexityTier.COMPLEX.value
    ):
        tier = ComplexityTier(tier.value + 1)
        escalated = True
        reason = (
            f"classifier said tier {prediction.tier.value} at only "
            f"{prediction.confidence:.0%} confidence (< {rc.min_confidence:.0%}); "
            f"bumped to tier {tier.value}"
        )

    chain = rc.chain_for(tier)
    model = first_available(chain, registry)
    if model is None:
        # Nothing in the chain is usable -- fall back to whatever is cheapest
        # among available models rather than failing the request outright.
        available = registry.available()
        if not available:
            raise RuntimeError("no model is available for any tier; check .env / ollama")
        model = registry.cheapest(available)
        reason += f"; none of {chain} available, fell back to cheapest usable model"
    elif model.key != chain[0]:
        reason += f"; primary {chain[0]!r} unavailable, used {model.key!r}"

    return RoutingDecision(
        tier=tier,
        model=model,
        confidence=prediction.confidence,
        classifier_tier=prediction.tier,
        escalated_by_confidence=escalated,
        reason=reason,
    )


__all__ = [
    "route",
    "RoutingDecision",
    "RoutingConfig",
    "get_routing_config",
    "reload_routing_config",
    "first_available",
]
