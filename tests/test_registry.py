import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from autopilot.registry import ModelConfig, ModelRegistry
from autopilot.schemas import QualityTier, Usage


@pytest.fixture(scope="module")
def registry():
    return ModelRegistry.load()


def test_registry_loads_all_models(registry):
    assert len(registry) >= 8
    assert "claude-opus-5" in registry
    assert "llama3.2-1b-local" in registry


def test_unknown_key_lists_known_keys(registry):
    with pytest.raises(KeyError, match="unknown model key"):
        registry["gpt-9-turbo-ultra"]


def test_cost_math_matches_hand_calculation(registry):
    # Sonnet 5: $2.00/1M in, $10.00/1M out.
    m = registry["claude-sonnet-5"]
    cost = m.cost_for(Usage(input_tokens=1_000_000, output_tokens=1_000_000))
    assert cost == pytest.approx(12.00)

    cost = m.cost_for(Usage(input_tokens=1500, output_tokens=300))
    assert cost == pytest.approx(1500 * 2e-6 + 300 * 1e-5)


def test_local_models_are_free_but_counted(registry):
    m = registry["llama3.1-local"]
    assert m.is_local
    u = Usage(5000, 2000)
    assert m.cost_for(u) == 0.0
    assert u.total_tokens == 7000


def test_cheapest_picks_lowest_combined_price(registry):
    priced = [m for m in registry if not m.is_local]
    assert registry.cheapest(priced).key == "gpt-4o-mini"


def test_tier_filters(registry):
    high = {m.key for m in registry.by_tier(QualityTier.HIGH)}
    assert "claude-opus-5" in high and "gpt-4o" in high
    assert "claude-haiku-4-5" not in high


def test_duplicate_keys_rejected():
    m = ModelConfig(
        key="dup", provider="mock", model_id="x", display_name="X",
        input_cost_per_1m=1.0, output_cost_per_1m=1.0, avg_latency_ms=1.0,
        quality_tier=QualityTier.LOW, context_window=1000, max_output_tokens=100,
    )
    with pytest.raises(ValueError, match="duplicate"):
        ModelRegistry([m, m])


def test_every_model_declares_pricing_provenance(registry):
    # A model with unverified pricing is allowed, but must be explicit about it,
    # because the project's headline metric is a dollar figure.
    for m in registry:
        assert isinstance(m.pricing_verified, bool)
