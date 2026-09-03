import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from autopilot.registry import ModelRegistry
from autopilot.router import first_available, get_routing_config, route


@pytest.fixture(scope="module")
def registry():
    return ModelRegistry.load()


def test_routing_config_loads():
    rc = get_routing_config()
    assert set(rc.tiers) == {1, 2, 3}
    assert rc.escalation_target


def test_simple_extraction_routes_to_cheap_tier(registry):
    d = route(
        "Extract the order number and shipping city from this line and return JSON: "
        '"Order 88213 dispatched to Rotterdam on 12 May."',
        registry,
    )
    assert d.tier.value == 1
    assert d.model.key in get_routing_config().chain_for(d.tier)


def test_complex_reasoning_routes_to_top_tier(registry):
    d = route(
        "A train leaves station A at 9:00am traveling 60 km/h toward station B, 210 km away. "
        "A second train leaves B at 9:30am traveling toward A at 90 km/h. At what clock time do "
        "they meet, and how far from A? Show your reasoning step by step, then state the answer.",
        registry,
    )
    assert d.tier.value == 3


def test_first_available_skips_unusable_models(registry):
    m = first_available(["claude-opus-5", "gpt-4o", "mock-premium"], registry)
    assert m is not None
    assert m.key == "mock-premium"


def test_first_available_returns_none_when_nothing_usable(registry):
    assert first_available(["claude-opus-5", "gpt-4o"], registry) is None
