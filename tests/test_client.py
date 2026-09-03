import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from autopilot.client import fan_out, send_request
from autopilot.providers.mock_provider import MockProvider
from autopilot.registry import ModelRegistry
from autopilot.schemas import LLMRequest, ProviderError

pytestmark = pytest.mark.asyncio


@pytest.fixture(scope="module")
def registry():
    return ModelRegistry.load()


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Strip simulated latency so tests stay fast."""
    monkeypatch.setattr(MockProvider, "__init__", lambda self: setattr(self, "simulate_latency", False))
    from autopilot import providers
    providers.get_provider.cache_clear()
    yield
    providers.get_provider.cache_clear()


async def test_response_is_standardized(registry):
    m = registry["mock-cheap"]
    r = await send_request("Extract the total from: Invoice #12 for $40.", m)
    assert r.text
    assert r.model_key == "mock-cheap"
    assert r.provider == "mock"
    assert r.usage.input_tokens > 0 and r.usage.output_tokens > 0
    assert r.latency_ms >= 0
    assert r.cost_usd == pytest.approx(m.cost_for(r.usage))
    assert set(r.to_row()) >= {"request_id", "model_key", "cost_usd", "latency_ms"}


async def test_same_prompt_costs_more_on_premium_model(registry):
    prompt = "Write a short assessment of this tradeoff."
    cheap = await send_request(prompt, registry["mock-cheap"])
    premium = await send_request(prompt, registry["mock-premium"])
    assert premium.cost_usd > cheap.cost_usd


async def test_mock_is_deterministic(registry):
    m = registry["mock-cheap"]
    a = await send_request("stable prompt", m)
    b = await send_request("stable prompt", m)
    assert a.text == b.text
    assert a.cost_usd == b.cost_usd


async def test_request_id_is_preserved(registry):
    req = LLMRequest(prompt="hello", request_id="fixed-id-123")
    r = await send_request(req, registry["mock-cheap"])
    assert r.request_id == "fixed-id-123"


async def test_unavailable_provider_raises_with_reason(registry, monkeypatch):
    from autopilot import providers
    monkeypatch.setattr(providers, "provider_usable", lambda name, api_keys=None: False)
    monkeypatch.setattr("autopilot.client.provider_usable", lambda name, api_keys=None: False)
    monkeypatch.setattr("autopilot.client.unavailable_reason", lambda name: "no key")
    with pytest.raises(ProviderError, match="no key"):
        await send_request("x", registry["mock-cheap"])


async def test_fan_out_returns_errors_instead_of_raising(registry):
    models = [registry["mock-cheap"], registry["mock-premium"]]
    results = await fan_out("compare these", models)
    assert len(results) == 2
    assert all(not isinstance(r, Exception) for r in results)
