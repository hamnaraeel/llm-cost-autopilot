"""Phase 5 -- the FastAPI service.

One completion endpoint. The caller never names a model; the router picks
one and the response says which. Everything else (models, stats,
routing-config) exists to inspect or steer that decision, not to bypass it.

Run with:  uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from autopilot import config
from autopilot.pipeline import run_request
from autopilot.registry import ModelRegistry
from autopilot.router import get_routing_config, reload_routing_config
from autopilot.schemas import ProviderError
from autopilot.store import get_store

app = FastAPI(
    title="LLM Cost Autopilot",
    description="Complexity-routed LLM gateway: cheap models for easy requests, "
    "expensive ones for hard requests, verified automatically.",
    version="1.0",
)

_registry: ModelRegistry | None = None


def registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry.load()
    return _registry


# ---- schemas ----------------------------------------------------------------


class ChatCompletionRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    task: str | None = None
    max_tokens: int = 1024
    temperature: float = 0.0
    verify: bool = True


class ChatCompletionResponse(BaseModel):
    text: str
    request_id: str
    model_key: str
    provider: str
    complexity_tier: int
    classifier_confidence: float
    escalated: bool
    agreement_score: float | None
    input_tokens: int
    output_tokens: int
    cost_usd: float
    baseline_cost_usd: float
    latency_ms: float
    routing_reason: str


class RoutingConfigUpdate(BaseModel):
    tier: int
    primary: str
    fallback: list[str] | None = None


# ---- endpoints ----------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(req: ChatCompletionRequest) -> ChatCompletionResponse:
    try:
        result = await run_request(
            req.prompt,
            registry(),
            task=req.task,
            verify_quality=req.verify,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        )
    except ProviderError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    r, d, v = result.response, result.routing, result.verification
    return ChatCompletionResponse(
        text=r.text,
        request_id=r.request_id,
        model_key=r.model_key,
        provider=r.provider,
        complexity_tier=d.tier.value,
        classifier_confidence=d.confidence,
        escalated=v.escalated,
        agreement_score=v.agreement,
        input_tokens=r.usage.input_tokens,
        output_tokens=r.usage.output_tokens,
        cost_usd=r.cost_usd,
        baseline_cost_usd=result.baseline_cost_usd,
        latency_ms=r.latency_ms,
        routing_reason=d.reason,
    )


@app.get("/v1/models")
def list_models() -> dict:
    from autopilot.providers import provider_available

    return {
        "models": [
            {
                "key": m.key,
                "provider": m.provider,
                "display_name": m.display_name,
                "quality_tier": m.quality_tier.value,
                "input_cost_per_1m": m.input_cost_per_1m,
                "output_cost_per_1m": m.output_cost_per_1m,
                "avg_latency_ms": m.avg_latency_ms,
                "available": provider_available(m.provider),
                "pricing_verified": m.pricing_verified,
            }
            for m in registry()
        ]
    }


@app.get("/v1/stats")
def stats() -> dict:
    return get_store().summary()


@app.get("/v1/routing-config")
def get_config() -> dict:
    rc = get_routing_config()
    return {
        "tiers": {k: {"primary": v["primary"], "fallback": v.get("fallback", [])} for k, v in rc.tiers.items()},
        "escalation_target": rc.escalation_target,
        "min_confidence": rc.min_confidence,
    }


@app.put("/v1/routing-config")
def update_config(update: RoutingConfigUpdate) -> dict:
    if update.tier not in (1, 2, 3):
        raise HTTPException(status_code=400, detail="tier must be 1, 2, or 3")
    if update.primary not in registry():
        raise HTTPException(status_code=400, detail=f"unknown model key {update.primary!r}")

    path = config.ROUTING_YAML
    raw = yaml.safe_load(path.read_text())
    raw["tiers"][update.tier]["primary"] = update.primary
    if update.fallback is not None:
        raw["tiers"][update.tier]["fallback"] = update.fallback
    path.write_text(yaml.safe_dump(raw, sort_keys=False))

    reload_routing_config()
    return get_config()
