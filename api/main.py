"""Phase 5 -- the FastAPI service.

Two completion endpoints. `/v1/chat/completions` is OpenAI-request-shape
compatible: point an existing `openai` SDK client's `base_url` at this
service (optionally with `api_key="unused"`, since auth is per-provider
headers, not this field) and it works as a drop-in -- Autopilot picks the
model, not the caller. `/v1/route` is the original prompt-in/rich-metadata-
out shape, kept for direct inspection of a routing decision.

Bring-your-own-key: pass `X-OpenAI-Api-Key` / `X-Anthropic-Api-Key` /
`X-Groq-Api-Key` to route through *your* provider account instead of the
server's -- lets someone self-serve without the operator paying for their
usage. Keys are used for exactly one request and never stored or logged.

Run with:  uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import logging
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from autopilot import config
from autopilot.pipeline import run_request
from autopilot.registry import ModelRegistry
from autopilot.router import get_routing_config, reload_routing_config
from autopilot.schemas import ProviderError
from autopilot.seed_prompts import load_pool
from autopilot.store import get_store

log = logging.getLogger(__name__)

# A brand-new deployment (or a wiped volume) should never show a first-time
# visitor an empty dashboard -- seed a small, real batch on first boot only.
SEED_ON_EMPTY_N = 20

app = FastAPI(
    title="LLM Cost Autopilot",
    description="Complexity-routed LLM gateway: cheap models for easy requests, "
    "expensive ones for hard requests, verified automatically.",
    version="1.0",
)

# Meant to be called from other people's apps/browsers with per-request,
# header-scoped API keys rather than cookies or sessions -- there is no
# session state a permissive origin could hijack, so a wildcard is a
# reasonable default here rather than a security hole.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_registry: ModelRegistry | None = None


def registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry.load()
    return _registry


async def _seed_if_empty() -> None:
    """Best-effort: run a small real batch through the pipeline on first
    boot so the dashboard has data to show immediately. Fire-and-forget in
    the background -- never blocks startup, and a failed seed call is
    swallowed rather than crashing the app over demo traffic.
    """
    store = get_store()
    if store.count() > 0:
        return
    pool = load_pool()
    if not pool:
        return
    rng = random.Random(7)
    batch = [rng.choice(pool) for _ in range(SEED_ON_EMPTY_N)]
    reg = registry()
    for item in batch:
        try:
            await run_request(item["text"], reg, task=item["task"])
        except Exception:  # noqa: BLE001 -- one bad seed call must not stop the rest
            log.warning("seed request failed", exc_info=True)


@app.on_event("startup")
async def _on_startup() -> None:
    asyncio.create_task(_seed_if_empty())


async def byok_headers(
    x_openai_api_key: str | None = Header(default=None, alias="X-OpenAI-Api-Key"),
    x_anthropic_api_key: str | None = Header(default=None, alias="X-Anthropic-Api-Key"),
    x_groq_api_key: str | None = Header(default=None, alias="X-Groq-Api-Key"),
) -> dict[str, str]:
    """Provider name -> caller-supplied key, from the X-*-Api-Key headers."""
    keys = {}
    if x_openai_api_key:
        keys["openai"] = x_openai_api_key
    if x_anthropic_api_key:
        keys["anthropic"] = x_anthropic_api_key
    if x_groq_api_key:
        keys["groq"] = x_groq_api_key
    return keys


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
    used_mock_fallback: bool
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


class OAIMessage(BaseModel):
    role: str
    content: str


class OAIChatRequest(BaseModel):
    # `model` is accepted (the openai SDK always sends one) and ignored --
    # the whole point of this service is that the router picks the model,
    # not the caller. `extra="ignore"` tolerates SDK fields like `top_p`
    # this service doesn't act on, rather than rejecting the request.
    model_config = ConfigDict(extra="ignore")

    model: str = "autopilot"
    messages: list[OAIMessage] = Field(..., min_length=1)
    max_tokens: int = 1024
    temperature: float = 0.0
    stream: bool = False


class OAIChoiceMessage(BaseModel):
    role: str = "assistant"
    content: str


class OAIChoice(BaseModel):
    index: int = 0
    message: OAIChoiceMessage
    finish_reason: str = "stop"


class OAIUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class AutopilotMeta(BaseModel):
    """Non-standard extra info, namespaced so it can't collide with an
    OpenAI response field a strict client might validate against."""

    complexity_tier: int
    routed_model: str
    provider: str
    escalated: bool
    agreement_score: float | None
    cost_usd: float
    baseline_cost_usd: float
    used_mock_fallback: bool
    routing_reason: str


class OAIChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[OAIChoice]
    usage: OAIUsage
    autopilot: AutopilotMeta


def _messages_to_prompt(messages: list[OAIMessage]) -> tuple[str, str | None]:
    """Collapse an OpenAI-style message list into the flat (prompt, system)
    shape the rest of the pipeline speaks. A single user message passes
    through untouched -- that's the common case and what the classifier was
    trained on; multi-turn history is rendered as a labeled transcript so
    context isn't silently dropped, at the cost of being slightly out of the
    classifier's training distribution for long conversations.
    """
    system_parts = [m.content for m in messages if m.role == "system"]
    system = "\n".join(system_parts) or None
    convo = [m for m in messages if m.role != "system"]
    if not convo:
        raise HTTPException(status_code=400, detail="messages must include at least one non-system message")
    if len(convo) == 1 and convo[0].role == "user":
        return convo[0].content, system
    lines = [f"{'Human' if m.role == 'user' else 'Assistant'}: {m.content}" for m in convo]
    lines.append("Assistant:")
    return "\n\n".join(lines), system


# ---- endpoints ----------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/v1/chat/completions", response_model=OAIChatResponse)
async def openai_compatible_chat_completions(
    req: OAIChatRequest,
    api_keys: dict[str, str] = Depends(byok_headers),
) -> OAIChatResponse:
    """Drop-in for `client.chat.completions.create(...)` from the `openai`
    SDK -- just point `base_url` here. See the module docstring for the
    bring-your-own-key headers.
    """
    if req.stream:
        # Streaming would need to fail loudly, not silently return a full
        # response the caller's SSE parser can't make sense of.
        raise HTTPException(status_code=400, detail="stream=true is not supported yet; retry with stream=false")

    prompt, system = _messages_to_prompt(req.messages)
    try:
        result = await run_request(
            prompt,
            registry(),
            verify_quality=True,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            system=system,
            api_keys=api_keys or None,
        )
    except ProviderError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    r, d, v = result.response, result.routing, result.verification
    return OAIChatResponse(
        id=f"chatcmpl-{r.request_id}",
        created=int(time.time()),
        model=r.model_key,
        choices=[OAIChoice(message=OAIChoiceMessage(content=r.text), finish_reason=r.stop_reason or "stop")],
        usage=OAIUsage(
            prompt_tokens=r.usage.input_tokens,
            completion_tokens=r.usage.output_tokens,
            total_tokens=r.usage.total_tokens,
        ),
        autopilot=AutopilotMeta(
            complexity_tier=d.tier.value,
            routed_model=r.model_key,
            provider=r.provider,
            escalated=v.escalated,
            agreement_score=v.agreement,
            cost_usd=r.cost_usd,
            baseline_cost_usd=result.baseline_cost_usd,
            used_mock_fallback=r.provider == "mock",
            routing_reason=d.reason,
        ),
    )


@app.post("/v1/route", response_model=ChatCompletionResponse)
async def route_debug(
    req: ChatCompletionRequest,
    api_keys: dict[str, str] = Depends(byok_headers),
) -> ChatCompletionResponse:
    """Prompt-in, rich-metadata-out -- for inspecting a routing decision
    directly rather than integrating against the OpenAI-compatible shape.
    """
    try:
        result = await run_request(
            req.prompt,
            registry(),
            task=req.task,
            verify_quality=req.verify,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            api_keys=api_keys or None,
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
        used_mock_fallback=r.provider == "mock",
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
