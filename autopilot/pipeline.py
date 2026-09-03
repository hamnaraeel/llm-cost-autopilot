"""Wires route -> send -> verify -> log into the one call every entry point
(the load test script, the FastAPI endpoint) shares, so behavior can't drift
between them.
"""

from __future__ import annotations

from dataclasses import dataclass

from autopilot.client import send_request
from autopilot.registry import ModelRegistry
from autopilot.router import RoutingConfig, RoutingDecision, get_routing_config, route
from autopilot.schemas import LLMResponse
from autopilot.store import LogStore, RequestLog, get_store
from autopilot.verifier import VerificationResult, verify


@dataclass
class PipelineResult:
    response: LLMResponse
    routing: RoutingDecision
    verification: VerificationResult
    baseline_cost_usd: float


async def run_request(
    prompt: str,
    registry: ModelRegistry,
    *,
    task: str | None = None,
    routing_config: RoutingConfig | None = None,
    store: LogStore | None = None,
    verify_quality: bool = True,
    **send_kwargs,
) -> PipelineResult:
    rc = routing_config or get_routing_config()
    decision = route(prompt, registry, routing_config=rc)
    primary_response = await send_request(prompt, decision.model, **send_kwargs)

    if verify_quality:
        verification = await verify(prompt, primary_response, registry, routing_config=rc)
    else:
        verification = VerificationResult(
            agreement=1.0, escalated=False, final_response=primary_response,
            verifier_response=None, reason="verification disabled for this request",
        )
    final = verification.final_response

    # "What would this have cost if the escalation-target model had handled
    # it alone" -- the reference point the headline savings number is measured
    # against. Uses the final response's own token counts as the closest
    # available stand-in for "the same amount of work."
    baseline_model = registry[rc.escalation_target]
    baseline_cost = baseline_model.cost_for(final.usage)

    log_store = store or get_store()
    log_store.log(
        RequestLog(
            request_id=final.request_id,
            prompt=prompt,
            task=task,
            complexity_tier=decision.tier.value,
            classifier_confidence=decision.confidence,
            routed_model_key=decision.model.key,
            routed_provider=decision.model.provider,
            final_model_key=final.model_key,
            final_provider=final.provider,
            escalated=verification.escalated,
            agreement_score=verification.agreement,
            input_tokens=final.usage.input_tokens,
            output_tokens=final.usage.output_tokens,
            cost_usd=final.cost_usd,
            baseline_cost_usd=baseline_cost,
            latency_ms=final.latency_ms,
            note=f"{decision.reason} | {verification.reason}",
        )
    )

    return PipelineResult(
        response=final,
        routing=decision,
        verification=verification,
        baseline_cost_usd=baseline_cost,
    )


__all__ = ["run_request", "PipelineResult"]
