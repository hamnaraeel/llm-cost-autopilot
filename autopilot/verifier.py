"""Phase 3 -- async quality verification and auto-escalation.

After the router's chosen (cheap) model answers, the verifier optionally
fans the same prompt out to the escalation target (the highest-quality model
available) and scores how much the two answers agree. A low agreement score
means the cheap model's answer is suspect, so the verifier's answer replaces
it and the event is logged as an escalation.

Agreement is a lexical-overlap heuristic (weighted Jaccard over word sets,
penalized for large length mismatches) rather than a second LLM-as-judge
call, so verification stays fast and free to run on every request instead of
being a cost center itself. It is intentionally crude: good enough to catch
"the cheap model gave a short, unrelated, or truncated answer," not a
substitute for task-specific graders (e.g. exact-field-match for extraction,
label-match for classification) -- those are a natural place to extend this
once real usage data exists.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from autopilot.client import send_request
from autopilot.registry import ModelRegistry
from autopilot.router import RoutingConfig, get_routing_config, first_available
from autopilot.schemas import ComplexityTier, LLMResponse

_WORD_RE = re.compile(r"[a-z0-9']+")

# Below this agreement score, the cheap model's answer is discarded in favor
# of the verifier's. Calibrated against the offline mock provider's score
# distribution (see scripts/load_test.py output) so escalation fires for a
# realistic minority of requests in an offline demo run; recalibrate against
# real provider outputs before trusting this on live traffic.
DEFAULT_THRESHOLD = 0.63


def _words(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def score_agreement(a: str, b: str) -> float:
    """0..1 lexical agreement between two responses to the same prompt."""
    wa, wb = _words(a), _words(b)
    if not wa and not wb:
        return 1.0
    if not wa or not wb:
        return 0.0
    jaccard = len(wa & wb) / len(wa | wb)

    len_a, len_b = len(a.split()), len(b.split())
    longer, shorter = max(len_a, len_b), max(min(len_a, len_b), 1)
    # A cheap model that answers in 3 words when the reference used 80 is
    # suspicious even if every one of those 3 words also appears in the
    # reference -- penalize extreme length mismatch independent of Jaccard.
    length_ratio = shorter / longer

    return round(0.7 * jaccard + 0.3 * length_ratio, 4)


@dataclass
class VerificationResult:
    agreement: float
    escalated: bool
    final_response: LLMResponse
    verifier_response: LLMResponse | None
    reason: str


async def verify(
    prompt: str,
    primary_response: LLMResponse,
    registry: ModelRegistry,
    *,
    routing_config: RoutingConfig | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> VerificationResult:
    """Optionally re-check `primary_response` against the escalation target."""
    rc = routing_config or get_routing_config()
    escalation_chain = rc.chain_for(ComplexityTier.COMPLEX)

    if primary_response.model_key in escalation_chain:
        return VerificationResult(
            agreement=1.0,
            escalated=False,
            final_response=primary_response,
            verifier_response=None,
            reason="primary model is already an escalation-tier model; skipped verification",
        )

    verifier_model = first_available(escalation_chain, registry)
    if verifier_model is None:
        return VerificationResult(
            agreement=1.0,
            escalated=False,
            final_response=primary_response,
            verifier_response=None,
            reason="no escalation-tier model available; verification skipped",
        )

    verifier_response = await send_request(prompt, verifier_model)
    agreement = score_agreement(primary_response.text, verifier_response.text)
    escalated = agreement < threshold

    return VerificationResult(
        agreement=agreement,
        escalated=escalated,
        final_response=verifier_response if escalated else primary_response,
        verifier_response=verifier_response,
        reason=(
            f"agreement {agreement:.2f} < {threshold:.2f}, escalated to {verifier_model.key}"
            if escalated
            else f"agreement {agreement:.2f} >= {threshold:.2f}, kept {primary_response.model_key}"
        ),
    )


__all__ = ["verify", "score_agreement", "VerificationResult", "DEFAULT_THRESHOLD"]
