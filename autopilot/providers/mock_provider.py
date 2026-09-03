"""Deterministic offline provider.

Exists so the full pipeline -- routing, async verification, escalation,
dashboard, load test -- can be exercised end to end with no API keys and no
network. Output is a function of (prompt, model), so reruns are reproducible
and the verifier's agreement scores are stable.

`mock-premium` answers more thoroughly than `mock-cheap`, and `mock-cheap`
degrades on long/complex prompts. That is what makes escalation events
actually fire during an offline load test instead of never happening.
"""

from __future__ import annotations

import asyncio
import hashlib
import random

from autopilot.providers.base import Provider
from autopilot.registry import ModelConfig
from autopilot.schemas import LLMRequest, RawCompletion

_FILLER = (
    "the request", "the provided context", "each relevant field", "the key constraint",
    "the underlying tradeoff", "the expected output", "the supporting detail",
    "the stated requirement", "the edge case", "the final recommendation",
)


def _seed(prompt: str, model_id: str) -> int:
    return int(hashlib.sha256(f"{model_id}::{prompt}".encode()).hexdigest()[:12], 16)


class MockProvider(Provider):
    name = "mock"

    def __init__(self, simulate_latency: bool = True) -> None:
        self.simulate_latency = simulate_latency

    async def complete(self, req: LLMRequest, model: ModelConfig) -> RawCompletion:
        rng = random.Random(_seed(req.prompt, model.model_id))
        premium = model.quality_tier.value == "high"

        # Cheap models write less, and get terser as prompts get longer.
        prompt_words = max(len(req.prompt.split()), 1)
        base = 90 if premium else 32
        length_penalty = 0 if premium else min(20, prompt_words // 40)
        n_words = max(8, base - length_penalty + rng.randint(-6, 6))

        opener = (
            "Here is a complete response addressing "
            if premium
            else "Short answer regarding "
        )
        body = " ".join(rng.choice(_FILLER) for _ in range(n_words))
        text = f"{opener}{req.prompt.strip()[:60]}: {body}."

        if self.simulate_latency:
            jitter = rng.uniform(0.75, 1.25)
            await asyncio.sleep(model.avg_latency_ms / 1000.0 * jitter)

        # ~1.3 tokens/word is a decent stand-in for real tokenizers.
        return RawCompletion(
            text=text,
            input_tokens=int(prompt_words * 1.3) + 8,
            output_tokens=int(len(text.split()) * 1.3),
            stop_reason="end_turn",
            raw={"deterministic": True},
        )
