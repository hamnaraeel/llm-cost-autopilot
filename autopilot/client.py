"""The unified model interface.

    response = await send_request("Summarize this.", registry["claude-haiku-4-5"])

Everything above this layer -- router, verifier, API, dashboard -- deals only
in `LLMRequest` / `LLMResponse` and never touches a provider SDK.
"""

from __future__ import annotations

import asyncio
import logging
import time

from autopilot.providers import get_provider, provider_available, unavailable_reason
from autopilot.registry import ModelConfig, ModelRegistry
from autopilot.schemas import LLMRequest, LLMResponse, ProviderError, Usage

log = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 2
_BACKOFF_BASE_S = 0.75


def _estimate_tokens(text: str) -> int:
    """~4 chars/token. Only used when a provider omits usage counts."""
    return max(1, len(text) // 4)


async def send_request(
    prompt: str | LLMRequest,
    model: ModelConfig,
    *,
    system: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> LLMResponse:
    """Send one completion to one model and return a standardized response.

    Latency is measured around the provider call only, so it reflects the
    model rather than our own bookkeeping. Retries are attributed to the
    total, because that is what a caller actually waits for.
    """
    req = (
        prompt
        if isinstance(prompt, LLMRequest)
        else LLMRequest(
            prompt=prompt, system=system, max_tokens=max_tokens, temperature=temperature
        )
    )

    if not provider_available(model.provider):
        raise ProviderError(
            model.provider,
            f"provider unavailable for model {model.key!r}: {unavailable_reason(model.provider)}",
        )

    provider = get_provider(model.provider)
    started = time.perf_counter()
    last_err: ProviderError | None = None

    for attempt in range(max_retries + 1):
        try:
            raw = await provider.complete(req, model)
            break
        except ProviderError as e:
            last_err = e
            if not e.retryable or attempt == max_retries:
                raise
            delay = _BACKOFF_BASE_S * (2**attempt)
            log.warning(
                "retrying %s in %.2fs (attempt %d/%d): %s",
                model.key, delay, attempt + 1, max_retries, e,
            )
            await asyncio.sleep(delay)
    else:  # pragma: no cover - loop always breaks or raises
        raise last_err or ProviderError(model.provider, "exhausted retries")

    latency_ms = (time.perf_counter() - started) * 1000

    input_tokens = raw.input_tokens or _estimate_tokens(req.prompt + (req.system or ""))
    output_tokens = raw.output_tokens or _estimate_tokens(raw.text)
    usage = Usage(input_tokens=input_tokens, output_tokens=output_tokens)

    return LLMResponse(
        text=raw.text,
        model_key=model.key,
        model_id=model.model_id,
        provider=model.provider,
        usage=usage,
        latency_ms=latency_ms,
        cost_usd=model.cost_for(usage),
        request_id=req.request_id,
        stop_reason=raw.stop_reason,
        raw=raw.raw,
    )


def send_request_sync(prompt: str | LLMRequest, model: ModelConfig, **kwargs) -> LLMResponse:
    """Blocking convenience wrapper for scripts and notebooks."""
    return asyncio.run(send_request(prompt, model, **kwargs))


async def fan_out(
    prompt: str | LLMRequest,
    models: list[ModelConfig],
    *,
    concurrency: int = 4,
    **kwargs,
) -> list[LLMResponse | ProviderError]:
    """Send the same prompt to several models. Failures are returned, not raised."""
    sem = asyncio.Semaphore(concurrency)

    async def one(m: ModelConfig):
        async with sem:
            try:
                return await send_request(prompt, m, **kwargs)
            except ProviderError as e:
                return e

    return await asyncio.gather(*(one(m) for m in models))


__all__ = ["send_request", "send_request_sync", "fan_out", "ModelRegistry"]
