"""Groq: hosted inference for open models (Llama, etc.) on Groq's LPU hardware.

Groq's API is OpenAI-compatible, so this reuses the `openai` SDK pointed at
Groq's base URL instead of writing a bespoke HTTP client. It exists as a
fast, free-tier real cloud option -- unlike local Ollama, calls typically
return in under a couple of seconds even for a 70B model.
"""

from __future__ import annotations

from autopilot import config
from autopilot.providers.base import Provider
from autopilot.registry import ModelConfig
from autopilot.schemas import LLMRequest, ProviderError, RawCompletion

_BASE_URL = "https://api.groq.com/openai/v1"


class GroqProvider(Provider):
    name = "groq"

    def __init__(self, api_key: str | None = None) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key or config.GROQ_API_KEY, base_url=_BASE_URL)

    @classmethod
    def available(cls) -> bool:
        return bool(config.GROQ_API_KEY)

    @classmethod
    def unavailable_reason(cls) -> str | None:
        return None if cls.available() else "GROQ_API_KEY is not set"

    async def complete(self, req: LLMRequest, model: ModelConfig) -> RawCompletion:
        import openai

        messages: list[dict] = []
        if req.system:
            messages.append({"role": "system", "content": req.system})
        messages.append({"role": "user", "content": req.prompt})

        try:
            resp = await self._client.chat.completions.create(
                model=model.model_id,
                messages=messages,
                max_tokens=min(req.max_tokens, model.max_output_tokens),
                temperature=req.temperature,
            )
        except openai.RateLimitError as e:
            raise ProviderError(self.name, str(e), retryable=True) from e
        except openai.APIConnectionError as e:
            raise ProviderError(self.name, str(e), retryable=True) from e
        except openai.APIStatusError as e:
            raise ProviderError(
                self.name, f"{e.status_code}: {e}", retryable=e.status_code >= 500
            ) from e

        choice = resp.choices[0]
        usage = resp.usage
        return RawCompletion(
            text=choice.message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            stop_reason=choice.finish_reason,
            raw={"id": resp.id},
        )
