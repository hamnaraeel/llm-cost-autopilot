from __future__ import annotations

from autopilot import config
from autopilot.providers.base import Provider
from autopilot.registry import ModelConfig
from autopilot.schemas import LLMRequest, ProviderError, RawCompletion


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(self, api_key: str | None = None) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key or config.OPENAI_API_KEY)

    @classmethod
    def available(cls) -> bool:
        return bool(config.OPENAI_API_KEY)

    @classmethod
    def unavailable_reason(cls) -> str | None:
        return None if cls.available() else "OPENAI_API_KEY is not set"

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
                # `max_completion_tokens` is the forward-compatible field;
                # `max_tokens` is rejected by newer OpenAI model families.
                max_completion_tokens=min(req.max_tokens, model.max_output_tokens),
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
