from __future__ import annotations

from autopilot import config
from autopilot.providers.base import Provider
from autopilot.registry import ModelConfig
from autopilot.schemas import LLMRequest, ProviderError, RawCompletion


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self) -> None:
        from anthropic import AsyncAnthropic

        self._client = AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)

    @classmethod
    def available(cls) -> bool:
        return bool(config.ANTHROPIC_API_KEY)

    @classmethod
    def unavailable_reason(cls) -> str | None:
        return None if cls.available() else "ANTHROPIC_API_KEY is not set"

    async def complete(self, req: LLMRequest, model: ModelConfig) -> RawCompletion:
        import anthropic

        kwargs: dict = {
            "model": model.model_id,
            "max_tokens": min(req.max_tokens, model.max_output_tokens),
            "messages": [{"role": "user", "content": req.prompt}],
        }
        if req.system:
            kwargs["system"] = req.system

        try:
            msg = await self._client.messages.create(**kwargs)
        except anthropic.RateLimitError as e:
            raise ProviderError(self.name, str(e), retryable=True) from e
        except anthropic.APIConnectionError as e:
            raise ProviderError(self.name, str(e), retryable=True) from e
        except anthropic.APIStatusError as e:
            raise ProviderError(
                self.name, f"{e.status_code}: {e}", retryable=e.status_code >= 500
            ) from e

        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        return RawCompletion(
            text=text,
            input_tokens=msg.usage.input_tokens,
            output_tokens=msg.usage.output_tokens,
            stop_reason=msg.stop_reason,
            raw={"id": msg.id},
        )
