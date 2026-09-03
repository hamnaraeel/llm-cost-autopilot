"""Local inference via the Ollama daemon.

Ollama returns real token counts (`prompt_eval_count` / `eval_count`), so the
cost model stays honest even though the marginal dollar cost is zero.
"""

from __future__ import annotations

import httpx

from autopilot import config
from autopilot.providers.base import Provider
from autopilot.registry import ModelConfig
from autopilot.schemas import LLMRequest, ProviderError, RawCompletion

_TIMEOUT = httpx.Timeout(connect=5.0, read=300.0, write=30.0, pool=5.0)


class OllamaProvider(Provider):
    name = "ollama"

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=_TIMEOUT)

    @classmethod
    def available(cls) -> bool:
        try:
            r = httpx.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False

    @classmethod
    def unavailable_reason(cls) -> str | None:
        if cls.available():
            return None
        return f"ollama daemon not reachable at {config.OLLAMA_BASE_URL} (try `ollama serve`)"

    async def complete(self, req: LLMRequest, model: ModelConfig) -> RawCompletion:
        payload: dict = {
            "model": model.model_id,
            "prompt": req.prompt,
            "stream": False,
            "options": {
                "temperature": req.temperature,
                "num_predict": min(req.max_tokens, model.max_output_tokens),
            },
        }
        if req.system:
            payload["system"] = req.system

        try:
            r = await self._client.post("/api/generate", json=payload)
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise ProviderError(
                self.name,
                f"{e.response.status_code}: {e.response.text[:300]}",
                retryable=e.response.status_code >= 500,
            ) from e
        except httpx.HTTPError as e:
            raise ProviderError(self.name, str(e), retryable=True) from e

        d = r.json()
        return RawCompletion(
            text=d.get("response", ""),
            # Ollama omits these fields on some cached/empty responses.
            input_tokens=int(d.get("prompt_eval_count") or 0),
            output_tokens=int(d.get("eval_count") or 0),
            stop_reason=d.get("done_reason"),
            raw={"total_duration_ns": d.get("total_duration")},
        )

    async def aclose(self) -> None:
        await self._client.aclose()
