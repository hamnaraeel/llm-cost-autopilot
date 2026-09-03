"""Provider interface. One method to implement, one shape to return."""

from __future__ import annotations

from abc import ABC, abstractmethod

from autopilot.registry import ModelConfig
from autopilot.schemas import LLMRequest, RawCompletion


class Provider(ABC):
    name: str

    @abstractmethod
    async def complete(self, req: LLMRequest, model: ModelConfig) -> RawCompletion:
        """Issue one completion. Raise ProviderError on failure."""

    @classmethod
    def available(cls) -> bool:
        """True when this provider can actually be called (key set, daemon up)."""
        return True

    @classmethod
    def unavailable_reason(cls) -> str | None:
        return None
