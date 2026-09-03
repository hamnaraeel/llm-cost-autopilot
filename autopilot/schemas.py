"""Standardized request/response objects shared by every provider.

The whole point of this module is that nothing downstream -- the router, the
verifier, the logger, the dashboard -- ever sees a provider-specific payload.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QualityTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ComplexityTier(int, Enum):
    """Tier 1 simple, Tier 2 moderate, Tier 3 complex. See Phase 2."""

    SIMPLE = 1
    MODERATE = 2
    COMPLEX = 3


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class LLMRequest:
    """A provider-agnostic completion request."""

    prompt: str
    system: str | None = None
    max_tokens: int = 1024
    temperature: float = 0.0
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """The single shape every call returns, regardless of provider."""

    text: str
    model_key: str          # registry key, e.g. "claude-haiku-4-5"
    model_id: str           # provider-native id, e.g. "claude-haiku-4-5"
    provider: str
    usage: Usage
    latency_ms: float
    cost_usd: float
    request_id: str
    created_at: float = field(default_factory=time.time)
    stop_reason: str | None = None
    raw: dict[str, Any] | None = field(default=None, repr=False)

    @property
    def cost_per_1k_output(self) -> float:
        if self.usage.output_tokens == 0:
            return 0.0
        return self.cost_usd / self.usage.output_tokens * 1000

    def to_row(self) -> dict[str, Any]:
        """Flat dict for SQLite / JSONL logging (Phase 4)."""
        return {
            "request_id": self.request_id,
            "created_at": self.created_at,
            "provider": self.provider,
            "model_key": self.model_key,
            "model_id": self.model_id,
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
            "total_tokens": self.usage.total_tokens,
            "latency_ms": round(self.latency_ms, 2),
            "cost_usd": self.cost_usd,
            "stop_reason": self.stop_reason,
        }


@dataclass
class RawCompletion:
    """What a Provider hands back before cost/latency are attached."""

    text: str
    input_tokens: int
    output_tokens: int
    stop_reason: str | None = None
    raw: dict[str, Any] | None = None


class ProviderError(RuntimeError):
    """Any provider-side failure, normalized."""

    def __init__(self, provider: str, message: str, *, retryable: bool = False):
        super().__init__(f"[{provider}] {message}")
        self.provider = provider
        self.retryable = retryable
