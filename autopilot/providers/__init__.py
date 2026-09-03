"""Provider lookup. Instances are cached -- clients hold connection pools."""

from __future__ import annotations

from functools import lru_cache

from autopilot import config
from autopilot.providers.base import Provider

_CLASSES: dict[str, type[Provider]] = {}


def _classes() -> dict[str, type[Provider]]:
    if not _CLASSES:
        from autopilot.providers.anthropic_provider import AnthropicProvider
        from autopilot.providers.mock_provider import MockProvider
        from autopilot.providers.ollama_provider import OllamaProvider
        from autopilot.providers.openai_provider import OpenAIProvider

        _CLASSES.update(
            anthropic=AnthropicProvider,
            openai=OpenAIProvider,
            ollama=OllamaProvider,
            mock=MockProvider,
        )
    return _CLASSES


@lru_cache(maxsize=None)
def get_provider(name: str) -> Provider:
    try:
        cls = _classes()[name]
    except KeyError:
        raise KeyError(
            f"unknown provider {name!r}; known: {sorted(_classes())}"
        ) from None
    return cls()


def provider_available(name: str) -> bool:
    cls = _classes().get(name)
    if cls is None:
        return False
    if name == "mock" and not config.ENABLE_MOCK:
        return False
    return cls.available()


def unavailable_reason(name: str) -> str | None:
    cls = _classes().get(name)
    if cls is None:
        return f"unknown provider {name!r}"
    if name == "mock" and not config.ENABLE_MOCK:
        return "AUTOPILOT_ENABLE_MOCK is false"
    return cls.unavailable_reason()


__all__ = ["Provider", "get_provider", "provider_available", "unavailable_reason"]
