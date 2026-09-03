"""Provider lookup. Instances are cached -- clients hold connection pools."""

from __future__ import annotations

from functools import lru_cache

from autopilot import config
from autopilot.providers.base import Provider

_CLASSES: dict[str, type[Provider]] = {}


def _classes() -> dict[str, type[Provider]]:
    if not _CLASSES:
        from autopilot.providers.anthropic_provider import AnthropicProvider
        from autopilot.providers.groq_provider import GroqProvider
        from autopilot.providers.mock_provider import MockProvider
        from autopilot.providers.ollama_provider import OllamaProvider
        from autopilot.providers.openai_provider import OpenAIProvider

        _CLASSES.update(
            anthropic=AnthropicProvider,
            openai=OpenAIProvider,
            groq=GroqProvider,
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


def get_provider_instance(name: str, api_key: str | None = None) -> Provider:
    """Like `get_provider`, but builds an uncached instance when `api_key` is
    given -- a caller-supplied (bring-your-own-key) request must never be
    cached under the shared singleton, or one user's key would leak into
    another user's requests.
    """
    if api_key is None:
        return get_provider(name)
    try:
        cls = _classes()[name]
    except KeyError:
        raise KeyError(
            f"unknown provider {name!r}; known: {sorted(_classes())}"
        ) from None
    return cls(api_key=api_key)


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


def provider_usable(name: str, api_keys: dict[str, str] | None = None) -> bool:
    """`provider_available`, but a caller-supplied key for `name` also counts
    -- a provider the server has no key for is still usable for a request
    that brings its own (see the X-*-Api-Key headers on /v1/chat/completions).
    """
    if api_keys and name in api_keys:
        return True
    return provider_available(name)


__all__ = [
    "Provider",
    "get_provider",
    "get_provider_instance",
    "provider_available",
    "provider_usable",
    "unavailable_reason",
]
