"""Bring-your-own-key: a caller-supplied provider key must unlock routing to
that provider even when the server has no key of its own, and must never be
cached into the shared provider singleton.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autopilot.providers import get_provider, get_provider_instance, provider_usable
from autopilot.registry import ModelRegistry
from autopilot.router import first_available


def test_provider_usable_false_with_no_server_key_and_no_override():
    assert provider_usable("anthropic", None) is False


def test_provider_usable_true_when_caller_supplies_a_key():
    assert provider_usable("anthropic", {"anthropic": "sk-test-123"}) is True


def test_provider_usable_ignores_keys_for_other_providers():
    assert provider_usable("anthropic", {"openai": "sk-test-123"}) is False


def test_get_provider_instance_with_key_is_not_the_cached_singleton():
    default = get_provider("anthropic")
    byok = get_provider_instance("anthropic", "sk-caller-key")
    assert byok is not default


def test_get_provider_instance_without_key_returns_cached_singleton():
    default = get_provider("anthropic")
    same = get_provider_instance("anthropic", None)
    assert same is default


def test_first_available_unlocked_by_caller_key():
    registry = ModelRegistry.load()
    # claude-opus-5 has no server-side key in this environment, so it's
    # normally skipped -- a caller-supplied key should unlock it.
    without_key = first_available(["claude-opus-5"], registry)
    with_key = first_available(["claude-opus-5"], registry, {"anthropic": "sk-test"})
    assert without_key is None
    assert with_key is not None
    assert with_key.key == "claude-opus-5"
