import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from autopilot.registry import ModelRegistry
from autopilot.verifier import score_agreement, verify


def test_identical_text_scores_perfect_agreement():
    assert score_agreement("the sky is blue", "the sky is blue") == 1.0


def test_disjoint_text_scores_low_agreement():
    # No shared vocabulary at all -- only the length-ratio term contributes.
    assert score_agreement("apples oranges bananas", "quantum entropy lattice") == pytest.approx(0.3)


def test_empty_vs_nonempty_scores_zero():
    assert score_agreement("", "some real answer here") == 0.0


@pytest.fixture(scope="module")
def registry():
    return ModelRegistry.load()


@pytest.mark.asyncio
async def test_verify_skips_when_primary_is_escalation_tier(registry):
    from autopilot.client import send_request

    primary = await send_request("write a haiku about the sea", registry["mock-premium"])
    result = await verify("write a haiku about the sea", primary, registry)
    assert result.escalated is False
    assert result.verifier_response is None
