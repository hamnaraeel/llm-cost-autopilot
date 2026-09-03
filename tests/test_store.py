import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from autopilot.store import LogStore, RequestLog


@pytest.fixture
def store(tmp_path):
    return LogStore(path=tmp_path / "test.db")


def _entry(**overrides) -> RequestLog:
    defaults = dict(
        request_id="req-1",
        prompt="summarize this text please",
        task="summarization",
        complexity_tier=1,
        classifier_confidence=0.9,
        routed_model_key="mock-cheap",
        routed_provider="mock",
        final_model_key="mock-cheap",
        final_provider="mock",
        escalated=False,
        input_tokens=10,
        output_tokens=20,
        cost_usd=0.001,
        baseline_cost_usd=0.01,
        latency_ms=120.0,
        agreement_score=0.8,
    )
    defaults.update(overrides)
    return RequestLog(**defaults)


def test_log_and_read_back(store):
    store.log(_entry())
    rows = store.all_rows()
    assert len(rows) == 1
    assert rows[0]["request_id"] == "req-1"
    assert rows[0]["prompt_preview"] == "summarize this text please"
    assert "prompt" not in rows[0]  # raw prompt text is not stored verbatim


def test_summary_computes_savings(store):
    store.log(_entry(request_id="r1", cost_usd=1.0, baseline_cost_usd=4.0))
    store.log(_entry(request_id="r2", cost_usd=1.0, baseline_cost_usd=4.0, escalated=True))
    s = store.summary()
    assert s["requests"] == 2
    assert s["total_cost_usd"] == pytest.approx(2.0)
    assert s["baseline_cost_usd"] == pytest.approx(8.0)
    assert s["savings_pct"] == pytest.approx(75.0)
    assert s["escalation_rate"] == pytest.approx(50.0)


def test_empty_summary_has_zeroed_fields(store):
    s = store.summary()
    assert s["requests"] == 0
    assert s["savings_pct"] == 0.0
    assert s["mock_fallback_count"] == 0


def test_summary_flags_mock_fallback_usage(store):
    store.log(_entry(request_id="r1", final_provider="groq"))
    store.log(_entry(request_id="r2", final_provider="mock"))
    s = store.summary()
    assert s["mock_fallback_count"] == 1
