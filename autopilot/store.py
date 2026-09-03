"""Phase 4, step 1 -- the request log: SQLite + one JSON blob per row.

Every field the dashboard aggregates on lives in its own column (fast
queries, easy to reason about); the full row is also kept as JSON so nothing
is ever lost to a column this module forgot to add -- that's the "full audit
trail per request" the project spec asks for.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

from autopilot import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    request_id           TEXT PRIMARY KEY,
    created_at            REAL NOT NULL,
    prompt_hash           TEXT NOT NULL,
    prompt_preview         TEXT NOT NULL,
    task                  TEXT,
    complexity_tier        INTEGER NOT NULL,
    classifier_confidence   REAL NOT NULL,
    routed_model_key        TEXT NOT NULL,
    routed_provider         TEXT NOT NULL,
    final_model_key         TEXT NOT NULL,
    final_provider          TEXT NOT NULL,
    escalated              INTEGER NOT NULL,
    agreement_score         REAL,
    input_tokens           INTEGER NOT NULL,
    output_tokens           INTEGER NOT NULL,
    cost_usd               REAL NOT NULL,
    baseline_cost_usd        REAL NOT NULL,
    latency_ms             REAL NOT NULL,
    note                  TEXT,
    raw_json               TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_requests_created_at ON requests(created_at);
CREATE INDEX IF NOT EXISTS idx_requests_model ON requests(final_model_key);
"""


def prompt_hash(prompt: str) -> str:
    """Short, stable identifier for a prompt without storing it verbatim."""
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


@dataclass
class RequestLog:
    request_id: str
    prompt: str
    task: str | None
    complexity_tier: int
    classifier_confidence: float
    routed_model_key: str
    routed_provider: str
    final_model_key: str
    final_provider: str
    escalated: bool
    input_tokens: int
    output_tokens: int
    cost_usd: float
    baseline_cost_usd: float
    latency_ms: float
    agreement_score: float | None = None
    note: str | None = None
    created_at: float = field(default_factory=time.time)

    def to_row(self) -> dict:
        d = asdict(self)
        d["prompt_hash"] = prompt_hash(self.prompt)
        d["prompt_preview"] = self.prompt[:200]
        del d["prompt"]
        return d


class LogStore:
    def __init__(self, path: Path | None = None):
        self.path = path or config.DB_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(SCHEMA)
            self._migrate(conn)

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Add columns introduced after a DB already existed on disk."""
        existing = {row[1] for row in conn.execute("PRAGMA table_info(requests)")}
        if "final_provider" not in existing:
            conn.execute("ALTER TABLE requests ADD COLUMN final_provider TEXT NOT NULL DEFAULT 'unknown'")

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---- writes -------------------------------------------------------
    def log(self, entry: RequestLog) -> None:
        row = entry.to_row()
        row["escalated"] = int(row["escalated"])
        row["raw_json"] = json.dumps(row, default=str)
        cols = ", ".join(row)
        placeholders = ", ".join(f":{c}" for c in row)
        with self._conn() as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO requests ({cols}) VALUES ({placeholders})", row
            )

    # ---- reads ----------------------------------------------------------
    def all_rows(self) -> list[dict]:
        with self._conn() as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM requests ORDER BY created_at")]

    def count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]

    def summary(self) -> dict:
        """The headline numbers the dashboard and /v1/stats both need."""
        rows = self.all_rows()
        n = len(rows)
        if n == 0:
            return {
                "requests": 0, "total_cost_usd": 0.0, "baseline_cost_usd": 0.0,
                "savings_usd": 0.0, "savings_pct": 0.0, "escalation_rate": 0.0,
                "avg_agreement": None, "model_distribution": {}, "tier_distribution": {},
                "mock_fallback_count": 0,
            }
        total_cost = sum(r["cost_usd"] for r in rows)
        baseline_cost = sum(r["baseline_cost_usd"] for r in rows)
        savings = baseline_cost - total_cost
        escalations = sum(r["escalated"] for r in rows)
        agreements = [r["agreement_score"] for r in rows if r["agreement_score"] is not None]
        # Real providers were unreachable and the request fell all the way
        # through to the offline placeholder -- should be rare-to-never;
        # surfaced explicitly rather than left to blend into "final model".
        mock_fallback_count = sum(1 for r in rows if r["final_provider"] == "mock")

        model_dist: dict[str, int] = {}
        tier_dist: dict[int, int] = {}
        for r in rows:
            model_dist[r["final_model_key"]] = model_dist.get(r["final_model_key"], 0) + 1
            tier_dist[r["complexity_tier"]] = tier_dist.get(r["complexity_tier"], 0) + 1

        return {
            "requests": n,
            "total_cost_usd": round(total_cost, 6),
            "baseline_cost_usd": round(baseline_cost, 6),
            "savings_usd": round(savings, 6),
            "savings_pct": round(savings / baseline_cost * 100, 2) if baseline_cost > 0 else 0.0,
            "escalation_rate": round(escalations / n * 100, 2),
            "avg_agreement": round(sum(agreements) / len(agreements), 4) if agreements else None,
            "model_distribution": model_dist,
            "tier_distribution": tier_dist,
            "mock_fallback_count": mock_fallback_count,
        }


_singleton: LogStore | None = None


def get_store() -> LogStore:
    global _singleton
    if _singleton is None:
        _singleton = LogStore()
    return _singleton


__all__ = ["LogStore", "RequestLog", "get_store", "prompt_hash"]
