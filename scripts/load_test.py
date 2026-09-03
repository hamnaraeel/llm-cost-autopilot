#!/usr/bin/env python
"""Phase 6, step 1 -- push a realistic volume of diverse prompts through the
full pipeline (route -> send -> verify -> log) to populate the cost
dashboard with real numbers instead of a handful of manual test calls.

Pulls its prompt pool from the labeled dataset (tier 1/2/3) plus the Phase 1
baseline set -- ~176 hand-written, genuinely varied prompts spanning every
task type the classifier was trained on. To reach the 500-1000 request
volume called for in the spec without hand-writing that many prompts, the
pool is sampled with replacement; duplicates get fresh request IDs and (for
non-deterministic providers) fresh responses, so they still exercise the
pipeline honestly.

Usage:
    python scripts/load_test.py                    # 500 requests, offline (mock only)
    python scripts/load_test.py --n 1000
    python scripts/load_test.py --online            # also allow ollama if running
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from autopilot import config
from autopilot.pipeline import run_request
from autopilot.registry import ModelRegistry
from autopilot.schemas import ProviderError
from autopilot.store import LogStore

PROMPT_FILES = [
    config.DATA_DIR / "prompts" / "labeled_tier1.jsonl",
    config.DATA_DIR / "prompts" / "labeled_tier2.jsonl",
    config.DATA_DIR / "prompts" / "labeled_tier3.jsonl",
]
BASELINE_FILE = config.DATA_DIR / "prompts" / "baseline.yaml"


def load_pool() -> list[dict]:
    pool = []
    for path in PROMPT_FILES:
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                row = json.loads(line)
                pool.append({"text": row["text"], "task": row["task"]})
    for p in yaml.safe_load(BASELINE_FILE.read_text())["prompts"]:
        pool.append({"text": p["text"], "task": p["task"]})
    return pool


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="number of requests to send")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--online", action="store_true",
        help="allow real ollama calls; default forces offline (mock provider only) "
             "so a 500+ request run finishes in seconds instead of hours",
    )
    ap.add_argument("--fresh-db", action="store_true", help="wipe the log DB before running")
    args = ap.parse_args()

    if not args.online:
        # Ollama daemon may be running but slow (CPU inference); route
        # everything to the fast, deterministic mock provider for the load
        # test unless the caller explicitly asks for real local calls.
        config.OLLAMA_BASE_URL = "http://127.0.0.1:1"

    if args.fresh_db and config.DB_PATH.exists():
        config.DB_PATH.unlink()

    registry = ModelRegistry.load()
    store = LogStore()
    pool = load_pool()
    rng = random.Random(args.seed)
    batch = [rng.choice(pool) for _ in range(args.n)]

    print(f"Prompt pool: {len(pool)} unique prompts. Sending {args.n} requests "
          f"(concurrency {args.concurrency}, {'online' if args.online else 'offline/mock'}).\n")

    sem = asyncio.Semaphore(args.concurrency)
    done = 0
    errors = 0
    started = time.time()

    async def one(item: dict):
        nonlocal done, errors
        async with sem:
            try:
                await run_request(item["text"], registry, task=item["task"], store=store)
            except ProviderError as e:
                errors += 1
                print(f"  ERROR: {e}")
        done += 1
        if done % 50 == 0 or done == args.n:
            elapsed = time.time() - started
            print(f"  [{done:>4}/{args.n}] {elapsed:6.1f}s elapsed", flush=True)

    await asyncio.gather(*(one(item) for item in batch))

    elapsed = time.time() - started
    print(f"\nDone: {done - errors} ok, {errors} errors in {elapsed:.1f}s.")

    summary = store.summary()
    print("\n=== Cost summary ===")
    print(f"Total requests:      {summary['requests']}")
    print(f"Actual cost:         ${summary['total_cost_usd']:.4f}")
    print(f"Baseline cost (all-escalation-model): ${summary['baseline_cost_usd']:.4f}")
    print(f"Savings:             ${summary['savings_usd']:.4f}  ({summary['savings_pct']:.1f}%)")
    print(f"Escalation rate:     {summary['escalation_rate']:.1f}%")
    if summary["avg_agreement"] is not None:
        print(f"Avg agreement score: {summary['avg_agreement']:.2f}")
    print("\nModel distribution:")
    for k, v in sorted(summary["model_distribution"].items(), key=lambda kv: -kv[1]):
        print(f"  {k:<20} {v:>5}  ({v / summary['requests'] * 100:5.1f}%)")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
