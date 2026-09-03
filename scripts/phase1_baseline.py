#!/usr/bin/env python
"""Phase 1, step 3 -- send the same 10 prompts to every available model.

Produces three artifacts:
  data/baseline/calls.jsonl        one row per (prompt, model) call, with output text
  data/baseline/summary.json       per-model aggregates
  config/measured_latency.yaml     measured p50 latency, layered into the registry

Usage:
    python scripts/phase1_baseline.py                 # every available model
    python scripts/phase1_baseline.py --models llama3.2-1b-local mock-cheap
    python scripts/phase1_baseline.py --dry-run       # availability report only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from autopilot import config
from autopilot.client import send_request
from autopilot.providers import provider_available, unavailable_reason
from autopilot.registry import ModelConfig, ModelRegistry
from autopilot.schemas import LLMResponse, ProviderError

OUT_DIR = config.DATA_DIR / "baseline"
PROMPTS_PATH = config.DATA_DIR / "prompts" / "baseline.yaml"


def load_prompts() -> list[dict]:
    return yaml.safe_load(PROMPTS_PATH.read_text())["prompts"]


def availability_report(registry: ModelRegistry) -> tuple[list[ModelConfig], list[tuple[ModelConfig, str]]]:
    usable, blocked = [], []
    for m in registry:
        if provider_available(m.provider):
            usable.append(m)
        else:
            blocked.append((m, unavailable_reason(m.provider) or "unavailable"))
    return usable, blocked


def print_availability(usable, blocked) -> None:
    print("\n=== Model availability ===")
    for m in usable:
        price = "free (local)" if m.is_local else f"${m.input_cost_per_1m:.2f}/${m.output_cost_per_1m:.2f} per 1M"
        flag = "" if m.pricing_verified else "  [pricing UNVERIFIED]"
        print(f"  [ok]   {m.key:<20} {m.provider:<10} {m.quality_tier.value:<6} {price}{flag}")
    for m, why in blocked:
        print(f"  [skip] {m.key:<20} {m.provider:<10} -- {why}")
    print()


async def run_one(prompt: dict, model: ModelConfig, max_tokens: int) -> dict:
    started = time.time()
    try:
        resp: LLMResponse = await send_request(
            prompt["text"], model, max_tokens=max_tokens
        )
    except ProviderError as e:
        return {
            "prompt_id": prompt["id"], "tier": prompt["tier"], "task": prompt["task"],
            "model_key": model.key, "provider": model.provider,
            "ok": False, "error": str(e), "started_at": started,
        }
    row = resp.to_row()
    row.update(
        prompt_id=prompt["id"], tier=prompt["tier"], task=prompt["task"],
        ok=True, output_text=resp.text, started_at=started,
    )
    return row


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", help="registry keys; default = all available")
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    registry = ModelRegistry.load()
    usable, blocked = availability_report(registry)
    print_availability(usable, blocked)

    if args.dry_run:
        return 0

    targets = [registry[k] for k in args.models] if args.models else usable
    unusable = [m.key for m in targets if not provider_available(m.provider)]
    if unusable:
        print(f"error: requested models are not available: {unusable}")
        return 1
    if not targets:
        print("error: no available models. Start ollama, set an API key, or enable the mock provider.")
        return 1

    prompts = load_prompts()
    print(f"Running {len(prompts)} prompts x {len(targets)} models = "
          f"{len(prompts) * len(targets)} calls (concurrency {args.concurrency})\n")

    sem = asyncio.Semaphore(args.concurrency)
    done = 0
    total = len(prompts) * len(targets)

    async def guarded(p, m):
        nonlocal done
        async with sem:
            row = await run_one(p, m, args.max_tokens)
        done += 1
        status = "ok " if row["ok"] else "ERR"
        lat = f'{row.get("latency_ms", 0):7.0f}ms' if row["ok"] else "        "
        # flush: this script is usually run redirected to a log.
        print(f"  [{done:>3}/{total}] {status} {row['model_key']:<20} {row['prompt_id']} {lat}", flush=True)
        return row

    rows = await asyncio.gather(*(guarded(p, m) for m in targets for p in prompts))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "calls.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # ---- aggregate --------------------------------------------------------
    summary: dict[str, dict] = {}
    for m in targets:
        mrows = [r for r in rows if r["model_key"] == m.key]
        ok = [r for r in mrows if r["ok"]]
        if not ok:
            summary[m.key] = {"calls": len(mrows), "ok": 0, "errors": len(mrows)}
            continue
        lats = sorted(r["latency_ms"] for r in ok)
        summary[m.key] = {
            "display_name": m.display_name,
            "provider": m.provider,
            "quality_tier": m.quality_tier.value,
            "calls": len(mrows),
            "ok": len(ok),
            "errors": len(mrows) - len(ok),
            "total_cost_usd": round(sum(r["cost_usd"] for r in ok), 6),
            "avg_cost_usd": round(sum(r["cost_usd"] for r in ok) / len(ok), 8),
            "p50_latency_ms": round(statistics.median(lats), 1),
            "p95_latency_ms": round(lats[max(0, int(len(lats) * 0.95) - 1)], 1),
            "avg_input_tokens": round(sum(r["input_tokens"] for r in ok) / len(ok), 1),
            "avg_output_tokens": round(sum(r["output_tokens"] for r in ok) / len(ok), 1),
        }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---- feed measured latency back into the registry ---------------------
    measured = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": "Written by scripts/phase1_baseline.py; overrides avg_latency_ms in models.yaml.",
        "p50_latency_ms": {k: v["p50_latency_ms"] for k, v in summary.items() if v.get("ok")},
    }
    config.MEASURED_LATENCY_YAML.write_text(yaml.safe_dump(measured, sort_keys=False))

    # ---- report -----------------------------------------------------------
    print("\n=== Baseline summary ===")
    hdr = f"{'model':<20} {'tier':<7} {'ok':>5} {'p50 ms':>9} {'p95 ms':>9} {'$/call':>11} {'$/10 calls':>11}"
    print(hdr)
    print("-" * len(hdr))
    for key, s in sorted(summary.items(), key=lambda kv: kv[1].get("avg_cost_usd", 0)):
        if not s.get("ok"):
            print(f"{key:<20} {'-':<7} {0:>5} {'':>9} {'':>9}   all calls failed")
            continue
        print(f"{key:<20} {s['quality_tier']:<7} {s['ok']:>5} {s['p50_latency_ms']:>9.0f} "
              f"{s['p95_latency_ms']:>9.0f} {s['avg_cost_usd']:>11.6f} {s['total_cost_usd']:>11.6f}")

    priced = {k: s for k, s in summary.items() if s.get("ok") and s["total_cost_usd"] > 0}
    if len(priced) > 1:
        most = max(priced.values(), key=lambda s: s["total_cost_usd"])
        least = min(priced.values(), key=lambda s: s["total_cost_usd"])
        if least["total_cost_usd"] > 0:
            ratio = most["total_cost_usd"] / least["total_cost_usd"]
            print(f"\nSpread across priced models: {ratio:.1f}x "
                  f"({most['display_name']} vs {least['display_name']}) on identical prompts.")
    print(f"\nArtifacts: {OUT_DIR/'calls.jsonl'}, {OUT_DIR/'summary.json'}, {config.MEASURED_LATENCY_YAML}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
