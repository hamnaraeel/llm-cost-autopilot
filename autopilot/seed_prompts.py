"""The shared pool of diverse, hand-labeled example prompts -- one source so
scripts/load_test.py and the API's first-boot auto-seed (api/main.py) can't
drift apart into two different definitions of "representative traffic."
"""

from __future__ import annotations

import json

import yaml

from autopilot import config

PROMPT_FILES = [
    config.DATA_DIR / "prompts" / "labeled_tier1.jsonl",
    config.DATA_DIR / "prompts" / "labeled_tier2.jsonl",
    config.DATA_DIR / "prompts" / "labeled_tier3.jsonl",
]
BASELINE_FILE = config.DATA_DIR / "prompts" / "baseline.yaml"


def load_pool() -> list[dict]:
    pool = []
    for path in PROMPT_FILES:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                row = json.loads(line)
                pool.append({"text": row["text"], "task": row["task"]})
    if BASELINE_FILE.exists():
        for p in yaml.safe_load(BASELINE_FILE.read_text())["prompts"]:
            pool.append({"text": p["text"], "task": p["task"]})
    return pool


__all__ = ["load_pool"]
