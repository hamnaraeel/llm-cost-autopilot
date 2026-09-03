#!/usr/bin/env python
"""Phase 2, step 3 -- train the complexity classifier on the labeled dataset.

Loads data/prompts/labeled_tier{1,2,3}.jsonl, fits a RandomForest on the
interpretable features in autopilot/classifier/features.py, and reports
5-fold cross-validated accuracy. The trained model is written to
models/classifier.joblib.

Usage:
    python scripts/train_classifier.py
    python scripts/train_classifier.py --extra data/prompts/labeled_failures.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from autopilot import config
from autopilot.classifier.features import feature_vector
from autopilot.classifier.model import ComplexityClassifier

LABELED_DIR = config.DATA_DIR / "prompts"
DEFAULT_FILES = [
    LABELED_DIR / "labeled_tier1.jsonl",
    LABELED_DIR / "labeled_tier2.jsonl",
    LABELED_DIR / "labeled_tier3.jsonl",
]


def load_examples(paths: list[Path]) -> tuple[list[str], list[int]]:
    prompts, tiers = [], []
    for path in paths:
        if not path.exists():
            print(f"warning: {path} not found, skipping")
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompts.append(row["text"])
            tiers.append(int(row["tier"]))
    return prompts, tiers


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", nargs="*", default=[], help="additional labeled jsonl files")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    paths = list(DEFAULT_FILES) + [Path(p) for p in args.extra]
    prompts, tiers = load_examples(paths)
    if len(prompts) < 20:
        print(f"error: only {len(prompts)} labeled examples found; need at least 20")
        return 1

    counts = {t: tiers.count(t) for t in sorted(set(tiers))}
    print(f"Loaded {len(prompts)} labeled examples: {counts}")

    # With ~166 hand-labeled examples, a single 80/20 split is noisy enough
    # that one flipped example swings accuracy by 3 points. 5-fold stratified
    # cross-validation uses every example as held-out exactly once, which is
    # the more honest "held-out accuracy" at this dataset size.
    X = np.array([feature_vector(p) for p in prompts])
    y = np.array(tiers)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    cv_model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=args.seed)
    preds = cross_val_predict(cv_model, X, y, cv=skf)
    acc = float((preds == y).mean())

    print(f"=== 5-fold cross-validated evaluation (n={len(prompts)}) ===")
    print(classification_report(y, preds, digits=3))
    print("Confusion matrix (rows=true, cols=pred), tiers 1/2/3:")
    print(confusion_matrix(y, preds, labels=[1, 2, 3]))
    print(f"\nCross-validated accuracy: {acc:.1%}")
    if acc < 0.80:
        print("warning: below the 80% target -- consider adding labeled examples "
              "for the tiers with the most confusion.")

    # Refit on all data before saving -- the held-out split was only for
    # honest evaluation, the shipped model should see every labeled example.
    final = ComplexityClassifier.fit(prompts, tiers, random_state=args.seed)
    path = final.save()
    print(f"\nSaved model to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
