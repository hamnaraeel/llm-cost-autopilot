"""Trained complexity classifier: fit, persist, and predict.

The classifier sits in the request path, so `predict` has to be fast and to
never throw -- a broken classifier should degrade to a safe default tier, not
take the router down. Training is a separate, offline step (see
scripts/train_classifier.py); this module only knows how to load and use the
artifact that step produces.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from autopilot import config
from autopilot.classifier.features import FEATURE_NAMES, feature_vector
from autopilot.schemas import ComplexityTier

MODEL_PATH = config.ARTIFACT_DIR / "classifier.joblib"

# Used whenever there is no trained model on disk (fresh checkout, load
# failure) or the classifier's own confidence is too low to trust. Moderate
# is the safe middle: never as expensive as always-escalate, never as risky
# as always-cheap.
FALLBACK_TIER = ComplexityTier.MODERATE


@dataclass
class Prediction:
    tier: ComplexityTier
    confidence: float
    proba: dict[int, float]
    used_fallback: bool = False


class ComplexityClassifier:
    """Wraps a fitted sklearn model plus the feature order it expects."""

    def __init__(self, sk_model: RandomForestClassifier, feature_names: list[str]):
        self._model = sk_model
        self._feature_names = feature_names

    # ---- inference --------------------------------------------------------
    def predict(self, prompt: str) -> Prediction:
        try:
            x = np.array([feature_vector(prompt)])
            proba = self._model.predict_proba(x)[0]
            classes = self._model.classes_
            proba_by_class = {int(c): float(p) for c, p in zip(classes, proba)}
            best_idx = int(np.argmax(proba))
            tier = ComplexityTier(int(classes[best_idx]))
            confidence = float(proba[best_idx])
            return Prediction(tier=tier, confidence=confidence, proba=proba_by_class)
        except Exception:  # noqa: BLE001 - classifier must never take the router down
            return Prediction(
                tier=FALLBACK_TIER,
                confidence=0.0,
                proba={},
                used_fallback=True,
            )

    # ---- persistence --------------------------------------------------------
    def save(self, path: Path | None = None) -> Path:
        path = path or MODEL_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self._model, "feature_names": self._feature_names}, path)
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "ComplexityClassifier":
        path = path or MODEL_PATH
        blob = joblib.load(path)
        return cls(blob["model"], blob["feature_names"])

    @classmethod
    def fit(
        cls,
        prompts: list[str],
        tiers: list[int],
        *,
        n_estimators: int = 200,
        max_depth: int | None = 8,
        random_state: int = 13,
    ) -> "ComplexityClassifier":
        X = np.array([feature_vector(p) for p in prompts])
        y = np.array(tiers)
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
        model.fit(X, y)
        return cls(model, list(FEATURE_NAMES))


_singleton: ComplexityClassifier | None = None


def get_classifier() -> ComplexityClassifier:
    """Process-wide cached classifier; loads the trained artifact once."""
    global _singleton
    if _singleton is None:
        _singleton = ComplexityClassifier.load()
    return _singleton


def reload_classifier() -> ComplexityClassifier:
    """Force a reload from disk -- used after retraining (Phase 3 feedback loop)."""
    global _singleton
    _singleton = ComplexityClassifier.load()
    return _singleton
