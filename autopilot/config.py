"""Environment + path configuration. Loads .env once, at import."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # dotenv is optional at runtime
    def load_dotenv(*_a, **_kw):  # type: ignore[misc]
        return False

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "models"

load_dotenv(ROOT / ".env")

MODELS_YAML = CONFIG_DIR / "models.yaml"
ROUTING_YAML = CONFIG_DIR / "routing.yaml"
MEASURED_LATENCY_YAML = CONFIG_DIR / "measured_latency.yaml"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or None
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") or None
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
ENABLE_MOCK = os.getenv("AUTOPILOT_ENABLE_MOCK", "true").lower() in {"1", "true", "yes"}

DB_PATH = ROOT / os.getenv("AUTOPILOT_DB_PATH", "data/autopilot.db")

DATA_DIR.mkdir(exist_ok=True)
ARTIFACT_DIR.mkdir(exist_ok=True)
