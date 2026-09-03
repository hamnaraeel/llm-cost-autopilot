# LLM Cost Autopilot

An intelligent routing layer that sits in front of multiple LLM providers, scores
how complex each incoming request actually is, sends it to the cheapest model
that can handle it, and automatically double-checks (and escalates) the ones it's
not sure about. On the current offline demo run it holds **42.9% cost reduction**
versus routing every request to the reference (highest-quality) model, with a
**19.6% escalation rate** and **86.1% cross-validated accuracy** on the complexity
classifier.

This document explains the concepts the project demonstrates, how each piece was
actually built, and the tech stack behind it — read it alongside the code, not
instead of it.

---

## 1. The core idea

Every LLM request has a complexity that isn't uniform:

- *"Extract the total from this invoice"* is mechanical — a cheap, fast model
  nails it every time.
- *"Weigh the tradeoffs of this incident response and recommend what the team
  lead should do"* needs real reasoning — only a top-tier model should be trusted
  with it.

Most systems send every request to one model, sized for the hardest case. That's
correct but wasteful: it pays frontier-model prices for reformatting a list. The
Autopilot instead **classifies first, routes second, verifies third** — the same
model-tiering pattern production LLM platforms use to control cost without
sacrificing the worst-case answer quality.

---

## 2. Concepts demonstrated

| Concept | Where it lives | What it shows |
|---|---|---|
| **Provider abstraction** | `autopilot/providers/`, `autopilot/client.py` | One request/response shape (`LLMRequest`/`LLMResponse`) hides Anthropic, OpenAI, Ollama, and a deterministic offline mock behind a single interface, so nothing downstream ever imports a provider SDK. |
| **Cost accounting as a single source of truth** | `autopilot/registry.py` | `ModelConfig.cost_for()` is the *only* place a dollar amount is computed from token counts — the router's decisions and the dashboard's headline number can never drift apart. |
| **Interpretable feature engineering** | `autopilot/classifier/features.py` | 20 hand-designed features (verb lexicons, format/constraint detectors, length signals) instead of an embedding, because a router needs to be able to explain *why* it called a prompt "simple." |
| **Supervised complexity classification** | `autopilot/classifier/model.py`, `scripts/train_classifier.py` | A `RandomForestClassifier` trained on 166 hand-labeled examples across 3 tiers, evaluated with 5-fold stratified cross-validation (the honest choice at this dataset size — a single 80/20 split is noisy enough that one flipped label swings accuracy by 3 points). |
| **Confidence-aware routing** | `autopilot/router.py` | The routing map (`config/routing.yaml`) is a plain YAML tier → model table with a fallback chain per tier; a low-confidence classification gets bumped up a tier rather than trusted blindly. |
| **Async quality verification / LLM-as-judge pattern** | `autopilot/verifier.py` | The cheap model's answer is checked against the highest-quality available model, scored for agreement, and silently swapped out if it disagrees too much — the same "generate cheap, verify expensive, escalate on disagreement" pattern real cost-routing systems use. |
| **Escalation and auto-recovery** | `autopilot/verifier.py`, `autopilot/pipeline.py` | Every escalation is logged with the original model, the escalated model, and the disagreement score that triggered it. |
| **Structured audit logging** | `autopilot/store.py` | SQLite with one indexed column per queryable field *and* a full JSON blob per row, so the dashboard is fast and no data is ever silently dropped. |
| **Cost-savings measurement methodology** | `autopilot/pipeline.py`, `dashboard/app.py` | Every request also computes a counterfactual: what it would have cost if the *reference model alone* had handled it. Savings % is `(baseline - actual) / baseline`, not a guess. |
| **API-first design** | `api/main.py` | A FastAPI service where the caller never names a model — they get one back in the response, along with the tier, confidence, cost, and whether it was escalated. |
| **Config-driven behavior** | `config/models.yaml`, `config/routing.yaml` | Pricing and routing decisions live in YAML, not code, so a model swap or price change is a one-line edit with no redeploy of logic. |
| **Offline-reproducible pipelines** | `autopilot/providers/mock_provider.py` | A deterministic mock provider (seeded by hash of prompt+model) lets the entire pipeline — routing, verification, escalation, dashboard — be exercised and load-tested with zero API keys and zero network calls. |
| **Data visualization design system** | `dashboard/app.py` | Charts use a validated, colorblind-safe categorical palette assigned in fixed order (never re-cycled per value), a single-hue sequential ramp for the ordinal tier chart, and reserved status colors (green/red) for escalated vs. kept — not arbitrary chart-library defaults. |

---

## 3. How the pipeline actually works, end to end

```
prompt
  │
  ▼
┌─────────────────────┐   20 interpretable features (verb lexicons, length,
│ Complexity Classifier│ ← constraints, format hints...) extracted in
│  (RandomForest)      │   microseconds, no network call
└─────────┬────────────┘
          │ tier 1 / 2 / 3 + confidence
          ▼
┌─────────────────────┐   config/routing.yaml: tier → primary model,
│      Router          │ ← with a fallback chain if the primary provider
└─────────┬────────────┘   has no API key / is down
          │ chosen model
          ▼
┌─────────────────────┐   provider-agnostic call; standardized response
│  send_request()       │ ← with cost, latency, token usage attached
└─────────┬────────────┘   (autopilot/client.py)
          │ primary response
          ▼
┌─────────────────────┐   re-asks the same prompt to the escalation-tier
│  Async Verifier       │ ← model, scores lexical agreement; if it's too low,
└─────────┬────────────┘   the verifier's (better) answer wins instead
          │ final response + escalated? + agreement score
          ▼
┌─────────────────────┐   SQLite row: tier, model, cost, baseline cost,
│      Log Store        │ ← latency, agreement score, escalation flag,
└─────────┬────────────┘   full JSON audit trail
          │
          ▼
   Streamlit Dashboard  /  FastAPI /v1/stats
```

This whole chain is one function call — `autopilot/pipeline.py:run_request()` —
used identically by the load-test script and the API endpoint, so behavior can
never diverge between "how we tested it" and "how it actually runs."

---

## 4. How each phase was built

**Phase 1 — Unified model interface.** Defined `ModelConfig`/`ModelRegistry`
(`autopilot/registry.py`) with real published pricing for Claude Opus 5 / Sonnet
5 / Haiku 4.5, GPT-4o / GPT-4o mini, and local Ollama models, all costed per
1M tokens. Built `send_request()` as the one abstraction every provider call
goes through, with retry/backoff on retryable errors. Verified it end-to-end
with `scripts/phase1_baseline.py`, which fires the same 10 prompts at every
available model and measures real p50/p95 latency (written back into
`config/measured_latency.yaml`, layered on top of the seed values in
`models.yaml`).

**Phase 2 — Complexity classifier + routing map.** Wrote prose tier
definitions (`autopilot/classifier/tiers.py`) so future labeling stays
consistent, then hand-labeled 166 prompts across the three tiers
(`data/prompts/labeled_tier{1,2,3}.jsonl` — tier 3 was added this build; only
tiers 1–2 existed before). Extracted 20 interpretable features per prompt and
trained a `RandomForestClassifier` (`scripts/train_classifier.py`), evaluated
with 5-fold stratified cross-validation to get an honest accuracy number at
this dataset size. Built `config/routing.yaml` as the tier → model lookup
table with per-tier fallback chains, consumed by `autopilot/router.py`.

**Phase 3 — Async quality verification.** `autopilot/verifier.py` re-sends
a request to the escalation-tier model and scores lexical agreement (weighted
Jaccard over word sets + a length-ratio penalty) — deliberately a cheap
heuristic rather than a second LLM-judge call, so verification itself doesn't
become a cost center. Below a calibrated threshold, the verifier's answer
replaces the cheap model's and the event is logged as an escalation with the
disagreement score attached.

**Phase 4 — Logging and cost dashboard.** `autopilot/store.py` logs one row
per request to SQLite (`data/autopilot.db`) — routed model, final model, tier,
cost, a computed *baseline cost* (what the reference model alone would have
charged for the same output), latency, and the full row as JSON for audit.
`dashboard/app.py` (Streamlit + Altair) reads directly from that database and
renders the headline savings %, model/tier distribution, an agreement-score
histogram, an escalated-vs-kept breakdown, and a cumulative actual-vs-baseline
cost line chart.

**Phase 5 — API.** `api/main.py` (FastAPI) exposes one completion endpoint
where the caller supplies a prompt, not a model — the response tells them
which model answered, its tier, cost, and whether it was escalated — plus
`GET /v1/models`, `GET /v1/stats`, and `GET`/`PUT /v1/routing-config` to
inspect or steer the routing map live.

**Phase 6 (in progress) — Load test and polish.** `scripts/load_test.py`
samples the 176-prompt labeled pool with replacement to push realistic volume
(500+ requests) through the full pipeline offline (forcing the deterministic
mock provider so a run finishes in seconds, not hours of real API/Ollama
calls), then prints the same cost-savings summary the dashboard shows.
Containerization and a written case study are the remaining polish items —
see [Status](#6-status--whats-not-done-yet).

---

## 5. Tech stack

| Component | Tool / library | Why |
|---|---|---|
| Language | Python 3.11+ | Ecosystem compatibility with every LLM SDK and ML library used here. |
| LLM providers | `anthropic`, `openai`, Ollama (local, via `httpx`) | A real mix of cloud and local models — the whole point of a router. |
| Async I/O | `asyncio`, `httpx` | Non-blocking calls to slow network APIs (and local Ollama calls that take seconds). |
| Config validation / data shapes | `pydantic` (API layer), `dataclasses` (core) | Standardized request/response objects (`LLMRequest`/`LLMResponse`) and typed API schemas. |
| Router / API | **FastAPI** + `uvicorn` | Async-native, production-grade, auto-generated OpenAPI docs at `/docs`. |
| Classifier | **scikit-learn** (`RandomForestClassifier`), `numpy`, `pandas`, `joblib` | Lightweight, interpretable, fast enough to sit in the request path; `joblib` persists the trained model to `models/classifier.joblib`. |
| Config format | **YAML** (`pyyaml`) | Pricing (`models.yaml`) and routing decisions (`routing.yaml`) are data, not code. |
| Logging / audit trail | **SQLite** (stdlib `sqlite3`) + structured JSON per row | Zero-ops embedded database; full per-request audit trail. |
| Dashboard | **Streamlit** + **Altair** | Fast to build, reads live from the same SQLite file the pipeline writes to. |
| Testing | **pytest** + `pytest-asyncio` | 26 tests across registry, client, router, verifier, and store. |
| Environment | `python-dotenv` | Optional cloud API keys — the system runs fully offline without any. |

---

## 6. Status — what's not done yet

- **Cloud keys (OpenAI, Anthropic) are still not configured server-side** in
  this environment, so those three model tiers fall back to Groq (real,
  cloud-hosted, fast and cheap) by default — see `config/routing.yaml`. The
  routing and verification logic has been run against real, priced cloud
  inference via Groq, not just the offline mock.
- **The verifier's agreement scorer is a lexical heuristic**, not a real
  LLM-as-judge or task-specific grader (exact-field-match for extraction,
  label-match for classification, etc.) — documented as a known
  simplification in `autopilot/verifier.py`.
- **No weekly classifier retraining / feedback loop** from escalation data yet
  (the spec's Phase 3 step 4) — `autopilot/classifier/model.py` exposes
  `reload_classifier()` for this, but nothing calls it on a schedule.
  Explicitly out of scope until real usage data exists.
- **No streaming support** on `/v1/chat/completions` — `stream: true` returns
  a clear 400 rather than silently ignoring the flag.
- **No accounts, per-user usage tracking, or billing** — see §7 below for what
  exists instead (bring-your-own-key) and what a real multi-tenant version
  would need.

## 7. Using it as a drop-in for your own app

`POST /v1/chat/completions` matches the OpenAI chat completions request and
response shape, so pointing an existing `openai` SDK client at this service's
`base_url` works with no other code changes — the router picks the model, not
the caller:

```python
from openai import OpenAI

client = OpenAI(base_url="https://<your-deployment>/v1", api_key="unused")
resp = client.chat.completions.create(
    model="gpt-4o",  # accepted, ignored -- Autopilot decides what actually answers
    messages=[{"role": "user", "content": "Summarize this in two sentences: ..."}],
)
print(resp.choices[0].message.content)
print(resp.autopilot)  # extra, namespaced field: tier, cost, escalated, etc.
```

**Bring your own key.** By default the service uses whichever provider keys
*it* has configured. To route through your own provider account instead —
so the operator never pays for your usage — pass one or more headers:

```
X-OpenAI-Api-Key: sk-...
X-Anthropic-Api-Key: sk-ant-...
X-Groq-Api-Key: gsk_...
```

Each key is used for that one request only and is never logged or stored —
see `autopilot/pipeline.py`'s `api_keys` parameter and
`autopilot/providers/__init__.py:get_provider_instance`, which builds an
uncached provider instance per call rather than touching the shared,
server-key singleton. A key you supply unlocks routing to that provider even
if the server itself has none configured for it (`provider_usable()` in
`autopilot/providers/__init__.py`).

The original prompt-in/rich-metadata-out shape still exists at `POST
/v1/route`, for inspecting a routing decision directly instead of integrating
against the OpenAI-compatible shape.

This is the lightweight version of "let other people use this": no accounts,
no per-user dashboards, no billing — just an OpenAI-compatible endpoint
anyone can call, optionally with their own key. A real multi-tenant product
(per-user API keys issued by *this* service, usage quotas, a personal savings
dashboard, payment) is a materially bigger build and isn't started.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# optional — real cloud calls (falls back to mock/Ollama without these)
cp .env.example .env && $EDITOR .env

python scripts/phase1_baseline.py          # measure real latency per model
python scripts/train_classifier.py         # train + evaluate the classifier
python scripts/load_test.py --n 600        # generate demo traffic (offline by default)

streamlit run dashboard/app.py             # cost dashboard  -> http://localhost:8501
uvicorn api.main:app --reload --port 8000  # API + docs      -> http://localhost:8000/docs

pytest                                     # 26 tests
```
