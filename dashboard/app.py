"""Phase 4, step 2 -- the cost dashboard (money-shot metric + Phase 6 polish).

    streamlit run dashboard/app.py

Reads directly from the SQLite log every other component writes to, so the
numbers here can never drift from what the pipeline actually charged.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import altair as alt
import pandas as pd
import streamlit as st

from autopilot.registry import ModelRegistry
from autopilot.router import get_routing_config
from autopilot.store import get_store

# Fixed-order categorical palette (dataviz skill reference instance) --
# assigned by position, never cycled or re-sorted by value.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE_SEQ = {1: "#86b6ef", 2: "#3987e5", 3: "#184f95"}  # ordinal tier ramp, step >=250
STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"

st.set_page_config(page_title="LLM Cost Autopilot", page_icon="💸", layout="wide")

st.title("LLM Cost Autopilot")
st.caption(
    "A complexity-routed LLM gateway: cheap models handle easy requests, expensive "
    "models handle hard ones, and an async verifier catches the cheap model when it's wrong."
)

store = get_store()
summary = store.summary()

if summary["requests"] == 0:
    st.warning(
        "No requests logged yet. Run `python scripts/load_test.py` to generate demo "
        "traffic, or send requests through the API at `POST /v1/chat/completions`."
    )
    st.stop()

rows = store.all_rows()
df = pd.DataFrame(rows)
df["created_at_dt"] = pd.to_datetime(df["created_at"], unit="s")

# ---- mock-fallback flag -------------------------------------------------------
# mock-* is only reachable as tier 3's last-resort fallback (see
# config/routing.yaml) -- every real provider was unreachable. Status colors
# carry an icon + label, never color alone, per the status-palette convention.
mock_n = summary["mock_fallback_count"]
if mock_n > 0:
    st.error(
        f"⚠️ {mock_n} request{'s' if mock_n != 1 else ''} fell all the way through to the "
        "offline mock provider — every real model (cloud, Groq, local Ollama) was "
        "unreachable for that request. Answers for those rows are placeholder text, not real."
    )
else:
    st.success("✅ No requests hit the mock fallback — every answer above came from a real model.")

# ---- the money shot ---------------------------------------------------------
st.divider()
m1, m2, m3 = st.columns([2, 1, 1])
with m1:
    st.metric(
        "Cost reduction vs. routing everything to the reference model",
        f"{summary['savings_pct']:.1f}%",
        delta=f"-${summary['savings_usd']:,.4f} saved",
        delta_color="normal",
    )
with m2:
    st.metric("Total requests", f"{summary['requests']:,}")
with m3:
    st.metric("Escalation rate", f"{summary['escalation_rate']:.1f}%")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Actual cost", f"${summary['total_cost_usd']:.4f}")
k2.metric("Reference-model cost", f"${summary['baseline_cost_usd']:.4f}")
k3.metric("Avg agreement score", f"{summary['avg_agreement']:.2f}" if summary["avg_agreement"] is not None else "—")
k4.metric("Avg latency", f"{df['latency_ms'].mean():.0f} ms")

st.divider()

# ---- model + tier distribution ----------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("Which model handled the traffic")
    model_counts = df["final_model_key"].value_counts().reset_index()
    model_counts.columns = ["model", "requests"]
    model_order = list(model_counts["model"])
    color_scale = alt.Scale(domain=model_order, range=CATEGORICAL[: len(model_order)])
    chart = (
        alt.Chart(model_counts)
        .mark_bar(cornerRadiusEnd=4, size=22)
        .encode(
            y=alt.Y("model:N", sort="-x", title=None, axis=alt.Axis(labelColor=INK_SECONDARY)),
            x=alt.X("requests:Q", title="Requests", axis=alt.Axis(gridColor=GRID, labelColor=INK_SECONDARY)),
            color=alt.Color("model:N", scale=color_scale, legend=None),
            tooltip=["model", "requests"],
        )
        .properties(height=32 * len(model_order) + 20)
    )
    st.altair_chart(chart, use_container_width=True)

with right:
    st.subheader("Complexity tier distribution")
    tier_names = {1: "1 · simple", 2: "2 · moderate", 3: "3 · complex"}
    tier_counts = df["complexity_tier"].value_counts().reset_index()
    tier_counts.columns = ["tier", "requests"]
    tier_counts["label"] = tier_counts["tier"].map(tier_names)
    tier_counts = tier_counts.sort_values("tier")
    tier_scale = alt.Scale(domain=[tier_names[t] for t in sorted(tier_names)],
                            range=[BLUE_SEQ[t] for t in sorted(tier_names)])
    chart = (
        alt.Chart(tier_counts)
        .mark_bar(cornerRadiusEnd=4, size=36)
        .encode(
            x=alt.X("label:N", title=None, axis=alt.Axis(labelColor=INK_SECONDARY)),
            y=alt.Y("requests:Q", title="Requests", axis=alt.Axis(gridColor=GRID, labelColor=INK_SECONDARY)),
            color=alt.Color("label:N", scale=tier_scale, legend=None),
            tooltip=["label", "requests"],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, use_container_width=True)

st.divider()

# ---- quality + escalation ----------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("Agreement score distribution")
    st.caption("Verifier agreement between the routed model and the escalation-tier check.")
    chart = (
        alt.Chart(df)
        .mark_bar(color=CATEGORICAL[0], opacity=0.9)
        .encode(
            x=alt.X("agreement_score:Q", bin=alt.Bin(maxbins=20), title="Agreement score",
                     axis=alt.Axis(gridColor=GRID, labelColor=INK_SECONDARY)),
            y=alt.Y("count():Q", title="Requests", axis=alt.Axis(gridColor=GRID, labelColor=INK_SECONDARY)),
            tooltip=["count()"],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, use_container_width=True)

with right:
    st.subheader("Escalated vs. kept")
    esc_counts = df["escalated"].map({0: "kept", 1: "escalated"}).value_counts().reset_index()
    esc_counts.columns = ["outcome", "requests"]
    esc_scale = alt.Scale(domain=["kept", "escalated"], range=[STATUS_GOOD, STATUS_CRITICAL])
    chart = (
        alt.Chart(esc_counts)
        .mark_arc(innerRadius=70, cornerRadius=3)
        .encode(
            theta="requests:Q",
            color=alt.Color("outcome:N", scale=esc_scale, legend=alt.Legend(title=None, labelColor=INK_SECONDARY)),
            tooltip=["outcome", "requests"],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, use_container_width=True)

st.divider()

# ---- cost over time -----------------------------------------------------------
st.subheader("Cumulative cost: actual vs. reference-model baseline")
ts = df.sort_values("created_at_dt").copy()
ts["actual_cum"] = ts["cost_usd"].cumsum()
ts["baseline_cum"] = ts["baseline_cost_usd"].cumsum()
ts["request_n"] = range(1, len(ts) + 1)
long = ts[["request_n", "actual_cum", "baseline_cum"]].melt(
    id_vars="request_n", value_vars=["actual_cum", "baseline_cum"],
    var_name="series", value_name="cost_usd",
)
long["series"] = long["series"].map({"actual_cum": "Actual (routed)", "baseline_cum": "Reference model only"})
line_scale = alt.Scale(domain=["Actual (routed)", "Reference model only"], range=[CATEGORICAL[0], CATEGORICAL[1]])
chart = (
    alt.Chart(long)
    .mark_line(strokeWidth=2)
    .encode(
        x=alt.X("request_n:Q", title="Request #", axis=alt.Axis(gridColor=GRID, labelColor=INK_SECONDARY)),
        y=alt.Y("cost_usd:Q", title="Cumulative cost (USD)", axis=alt.Axis(gridColor=GRID, labelColor=INK_SECONDARY)),
        color=alt.Color("series:N", scale=line_scale, legend=alt.Legend(title=None, labelColor=INK_SECONDARY)),
        tooltip=["request_n", "series", "cost_usd"],
    )
    .properties(height=320)
)
st.altair_chart(chart, use_container_width=True)

st.divider()

# ---- model registry + routing map -------------------------------------------
st.subheader("Current routing map")
rc = get_routing_config()
registry = ModelRegistry.load()
route_rows = []
for tier, spec in sorted(rc.tiers.items()):
    m = registry[spec["primary"]]
    route_rows.append({
        "tier": tier,
        "primary model": spec["primary"],
        "quality": m.quality_tier.value,
        "$/1M in": m.input_cost_per_1m,
        "$/1M out": m.output_cost_per_1m,
        "fallback chain": " -> ".join(spec.get("fallback", [])),
    })
st.dataframe(pd.DataFrame(route_rows), hide_index=True, use_container_width=True)

st.subheader("Recent requests")
recent = df.sort_values("created_at_dt", ascending=False).head(50)[
    ["created_at_dt", "task", "complexity_tier", "routed_model_key", "final_model_key",
     "final_provider", "escalated", "agreement_score", "cost_usd", "latency_ms", "prompt_preview"]
]
st.dataframe(recent, hide_index=True, use_container_width=True)
