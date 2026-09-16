"""Streamlit drift-monitoring dashboard.

Run: streamlit run src/monitoring/dashboard.py
Panels: traffic + score distribution over time, per-feature PSI table with
status colouring, reference-vs-current histograms, and registry state.
"""
from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from .drift import compute_drift, prediction_drift

st.set_page_config(page_title="Model Monitoring", layout="wide")

INFERENCE_LOG = Path("data/inference_log.parquet")
REFERENCE = Path("artifacts/reference_profile.json")

COLORS = {"stable": "#1a7f37", "warning": "#bf8700", "alert": "#cf222e"}


@st.cache_data(ttl=30)
def load_log() -> pd.DataFrame:
    if not INFERENCE_LOG.exists():
        return pd.DataFrame()
    df = pd.read_parquet(INFERENCE_LOG)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.sort_values("ts")


@st.cache_data(ttl=30)
def load_reference() -> dict:
    return json.loads(REFERENCE.read_text()) if REFERENCE.exists() else {}


df = load_log()
profile = load_reference()

st.title("Model monitoring")

if df.empty or not profile:
    st.warning("No inference log or reference profile yet. Run training, then send traffic to /predict.")
    st.stop()

window = st.sidebar.selectbox("Comparison window", ["1d", "3d", "7d", "14d"], index=2)
cutoff = df["ts"].max() - pd.Timedelta(window)
current = df[df["ts"] >= cutoff]
baseline = df[df["ts"] < cutoff]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Requests (window)", f"{len(current):,}")
c2.metric("Mean score", f"{current['score'].mean():.3f}",
          delta=f"{current['score'].mean() - df['score'].mean():+.3f}")
c3.metric("Decline rate", f"{(current.decision == 'decline').mean():.2%}")
c4.metric("Model", str(current["model_version"].iloc[-1])[:8])

results = compute_drift(profile, current)
worst = "alert" if any(r.status == "alert" for r in results) else (
    "warning" if any(r.status == "warning" for r in results) else "stable")
st.markdown(
    f"### Overall status: <span style='color:{COLORS[worst]}'>{worst.upper()}</span>",
    unsafe_allow_html=True,
)

st.subheader("Feature drift")
table = pd.DataFrame([{
    "feature": r.feature, "test": r.test, "statistic": round(r.statistic, 4),
    "p_value": None if r.p_value is None or np.isnan(r.p_value) else round(r.p_value, 5),
    "status": r.status, "n_current": r.n_current,
} for r in results]).sort_values("statistic", ascending=False)

st.dataframe(
    table.style.map(lambda v: f"color: {COLORS.get(v, '')}", subset=["status"]),
    use_container_width=True, hide_index=True,
)

if not baseline.empty:
    pd_res = prediction_drift(baseline["score"].to_numpy(), current["score"].to_numpy())
    st.subheader("Prediction drift")
    st.write(
        f"PSI **{pd_res.statistic:.4f}** ({pd_res.status}) · "
        f"KS p-value {pd_res.p_value:.2e}"
    )

left, right = st.columns(2)
with left:
    st.subheader("Score distribution over time")
    hourly = (current.set_index("ts").resample("1h")["score"]
              .agg(["mean", "count"]).reset_index().dropna())
    st.altair_chart(
        alt.Chart(hourly).mark_line(point=False).encode(
            x="ts:T", y=alt.Y("mean:Q", title="mean score"),
        ).properties(height=260),
        use_container_width=True,
    )

with right:
    st.subheader("Reference vs current")
    numeric_feats = list(profile.get("numeric", {}))
    feat = st.selectbox("Feature", numeric_feats)
    ref_q = profile["numeric"][feat]["quantiles"]
    rng = np.random.default_rng(0)
    ref_sample = np.interp(rng.uniform(0, 1, len(current)), np.linspace(0, 1, len(ref_q)), ref_q)
    comp = pd.concat([
        pd.DataFrame({"value": ref_sample, "window": "reference"}),
        pd.DataFrame({"value": current[feat].astype(float), "window": "current"}),
    ])
    st.altair_chart(
        alt.Chart(comp).mark_area(opacity=0.45, interpolate="step").encode(
            x=alt.X("value:Q", bin=alt.Bin(maxbins=40)),
            y=alt.Y("count()", stack=None),
            color="window:N",
        ).properties(height=260),
        use_container_width=True,
    )

st.caption("PSI cutoffs: <0.10 stable · 0.10-0.25 warning · >=0.25 alert")
