"""Drift detection over the inference log.

Three things are measured, and they are not the same thing:
  1. Covariate drift  -- has the input distribution moved? (PSI / chi-square)
  2. Prediction drift -- has the score distribution moved? (PSI on scores)
  3. Concept drift    -- has P(y|x) moved? Only computable once labels land,
                         so it is reported on a lagged window.

PSI thresholds used throughout: <0.10 stable, 0.10-0.25 warning, >=0.25 alert.
Those are the conventional credit-risk cutoffs; they are heuristics, not tests,
which is why the KS/chi-square p-values are reported alongside.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

Status = Literal["stable", "warning", "alert"]


@dataclass
class DriftResult:
    feature: str
    test: str
    statistic: float
    p_value: float | None
    status: Status
    n_reference: int
    n_current: int


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10, eps: float = 1e-6) -> float:
    """Population Stability Index using reference quantile edges.

    Edges come from the reference window, not from the pooled data -- pooling
    lets the current window define its own bins and systematically understates
    drift.
    """
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(reference, bins=edges)[0] / max(len(reference), 1) + eps
    cur_pct = np.histogram(current, bins=edges)[0] / max(len(current), 1) + eps
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def categorical_psi(ref_props: dict, current: pd.Series, eps: float = 1e-6) -> tuple[float, float]:
    """Returns (psi, chi-square p-value) for a categorical feature."""
    cur_props = current.value_counts(normalize=True).to_dict()
    categories = sorted(set(ref_props) | set(cur_props))
    r = np.array([ref_props.get(c, 0) for c in categories]) + eps
    c = np.array([cur_props.get(c, 0) for c in categories]) + eps
    value = float(np.sum((c - r) * np.log(c / r)))

    observed = np.array([(current == cat).sum() for cat in categories])
    expected = r / r.sum() * observed.sum()
    keep = expected >= 5  # chi-square is unreliable on sparse cells
    if keep.sum() >= 2:
        _, p = stats.chisquare(observed[keep], expected[keep] * observed[keep].sum() / expected[keep].sum())
    else:
        p = float("nan")
    return value, float(p)


def _status(value: float) -> Status:
    if value >= 0.25:
        return "alert"
    if value >= 0.10:
        return "warning"
    return "stable"


def compute_drift(
    reference_profile: dict,
    current: pd.DataFrame,
    reference_raw: pd.DataFrame | None = None,
) -> list[DriftResult]:
    results: list[DriftResult] = []

    for feat, prof in reference_profile.get("numeric", {}).items():
        if feat not in current.columns:
            continue
        cur = current[feat].dropna().astype(float).to_numpy()
        if len(cur) < 30:
            continue
        # Reconstruct a reference sample from stored quantiles when the raw
        # training frame is not shipped alongside the model.
        ref = (
            reference_raw[feat].dropna().astype(float).to_numpy()
            if reference_raw is not None and feat in reference_raw
            else _sample_from_quantiles(prof["quantiles"], n=len(cur))
        )
        value = psi(ref, cur)
        ks = stats.ks_2samp(ref, cur)
        results.append(
            DriftResult(feat, "psi", value, float(ks.pvalue), _status(value), len(ref), len(cur))
        )

    for feat, ref_props in reference_profile.get("categorical", {}).items():
        if feat not in current.columns:
            continue
        cur = current[feat].dropna().astype(str)
        if len(cur) < 30:
            continue
        value, p = categorical_psi(ref_props, cur)
        results.append(
            DriftResult(feat, "chi2", value, p, _status(value), 0, len(cur))
        )

    return results


def prediction_drift(reference_scores: np.ndarray, current_scores: np.ndarray) -> DriftResult:
    value = psi(reference_scores, current_scores)
    ks = stats.ks_2samp(reference_scores, current_scores)
    return DriftResult(
        "prediction_score", "psi", value, float(ks.pvalue), _status(value),
        len(reference_scores), len(current_scores),
    )


def _sample_from_quantiles(quantiles: list[float], n: int, seed: int = 0) -> np.ndarray:
    """Inverse-CDF sampling from a stored quantile grid."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 1, n)
    grid = np.linspace(0, 1, len(quantiles))
    return np.interp(u, grid, quantiles)


def run(
    inference_log: str = "data/inference_log.parquet",
    reference_profile: str = "artifacts/reference_profile.json",
    window: str = "7d",
    out: str | None = "artifacts/drift_report.json",
) -> dict:
    prof = json.loads(Path(reference_profile).read_text())
    log_df = pd.read_parquet(inference_log)
    log_df["ts"] = pd.to_datetime(log_df["ts"], utc=True)
    cutoff = log_df["ts"].max() - pd.Timedelta(window)
    current = log_df[log_df["ts"] >= cutoff]

    results = compute_drift(prof, current)
    report = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "window": window,
        "n_current": int(len(current)),
        "worst_status": _worst([r.status for r in results]),
        "features": [asdict(r) for r in results],
    }
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(report, indent=2, default=str))
    return report


def _worst(statuses: list[str]) -> str:
    for level in ("alert", "warning"):
        if level in statuses:
            return level
    return "stable"


if __name__ == "__main__":
    print(json.dumps(run(), indent=2)[:2000])
