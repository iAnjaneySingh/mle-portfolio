from __future__ import annotations

import numpy as np
import pandas as pd

from src.monitoring.drift import categorical_psi, compute_drift, psi


def test_psi_is_zero_for_identical_distributions():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 20_000)
    assert psi(x, x) < 0.01


def test_psi_detects_mean_shift():
    rng = np.random.default_rng(2)
    ref = rng.normal(0, 1, 20_000)
    cur = rng.normal(1.5, 1, 20_000)
    assert psi(ref, cur) > 0.25


def test_psi_is_monotone_in_shift_size():
    rng = np.random.default_rng(3)
    ref = rng.normal(0, 1, 20_000)
    vals = [psi(ref, rng.normal(s, 1, 20_000)) for s in (0.1, 0.5, 1.0, 2.0)]
    assert vals == sorted(vals)


def test_categorical_psi_flags_new_dominant_category():
    ref = {"grocery": 0.6, "fuel": 0.4}
    cur = pd.Series(["crypto"] * 700 + ["grocery"] * 300)
    value, _ = categorical_psi(ref, cur)
    assert value > 0.25


def test_compute_drift_end_to_end():
    rng = np.random.default_rng(4)
    profile = {
        "numeric": {"amount_log": {
            "quantiles": np.quantile(rng.normal(3, 1, 10_000), np.linspace(0, 1, 11)).tolist(),
            "mean": 3.0, "std": 1.0,
        }},
        "categorical": {"merchant_risk_bucket": {"grocery": 0.7, "fuel": 0.3}},
    }
    current = pd.DataFrame({
        "amount_log": rng.normal(5.0, 1, 3_000),
        "merchant_risk_bucket": rng.choice(["crypto", "grocery"], 3_000, p=[0.8, 0.2]),
    })
    results = {r.feature: r for r in compute_drift(profile, current)}
    assert results["amount_log"].status == "alert"
    assert results["merchant_risk_bucket"].status == "alert"
