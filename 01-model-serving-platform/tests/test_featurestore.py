from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.featurestore.offline import OfflineStore
from src.featurestore.spec import TXN_RISK_V1, Feature, FeatureView


def _events(n=50):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "account_id": [f"a{i % 5}" for i in range(n)],
        "event_ts": pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC"),
        "amount": rng.uniform(5, 500, n),
        "amount_mean_30d": rng.uniform(20, 200, n),
        "txn_count_24h": rng.integers(0, 10, n),
        "seconds_since_prev_txn": rng.uniform(0, 100_000, n),
        "merchant_category": rng.choice(["grocery", "crypto"], n),
        "country": rng.choice(["US", "IN"], n),
        "home_country": "US",
        "is_fraud": rng.integers(0, 2, n),
    })


def test_version_changes_when_transform_changes():
    v1 = TXN_RISK_V1.version
    mutated = FeatureView(
        name="txn_risk",
        entity_key="account_id",
        features=[*TXN_RISK_V1.features[:-1],
                  Feature("merchant_risk_bucket", "categorical",
                          lambda df: df["merchant_category"].fillna("OTHER").astype(str))],
    )
    assert mutated.version != v1, "feature version must track transform source"


def test_compute_produces_all_columns():
    out = TXN_RISK_V1.compute(_events())
    assert set(TXN_RISK_V1.feature_names).issubset(out.columns)
    assert out["is_foreign"].isin([0, 1]).all()


def test_point_in_time_join_never_leaks_future(tmp_path):
    events = _events(80)
    store = OfflineStore(tmp_path)
    feats = TXN_RISK_V1.compute(events)
    feats["event_ts"] = events["event_ts"].values
    store.write(TXN_RISK_V1, feats)

    # Label rows placed 30 minutes BEFORE each feature row: nothing may join.
    entity_df = events[["account_id", "event_ts", "is_fraud"]].copy()
    entity_df["event_ts"] = entity_df["event_ts"] - pd.Timedelta("90m")
    joined = store.get_training_frame(TXN_RISK_V1, entity_df, tolerance=pd.Timedelta("30m"))
    assert joined["amount_log"].isna().all(), "future features leaked into the past"


def test_point_in_time_join_matches_past_rows(tmp_path):
    events = _events(80)
    store = OfflineStore(tmp_path)
    feats = TXN_RISK_V1.compute(events)
    feats["event_ts"] = events["event_ts"].values
    store.write(TXN_RISK_V1, feats)

    entity_df = events[["account_id", "event_ts", "is_fraud"]].copy()
    entity_df["event_ts"] = entity_df["event_ts"] + pd.Timedelta("10m")
    joined = store.get_training_frame(TXN_RISK_V1, entity_df)
    assert joined["amount_log"].notna().mean() > 0.95


def test_missing_offline_data_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        OfflineStore(tmp_path).read(TXN_RISK_V1)
