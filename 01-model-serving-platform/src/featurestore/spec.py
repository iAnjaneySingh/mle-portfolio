"""Single source of truth for feature definitions.

The whole point of this module: training code, materialization jobs and the
online serving path all import the SAME transformation functions. If a feature
is redefined in two places, training/serving skew is inevitable. Here it can't
happen structurally -- there is exactly one implementation.
"""
from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass, field
from typing import Callable, Literal, Sequence

import numpy as np
import pandas as pd

Dtype = Literal["float", "int", "categorical"]


@dataclass(frozen=True)
class Feature:
    name: str
    dtype: Dtype
    transform: Callable[[pd.DataFrame], pd.Series]
    description: str = ""
    drift_test: Literal["psi", "chi2"] = "psi"

    @property
    def code_hash(self) -> str:
        """Hash of the transform source. Any edit changes the feature version."""
        src = inspect.getsource(self.transform)
        return hashlib.sha256(src.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class FeatureView:
    """A named, versioned group of features keyed by a single entity."""

    name: str
    entity_key: str
    features: Sequence[Feature]
    ttl_seconds: int = 3600
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def feature_names(self) -> list[str]:
        return [f.name for f in self.features]

    @property
    def version(self) -> str:
        """Deterministic version derived from the feature implementations.

        Materialized rows are written under this version, and the serving layer
        refuses to answer with a version the model was not trained against.
        """
        joined = "|".join(f"{f.name}:{f.code_hash}" for f in self.features)
        return hashlib.sha256(joined.encode()).hexdigest()[:12]

    def compute(self, events: pd.DataFrame) -> pd.DataFrame:
        """Apply every transform to a raw event frame."""
        out = pd.DataFrame(index=events.index)
        out[self.entity_key] = events[self.entity_key].values
        for f in self.features:
            out[f.name] = f.transform(events)
        return out

    def get(self, name: str) -> Feature:
        for f in self.features:
            if f.name == name:
                return f
        raise KeyError(f"{name} not in feature view {self.name}")


# --------------------------------------------------------------------------
# Example domain: per-account transaction risk scoring.
# --------------------------------------------------------------------------

def _amount_log(df: pd.DataFrame) -> pd.Series:
    return np.log1p(df["amount"].clip(lower=0)).astype("float64")


def _txn_count_24h(df: pd.DataFrame) -> pd.Series:
    return df["txn_count_24h"].fillna(0).astype("float64")


def _amount_vs_mean_30d(df: pd.DataFrame) -> pd.Series:
    mean = df["amount_mean_30d"].replace(0, np.nan)
    return (df["amount"] / mean).fillna(1.0).clip(0, 50).astype("float64")


def _seconds_since_prev(df: pd.DataFrame) -> pd.Series:
    return df["seconds_since_prev_txn"].fillna(86400).clip(0, 86400).astype("float64")


def _merchant_risk_bucket(df: pd.DataFrame) -> pd.Series:
    return df["merchant_category"].fillna("unknown").astype(str)


def _is_foreign(df: pd.DataFrame) -> pd.Series:
    return (df["country"] != df["home_country"]).astype("int64")


TXN_RISK_V1 = FeatureView(
    name="txn_risk",
    entity_key="account_id",
    ttl_seconds=900,
    tags={"owner": "mle-platform", "domain": "risk"},
    features=[
        Feature("amount_log", "float", _amount_log, "log1p of transaction amount"),
        Feature("txn_count_24h", "float", _txn_count_24h, "rolling 24h txn count"),
        Feature("amount_vs_mean_30d", "float", _amount_vs_mean_30d, "amount / 30d mean"),
        Feature("seconds_since_prev", "float", _seconds_since_prev, "recency of last txn"),
        Feature("is_foreign", "int", _is_foreign, "txn country != home country"),
        Feature(
            "merchant_risk_bucket",
            "categorical",
            _merchant_risk_bucket,
            "merchant category code",
            drift_test="chi2",
        ),
    ],
)

REGISTRY: dict[str, FeatureView] = {TXN_RISK_V1.name: TXN_RISK_V1}


def get_view(name: str) -> FeatureView:
    return REGISTRY[name]
