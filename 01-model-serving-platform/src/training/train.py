"""Train, evaluate and register a model in MLflow.

What this does beyond `model.fit()`:
  * Builds the training frame through the offline store's point-in-time join,
    so there is no label leakage.
  * Splits temporally, never randomly -- random splits on time-series-shaped
    data are the single most common way portfolio metrics get inflated.
  * Logs the feature-view version as a tag and the training feature
    distribution as an artifact. Serving reads the former to reject skew; the
    drift monitor reads the latter as its reference window.
  * Registers the model and promotes it via an alias (`champion`), which is the
    modern replacement for MLflow's deprecated stages API.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ..featurestore.offline import OfflineStore
from ..featurestore.spec import get_view

log = logging.getLogger(__name__)
MODEL_NAME = "txn_risk_classifier"


def build_pipeline(numeric: list[str], categorical: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", "passthrough", numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), categorical),
        ]
    )
    clf = HistGradientBoostingClassifier(
        max_iter=350,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.12,
        random_state=7,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def temporal_split(df: pd.DataFrame, test_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("event_ts")
    cut = int(len(df) * (1 - test_frac))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def train(
    events_path: str,
    view_name: str = "txn_risk",
    tracking_uri: str = "http://localhost:5000",
    experiment: str = "txn-risk",
    register: bool = True,
) -> dict:
    view = get_view(view_name)
    events = pd.read_parquet(events_path)

    entity_df = events[[view.entity_key, "event_ts", "is_fraud"]].copy()
    frame = OfflineStore().get_training_frame(view, entity_df).dropna(subset=view.feature_names)

    numeric = [f.name for f in view.features if f.dtype in ("float", "int")]
    categorical = [f.name for f in view.features if f.dtype == "categorical"]

    train_df, test_df = temporal_split(frame)
    X_tr, y_tr = train_df[view.feature_names], train_df["is_fraud"]
    X_te, y_te = test_df[view.feature_names], test_df["is_fraud"]

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    with mlflow.start_run() as run:
        pipe = build_pipeline(numeric, categorical)
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]

        metrics = {
            "roc_auc": float(roc_auc_score(y_te, proba)),
            "pr_auc": float(average_precision_score(y_te, proba)),
            "brier": float(brier_score_loss(y_te, proba)),
            "recall_at_1pct_alert_rate": _recall_at_alert_rate(y_te.values, proba, 0.01),
            "base_rate": float(y_te.mean()),
        }
        params = {
            "n_train": len(train_df),
            "n_test": len(test_df),
            "split": "temporal",
            "feature_view": view.name,
        }
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tags({
            "feature_view": view.name,
            # Serving compares this to the live store's version and 409s on mismatch.
            "feature_view_version": view.version,
            "train_end_ts": str(train_df["event_ts"].max()),
        })

        # Reference distribution for the drift monitor.
        ref_path = Path("artifacts/reference_profile.json")
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        ref_path.write_text(json.dumps(_profile(X_tr, numeric, categorical), indent=2))
        mlflow.log_artifact(str(ref_path), artifact_path="monitoring")

        signature = infer_signature(X_te.head(200), proba[:200])
        mlflow.sklearn.log_model(
            pipe,
            name="model",
            signature=signature,
            input_example=X_te.head(5),
            registered_model_name=MODEL_NAME if register else None,
        )

        if register:
            client = mlflow.MlflowClient()
            versions = client.search_model_versions(f"name='{MODEL_NAME}'")
            latest = max(versions, key=lambda v: int(v.version))
            client.set_registered_model_alias(MODEL_NAME, "champion", latest.version)
            log.info("registered %s v%s as @champion", MODEL_NAME, latest.version)

        log.info("run %s metrics=%s", run.info.run_id, metrics)
        return {"run_id": run.info.run_id, **metrics}


def _recall_at_alert_rate(y: np.ndarray, proba: np.ndarray, rate: float) -> float:
    """The metric a fraud team actually cares about: catch rate at a fixed
    review budget, rather than a threshold-free aggregate."""
    k = max(1, int(len(proba) * rate))
    top = np.argsort(-proba)[:k]
    positives = y.sum()
    return float(y[top].sum() / positives) if positives else 0.0


def _profile(X: pd.DataFrame, numeric: list[str], categorical: list[str]) -> dict:
    prof: dict = {"numeric": {}, "categorical": {}}
    for c in numeric:
        q = np.quantile(X[c].astype(float), np.linspace(0, 1, 11)).tolist()
        prof["numeric"][c] = {"quantiles": q, "mean": float(X[c].mean()), "std": float(X[c].std())}
    for c in categorical:
        prof["categorical"][c] = X[c].value_counts(normalize=True).to_dict()
    return prof


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="data/events.parquet")
    ap.add_argument("--tracking-uri", default="http://localhost:5000")
    ap.add_argument("--no-register", action="store_true")
    args = ap.parse_args()
    print(train(args.events, tracking_uri=args.tracking_uri, register=not args.no_register))


if __name__ == "__main__":
    main()
