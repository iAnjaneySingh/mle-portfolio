"""FastAPI inference service.

Production-shaped details that separate this from a toy endpoint:
  * The model is pulled from the MLflow registry by alias (`models:/name@champion`),
    not from a pickle on disk, and can be hot-reloaded with POST /reload.
  * On startup the service compares the feature-view version the model was
    trained on against the version the live feature store is serving. A
    mismatch makes /health report `degraded` and predictions return 409 rather
    than quietly scoring on the wrong features.
  * Every prediction is appended to an inference log (parquet) with the exact
    feature vector used. That log is the input to the drift monitor -- you
    cannot monitor what you never wrote down.
  * Prometheus metrics for latency, cache misses and score distribution.
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from ..featurestore.online import OnlineStore
from ..featurestore.spec import get_view
from .schemas import (
    BatchPredictRequest,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)

log = logging.getLogger(__name__)

MODEL_URI = os.getenv("MODEL_URI", "models:/txn_risk_classifier@champion")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
VIEW_NAME = os.getenv("FEATURE_VIEW", "txn_risk")
INFERENCE_LOG = Path(os.getenv("INFERENCE_LOG", "data/inference_log.parquet"))
REVIEW_T = float(os.getenv("REVIEW_THRESHOLD", "0.35"))
DECLINE_T = float(os.getenv("DECLINE_THRESHOLD", "0.80"))

PRED_LATENCY = Histogram("prediction_latency_seconds", "end-to-end predict latency")
FEATURE_MISS = Counter("feature_store_miss_total", "entities absent from online store")
PREDICTIONS = Counter("predictions_total", "predictions served", ["decision"])
SCORE_HIST = Histogram("prediction_score", "score distribution", buckets=[i / 10 for i in range(11)])

STATE: dict[str, Any] = {
    "model": None,
    "model_version": None,
    "trained_fv_version": None,
    "buffer": [],
}


def _load_model() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    STATE["model"] = mlflow.pyfunc.load_model(MODEL_URI)
    meta = STATE["model"].metadata
    STATE["model_version"] = getattr(meta, "run_id", "unknown")
    try:
        client = mlflow.MlflowClient()
        run = client.get_run(STATE["model_version"])
        STATE["trained_fv_version"] = run.data.tags.get("feature_view_version")
    except Exception:  # registry unreachable -> skew check reports "unknown"
        STATE["trained_fv_version"] = None
    log.info("loaded %s (run=%s fv=%s)", MODEL_URI, STATE["model_version"], STATE["trained_fv_version"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _load_model()
    except Exception as exc:  # serve /health even with no model, so k8s can report why
        log.error("model load failed: %s", exc)
    yield
    _flush()


app = FastAPI(title="Model Serving Platform", version="1.0.0", lifespan=lifespan)
store = OnlineStore(REDIS_URL)
view = get_view(VIEW_NAME)


def _skew_status() -> str:
    trained = STATE["trained_fv_version"]
    if trained is None:
        return "unknown"
    return "pass" if trained == view.version else "fail"


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    skew = _skew_status()
    ok = STATE["model"] is not None and store.ping() and skew != "fail"
    return HealthResponse(
        status="ok" if ok else "degraded",
        model_loaded=STATE["model"] is not None,
        model_version=STATE["model_version"],
        redis=store.ping(),
        feature_view=view.name,
        feature_view_version=view.version,
        skew_check=skew,  # type: ignore[arg-type]
    )


@app.post("/reload")
def reload_model() -> dict:
    _load_model()
    return {"reloaded": True, "model_version": STATE["model_version"]}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    return _predict_one(req)


@app.post("/predict/batch", response_model=list[PredictResponse])
def predict_batch(req: BatchPredictRequest) -> list[PredictResponse]:
    return [_predict_one(r) for r in req.requests]


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _predict_one(req: PredictRequest) -> PredictResponse:
    t0 = time.perf_counter()
    if STATE["model"] is None:
        raise HTTPException(503, "model not loaded")
    if _skew_status() == "fail":
        raise HTTPException(
            409,
            f"feature view skew: model trained on {STATE['trained_fv_version']}, "
            f"store serving {view.version}",
        )

    stored = store.read(view, req.account_id)
    if stored is None:
        FEATURE_MISS.inc()
        source = "override_only"
        stored = {}
    else:
        source = "online_store+override" if req.overrides else "online_store"

    features = {**stored, **{k: v for k, v in req.overrides.items() if k in view.feature_names}}
    missing = [n for n in view.feature_names if features.get(n) is None]
    if missing:
        raise HTTPException(422, f"missing features for {req.account_id}: {missing}")

    X = pd.DataFrame([features])[view.feature_names]
    raw = STATE["model"].predict(X)
    score = float(raw[0] if not hasattr(raw, "iloc") else raw.iloc[0])
    decision = "decline" if score >= DECLINE_T else "review" if score >= REVIEW_T else "approve"

    latency = time.perf_counter() - t0
    PRED_LATENCY.observe(latency)
    SCORE_HIST.observe(score)
    PREDICTIONS.labels(decision=decision).inc()

    rid = str(uuid.uuid4())
    _buffer({
        "request_id": rid,
        "ts": pd.Timestamp.utcnow(),
        "account_id": req.account_id,
        "score": score,
        "decision": decision,
        "model_version": STATE["model_version"],
        "feature_view_version": view.version,
        **features,
    })

    return PredictResponse(
        account_id=req.account_id,
        score=score,
        decision=decision,  # type: ignore[arg-type]
        model_version=STATE["model_version"] or "unknown",
        feature_view_version=view.version,
        features_source=source,  # type: ignore[arg-type]
        latency_ms=round(latency * 1000, 3),
        request_id=rid,
    )


def _buffer(row: dict, flush_at: int = 200) -> None:
    STATE["buffer"].append(row)
    if len(STATE["buffer"]) >= flush_at:
        _flush()


def _flush() -> None:
    if not STATE["buffer"]:
        return
    df = pd.DataFrame(STATE["buffer"])
    STATE["buffer"] = []
    INFERENCE_LOG.parent.mkdir(parents=True, exist_ok=True)
    if INFERENCE_LOG.exists():
        df = pd.concat([pd.read_parquet(INFERENCE_LOG), df], ignore_index=True)
    df.to_parquet(INFERENCE_LOG, index=False)
