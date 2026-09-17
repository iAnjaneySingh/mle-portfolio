"""FastAPI model-serving app: loads the MLflow-registered model, serves
predictions, runs periodic drift checks against a rolling buffer of recent
requests, and exposes Prometheus metrics.
"""

import json
from collections import deque
from pathlib import Path
from typing import List

from contextlib import asynccontextmanager

import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel
from starlette.responses import Response

from .monitoring import DriftDetector

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"

DRIFT_CHECK_WINDOW = 20  # run a drift check once this many requests have accumulated

_state = {"model": None, "model_uri": None, "detector": None, "buffer": deque(maxlen=DRIFT_CHECK_WINDOW)}


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="Model Serving Platform", lifespan=lifespan)

predictions_total = Counter("predictions_total", "Total prediction requests served")
drift_checks_total = Counter("drift_checks_total", "Total drift checks run")
drift_detected_total = Counter("drift_detected_total", "Total drift checks that flagged drift")
model_loaded_gauge = Gauge("model_loaded", "1 if a model is currently loaded, else 0")


class PredictRequest(BaseModel):
    features: List[float]


class PredictResponse(BaseModel):
    prediction: int
    model_uri: str
    drift_check: dict | None = None


def load_model():
    model_uri_path = ARTIFACTS_DIR / "model_uri.txt"
    reference_path = ARTIFACTS_DIR / "reference_distribution.json"

    if not model_uri_path.exists() or not reference_path.exists():
        raise RuntimeError(
            "No trained model found. Run `python -m serving.train` before starting the API."
        )

    model_uri = model_uri_path.read_text().strip()
    _state["model"] = mlflow.pyfunc.load_model(model_uri)
    _state["model_uri"] = model_uri

    with open(reference_path) as f:
        reference = json.load(f)
    _state["detector"] = DriftDetector(
        reference_samples=reference["reference_samples"],
        feature_names=reference["feature_names"],
    )
    model_loaded_gauge.set(1)


@app.get("/health")
def health():
    return {"status": "ok", "model_uri": _state["model_uri"]}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    expected_features = len(_state["detector"].feature_names)
    if len(req.features) != expected_features:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {expected_features} features, got {len(req.features)}",
        )

    prediction = int(_state["model"].predict([req.features])[0])
    predictions_total.inc()

    _state["buffer"].append(req.features)

    drift_result = None
    if len(_state["buffer"]) == DRIFT_CHECK_WINDOW:
        drift_result = _state["detector"].check(list(_state["buffer"]))
        drift_checks_total.inc()
        if drift_result["any_drift"]:
            drift_detected_total.inc()

    return PredictResponse(prediction=prediction, model_uri=_state["model_uri"], drift_check=drift_result)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
