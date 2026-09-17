"""End-to-end API tests. A session-scoped fixture trains a real model
before the app starts, so these tests exercise the actual FastAPI app
against an actual MLflow-loaded model — not mocks.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="session", autouse=True)
def trained_model():
    from serving.train import train_and_register
    train_and_register(n_samples=500, random_state=1)


@pytest.fixture(scope="session")
def client(trained_model):
    from serving.app import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_uri"].startswith("runs:/")


def test_predict_returns_valid_response(client):
    resp = client.post("/predict", json={"features": [0.1, 0.2, -0.3, 0.4, -0.5]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] in (0, 1)
    assert "model_uri" in body


def test_predict_rejects_wrong_feature_count(client):
    resp = client.post("/predict", json={"features": [0.1, 0.2]})
    assert resp.status_code == 422


def test_metrics_endpoint_exposes_prometheus_format(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert b"predictions_total" in resp.content


def test_drift_check_fires_after_window_fills(client):
    # send enough requests to fill the drift-check rolling window
    last_resp = None
    for _ in range(20):
        last_resp = client.post("/predict", json={"features": [0.1, 0.2, -0.3, 0.4, -0.5]})
    body = last_resp.json()
    assert body["drift_check"] is not None
    assert "any_drift" in body["drift_check"]
