"""Serving-layer tests with the model and store stubbed out.

The point of these is the *contract*: skew rejection, missing-feature 422s,
and the shape of the inference log -- not the model's accuracy.
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.serving import app as app_module


class _StubModel:
    metadata = type("M", (), {"run_id": "stub-run"})()

    def predict(self, X):
        return np.array([0.91])


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "INFERENCE_LOG", tmp_path / "log.parquet")
    app_module.STATE.update({
        "model": _StubModel(),
        "model_version": "stub-run",
        "trained_fv_version": app_module.view.version,
        "buffer": [],
    })
    monkeypatch.setattr(app_module.store, "ping", lambda: True)
    with TestClient(app_module.app) as c:
        yield c


def _full_features():
    return {
        "amount_log": 4.2, "txn_count_24h": 3.0, "amount_vs_mean_30d": 1.4,
        "seconds_since_prev": 900.0, "is_foreign": 1, "merchant_risk_bucket": "crypto",
    }


def test_health_ok(client, monkeypatch):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["skew_check"] == "pass"


def test_predict_with_overrides_only(client, monkeypatch):
    monkeypatch.setattr(app_module.store, "read", lambda *a, **k: None)
    r = client.post("/predict", json={"account_id": "a1", "overrides": _full_features()})
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "decline"
    assert body["features_source"] == "override_only"


def test_missing_features_return_422(client, monkeypatch):
    monkeypatch.setattr(app_module.store, "read", lambda *a, **k: None)
    r = client.post("/predict", json={"account_id": "a1", "overrides": {"amount_log": 4.2}})
    assert r.status_code == 422


def test_feature_view_skew_returns_409(client, monkeypatch):
    app_module.STATE["trained_fv_version"] = "deadbeefcafe"
    monkeypatch.setattr(app_module.store, "read", lambda *a, **k: _full_features())
    r = client.post("/predict", json={"account_id": "a1"})
    assert r.status_code == 409
    assert client.get("/health").json()["status"] == "degraded"


def test_store_values_are_used_when_present(client, monkeypatch):
    monkeypatch.setattr(app_module.store, "read", lambda *a, **k: _full_features())
    body = client.post("/predict", json={"account_id": "a1"}).json()
    assert body["features_source"] == "online_store"
