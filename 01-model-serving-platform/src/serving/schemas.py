from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    account_id: str = Field(..., min_length=1, max_length=64)
    # Request-time overrides let callers pass features that only exist at the
    # moment of the transaction (amount, merchant) while everything historical
    # comes from the online store.
    overrides: dict[str, Any] = Field(default_factory=dict)


class BatchPredictRequest(BaseModel):
    requests: list[PredictRequest] = Field(..., max_length=500)


class PredictResponse(BaseModel):
    account_id: str
    score: float
    decision: Literal["approve", "review", "decline"]
    model_version: str
    feature_view_version: str
    features_source: Literal["online_store", "online_store+override", "override_only"]
    latency_ms: float
    request_id: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None
    redis: bool
    feature_view: str
    feature_view_version: str
    skew_check: Literal["pass", "fail", "unknown"]
