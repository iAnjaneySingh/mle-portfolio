# Model Serving Platform

An end-to-end MLOps stack: versioned feature store (Redis online + parquet
offline), MLflow model registry, FastAPI inference service, and a drift
monitoring dashboard. The emphasis is on the parts that break in production,
not on the model itself.

```
raw events ──► materialize ──┬─► offline store (parquet, point-in-time joins) ──► training ──► MLflow registry
                             └─► online store (Redis, TTL'd hashes)                                   │
                                          │                                                           │
                                          └──────────────► FastAPI /predict ◄─────────────────────────┘
                                                                  │
                                                          inference log (parquet)
                                                                  │
                                                       drift job ──► Streamlit dashboard
```

## What is actually interesting here

**One definition of a feature, ever.** `src/featurestore/spec.py` holds a
`FeatureView` whose version is a hash of the *source code* of every transform in
it. Change a transform, the version changes, materialized rows land in a new
Redis keyspace, and the serving layer — which compares the store's version
against the version tagged on the trained model — returns `409` instead of
scoring garbage. Training/serving skew becomes a deployment error rather than a
silent accuracy regression.

**Point-in-time correct training data.** `OfflineStore.get_training_frame` uses
`merge_asof(direction="backward")` with a tolerance, so a label row can only see
feature rows that existed at or before its own timestamp. There is a test
(`test_point_in_time_join_never_leaks_future`) that fails loudly if that breaks.

**Temporal splits and a decision-relevant metric.** Training splits by time, not
randomly, and logs `recall_at_1pct_alert_rate` alongside ROC-AUC — the catch rate
at a fixed review budget, which is what a risk team funds headcount against.

**Registry, not a pickle.** The service loads `models:/txn_risk_classifier@champion`
from MLflow and supports `POST /reload` for hot swaps. Aliases are used instead
of the deprecated stage API.

**Monitoring distinguishes three drifts.** Covariate (PSI + KS per feature),
prediction (PSI on the score distribution), and concept (lagged, once labels
arrive). PSI bin edges come from the *reference* window — pooling the windows is
a common bug that systematically understates drift. The synthetic data generator
deliberately injects a distribution shift in the final 15% of the timeline so the
dashboard has a real alert to show.

## Run it

```bash
pip install -r requirements.txt
make up              # redis + mlflow
make data            # synthetic events with an injected drift tail
make materialize     # offline parquet + online Redis
make train           # trains, evaluates, registers @champion
make serve           # FastAPI on :8000
make traffic         # replay 3k requests, 40% from the drifted tail
make dashboard       # Streamlit on :8501
make test
```

Or `docker compose up --build` for the whole thing.

## API

| Endpoint | Purpose |
|---|---|
| `POST /predict` | Single score. Features come from Redis, with per-request overrides for values only known at transaction time. |
| `POST /predict/batch` | Up to 500 entities per call, pipelined Redis reads. |
| `GET /health` | Model loaded, Redis reachable, **feature-view skew check**. |
| `POST /reload` | Pull the current `@champion` without a restart. |
| `GET /metrics` | Prometheus: latency histogram, feature-store miss counter, score distribution. |

```bash
curl -s localhost:8000/predict -H 'content-type: application/json' -d '{
  "account_id": "acct_00042",
  "overrides": {"amount_log": 6.4, "merchant_risk_bucket": "crypto"}
}' | jq
```

## Known limits

- Single-node Redis with no replication; a real deployment needs Cluster or a
  managed store, and the key layout already shards cleanly by entity.
- Concept drift needs a label-delivery pipeline that this repo stubs out.
- The inference log is parquet-on-disk. At real volume that becomes Kafka into a
  lakehouse table; the writer is isolated in `_flush()` for exactly that swap.
