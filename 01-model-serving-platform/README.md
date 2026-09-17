# MLOps Model-Serving Platform

FastAPI model serving backed by an MLflow-tracked model, with covariate
drift detection and Prometheus metrics — the actual pieces of a serving
platform, built and verified, not just described.

## What it handles

- **MLflow-tracked training**: `serving/train.py` trains a model, logs
  params/metrics/the model itself to a local MLflow registry, and saves the
  training feature distribution as a drift-detection reference
- **Model loading at serve time**: the FastAPI app loads the model via its
  MLflow run URI (`runs:/<run_id>/model`) at startup — the serving code
  never touches a raw pickle file directly, it goes through MLflow's model
  interface
- **Covariate drift detection**: a rolling buffer of the last 20 requests is
  compared against the training-time reference distribution via a two-sample
  KS test per feature (`serving/monitoring.py`) — flagged in the response,
  not silently ignored
- **Prometheus metrics**: `/metrics` exposes real counters
  (`predictions_total`, `drift_checks_total`, `drift_detected_total`) in
  standard Prometheus text format

## Architecture

```
train.py --logs model+metrics--> MLflow (local ./mlruns)
                                       |
                              model_uri (runs:/...)
                                       |
                     app.py loads model at startup
                                       |
              POST /predict --> prediction + rolling drift check
                                       |
                        GET /metrics --> Prometheus counters
```

## Run it

```bash
pip install -r requirements-dev.txt
python -m serving.train        # trains + registers a model, ~2 sec
uvicorn serving.app:app --port 8000
```

Verified against a live server in this environment:

```
$ curl localhost:8000/health
{"status":"ok","model_uri":"runs:/29369e8d.../model"}

$ curl -X POST localhost:8000/predict -d '{"features":[0.1,0.2,-0.3,0.4,-0.5]}'
{"prediction":1,"model_uri":"runs:/29369e8d.../model","drift_check":null}

$ curl localhost:8000/metrics | grep predictions_total
predictions_total 1.0
```

(`drift_check` is `null` until 20 requests have accumulated in the rolling
buffer — a single request is too small a sample for a meaningful
distribution comparison.)

## Run tests

```bash
pytest tests/ -v
```

8 tests, all executed against real components — a real trained model, a
real running FastAPI app (via `TestClient`), real KS-test drift detection
on synthetic shifted/unshifted distributions. No mocks standing in for the
actual model or actual statistical test.

## Two real bugs found and fixed while building this

Documented here on purpose — this is what "actually building and testing
it" catches that "describing the architecture" doesn't:

1. **MLflow's sklearn flavor defaults to `skops` serialization**, which
   refused to save a `RandomForestClassifier` because it flags
   `sklearn.tree._tree.Tree` as an untrusted type (a legitimate security
   default — that type's raw node indices aren't bounds-checked on load).
   Fixed by explicitly using `serialization_format="pickle"` in
   `train.py`, which is the documented, supported alternative for models
   you trained yourself and trust.
2. **`scipy.stats.ks_2samp` returns numpy bool types**, not Python
   `bool` — `np.True_ is True` is `False` in Python, which silently broke
   an `is True` assertion in the original test. Fixed by explicitly
   casting with `bool(...)` in `monitoring.py`.
