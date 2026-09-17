MLE Portfolio

Three systems built around ML/trading failure modes that don't announce
themselves — a model silently served on stale features, a backtest that
looks profitable because of lookahead, feature logic that quietly diverges
between training and serving.

| Project | What it does | Tests |
|---|---|---|
| [`01-model-serving-platform`](./01-model-serving-platform) | FastAPI model serving on an MLflow registry, with covariate drift detection | 8/8 passing |
| [`02-backtest-harness`](./02-backtest-harness) | No-lookahead backtest engine with Sharpe/Sortino/drawdown/win-rate, MLflow-tracked runs | 11/11 passing |
| [`03-streaming-features`](./03-streaming-features) | A single feature-computation function shared by a pandas batch path and a Spark Structured Streaming path | 7/7 passing |
=======
Three systems built around ML/trading failure modes that don't announce themselves — a model silently served on stale features, a backtest that looks profitable because of lookahead, feature logic that quietly diverges between training and serving.

Project	What it does	Tests
01-model-serving-platform	FastAPI model serving on an MLflow registry, with covariate drift detection	8/8 passing
02-backtest-harness	No-lookahead backtest engine with Sharpe/Sortino/drawdown/win-rate, MLflow-tracked runs	11/11 passing
03-streaming-features	A single feature-computation function shared by a pandas batch path and a Spark Structured Streaming path	7/7 passing
01 — Model Serving Platform

A FastAPI service that loads a trained model through MLflow's model interface and flags covariate drift on live traffic instead of leaving that to a dashboard nobody's watching.

Design

Training run logs params, metrics, and the model to a local MLflow registry, and snapshots the training feature distribution as a drift reference
The API loads the model by its MLflow run URI at startup — never a raw pickle
A rolling window of the last 20 requests is compared against the reference distribution via a two-sample KS test, per feature
/metrics exposes Prometheus counters (predictions_total, drift_checks_total, drift_detected_total)

Scope: covariate drift only (no PSI/concept drift), no feature-store layer, no feature-version enforcement. A deliberately small, correct surface rather than a large speculative one.

bash
python -m serving.train
uvicorn serving.app:app --port 8000
pytest tests/ -v
02 — Backtest Harness
 fd9be1af44236f8806fa8ab8b0f710bb27e79f92

Built around one non-negotiable rule: a strategy's position on day t is only exposed to day t+1's return. The engine enforces this — it isn't left to the strategy implementation to get right.

Design

A FastAPI service that loads a trained model through MLflow's model
interface and flags covariate drift on live traffic instead of leaving that
to a dashboard nobody's watching.

**Design**
- Training run logs params, metrics, and the model to a local MLflow
  registry, and snapshots the training feature distribution as a drift
  reference
- The API loads the model by its MLflow run URI at startup — never a raw
  pickle
- A rolling window of the last 20 requests is compared against the
  reference distribution via a two-sample KS test, per feature
- `/metrics` exposes Prometheus counters (`predictions_total`,
  `drift_checks_total`, `drift_detected_total`)

**Scope**: covariate drift only (no PSI/concept drift), no feature-store
layer, no feature-version enforcement. A deliberately small, correct
surface rather than a large speculative one.

```bash
python -m serving.train
uvicorn serving.app:app --port 8000
pytest tests/ -v
```

---

## 02 — Backtest Harness

Built around one non-negotiable rule: a strategy's position on day *t* is
only exposed to day *t+1*'s return. The engine enforces this — it isn't
left to the strategy implementation to get right.

**Design**
- Sharpe, Sortino, max drawdown, CAGR, and win rate, each verified against
  hand-computed expected values in the test suite
- Transaction costs charged on every position change, so signal churn has
  a real cost in the simulation
- Every run — parameters, metrics, full equity curve — logged to MLflow
  for side-by-side comparison across parameter sweeps
- A `Strategy` interface any signal source can implement, including a
  multi-agent conviction score

**Scope**: single-pass backtest metrics; no Deflated Sharpe Ratio or
walk-forward validation yet (see Roadmap).

```bash
python scripts/run_backtest.py
mlflow ui
pytest tests/ -v
```

Sample run against the included synthetic price series:
```
[ma_crossover_fast5_slow20]   Sharpe: 0.08   MaxDD: -10.07%  WinRate: 45.60%
[ma_crossover_fast10_slow30]  Sharpe: 0.07   MaxDD: -7.91%   WinRate: 46.60%
[ma_crossover_fast20_slow60]  Sharpe: -0.18  MaxDD: -17.38%  WinRate: 42.00%
```

---

## 03 — Streaming Features

The core problem: a feature computed one way in batch (training) and a
different way in streaming (serving) drifts apart silently. The fix here
is structural rather than procedural — **one function**, imported by both
paths, so there's no second implementation to fall out of sync.

**Design**
- `shared_features.py` — rolling VWAP, rolling volatility, momentum, in
  pure pandas
- `batch_features.py` — calls the shared function directly against
  historical data; produces the training feature set
- `streaming_job.py` — a Spark Structured Streaming job calling the same
  function inside `foreachBatch`, reading from a file source that
  simulates ticks arriving over time

**Requirements**: the streaming path needs a local Spark + Java
installation (not bundled). The batch path and the shared feature logic
run with just `pip install -r requirements.txt`.

```bash
pytest tests/ -v
python features/batch_features.py
```

---

## Roadmap

- [ ] Deflated Sharpe Ratio + purged/embargoed walk-forward validation for the backtest harness
- [ ] PSI and concept-drift detection alongside covariate drift in the serving platform
- [ ] Parity harness quantifying batch/streaming divergence under out-of-order event arrival
- [ ] Feature-version enforcement in the serving API

## Running everything

```bash
pip install -r requirements-dev.txt   # per project
pytest tests/ -v                      # per project
```

## Why these three

Not three unrelated demos of Redis, Spark, FastAPI, and MLflow — three
takes on the same question: what happens when a system keeps producing
plausible output after the thing that made it correct quietly stops being
true? A crashed service is easy to notice. A model serving slightly-wrong
features, or a backtest with unflagged lookahead, is not. Where that could
be turned into a test or a hard failure, it was.
=======
Sharpe, Sortino, max drawdown, CAGR, and win rate, each verified against hand-computed expected values in the test suite
Transaction costs charged on every position change, so signal churn has a real cost in the simulation
Every run — parameters, metrics, full equity curve — logged to MLflow for side-by-side comparison across parameter sweeps
A Strategy interface any signal source can implement, including a multi-agent conviction score

Scope: single-pass backtest metrics; no Deflated Sharpe Ratio or walk-forward validation yet (see Roadmap).

bash
python scripts/run_backtest.py
mlflow ui
pytest tests/ -v

Sample run against the included synthetic price series:

[ma_crossover_fast5_slow20]   Sharpe: 0.08   MaxDD: -10.07%  WinRate: 45.60%
[ma_crossover_fast10_slow30]  Sharpe: 0.07   MaxDD: -7.91%   WinRate: 46.60%
[ma_crossover_fast20_slow60]  Sharpe: -0.18  MaxDD: -17.38%  WinRate: 42.00%
03 — Streaming Features

The core problem: a feature computed one way in batch (training) and a different way in streaming (serving) drifts apart silently. The fix here is structural rather than procedural — one function, imported by both paths, so there's no second implementation to fall out of sync.

Design

shared_features.py — rolling VWAP, rolling volatility, momentum, in pure pandas
batch_features.py — calls the shared function directly against historical data; produces the training feature set
streaming_job.py — a Spark Structured Streaming job calling the same function inside foreachBatch, reading from a file source that simulates ticks arriving over time

Requirements: the streaming path needs a local Spark + Java installation (not bundled). The batch path and the shared feature logic run with just pip install -r requirements.txt.

bash
pytest tests/ -v
python features/batch_features.py
Roadmap
 Deflated Sharpe Ratio + purged/embargoed walk-forward validation for the backtest harness
 PSI and concept-drift detection alongside covariate drift in the serving platform
 Parity harness quantifying batch/streaming divergence under out-of-order event arrival
 Feature-version enforcement in the serving API
Running everything
bash
pip install -r requirements-dev.txt   # per project
pytest tests/ -v                      # per project
Why these three

Not three unrelated demos of Redis, Spark, FastAPI, and MLflow — three takes on the same question: what happens when a system keeps producing plausible output after the thing that made it correct quietly stops being true? A crashed service is easy to notice. A model serving slightly-wrong features, or a backtest with unflagged lookahead, is not. Where that could be turned into a test or a hard failure, it was.
>>>>>>> fd9be1af44236f8806fa8ab8b0f710bb27e79f92
