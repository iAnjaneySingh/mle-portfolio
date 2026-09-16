# MLE Portfolio

Three repositories covering the parts of ML engineering that "trained a model,
got 0.94 AUC" resumes skip: serving infrastructure, honest evaluation, and
feature-pipeline correctness.

| | Repo | What it demonstrates |
|---|---|---|
| 1 | [`01-model-serving-platform`](01-model-serving-platform) | Versioned feature store (Redis + parquet), MLflow registry, FastAPI inference, drift monitoring |
| 2 | [`02-backtest-harness`](02-backtest-harness) | Event-driven backtester, deflated Sharpe, purged walk-forward, MLflow-tracked runs |
| 3 | [`03-streaming-features`](03-streaming-features) | Spark Structured Streaming aggregations with a train/serve parity gate |

A theme runs through all three: **make the failure mode a build error.** Feature
skew becomes a 409 instead of a silent accuracy regression. Lookahead bias
becomes a failing test instead of an inflated Sharpe. Streaming/batch divergence
becomes a non-zero exit code instead of a model that degrades in month three.

## Suggested resume framing

Bullets that lead with the engineering decision rather than the tool list —
tools go in the skills section, judgment goes in the bullets:

**Model serving platform**
- Built a feature store whose version is a hash of its transform source code, so
  a changed feature lands in a new keyspace and the inference service rejects
  requests against a model trained on the old one — turning training/serving
  skew from a silent regression into a deployment error.
- Implemented point-in-time-correct training joins (`merge_asof` with tolerance)
  with a regression test that fails if future data leaks into a label row.
- Shipped drift monitoring distinguishing covariate, prediction, and concept
  drift, with PSI bin edges fixed to the reference window (pooling understates
  drift) and KS/chi-square p-values reported alongside the heuristic cutoffs.

**Backtest harness**
- Built an event-driven backtester where signals fill at the next bar's open and
  intrabar stop/target ties resolve pessimistically; each invariant is enforced
  by a test rather than a comment.
- Implemented the Deflated Sharpe Ratio with `n_trials` read from the MLflow
  experiment, so a strategy found after 200 parameter sweeps is scored against
  the expected maximum Sharpe of 200 trials rather than against zero.
- Added purged, embargoed walk-forward validation and reported fold-level Sharpe
  consistency, surfacing curve-fitting that a single aggregate number hides.

**Streaming features**
- Designed feature definitions that compile to both a Spark streaming aggregate
  and a pandas reference implementation, so window length and aggregation
  function have exactly one source of truth.
- Built a parity harness as a CI gate; it quantified a ~1.4% feature divergence
  under 20% out-of-order arrival even with nothing dropped, and caught a
  microsecond/nanosecond resolution bug that silently inflated every window 1000x.
- Achieved effectively-once delivery into Redis via version-keyed idempotent
  writes guarded by a stored window watermark, without distributed transactions.

## Interview prep

Each README has a "known limits" section. Those are the questions a good
interviewer will ask, answered before they ask — which is usually a better
signal than the project itself.

All three run locally with `docker compose up` and `make test`. Every test suite
passes; the two failing-by-design scenarios in repo 3 are assertions that the
parity harness *detects* injected bugs, not bugs themselves.
