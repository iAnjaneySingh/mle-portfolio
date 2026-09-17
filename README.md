# MLE Portfolio

I built these three repositories to work through some of the ML engineering problems that are easy to skip when the main goal is just to train a model and report a metric.

The common problem I kept coming back to was:

**How do I make it harder for an ML system to be wrong without noticing?**

So the three projects focus on different failure modes:

| Repo                                                     | What I built                                                               | Failure mode I wanted to catch                                        |
| -------------------------------------------------------- | -------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| [`01-model-serving-platform`](01-model-serving-platform) | Feature store + model registry + inference API + drift monitoring          | Training and serving using different feature definitions              |
| [`02-backtest-harness`](02-backtest-harness)             | Event-driven backtester + statistical evaluation + walk-forward validation | Backtests looking good because of leakage or repeated experimentation |
| [`03-streaming-features`](03-streaming-features)         | Spark streaming features + pandas reference implementation + parity CI     | Batch and streaming producing different features                      |

The theme across all three is pretty simple:

> **If a failure can be detected automatically, I don't want it to be a comment in the code or a step in someone's memory. I want it to fail.**

---

## 01 — Model Serving Platform

[`01-model-serving-platform`](01-model-serving-platform)

I started with the problem of feature skew.

A model might be trained with one version of a feature transformation and then receive a slightly different version of that feature in production. Nothing necessarily crashes. The API still returns a prediction. The model might even look healthy for a while.

That's a particularly uncomfortable failure because the system is technically working.

### Versioning the features

The feature store uses Redis and Parquet.

Instead of manually maintaining a feature version such as `v1`, `v2`, etc., the version is derived from a hash of the transformation source code.

That means changing the transformation changes the feature version.

The model records which feature version it was trained against, and the inference service checks that version before accepting a request.

So this:

```text
model trained with feature_version=A
request contains feature_version=B
```

doesn't result in a prediction.

It results in a **409**.

That was intentional. I'd rather have an obvious deployment failure than a model silently receiving a different representation of the data.

### Point-in-time correctness

I also wanted to make sure the training data itself wasn't cheating.

The training joins use `merge_asof` with a tolerance so that a training row only sees information that would actually have been available at that point in time.

There is a regression test specifically for future-data leakage.

The useful part of that test isn't that it passes today. It's that if someone changes the join logic later and starts pulling a future observation into a label row, the test fails.

### Drift monitoring

The serving project also includes drift monitoring for:

* Covariate drift
* Prediction drift
* Concept drift

For PSI, I keep the bin edges from the reference window fixed.

This matters because pooling the reference and current distributions to create the bins can make drift look smaller than it actually is. The reference distribution is supposed to remain the reference.

I also report KS and chi-square p-values alongside the heuristic thresholds rather than pretending one number is enough to decide whether something has drifted.

The goal here isn't to build a magical drift detector. It's to make the assumptions behind the monitoring visible.

---

# 02 — Backtest Harness

[`02-backtest-harness`](02-backtest-harness)

The second project came from a different kind of problem: **it's surprisingly easy to write a backtest that gives you a very convincing wrong answer.**

I wanted the execution model and statistical assumptions to be explicit rather than buried inside a large simulation function.

### Event-driven execution

The backtester is event-driven.

Signals generated from a bar don't magically get filled at that same bar's price. Orders fill at the **next bar's open**.

For trades where both a stop and target could have been hit inside the same bar, the simulator resolves the tie pessimistically.

These aren't particularly complicated rules, but they matter.

More importantly, each of these assumptions has a test.

I don't want the backtest to be correct because someone read the implementation carefully. I want the behavior to be enforced.

### The Sharpe ratio problem

Another thing I wanted to handle was repeated experimentation.

Suppose I try 200 parameter combinations and eventually find one with a Sharpe of 2.

Looking at `Sharpe = 2` without knowing about those 200 trials leaves out an important part of the story.

The harness implements the **Deflated Sharpe Ratio**, with `n_trials` pulled from the MLflow experiment rather than manually typed into the evaluation code.

So the evaluation knows that the strategy came from a search involving 200 trials and adjusts the interpretation accordingly.

This is one of those details that is easy to leave out because the resulting Sharpe ratio is less exciting.

I'd rather have the evaluation reflect how the strategy was actually found.

### Walk-forward validation

The backtester also uses purged and embargoed walk-forward validation.

Instead of producing one Sharpe ratio for the entire dataset, it reports the result for each fold.

That makes it easier to see something like:

```text
Fold 1: strong
Fold 2: strong
Fold 3: weak
Fold 4: negative
```

rather than hiding everything behind one aggregate number.

The point isn't that every fold needs to look identical. It's that a single summary statistic shouldn't be allowed to hide instability.

All runs are tracked in MLflow so the evaluation is tied back to the experiment that produced it.

---

# 03 — Streaming Features

[`03-streaming-features`](03-streaming-features)

This one is probably the most practical of the three.

The question was:

**If the training pipeline calculates a feature in pandas and production calculates the same feature in Spark Structured Streaming, how do I know they actually mean the same thing?**

It's very easy for these implementations to slowly diverge.

One might use a 5-minute window while the other uses 300 seconds. One might handle late events differently. One might use a different timestamp resolution.

Everything can look reasonable when you inspect the code.

### One feature definition, two implementations

The feature definitions are designed so the same definition can be used for:

* A Spark Structured Streaming aggregation
* A pandas reference implementation

The point is to avoid having the window size, aggregation function, and related semantics independently defined in two places.

The pandas implementation gives me something straightforward to compare the streaming implementation against.

### The parity test found an actual problem

The parity harness isn't just a unit test around the happy path.

Under **20% out-of-order event arrival**, it measured roughly **1.4% feature divergence**, even though no records were dropped.

That was useful because nothing about the pipeline necessarily looked broken.

The events were there.

The aggregation was running.

The job was producing values.

The two implementations simply weren't producing the same values.

That's exactly the kind of problem I wanted this project to expose.

### The timestamp bug

The parity tests also caught a timestamp resolution issue.

One side of the pipeline was effectively treating timestamps at microsecond resolution while the other was operating at nanosecond resolution.

The result was a window that was effectively **1,000× larger than intended**.

That is a pretty easy bug to miss when looking at a feature definition like:

```text
window = 5 minutes
aggregation = mean
```

The code can look completely reasonable while the actual time arithmetic is wrong.

The parity test made the discrepancy visible.

### Redis delivery

The streaming pipeline writes into Redis using version-keyed idempotent writes guarded by a stored window watermark.

The intent is effectively-once behavior without introducing distributed transactions.

If the same window is processed again, the versioned write doesn't create a second logical state.

Again, the interesting part for me wasn't adding another technology to the stack. It was defining what should happen when the same data gets processed twice and then making the implementation follow that rule.

---

# A note about the failing tests in repo 3

If you run the complete test suite, you'll see two scenarios in repo 3 that are designed to fail.

They are there deliberately.

The tests inject known bugs into the feature calculation and check that the parity harness detects them.

So the test is effectively asking:

```text
"Can I break the streaming implementation in this specific way?"
```

and expecting:

```text
"Yes — CI caught it."
```

They are assertions about the **detection mechanism**, not unresolved failures in the implementation.

---

# Running the projects

Each repository is intended to run locally.

The basic setup is:

```bash
docker compose up
make test
```

The dependencies are containerized where appropriate so that the projects don't require a large external environment just to reproduce the tests.

---

# What I was trying to learn

These aren't meant to be three unrelated demos of Redis, Spark, FastAPI, or MLflow.

I was more interested in the engineering questions underneath them:

**Serving**

> What happens when the feature definition changes after the model was trained?

**Evaluation**

> What happens when the reported result came after trying hundreds of alternatives?

**Streaming**

> What happens when production computes a feature slightly differently from training?

In each case, the dangerous failure is one that produces a plausible result.

A crashed service is usually easy to notice.

A service returning predictions using the wrong feature version is much harder.

A backtest throwing an exception is obvious.

A backtest quietly benefiting from lookahead bias is not.

A streaming job going down is visible.

A streaming job producing features that are 1.4% different from training is much easier to miss.

That's why the design choice across these projects is consistent:

**turn those assumptions into tests, gates, version checks, and explicit errors wherever possible.**

That's the part of ML engineering I wanted this portfolio to spend time on.
