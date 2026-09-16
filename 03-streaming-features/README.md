# Real-Time Feature Computation with Train/Serve Parity

Spark Structured Streaming computes windowed aggregations from Kafka into a
Redis online store and a Delta offline store. A batch backfill computes the same
features for training. A parity harness proves the two agree — and measures
exactly where and why they can't.

Train/serve skew is the most common way a working model degrades in production
without anyone noticing, because both halves are individually correct and
nothing errors. This repo is an argument about how to make that structurally
hard.

## The core idea: one declaration, two compilers

```python
WindowedFeature("amount_sum_1h", source_column="amount", agg="sum", window="1 hour")
```

That declaration compiles two ways:

- `to_spark_agg()` → a Spark aggregate column used by the streaming job
- `compute_pandas()` → the exact reference implementation used for training backfill

Neither compiler owns the window length, the aggregation function, or the source
column, so neither can drift from the other on those. `FeatureGroup.version()` is
a hash of the declarations, so changing a window changes the Redis keyspace
rather than silently corrupting it.

What the compilers *can* still disagree on is semantics — boundary inclusivity,
null handling, `ddof`, whether the current event is in its own window. That is
what the parity harness is for.

## What the parity harness found

`tests/test_train_serve_parity.py` replays events through an incremental
aggregator modelling Spark's state store (arrival-order consumption, event-time
state, watermark frontier) and compares against the batch reference:

| Scenario | Result |
|---|---|
| In-order stream | **exact parity**, all 9 features |
| 20% of events up to 2 min late, 10-min watermark | **~98.6% row match** — nothing dropped, yet features still differ |
| Same events replayed in event-time order (converged state) | **exact parity** again |
| 25% late up to 1 hour, 1-min watermark | rows dropped; divergence quantified per feature |

The middle row is the useful finding, and it took building the harness to see it.
Nothing was dropped, both implementations are correct, and they still disagree —
because when a late event is scored online, other events belonging in its window
haven't arrived yet. The batch job, running afterwards, sees all of them. This is
irreducible for any streaming system: it's the cost of answering *now* instead of
*later*. The point is that it's a measured number rather than an assumed zero. If
a model is sensitive at the 1% level, the fix is to train on features
reconstructed the way serving computes them — not to pretend the gap doesn't
exist.

Two real bugs this caught while being written, both invisible to any unit test on
either side alone:

- pandas 2.x can carry **microsecond** resolution, so `astype("int64")` yields
  µs while `Timedelta.value` is ns — a 1000× window inflation that produces
  perfectly plausible numbers.
- Appending to the state buffer in **arrival** order and evicting from the head
  corrupts every window as soon as one event arrives late. State has to be held
  in event-time order and evicted against the watermark frontier.

## Streaming job design

`src/streaming/job.py`, with the reasoning inline:

- **Event time, not processing time.** Processing-time windows make features a
  function of your Kafka lag — a skew generator by construction.
- **Watermarks bound state and drop data.** Both halves are true; the job counts
  drops so the tradeoff is observable.
- **`update` output mode, not `append`.** `append` holds every window until its
  watermark expires, adding a full watermark of latency to every online feature.
- **Effectively-once without distributed transactions.** The Kafka source is
  exactly-once into Delta but at-least-once into an arbitrary sink. Redis writes
  are keyed by `(group, version, entity)` and are pure overwrites guarded by a
  stored `_window_end`, so replaying a batch is a no-op and an out-of-order
  micro-batch can't overwrite newer state.
- **Delta before Redis.** If the job dies between sinks, the online store is
  stale but never *ahead of* the training record.
- **9 features, 3 aggregations.** Features sharing a window spec are collapsed
  into one `groupBy`, so it's 3 state stores rather than 9.

## Serving

`src/serving/predict.py` returns a score plus `staleness_seconds`, `fresh`, and
`degraded`. Every online vector carries the `window_end` it was computed from; a
model quietly scoring hour-old features looks perfectly healthy on every
dashboard you have, which is why staleness is part of the response rather than a
log line.

## Run it

```bash
pip install -r requirements.txt
make up           # kafka + redis
make produce      # synthetic stream: bursts, out-of-order events, duplicates
make stream       # spark-submit the streaming job
make dump         # same generator, to parquet
make backfill     # batch features from the same declarations
make parity       # CI gate: batch vs live online store, non-zero exit on failure
make test
```

The producer deliberately injects out-of-order events, events beyond the
watermark, bursts, and duplicate `event_id`s. A stream that arrives perfectly
ordered proves nothing.

## Known limits

- One entity key per feature group. Cross-entity features (merchant-level
  aggregates joined onto user events) need a stream-stream join with its own
  watermark on both sides.
- `approx_count_distinct` is HyperLogLog in Spark and exact `nunique` in the
  pandas reference, so parity on that feature holds only within HLL's error
  bound. It is called out as approximate rather than quietly tolerated.
- The Redis online store is single-node. Key layout already shards by entity, so
  Cluster is a config change rather than a rewrite.
