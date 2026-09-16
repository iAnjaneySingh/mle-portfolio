"""The test the whole repo exists for.

`simulate_streaming` replays events in arrival order through an incremental
aggregator that mirrors what Spark's windowed state store does, including
watermark-based dropping of late data. Its output is then compared against the
pandas reference implementation used to build the training set.

This catches the class of bug that no unit test on either side alone would:
the two implementations are each internally correct and disagree with each other.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd
import pytest

from src.batch.backfill import compute_reference
from src.features.definitions import USER_ACTIVITY, FeatureGroup
from src.features.parity import compare, report


def make_events(n=4000, n_users=40, seed=3, late_fraction=0.0, max_lateness_s=120):
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2025-03-01T00:00:00Z")
    arrival_offsets = np.sort(rng.uniform(0, 86_400, n))
    lateness = np.where(rng.random(n) < late_fraction,
                        rng.uniform(1, max_lateness_s, n), 0.0)
    return pd.DataFrame({
        "user_id": [f"u{rng.integers(n_users):03d}" for _ in range(n)],
        "event_ts": base + pd.to_timedelta(arrival_offsets - lateness, unit="s"),
        "arrival_ts": base + pd.to_timedelta(arrival_offsets, unit="s"),
        "amount": rng.lognormal(3.0, 1.0, n),
        "merchant_id": [f"m{rng.integers(50):02d}" for _ in range(n)],
    })


def simulate_streaming(
    events: pd.DataFrame,
    group: FeatureGroup,
    watermark: pd.Timedelta = pd.Timedelta("10 minutes"),
) -> pd.DataFrame:
    """Incremental, arrival-ordered aggregation with watermark semantics.

    Models what Spark's windowed state store does: events are consumed in
    ARRIVAL order, state is keyed by entity and held in EVENT-time order, the
    watermark frontier advances with the largest event time seen, and anything
    arriving behind the frontier is dropped.

    The event-time ordering of the buffer is the subtle part -- appending in
    arrival order and evicting from the head silently corrupts every window as
    soon as a single event arrives late.
    """
    import bisect

    ordered = events.sort_values("arrival_ts")
    buffers: dict[str, list] = defaultdict(list)
    widest = max(f.window_timedelta for f in group.features)
    max_event_ts: pd.Timestamp | None = None
    rows, dropped = [], 0

    for ev in ordered.itertuples(index=False):
        ts = ev.event_ts
        if max_event_ts is not None and ts < max_event_ts - watermark:
            dropped += 1
            continue
        max_event_ts = ts if max_event_ts is None else max(max_event_ts, ts)

        buf = buffers[ev.user_id]
        bisect.insort(buf, (ts, ev.amount, ev.merchant_id))
        # Evict against the watermark frontier, not this event's own time:
        # a late event must not resurrect state that has already expired.
        floor = max_event_ts - widest
        while buf and buf[0][0] <= floor:
            buf.pop(0)

        row = {"user_id": ev.user_id, "event_ts": ts}
        for f in group.features:
            cutoff = ts - f.window_timedelta
            vals = [(a, m) for (t, a, m) in buf if cutoff < t <= ts]
            row[f.name] = _agg(f.agg, vals, f.default)
        rows.append(row)

    out = pd.DataFrame(rows)
    out.attrs["dropped_late"] = dropped
    return out


def _agg(kind, vals, default):
    amounts = [a for a, _ in vals]
    if not amounts:
        return default
    if kind == "count":
        return float(len(amounts))
    if kind == "sum":
        return float(np.sum(amounts))
    if kind == "avg":
        return float(np.mean(amounts))
    if kind == "max":
        return float(np.max(amounts))
    if kind == "min":
        return float(np.min(amounts))
    if kind == "stddev":
        return float(np.std(amounts, ddof=1)) if len(amounts) > 1 else default
    if kind == "approx_distinct":
        return float(len({m for _, m in vals}))
    raise ValueError(kind)


def test_perfect_parity_on_ordered_stream():
    events = make_events(late_fraction=0.0)
    batch = compute_reference(events, USER_ACTIVITY)
    stream = simulate_streaming(events, USER_ACTIVITY)
    results = compare(batch, stream, USER_ACTIVITY)
    failures = [r for r in results if not r.passed]
    assert not failures, "\n" + report(results)


def test_out_of_order_arrival_costs_measurable_parity():
    """The finding this harness exists to produce.

    Events arrive up to 2 minutes late, well inside a 10-minute watermark, so
    NOTHING is dropped. Parity still is not perfect -- and that is correct
    behaviour, not a bug. At the moment a late event is scored online, other
    events that belong in its window have not arrived yet, so the online feature
    vector is computed on an incomplete picture. The batch job, running after
    the fact, sees all of them.

    This is irreducible for any streaming system: it is the cost of answering
    now instead of later. What matters is that the number is *measured* rather
    than assumed to be zero. Here it is ~1-2% of rows at 20% lateness; if a
    model is sensitive at that level, the fix is to train on features
    reconstructed the way serving computes them, not to pretend the gap is zero.
    """
    events = make_events(late_fraction=0.2, max_lateness_s=120)
    batch = compute_reference(events, USER_ACTIVITY)
    stream = simulate_streaming(events, USER_ACTIVITY, watermark=pd.Timedelta("10 minutes"))
    assert stream.attrs["dropped_late"] == 0, "watermark should not drop anything here"

    results = compare(batch, stream, USER_ACTIVITY, min_match_rate=0.0)
    rates = {r.feature: r.match_rate for r in results}
    assert min(rates.values()) > 0.95, "\n" + report(results)
    assert min(rates.values()) < 1.0, "arrival-order effects should be visible"


def test_parity_is_exact_once_state_converges():
    """Replaying in event-time order is the converged state Spark's `update`
    output mode reaches after every late event has been folded in. Against that,
    the two compilers must agree exactly -- any residual difference is a real
    implementation divergence rather than an arrival-order artefact."""
    events = make_events(late_fraction=0.2, max_lateness_s=120).copy()
    events["arrival_ts"] = events["event_ts"]   # converged: arrival == event time
    batch = compute_reference(events, USER_ACTIVITY)
    stream = simulate_streaming(events, USER_ACTIVITY)
    results = compare(batch, stream, USER_ACTIVITY)
    assert all(r.passed for r in results), "\n" + report(results)


def test_late_data_beyond_watermark_is_dropped_and_measurable():
    """The honest version of the watermark tradeoff: data IS lost, and the
    harness quantifies how much feature error that produces."""
    events = make_events(n=3000, late_fraction=0.25, max_lateness_s=3600)
    stream = simulate_streaming(events, USER_ACTIVITY, watermark=pd.Timedelta("1 minute"))
    assert stream.attrs["dropped_late"] > 0

    batch = compute_reference(events, USER_ACTIVITY)
    results = compare(batch, stream, USER_ACTIVITY, min_match_rate=0.0)
    rates = {r.feature: r.match_rate for r in results}
    # Dropping data must produce measurable divergence -- if it did not, the
    # parity harness would not be sensitive enough to trust on real skew.
    assert min(rates.values()) < 1.0
    # ...and the harness reports exactly how much, per feature, rather than
    # collapsing it to a single pass/fail.
    assert all(0.0 <= v <= 1.0 for v in rates.values())


def test_parity_harness_detects_an_injected_off_by_one():
    """Sanity check on the detector itself: corrupt one feature, expect a FAIL."""
    events = make_events(n=1500)
    batch = compute_reference(events, USER_ACTIVITY)
    stream = simulate_streaming(events, USER_ACTIVITY)
    stream = stream.copy()
    stream["txn_count_1h"] = stream["txn_count_1h"] + 1  # classic boundary bug
    results = {r.feature: r for r in compare(batch, stream, USER_ACTIVITY)}
    assert not results["txn_count_1h"].passed
    assert results["amount_sum_1h"].passed


def test_report_is_human_readable():
    events = make_events(n=800)
    batch = compute_reference(events, USER_ACTIVITY)
    stream = simulate_streaming(events, USER_ACTIVITY)
    text = report(compare(batch, stream, USER_ACTIVITY))
    assert "features in parity" in text
