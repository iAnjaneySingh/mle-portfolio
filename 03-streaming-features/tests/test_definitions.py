from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.definitions import USER_ACTIVITY, FeatureGroup, WindowedFeature


def _events():
    """Two users, hand-placed timestamps so every window boundary is checkable."""
    return pd.DataFrame({
        "user_id": ["u1", "u1", "u1", "u2", "u1", "u2"],
        "event_ts": pd.to_datetime([
            "2025-01-01T00:00:00Z", "2025-01-01T00:02:00Z", "2025-01-01T00:04:00Z",
            "2025-01-01T00:04:30Z", "2025-01-01T00:20:00Z", "2025-01-01T00:59:00Z",
        ]),
        "amount": [10.0, 20.0, 30.0, 5.0, 40.0, 7.0],
        "merchant_id": ["a", "b", "a", "c", "d", "c"],
    })


def test_count_5m_window_boundaries():
    f = WindowedFeature("c5", "amount", "count", "5 minutes")
    got = f.compute_pandas(_events(), "user_id")
    # u1 events at 0, 2, 4 min are all inside a 5-minute trailing window;
    # the 20-minute event sees only itself. u2's two events are 55 min apart.
    assert list(got) == [1, 2, 3, 1, 1, 1]


def test_sum_1h_accumulates_per_entity_only():
    f = WindowedFeature("s1h", "amount", "sum", "1 hour")
    got = f.compute_pandas(_events(), "user_id")
    assert got[2] == pytest.approx(60.0)     # u1: 10+20+30
    assert got[3] == pytest.approx(5.0)      # u2 is unaffected by u1's volume
    assert got[4] == pytest.approx(100.0)    # u1: 10+20+30+40


def test_avg_matches_sum_over_count():
    ev = _events()
    s = WindowedFeature("s", "amount", "sum", "1 hour").compute_pandas(ev, "user_id")
    c = WindowedFeature("c", "amount", "count", "1 hour").compute_pandas(ev, "user_id")
    a = WindowedFeature("a", "amount", "avg", "1 hour").compute_pandas(ev, "user_id")
    assert np.allclose(a, s / c)


def test_default_fills_undefined_aggregations():
    f = WindowedFeature("std", "amount", "stddev", "5 minutes", default=0.0)
    got = f.compute_pandas(_events(), "user_id")
    assert got[0] == 0.0, "stddev of a single observation must fall back to the default"


def test_approx_distinct_counts_unique_merchants():
    f = WindowedFeature("dm", "merchant_id", "approx_distinct", "1 hour")
    got = f.compute_pandas(_events(), "user_id")
    assert got[2] == 2   # u1 saw merchants a, b, a


def test_version_changes_with_window_length():
    a = FeatureGroup("g", "user_id", (WindowedFeature("x", "amount", "sum", "1 hour"),))
    b = FeatureGroup("g", "user_id", (WindowedFeature("x", "amount", "sum", "2 hours"),))
    assert a.version() != b.version()


def test_version_changes_with_aggregation():
    a = FeatureGroup("g", "user_id", (WindowedFeature("x", "amount", "sum", "1 hour"),))
    b = FeatureGroup("g", "user_id", (WindowedFeature("x", "amount", "avg", "1 hour"),))
    assert a.version() != b.version()


def test_group_collapses_shared_windows():
    # 9 features but only 3 distinct window specs -> 3 streaming aggregations,
    # not one per feature. This is the difference between 3 state stores and 9.
    assert len(USER_ACTIVITY.by_window()) == 3
    assert len(USER_ACTIVITY.features) == 9
