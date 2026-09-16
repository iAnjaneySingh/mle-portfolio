"""Declarative feature specs that compile to BOTH Spark and pandas.

This is the crux of the repo. The usual training/serving skew story is:

    training:  df.groupby("user").rolling("1h").mean()      # pandas, offline
    serving:   window(col("ts"), "1 hour") ... avg("amount") # Spark, streaming

Two implementations of "the same" feature, written months apart, that silently
disagree on window boundary inclusivity, on how nulls are counted, on whether
the current event is included. The model trains on one and scores on the other,
and nobody notices until the offline AUC stops predicting online performance.

Here a feature is declared once as data. Two compilers consume that declaration:
  * `to_spark_agg()`   -> a Spark aggregate Column for a streaming windowed agg
  * `compute_pandas()` -> the reference implementation used for training backfill

Neither compiler can invent a different window length or aggregation function,
because neither owns those values. What the compilers *can* still get wrong is
boundary semantics -- so `src/features/parity.py` computes both over identical
events and asserts equality. The parity test is the contract; this file just
makes the contract cheap to satisfy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

AggFn = Literal["count", "sum", "avg", "max", "min", "stddev", "approx_distinct"]


@dataclass(frozen=True)
class WindowedFeature:
    """One event-time windowed aggregation over a single entity key."""

    name: str
    source_column: str
    agg: AggFn
    window: str                  # e.g. "1 hour", "24 hours"
    slide: str | None = None     # None = tumbling; set for sliding windows
    default: float = 0.0
    description: str = ""

    # -- shared semantics --------------------------------------------------
    @property
    def window_timedelta(self) -> pd.Timedelta:
        return pd.Timedelta(self.window)

    @property
    def slide_timedelta(self) -> pd.Timedelta:
        return pd.Timedelta(self.slide) if self.slide else self.window_timedelta

    # -- compiler 1: Spark -------------------------------------------------
    def to_spark_agg(self):
        """Returns a pyspark.sql.Column aliased to this feature's name."""
        from pyspark.sql import functions as F

        col = F.col(self.source_column)
        fn = {
            "count": lambda c: F.count(c),
            "sum": lambda c: F.sum(c),
            "avg": lambda c: F.avg(c),
            "max": lambda c: F.max(c),
            "min": lambda c: F.min(c),
            "stddev": lambda c: F.stddev(c),
            "approx_distinct": lambda c: F.approx_count_distinct(c),
        }[self.agg]
        return F.coalesce(fn(col).cast("double"), F.lit(self.default)).alias(self.name)

    # -- compiler 2: pandas (reference) ------------------------------------
    def compute_pandas(
        self, events: pd.DataFrame, entity_key: str, ts_col: str = "event_ts"
    ) -> np.ndarray:
        """Trailing window ending at (and including) each event's timestamp.

        Boundary semantics are stated here once, so the Spark side can be tested
        against them rather than guessed at: the window is **(t - w, t]** --
        open on the left, closed on the right, current event included.

        This is the reference implementation: exact, single-machine, O(n * k)
        where k is the window occupancy. `src/batch/backfill.compute_spark`
        is the distributed version, and the parity harness is what proves they
        agree.
        """
        df = events.sort_values(ts_col).reset_index(drop=True)
        # Work in integer nanoseconds. Two traps here, both silent:
        #   * a tz-aware Series comes back from to_numpy() as an object array of
        #     Timestamps, which searchsorted cannot compare against timedelta64;
        #   * pandas 2.x may carry microsecond resolution, so astype("int64")
        #     without pinning [ns] yields microseconds while Timedelta.value is
        #     nanoseconds -- a 1000x window inflation that no test on either
        #     compiler alone would catch.
        ts = (pd.to_datetime(df[ts_col], utc=True)
              .astype("datetime64[ns, UTC]").astype("int64").to_numpy())
        vals = df[self.source_column].to_numpy()
        out = np.full(len(df), float(self.default), dtype=float)
        w = int(self.window_timedelta.value)

        for positions in df.groupby(entity_key).indices.values():
            gt, gv = ts[positions], vals[positions]
            starts = np.searchsorted(gt, gt - w, side="right")
            for j in range(len(positions)):
                out[positions[j]] = self._apply(gv[starts[j]: j + 1])
        return out

    def _apply(self, values) -> float:
        if len(values) == 0:
            return float(self.default)
        if self.agg == "approx_distinct":
            return float(len({v for v in values if v is not None and v == v}))
        arr = np.asarray(values, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            return float(self.default)
        if self.agg == "count":
            return float(len(arr))
        if self.agg == "sum":
            return float(arr.sum())
        if self.agg == "avg":
            return float(arr.mean())
        if self.agg == "max":
            return float(arr.max())
        if self.agg == "min":
            return float(arr.min())
        if self.agg == "stddev":
            # ddof=1 to match Spark's stddev(), which is the SAMPLE stddev.
            # pandas' Series.std() also defaults to ddof=1; numpy's does not.
            return float(arr.std(ddof=1)) if len(arr) > 1 else float(self.default)
        raise ValueError(f"unsupported aggregation {self.agg}")


@dataclass(frozen=True)
class FeatureGroup:
    name: str
    entity_key: str
    features: tuple[WindowedFeature, ...]
    watermark: str = "10 minutes"
    ttl_seconds: int = 3600

    @property
    def feature_names(self) -> list[str]:
        return [f.name for f in self.features]

    @property
    def windows(self) -> list[str]:
        """Distinct window lengths -- the streaming job runs one aggregation
        per window and unions the results, rather than one job per feature."""
        seen, out = set(), []
        for f in self.features:
            key = (f.window, f.slide)
            if key not in seen:
                seen.add(key)
                out.append(f.window)
        return out

    def by_window(self) -> dict[tuple[str, str | None], list[WindowedFeature]]:
        groups: dict[tuple[str, str | None], list[WindowedFeature]] = {}
        for f in self.features:
            groups.setdefault((f.window, f.slide), []).append(f)
        return groups

    def version(self) -> str:
        import hashlib

        payload = "|".join(
            f"{f.name}:{f.source_column}:{f.agg}:{f.window}:{f.slide}" for f in self.features
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


# --------------------------------------------------------------------------
# Domain: per-user activity features for a real-time risk model.
# --------------------------------------------------------------------------

USER_ACTIVITY = FeatureGroup(
    name="user_activity",
    entity_key="user_id",
    watermark="10 minutes",
    ttl_seconds=7200,
    features=(
        WindowedFeature("txn_count_5m", "amount", "count", "5 minutes",
                        description="velocity: burst detection"),
        WindowedFeature("txn_count_1h", "amount", "count", "1 hour"),
        WindowedFeature("amount_sum_1h", "amount", "sum", "1 hour"),
        WindowedFeature("amount_avg_1h", "amount", "avg", "1 hour"),
        WindowedFeature("amount_max_1h", "amount", "max", "1 hour"),
        WindowedFeature("amount_std_1h", "amount", "stddev", "1 hour", default=0.0),
        WindowedFeature("distinct_merchants_1h", "merchant_id", "approx_distinct", "1 hour"),
        WindowedFeature("txn_count_24h", "amount", "count", "24 hours"),
        WindowedFeature("amount_sum_24h", "amount", "sum", "24 hours"),
    ),
)

GROUPS = {USER_ACTIVITY.name: USER_ACTIVITY}


def get_group(name: str) -> FeatureGroup:
    return GROUPS[name]
