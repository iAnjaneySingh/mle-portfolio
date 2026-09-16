"""Batch backfill using the SAME feature specs as the streaming job.

Two modes, both driven by `FeatureGroup`:

  `compute_reference(events)` -- pure pandas, exact trailing windows. This is the
  reference implementation the parity harness scores Spark against. It is exact
  but does not scale past one machine.

  `compute_spark(events)`     -- Spark batch over the same specs, for real
  historical volumes. Identical aggregation semantics, distributed execution.

Both emit one row per event with features computed over (t - window, t], which
is the point-in-time correct definition: an event's feature vector reflects only
what had happened when that event occurred.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..features.definitions import FeatureGroup, get_group


def compute_reference(events: pd.DataFrame, group: FeatureGroup) -> pd.DataFrame:
    """Exact trailing-window features, one row per event."""
    df = events.sort_values("event_ts").reset_index(drop=True)
    df["event_ts"] = pd.to_datetime(df["event_ts"], utc=True)

    out = df[[group.entity_key, "event_ts"]].copy()
    for feat in group.features:
        out[feat.name] = feat.compute_pandas(df, group.entity_key)
    out.attrs["feature_group_version"] = group.version()
    return out


def compute_spark(events_path: str, group: FeatureGroup, out_path: str) -> str:
    """Distributed backfill. Uses a range join on the window bounds rather than
    a rolling API, because Spark has no per-key trailing-window aggregate."""
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    spark = SparkSession.builder.appName("feature-backfill").getOrCreate()
    events = spark.read.parquet(events_path)

    result = events.select(group.entity_key, "event_ts")
    for feat in group.features:
        seconds = int(feat.window_timedelta.total_seconds())
        # rangeBetween on an epoch-second column gives an exact trailing window,
        # matching the pandas reference's (t - w, t] semantics.
        w = (
            Window.partitionBy(group.entity_key)
            .orderBy(F.col("event_ts").cast("long"))
            .rangeBetween(-seconds + 1, 0)
        )
        expr = {
            "count": F.count, "sum": F.sum, "avg": F.avg,
            "max": F.max, "min": F.min, "stddev": F.stddev,
            "approx_distinct": F.approx_count_distinct,
        }[feat.agg]
        events = events.withColumn(
            feat.name, F.coalesce(expr(F.col(feat.source_column)).over(w), F.lit(feat.default))
        )

    cols = [group.entity_key, "event_ts", *group.feature_names]
    events.select(*cols).write.mode("overwrite").parquet(out_path)
    del result
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--out", default="data/training_features.parquet")
    ap.add_argument("--group", default="user_activity")
    ap.add_argument("--engine", choices=["pandas", "spark"], default="pandas")
    args = ap.parse_args()

    group = get_group(args.group)
    if args.engine == "spark":
        print(compute_spark(args.events, group, args.out))
        return

    events = pd.read_parquet(args.events)
    out = compute_reference(events, group)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"wrote {len(out):,} rows -> {args.out} (group version {group.version()})")


if __name__ == "__main__":
    main()
