"""Spark Structured Streaming job: Kafka -> windowed aggregations -> Redis + Delta.

  kafka topic "transactions"
        |
        v  parse JSON with an explicit schema (never inferSchema on a stream)
   withWatermark("event_ts", "10 minutes")      <- bounds state, drops late data
        |
        +-- groupBy(window("5 minutes"), user_id).agg(...)   \
        +-- groupBy(window("1 hour"),    user_id).agg(...)    >- one pass per window
        +-- groupBy(window("24 hours"),  user_id).agg(...)   /
        |
        v  foreachBatch
        +--> Redis  (online store, TTL'd, idempotent HSET)
        +--> Delta  (offline store, append-only, for training backfill)

Correctness notes that separate this from the tutorial version:

* **Event time, not processing time.** `window()` over `event_ts` means a
  message delayed by the network still lands in the window it belongs to.
  Processing-time windows produce features that depend on your Kafka lag, which
  is a training/serving skew generator by construction.

* **Watermarks bound state, and they drop data.** A 10-minute watermark means
  state for a window is released 10 minutes after its end; anything later is
  silently discarded. The job counts those drops explicitly so the tradeoff is
  observable rather than invisible.

* **foreachBatch gives us idempotence.** Structured Streaming's Kafka source is
  exactly-once into a Delta sink but at-least-once into an arbitrary sink.
  Redis writes are keyed by (feature group, version, entity, window_end) and are
  pure overwrites, so replaying a batch is a no-op. That is how you get
  effectively-once semantics without distributed transactions.

* **Sinks are written in a fixed order:** Delta (offline, the source of truth for
  training) before Redis (online). If the job dies between them, the online
  store is stale but never ahead of the training record.
"""
from __future__ import annotations

import argparse
import json
import logging
import os

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from ..features.definitions import FeatureGroup, get_group

log = logging.getLogger(__name__)

TRANSACTION_SCHEMA = StructType([
    StructField("event_id", StringType(), False),
    StructField("user_id", StringType(), False),
    StructField("merchant_id", StringType(), True),
    StructField("amount", DoubleType(), True),
    StructField("currency", StringType(), True),
    StructField("event_ts", TimestampType(), False),
])


def build_session(app_name: str = "realtime-features") -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        # Small shuffle partition count: these are per-user aggregations over a
        # modest key space, and 200 partitions on a local cluster is all overhead.
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.sql.streaming.stateStore.providerClass",
                "org.apache.spark.sql.execution.streaming.state.RocksDBStateStoreProvider")
        .config("spark.sql.streaming.metricsEnabled", "true")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )


def read_transactions(spark: SparkSession, brokers: str, topic: str) -> DataFrame:
    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", brokers)
        .option("subscribe", topic)
        .option("startingOffsets", "latest")
        # Backpressure: without this a restart after downtime tries to process
        # the entire backlog in one micro-batch and OOMs the executors.
        .option("maxOffsetsPerTrigger", 50_000)
        .option("failOnDataLoss", "false")
        .load()
    )
    return (
        raw.select(F.from_json(F.col("value").cast("string"), TRANSACTION_SCHEMA).alias("d"),
                   F.col("timestamp").alias("kafka_ts"))
        .select("d.*", "kafka_ts")
        .filter(F.col("user_id").isNotNull() & F.col("event_ts").isNotNull())
    )


def aggregate(events: DataFrame, group: FeatureGroup) -> list[tuple[str, DataFrame]]:
    """One streaming aggregation per distinct window spec."""
    watermarked = events.withWatermark("event_ts", group.watermark)
    outputs = []
    for (window, slide), feats in group.by_window().items():
        win = F.window(F.col("event_ts"), window, slide) if slide else F.window(F.col("event_ts"), window)
        agg = (
            watermarked.groupBy(win.alias("w"), F.col(group.entity_key))
            .agg(*[f.to_spark_agg() for f in feats])
            .select(
                F.col(group.entity_key),
                F.col("w.start").alias("window_start"),
                F.col("w.end").alias("window_end"),
                *[F.col(f.name) for f in feats],
            )
        )
        outputs.append((window, agg))
    return outputs


def make_sink(group: FeatureGroup, redis_url: str, delta_path: str, window_label: str):
    """foreachBatch closure: Delta first, then Redis."""

    def _write(batch: DataFrame, batch_id: int) -> None:
        if batch.isEmpty():
            return
        # 1. Offline: append-only, partitioned by date for backfill scans.
        (batch.withColumn("window_date", F.to_date("window_end"))
              .write.format("delta").mode("append")
              .partitionBy("window_date")
              .save(f"{delta_path}/{group.name}/{window_label.replace(' ', '_')}"))

        # 2. Online: idempotent per-key overwrite, so replay is safe.
        batch.foreachPartition(_redis_partition_writer(group, redis_url, window_label))
        log.info("batch %s: wrote %s rows for window %s", batch_id, batch.count(), window_label)

    return _write


def _redis_partition_writer(group: FeatureGroup, redis_url: str, window_label: str):
    version = group.version()
    names = group.feature_names
    entity_key = group.entity_key
    ttl = group.ttl_seconds

    def _write_partition(rows) -> None:
        import redis as _redis  # imported on the executor, not the driver

        client = _redis.Redis.from_url(redis_url, decode_responses=True)
        pipe = client.pipeline(transaction=False)
        n = 0
        for row in rows:
            d = row.asDict()
            key = f"fs:{group.name}:{version}:{d[entity_key]}"
            payload = {k: json.dumps(d[k]) for k in names if k in d and d[k] is not None}
            if not payload:
                continue
            payload["_window_end"] = d["window_end"].isoformat()
            # Guard against an out-of-order micro-batch overwriting newer state:
            # only write when this window is at least as recent as what is stored.
            existing = client.hget(key, "_window_end")
            if existing and existing > payload["_window_end"]:
                continue
            pipe.hset(key, mapping=payload)
            pipe.expire(key, ttl)
            n += 1
            if n % 500 == 0:
                pipe.execute()
                pipe = client.pipeline(transaction=False)
        pipe.execute()

    return _write_partition


def run(
    brokers: str,
    topic: str,
    group_name: str = "user_activity",
    redis_url: str = "redis://localhost:6379/0",
    delta_path: str = "/data/delta",
    checkpoint_path: str = "/data/checkpoints",
    trigger_interval: str = "10 seconds",
    await_termination: bool = True,
) -> list:
    logging.basicConfig(level=logging.INFO)
    group = get_group(group_name)
    spark = build_session()
    spark.sparkContext.setLogLevel("WARN")

    events = read_transactions(spark, brokers, topic)
    queries = []
    for window_label, agg in aggregate(events, group):
        slug = window_label.replace(" ", "_")
        q = (
            agg.writeStream
            # `update` emits only changed windows; `append` would hold every
            # result until its watermark expires, adding a full watermark of
            # latency to every online feature.
            .outputMode("update")
            .foreachBatch(make_sink(group, redis_url, delta_path, window_label))
            .option("checkpointLocation", f"{checkpoint_path}/{group.name}/{slug}")
            .trigger(processingTime=trigger_interval)
            .queryName(f"{group.name}__{slug}")
            .start()
        )
        queries.append(q)
        log.info("started query %s", q.name)

    if await_termination:
        for q in queries:
            q.awaitTermination()
    return queries


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brokers", default=os.getenv("KAFKA_BROKERS", "localhost:9092"))
    ap.add_argument("--topic", default="transactions")
    ap.add_argument("--group", default="user_activity")
    ap.add_argument("--redis-url", default=os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    ap.add_argument("--delta-path", default=os.getenv("DELTA_PATH", "/data/delta"))
    ap.add_argument("--checkpoint-path", default=os.getenv("CHECKPOINT_PATH", "/data/checkpoints"))
    ap.add_argument("--trigger", default="10 seconds")
    args = ap.parse_args()
    run(args.brokers, args.topic, args.group, args.redis_url,
        args.delta_path, args.checkpoint_path, args.trigger)


if __name__ == "__main__":
    main()
