"""Online/serving feature computation via Spark Structured Streaming.

Watches a directory for new tick files (simulating a real-time feed — swap
the file source for a Kafka source in production, as in the companion
event-pipeline project) and computes the SAME features as batch_features.py
via `compute_features`, using foreachBatch to drop into pandas per micro-batch.

Requires a local Spark installation (pip install pyspark) and Java 8/11/17.
Run `scripts/stream_ticks_to_dir.py` in another terminal first to simulate
ticks arriving over time.
"""

import sys
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, TimestampType

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features.shared_features import compute_features

TICK_SCHEMA = StructType([
    StructField("symbol", StringType()),
    StructField("timestamp", TimestampType()),
    StructField("price", DoubleType()),
    StructField("volume", IntegerType()),
])


def process_micro_batch(micro_batch_df, batch_id: int):
    """Called once per micro-batch. Converts to pandas, applies the shared
    feature functions per symbol, and writes results out.

    Converting to pandas caps this pattern's throughput at what fits in a
    single executor's memory per micro-batch — fine for a moderate tick
    volume, and it's the direct trade you're making in exchange for reusing
    training-side pandas logic verbatim instead of reimplementing it in Spark
    SQL. Past that scale, `compute_features`'s window logic would need a
    Spark-native rewrite (or a pandas UDF grouped by symbol via
    `applyInPandas`, which keeps the same function but distributes the groups).
    """
    if micro_batch_df.rdd.isEmpty():
        return

    pdf = micro_batch_df.toPandas()
    results = []
    for symbol, group in pdf.groupby("symbol"):
        group = group.sort_values("timestamp")
        results.append(compute_features(group))

    import pandas as pd
    out = pd.concat(results)

    out_dir = Path(__file__).resolve().parent.parent / "data" / "streaming_features"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"batch_{batch_id}.csv"
    out.to_csv(out_path, index=False)
    print(f"[batch {batch_id}] wrote {len(out)} feature rows to {out_path}")


def run():
    spark = SparkSession.builder.appName("streaming-feature-pipeline").master("local[2]").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    incoming_dir = str(Path(__file__).resolve().parent.parent / "data" / "incoming_ticks")
    Path(incoming_dir).mkdir(parents=True, exist_ok=True)

    stream_df = (
        spark.readStream
        .schema(TICK_SCHEMA)
        .option("header", "true")
        .option("maxFilesPerTrigger", 1)
        .csv(incoming_dir)
        .withWatermark("timestamp", "1 minute")
    )

    query = (
        stream_df.writeStream
        .foreachBatch(process_micro_batch)
        .outputMode("append")
        .trigger(processingTime="10 seconds")
        .start()
    )
    query.awaitTermination()


if __name__ == "__main__":
    run()
