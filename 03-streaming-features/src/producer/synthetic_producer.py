"""Kafka producer that emits transactions with realistic pathologies.

A stream that arrives perfectly ordered proves nothing. This producer injects:
  * out-of-order events (bounded by `max_lateness`)
  * a configurable fraction of events later than the watermark, which SHOULD be
    dropped -- the parity harness then tells you exactly how much feature error
    that costs
  * bursts, so velocity features have something to fire on
  * duplicate event_ids, to exercise idempotent writes
"""
from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timedelta, timezone

import numpy as np

MERCHANTS = [f"m_{i:03d}" for i in range(200)]


def make_event(user_id: str, ts: datetime, rng: np.random.Generator) -> dict:
    return {
        "event_id": f"e_{rng.integers(0, 2**40)}",
        "user_id": user_id,
        "merchant_id": rng.choice(MERCHANTS),
        "amount": float(np.round(rng.lognormal(3.2, 1.1), 2)),
        "currency": "USD",
        "event_ts": ts.isoformat(),
    }


def generate_stream(
    n_users: int = 500,
    rate_per_second: int = 200,
    duration_seconds: int = 600,
    late_fraction: float = 0.05,
    very_late_fraction: float = 0.01,
    max_lateness_seconds: int = 300,
    watermark_seconds: int = 600,
    burst_probability: float = 0.02,
    duplicate_fraction: float = 0.005,
    seed: int = 13,
):
    """Yields (event, is_beyond_watermark) tuples."""
    rng = np.random.default_rng(seed)
    users = [f"u_{i:05d}" for i in range(n_users)]
    now = datetime.now(timezone.utc)

    for second in range(duration_seconds):
        wall = now + timedelta(seconds=second)
        count = rate_per_second * (10 if rng.random() < burst_probability else 1)
        for _ in range(count):
            user = rng.choice(users)
            beyond = False
            if rng.random() < very_late_fraction:
                lag = watermark_seconds + rng.integers(1, 600)
                beyond = True
            elif rng.random() < late_fraction:
                lag = rng.integers(1, max_lateness_seconds)
            else:
                lag = 0
            ev = make_event(user, wall - timedelta(seconds=int(lag)), rng)
            yield ev, beyond
            if rng.random() < duplicate_fraction:
                yield dict(ev), beyond


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brokers", default="localhost:9092")
    ap.add_argument("--topic", default="transactions")
    ap.add_argument("--rate", type=int, default=200)
    ap.add_argument("--duration", type=int, default=600)
    ap.add_argument("--users", type=int, default=500)
    ap.add_argument("--dump", default=None, help="write to parquet instead of Kafka")
    args = ap.parse_args()

    stream = generate_stream(args.users, args.rate, args.duration)

    if args.dump:
        import pandas as pd

        rows = [e for e, _ in stream]
        df = pd.DataFrame(rows)
        df["event_ts"] = pd.to_datetime(df["event_ts"], utc=True)
        df.to_parquet(args.dump, index=False)
        print(f"wrote {len(df):,} events -> {args.dump}")
        return

    from kafka import KafkaProducer

    producer = KafkaProducer(
        bootstrap_servers=args.brokers,
        value_serializer=lambda v: json.dumps(v).encode(),
        # Key by user so all of a user's events land on one partition. Without
        # this, per-user aggregations still work but state is spread across
        # every partition, and ordering guarantees vanish.
        key_serializer=lambda k: k.encode(),
        linger_ms=20,
        compression_type="gzip",
    )
    sent = late = 0
    t0 = time.perf_counter()
    for ev, beyond in stream:
        producer.send(args.topic, key=ev["user_id"], value=ev)
        sent += 1
        late += beyond
        if sent % args.rate == 0:
            time.sleep(max(0.0, 1.0 - (time.perf_counter() - t0) % 1.0))
    producer.flush()
    print(f"sent {sent:,} events ({late:,} beyond watermark, expected to be dropped)")


if __name__ == "__main__":
    main()
