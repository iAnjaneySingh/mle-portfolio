"""CI gate: fail the build if training and serving features have diverged.

Compares the offline training features against features read back from the live
online store for the same (entity, event_ts) pairs. Exits non-zero on failure,
so it can sit in a pipeline stage rather than a notebook.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from src.features.definitions import get_group
from src.features.parity import compare, report
from src.serving.predict import OnlineFeatureClient


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True, help="parquet of offline training features")
    ap.add_argument("--group", default="user_activity")
    ap.add_argument("--redis-url", default="redis://localhost:6379/0")
    ap.add_argument("--sample", type=int, default=2000)
    ap.add_argument("--min-match-rate", type=float, default=0.99)
    args = ap.parse_args()

    group = get_group(args.group)
    batch = pd.read_parquet(args.batch)
    # Only the latest row per entity is comparable: the online store keeps
    # current state, not history.
    latest = batch.sort_values("event_ts").groupby(group.entity_key).tail(1)
    latest = latest.sample(min(args.sample, len(latest)), random_state=0)

    client = OnlineFeatureClient(args.redis_url, args.group)
    rows = []
    for entity_id, ts in zip(latest[group.entity_key], latest["event_ts"]):
        fv = client.get(str(entity_id))
        if not fv.complete:
            continue
        rows.append({group.entity_key: entity_id, "event_ts": ts, **fv.values})

    if not rows:
        print("no online features found -- is the streaming job running?", file=sys.stderr)
        return 2

    online = pd.DataFrame(rows)
    results = compare(latest, online, group,
                      key_cols=(group.entity_key, "event_ts"),
                      min_match_rate=args.min_match_rate)
    print(report(results))
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
