"""Replay events against the live API, including a drifted tail, so the
monitoring dashboard has real traffic to chart."""
from __future__ import annotations

import argparse
import random
import time

import httpx
import pandas as pd

from src.featurestore.spec import get_view


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--events", default="data/events.parquet")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--drift-frac", type=float, default=0.4,
                    help="fraction of traffic drawn from the drifted tail")
    args = ap.parse_args()

    view = get_view("txn_risk")
    events = pd.read_parquet(args.events)
    feats = view.compute(events)

    head, tail = feats.iloc[: int(0.85 * len(feats))], feats.iloc[int(0.85 * len(feats)) :]
    ok = err = 0
    with httpx.Client(base_url=args.url, timeout=10.0) as client:
        t0 = time.perf_counter()
        for _ in range(args.n):
            pool = tail if random.random() < args.drift_frac else head
            row = pool.sample(1).iloc[0]
            payload = {
                "account_id": str(row[view.entity_key]),
                "overrides": {n: _py(row[n]) for n in view.feature_names},
            }
            r = client.post("/predict", json=payload)
            ok += r.status_code == 200
            err += r.status_code != 200
        dt = time.perf_counter() - t0
    print(f"{ok} ok / {err} errors in {dt:.1f}s ({args.n / dt:.0f} rps)")


def _py(v):
    return v.item() if hasattr(v, "item") else v


if __name__ == "__main__":
    main()
