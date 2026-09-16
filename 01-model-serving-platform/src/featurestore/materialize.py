"""Materialization: raw events -> offline parquet -> Redis online store.

Run: python -m src.featurestore.materialize --events data/events.parquet
In production this is the same code path a scheduled Airflow task would call.
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd

from .offline import OfflineStore
from .online import OnlineStore
from .spec import get_view

log = logging.getLogger(__name__)


def materialize(events: pd.DataFrame, view_name: str, redis_url: str) -> dict:
    view = get_view(view_name)
    computed = view.compute(events)
    computed["event_ts"] = pd.to_datetime(events["event_ts"], utc=True).values

    offline = OfflineStore()
    path = offline.write(view, computed)

    # Online store only needs the latest row per entity.
    latest = (
        computed.sort_values("event_ts")
        .groupby(view.entity_key, as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    online = OnlineStore(redis_url)
    n = online.write(view, latest)

    return {
        "view": view.name,
        "version": view.version,
        "offline_path": str(path),
        "offline_rows": len(computed),
        "online_entities": n,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--view", default="txn_risk")
    ap.add_argument("--redis-url", default="redis://localhost:6379/0")
    args = ap.parse_args()
    print(materialize(pd.read_parquet(args.events), args.view, args.redis_url))


if __name__ == "__main__":
    main()
