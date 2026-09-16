"""Redis-backed online feature store.

Design notes worth defending in an interview:
  * Values live in one hash per (view, version, entity), so a read is a single
    round trip and HMGET fetches only the features a model declares.
  * The feature-view version is part of the key. Rolling out a changed
    transform writes to a new keyspace instead of silently corrupting the old
    one; stale keys expire on their own TTL.
  * Writes go through a pipeline: a 50k-row materialization is ~50 round trips
    at batch_size=1000, not 50k.
"""
from __future__ import annotations

import json
import logging
import math
from typing import Any, Iterable, Sequence

import pandas as pd
import redis

from .spec import FeatureView

log = logging.getLogger(__name__)


class OnlineStore:
    def __init__(self, url: str = "redis://localhost:6379/0", batch_size: int = 1000):
        self.client = redis.Redis.from_url(url, decode_responses=True)
        self.batch_size = batch_size

    @staticmethod
    def _key(view: FeatureView, entity_id: str) -> str:
        return f"fs:{view.name}:{view.version}:{entity_id}"

    # -- writes ------------------------------------------------------------
    def write(self, view: FeatureView, frame: pd.DataFrame) -> int:
        """Materialize a computed feature frame into Redis. Returns row count."""
        missing = set(view.feature_names) - set(frame.columns)
        if missing:
            raise ValueError(f"feature frame missing columns: {sorted(missing)}")

        written = 0
        pipe = self.client.pipeline(transaction=False)
        for i, row in enumerate(frame.to_dict(orient="records"), start=1):
            key = self._key(view, str(row[view.entity_key]))
            payload = {n: json.dumps(_jsonable(row[n])) for n in view.feature_names}
            pipe.hset(key, mapping=payload)
            pipe.expire(key, view.ttl_seconds)
            written += 1
            if i % self.batch_size == 0:
                pipe.execute()
                pipe = self.client.pipeline(transaction=False)
        pipe.execute()
        log.info("materialized %s rows into %s@%s", written, view.name, view.version)
        return written

    # -- reads -------------------------------------------------------------
    def read(
        self,
        view: FeatureView,
        entity_id: str,
        features: Sequence[str] | None = None,
    ) -> dict[str, Any] | None:
        names = list(features or view.feature_names)
        values = self.client.hmget(self._key(view, entity_id), names)
        if all(v is None for v in values):
            return None
        return {n: (json.loads(v) if v is not None else None) for n, v in zip(names, values)}

    def read_many(
        self,
        view: FeatureView,
        entity_ids: Iterable[str],
        features: Sequence[str] | None = None,
    ) -> dict[str, dict[str, Any] | None]:
        names = list(features or view.feature_names)
        ids = list(entity_ids)
        pipe = self.client.pipeline(transaction=False)
        for eid in ids:
            pipe.hmget(self._key(view, eid), names)
        results = pipe.execute()
        out: dict[str, dict[str, Any] | None] = {}
        for eid, values in zip(ids, results):
            if all(v is None for v in values):
                out[eid] = None
            else:
                out[eid] = {
                    n: (json.loads(v) if v is not None else None)
                    for n, v in zip(names, values)
                }
        return out

    def ping(self) -> bool:
        try:
            return bool(self.client.ping())
        except redis.RedisError:
            return False


def _jsonable(v: Any) -> Any:
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float) and math.isnan(v):
        return None
    return v
