"""Online scoring against the streaming feature store.

The freshness check is the part that matters. Every online feature vector
carries the window_end it was computed from; if that is older than
`max_staleness`, the request is served in a degraded mode rather than scored on
features from an hour ago. A model silently scoring stale features looks
perfectly healthy on every dashboard you have.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import redis

from ..features.definitions import FeatureGroup, get_group

log = logging.getLogger(__name__)


@dataclass
class FeatureVector:
    entity_id: str
    values: dict[str, float]
    window_end: datetime | None
    staleness_seconds: float
    complete: bool

    def is_fresh(self, max_staleness: timedelta) -> bool:
        return self.window_end is not None and self.staleness_seconds <= max_staleness.total_seconds()


class OnlineFeatureClient:
    def __init__(self, redis_url: str = "redis://localhost:6379/0", group_name: str = "user_activity"):
        self.client = redis.Redis.from_url(redis_url, decode_responses=True)
        self.group: FeatureGroup = get_group(group_name)
        self.version = self.group.version()

    def _key(self, entity_id: str) -> str:
        return f"fs:{self.group.name}:{self.version}:{entity_id}"

    def get(self, entity_id: str) -> FeatureVector:
        names = [*self.group.feature_names, "_window_end"]
        raw = self.client.hmget(self._key(entity_id), names)
        mapping = dict(zip(names, raw))

        window_end = None
        if mapping.get("_window_end"):
            window_end = datetime.fromisoformat(mapping["_window_end"])
            if window_end.tzinfo is None:
                window_end = window_end.replace(tzinfo=timezone.utc)

        values, missing = {}, []
        for f in self.group.features:
            v = mapping.get(f.name)
            if v is None:
                missing.append(f.name)
                values[f.name] = f.default
            else:
                values[f.name] = float(json.loads(v))

        staleness = ((datetime.now(timezone.utc) - window_end).total_seconds()
                     if window_end else float("inf"))
        if missing:
            log.debug("cold-start defaults for %s: %s", entity_id, missing)
        return FeatureVector(entity_id, values, window_end, staleness, not missing)


class Scorer:
    def __init__(
        self,
        model,
        client: OnlineFeatureClient,
        max_staleness: timedelta = timedelta(minutes=5),
    ):
        self.model = model
        self.client = client
        self.max_staleness = max_staleness

    def score(self, entity_id: str) -> dict:
        fv = self.client.get(entity_id)
        fresh = fv.is_fresh(self.max_staleness)
        order = self.client.group.feature_names
        x = [[fv.values[n] for n in order]]
        score = float(self.model.predict_proba(x)[0][1]) if hasattr(self.model, "predict_proba") \
            else float(self.model.predict(x)[0])
        return {
            "entity_id": entity_id,
            "score": score,
            "feature_group_version": self.client.version,
            "features_complete": fv.complete,
            "staleness_seconds": round(fv.staleness_seconds, 2),
            "fresh": fresh,
            # A stale vector still returns a score, flagged. The caller decides
            # whether to fall back; silently returning it unlabelled is the bug.
            "degraded": not fresh or not fv.complete,
        }
