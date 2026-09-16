"""Offline store: parquet-backed, with point-in-time correct joins.

The join below is the piece most portfolio projects get wrong. A naive
merge(labels, features, on='account_id') leaks the future into training. We use
merge_asof with direction='backward' so every label row only sees feature rows
that existed at or before its own event timestamp.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .spec import FeatureView


class OfflineStore:
    def __init__(self, root: str | Path = "data/offline"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, view: FeatureView) -> Path:
        return self.root / f"{view.name}__{view.version}.parquet"

    def write(self, view: FeatureView, frame: pd.DataFrame) -> Path:
        if "event_ts" not in frame.columns:
            raise ValueError("offline feature frames must carry an event_ts column")
        path = self._path(view)
        frame.sort_values("event_ts").to_parquet(path, index=False)
        return path

    def read(self, view: FeatureView) -> pd.DataFrame:
        path = self._path(view)
        if not path.exists():
            raise FileNotFoundError(
                f"no materialized offline data for {view.name}@{view.version}"
            )
        return pd.read_parquet(path)

    def get_training_frame(
        self,
        view: FeatureView,
        entity_df: pd.DataFrame,
        tolerance: pd.Timedelta | None = pd.Timedelta("7d"),
    ) -> pd.DataFrame:
        """Point-in-time join of features onto (entity, timestamp, label) rows."""
        for col in (view.entity_key, "event_ts"):
            if col not in entity_df.columns:
                raise ValueError(f"entity_df must contain {col}")

        left = entity_df.copy()
        right = self.read(view)
        left["event_ts"] = pd.to_datetime(left["event_ts"], utc=True)
        right["event_ts"] = pd.to_datetime(right["event_ts"], utc=True)
        left = left.sort_values("event_ts")
        right = right.sort_values("event_ts")

        joined = pd.merge_asof(
            left,
            right[[view.entity_key, "event_ts", *view.feature_names]],
            on="event_ts",
            by=view.entity_key,
            direction="backward",
            allow_exact_matches=True,
            tolerance=tolerance,
        )
        joined.attrs["feature_view_version"] = view.version
        return joined
