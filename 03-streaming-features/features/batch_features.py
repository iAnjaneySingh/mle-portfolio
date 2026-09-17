"""Offline/training feature computation: reads historical ticks and applies
the exact same `compute_features` used by the streaming job, per symbol.

This is what a training pipeline would call to build a feature set for
model training — and because it shares shared_features.py with the
streaming path, whatever the model was trained on is what serving computes.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.shared_features import compute_features


def build_training_features(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    results = []
    for symbol, group in df.groupby("symbol"):
        group = group.sort_values("timestamp")
        results.append(compute_features(group))
    return pd.concat(results).sort_values(["symbol", "timestamp"]).reset_index(drop=True)


if __name__ == "__main__":
    data_path = Path(__file__).resolve().parent.parent / "data" / "sample_ticks.csv"
    out_path = Path(__file__).resolve().parent.parent / "data" / "training_features.csv"

    features_df = build_training_features(str(data_path))
    features_df.to_csv(out_path, index=False)
    print(f"Wrote {len(features_df)} rows of training features to {out_path}")
    print(features_df.tail())
