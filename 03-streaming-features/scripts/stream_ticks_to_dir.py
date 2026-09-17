"""Simulates ticks arriving in real time: reads sample_ticks.csv and writes
small chunks into data/incoming_ticks/ every couple seconds, for the
Structured Streaming file source to pick up.
"""

import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "sample_ticks.csv"
DEST = ROOT / "data" / "incoming_ticks"


def main(chunk_size: int = 20, delay_seconds: float = 3.0):
    DEST.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SRC)

    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size]
        out_path = DEST / f"ticks_{i:05d}.csv"
        chunk.to_csv(out_path, index=False)
        print(f"Wrote {len(chunk)} ticks to {out_path}")
        time.sleep(delay_seconds)


if __name__ == "__main__":
    main()
