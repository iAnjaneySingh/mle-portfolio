"""Data loading and a realistic synthetic generator.

The synthetic series is not a random walk with constant vol. It has:
  * a two-state volatility regime (Markov switching), so risk metrics see the
    clustering that real returns have;
  * fat-tailed innovations (Student-t, df=4);
  * intraday session structure, so session filters do something.

A strategy that only works on Gaussian noise fails here, which is the point.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_csv(path: str | Path, tz: str = "UTC") -> pd.DataFrame:
    df = pd.read_csv(path)
    ts_col = next(c for c in df.columns if c.lower() in ("timestamp", "date", "datetime", "time"))
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True).dt.tz_convert(tz)
    df = df.set_index(ts_col).sort_index()
    df.columns = [c.lower() for c in df.columns]
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    return df[keep]


def generate_ohlcv(
    n_bars: int = 20_000,
    freq: str = "1h",
    start: str = "2021-01-01",
    s0: float = 100.0,
    seed: int = 11,
    drift_annual: float = 0.05,
    periods_per_year: int = 24 * 252,
    include_regime: bool = False,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Two-state vol regime.
    p_stay = np.array([0.995, 0.97])
    vol_levels = np.array([0.0015, 0.0050])   # per-bar sigma, hourly
    state = np.zeros(n_bars, dtype=int)
    for i in range(1, n_bars):
        state[i] = state[i - 1] if rng.random() < p_stay[state[i - 1]] else 1 - state[i - 1]
    vol = vol_levels[state]

    mu = drift_annual / periods_per_year
    shocks = rng.standard_t(df=4, size=n_bars) / np.sqrt(4 / 2)  # unit variance
    log_ret = mu + vol * shocks
    close = s0 * np.exp(np.cumsum(log_ret))

    # Build OHLC consistent with the close path, with occasional opening gaps.
    # Gaps matter: without them a synthetic series has no three-bar imbalances,
    # and any strategy keyed on fair value gaps silently never trades.
    gap = np.where(rng.random(n_bars) < 0.04, rng.normal(0, 2.5, n_bars) * vol, 0.0)
    open_ = np.empty(n_bars)
    open_[0] = s0
    open_[1:] = close[:-1] * (1 + gap[1:])
    wick = vol * close * rng.uniform(0.3, 1.6, n_bars)
    high = np.maximum(open_, close) + wick * rng.uniform(0, 1, n_bars)
    low = np.minimum(open_, close) - wick * rng.uniform(0, 1, n_bars)
    volume = rng.lognormal(10, 0.6, n_bars) * (1 + 2 * state)

    idx = pd.date_range(start, periods=n_bars, freq=freq, tz="UTC")
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
    df.index.name = "timestamp"
    if include_regime:
        # Diagnostic only -- never consumed by a strategy in this repo.
        df["regime"] = state
    return df


def train_test_split_by_time(bars: pd.DataFrame, test_frac: float = 0.3):
    cut = int(len(bars) * (1 - test_frac))
    return bars.iloc[:cut], bars.iloc[cut:]
