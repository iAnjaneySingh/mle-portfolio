"""The single source of truth for feature logic.

This module is imported by BOTH the offline/training path (batch_features.py,
plain pandas over historical data) and the online/serving path
(streaming_job.py, called from Spark's foreachBatch on each micro-batch).

That's the actual fix for training/serving skew: not "make sure both
implementations agree," but "there is only one implementation." If you find
yourself writing the feature logic twice — once in SQL/Spark, once in
Python for training — that gap is where skew bugs live.
"""

import numpy as np
import pandas as pd


def rolling_vwap(prices: pd.Series, volumes: pd.Series, window: int) -> pd.Series:
    """Volume-weighted average price over a trailing window of ticks."""
    notional = prices * volumes
    return notional.rolling(window, min_periods=1).sum() / volumes.rolling(window, min_periods=1).sum()


def rolling_volatility(prices: pd.Series, window: int) -> pd.Series:
    """Rolling standard deviation of simple returns over a trailing window."""
    returns = prices.pct_change().fillna(0.0)
    return returns.rolling(window, min_periods=2).std(ddof=0).fillna(0.0)


def momentum(prices: pd.Series, window: int) -> pd.Series:
    """Percent change vs. the price `window` ticks ago."""
    shifted = prices.shift(window)
    return ((prices - shifted) / shifted).fillna(0.0)


def compute_features(df: pd.DataFrame, vwap_window: int = 20,
                      vol_window: int = 20, momentum_window: int = 10) -> pd.DataFrame:
    """Apply all three features to a single symbol's tick data.

    `df` must be sorted by timestamp ascending and contain columns:
    price, volume. Returns a copy with feature columns added.
    """
    out = df.copy()
    out["vwap"] = rolling_vwap(out["price"], out["volume"], vwap_window)
    out["volatility"] = rolling_volatility(out["price"], vol_window)
    out["momentum"] = momentum(out["price"], momentum_window)
    return out
