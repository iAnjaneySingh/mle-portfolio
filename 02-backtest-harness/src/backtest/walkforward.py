"""Purged, embargoed walk-forward evaluation.

Two problems with naive rolling splits on financial data:

  1. Labels built from forward-looking windows overlap the split boundary, so
     training rows leak into the test period. Fix: *purge* training rows whose
     label horizon extends into the test window.
  2. Serial correlation means the bars immediately after the test window are
     still informationally close to it. Fix: an *embargo* gap.

Both come from Lopez de Prado, Advances in Financial Machine Learning, ch. 7.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np
import pandas as pd

from .engine import BacktestEngine
from .metrics import build_report


@dataclass
class Fold:
    index: int
    train: pd.DataFrame
    test: pd.DataFrame


def purged_walk_forward_splits(
    bars: pd.DataFrame,
    n_folds: int = 5,
    train_frac: float = 0.6,
    label_horizon_bars: int = 20,
    embargo_frac: float = 0.01,
    anchored: bool = False,
) -> Iterator[Fold]:
    n = len(bars)
    embargo = int(n * embargo_frac)
    test_size = int(n * (1 - train_frac) / n_folds)
    train_size = int(n * train_frac)

    for i in range(n_folds):
        test_start = train_size + i * test_size
        test_end = min(test_start + test_size, n)
        if test_end - test_start < 30:
            break
        train_start = 0 if anchored else max(0, test_start - train_size)
        # Purge: drop training bars whose forward label window crosses into test.
        train_end = max(train_start, test_start - label_horizon_bars - embargo)
        yield Fold(i, bars.iloc[train_start:train_end], bars.iloc[test_start:test_end])


def run_walk_forward(
    bars: pd.DataFrame,
    strategy_factory: Callable[[pd.DataFrame], object],
    engine: BacktestEngine | None = None,
    periods_per_year: int = 252,
    **split_kwargs,
) -> pd.DataFrame:
    """Fit-on-train, evaluate-on-test, repeatedly. Returns one row per fold.

    `strategy_factory(train_bars) -> strategy` is where parameter selection
    happens. Because it only ever receives the training slice, an overfit
    selection procedure will show up as degraded out-of-sample folds instead of
    being hidden by a single in-sample number.
    """
    engine = engine or BacktestEngine()
    rows = []
    for fold in purged_walk_forward_splits(bars, **split_kwargs):
        strategy = strategy_factory(fold.train)
        result = engine.run(fold.test, strategy)
        rep = build_report(result, periods_per_year=periods_per_year, bootstrap=False)
        rows.append({
            "fold": fold.index,
            "train_start": fold.train.index[0], "train_end": fold.train.index[-1],
            "test_start": fold.test.index[0], "test_end": fold.test.index[-1],
            "n_train": len(fold.train), "n_test": len(fold.test),
            **rep.to_dict(),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df.attrs["summary"] = {
            "mean_sharpe": float(df.sharpe.mean()),
            "std_sharpe": float(df.sharpe.std(ddof=1)) if len(df) > 1 else 0.0,
            "folds_positive": int((df.sharpe > 0).sum()),
            "worst_drawdown": float(df.max_drawdown.min()),
            # Consistency matters more than the mean: a strategy that is great
            # in one fold and flat in four is a curve fit.
            "sharpe_consistency": float((df.sharpe > 0).mean()),
        }
    return df


def combinatorial_purged_cv_paths(n_groups: int = 6, n_test_groups: int = 2) -> int:
    """Number of distinct backtest paths CPCV generates -- useful for setting
    `n_trials` in the deflated Sharpe calculation honestly."""
    from math import comb

    return comb(n_groups, n_test_groups)
