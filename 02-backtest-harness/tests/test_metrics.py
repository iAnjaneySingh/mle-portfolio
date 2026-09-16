from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.metrics import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    max_drawdown,
    max_drawdown_duration,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
    sortino_ratio,
    trade_stats,
    ulcer_index,
)


def _returns(mu, sigma, n=2520, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    return pd.Series(rng.normal(mu, sigma, n), index=idx)


def test_sharpe_matches_closed_form():
    r = _returns(0.0005, 0.01, n=100_000)
    expected = 0.0005 / 0.01 * np.sqrt(252)
    assert sharpe_ratio(r, 252) == pytest.approx(expected, rel=0.05)


def test_sharpe_of_zero_variance_is_zero():
    r = pd.Series(np.zeros(500), index=pd.date_range("2020-01-01", periods=500, freq="B"))
    assert sharpe_ratio(r, 252) == 0.0


def test_sortino_exceeds_sharpe_for_right_skewed_returns():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.lognormal(-5, 1, 5000) - 0.004)
    assert sortino_ratio(r, 252) > sharpe_ratio(r, 252)


def test_max_drawdown_on_known_path():
    eq = pd.Series([100, 120, 90, 110, 60, 80],
                   index=pd.date_range("2020-01-01", periods=6, freq="D"))
    assert max_drawdown(eq) == pytest.approx(60 / 120 - 1)


def test_drawdown_duration_counts_days_underwater():
    eq = pd.Series([100, 90, 95, 101],
                   index=pd.date_range("2020-01-01", periods=4, freq="D"))
    assert max_drawdown_duration(eq) == pytest.approx(2.0)


def test_ulcer_index_penalises_prolonged_drawdown():
    idx = pd.date_range("2020-01-01", periods=40, freq="D")
    quick = pd.Series([100] * 19 + [70] + [100] * 20, index=idx)
    slow = pd.Series([100] * 10 + [70] * 20 + [100] * 10, index=idx)
    assert ulcer_index(slow) > ulcer_index(quick)


def test_psr_rises_with_sample_size():
    short = probabilistic_sharpe_ratio(_returns(0.0004, 0.01, n=120), 252)
    long = probabilistic_sharpe_ratio(_returns(0.0004, 0.01, n=5040), 252)
    assert long > short


def test_expected_max_sharpe_grows_with_trials():
    vals = [expected_max_sharpe(n, 0.01) for n in (2, 10, 100, 1000)]
    assert vals == sorted(vals)


def test_deflated_sharpe_falls_as_trials_grow():
    r = _returns(0.0004, 0.01, n=2520)
    one = deflated_sharpe_ratio(r, 252, n_trials=1)
    many = deflated_sharpe_ratio(r, 252, n_trials=500)
    assert many < one, "multiple-testing correction must reduce confidence"


def test_trade_stats_on_hand_built_ledger():
    df = pd.DataFrame({
        "pnl": [100, -50, 200, -50, -50],
        "bars_held": [5, 3, 8, 2, 4],
    })
    s = trade_stats(df)
    assert s["n_trades"] == 5
    assert s["win_rate"] == pytest.approx(0.4)
    assert s["profit_factor"] == pytest.approx(300 / 150)
    assert s["payoff_ratio"] == pytest.approx(150 / 50)
    # loss sequence is F,T,F,T,T -> longest run is 2
    assert s["max_consecutive_losses"] == 2
    # 0.4 * 3 - 0.6 = 0.6R expectancy
    assert s["expectancy_r"] == pytest.approx(0.6)


def test_trade_stats_handles_empty_ledger():
    assert trade_stats(pd.DataFrame())["n_trades"] == 0
