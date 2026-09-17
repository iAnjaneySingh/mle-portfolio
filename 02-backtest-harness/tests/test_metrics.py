import numpy as np
import pandas as pd
import pytest

from backtest.metrics import sharpe_ratio, sortino_ratio, max_drawdown, cagr, win_rate, summarize


def test_sharpe_ratio_known_value():
    # constant positive return, zero volatility -> function should not blow up
    returns = pd.Series([0.0, 0.0, 0.0])
    assert sharpe_ratio(returns) == 0.0

    # simple case: verify against manual annualization
    returns = pd.Series([0.01, -0.01, 0.02, 0.0, 0.01])
    expected = np.sqrt(252) * returns.mean() / returns.std(ddof=0)
    assert sharpe_ratio(returns) == pytest.approx(expected)


def test_max_drawdown_simple_series():
    equity = pd.Series([100, 110, 90, 95, 120, 80])
    # peak 110 -> trough 90 => -18.18%; peak 120 -> trough 80 => -33.33% (larger)
    dd = max_drawdown(equity)
    assert dd == pytest.approx((80 - 120) / 120)


def test_max_drawdown_monotonic_increase_is_zero():
    equity = pd.Series([100, 105, 110, 120])
    assert max_drawdown(equity) == pytest.approx(0.0)


def test_win_rate():
    returns = pd.Series([0.01, -0.02, 0.03, 0.0, -0.01, 0.02])
    # positive: 0.01, 0.03, 0.02 -> 3 out of 6
    assert win_rate(returns) == pytest.approx(0.5)


def test_win_rate_empty():
    assert win_rate(pd.Series([], dtype=float)) == 0.0


def test_cagr_doubling_in_one_year():
    idx = pd.bdate_range("2023-01-01", periods=252)
    equity = pd.Series(np.linspace(100, 200, 252), index=idx)
    result = cagr(equity, periods_per_year=252)
    assert result == pytest.approx(1.0, rel=0.05)  # ~100% annual growth


def test_summarize_returns_all_keys():
    idx = pd.bdate_range("2023-01-01", periods=10)
    equity = pd.Series(np.linspace(100, 105, 10), index=idx)
    returns = equity.pct_change().fillna(0.0)
    result = summarize(equity, returns)
    for key in ["sharpe_ratio", "sortino_ratio", "max_drawdown", "cagr", "win_rate", "total_return", "num_periods"]:
        assert key in result
