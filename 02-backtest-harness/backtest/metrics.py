"""Performance metrics computed from a backtest's daily returns / equity curve.

All functions take plain pandas Series so they're trivially unit-testable
against hand-computed values, independent of the simulation engine.
"""

import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio from a series of periodic returns."""
    excess = returns - risk_free / periods_per_year
    if excess.std(ddof=0) == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * excess.mean() / excess.std(ddof=0))


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    """Like Sharpe, but only penalizes downside volatility."""
    excess = returns - risk_free / periods_per_year
    downside = excess[excess < 0]
    downside_std = downside.std(ddof=0)
    if downside_std == 0 or len(downside) == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * excess.mean() / downside_std)


def max_drawdown(equity_curve: pd.Series) -> float:
    """Largest peak-to-trough decline, as a negative fraction (e.g. -0.23 = -23%)."""
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    return float(drawdown.min())


def cagr(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    """Compound annual growth rate implied by the equity curve's start/end and length."""
    if len(equity_curve) < 2 or equity_curve.iloc[0] <= 0:
        return 0.0
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    years = len(equity_curve) / periods_per_year
    if years <= 0:
        return 0.0
    return float(total_return ** (1 / years) - 1)


def win_rate(trade_returns: pd.Series) -> float:
    """Fraction of closed trades (or return-bearing periods) that were profitable."""
    if len(trade_returns) == 0:
        return 0.0
    return float((trade_returns > 0).sum() / len(trade_returns))


def summarize(equity_curve: pd.Series, returns: pd.Series, periods_per_year: int = 252) -> dict:
    """Bundle of all metrics, in the shape logged to MLflow."""
    return {
        "sharpe_ratio": sharpe_ratio(returns, periods_per_year=periods_per_year),
        "sortino_ratio": sortino_ratio(returns, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown(equity_curve),
        "cagr": cagr(equity_curve, periods_per_year=periods_per_year),
        "win_rate": win_rate(returns),
        "total_return": float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) if len(equity_curve) > 1 else 0.0,
        "num_periods": int(len(returns)),
    }
