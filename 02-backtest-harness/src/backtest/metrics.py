"""Performance and risk metrics.

Beyond the usual Sharpe/drawdown/win-rate, this module carries the statistics
that tell you whether a headline number survives contact with reality:

  * Probabilistic Sharpe Ratio (PSR)  -- P(true SR > benchmark), correcting for
    the skew and fat tails that inflate naive Sharpe on trading returns.
  * Deflated Sharpe Ratio (DSR)       -- PSR with the benchmark raised to the
    expected maximum Sharpe across N independent backtest trials. This is the
    multiple-testing correction. If you tried 200 parameter sets, a Sharpe of
    1.8 is unremarkable, and DSR says so.
  * Stationary bootstrap CI on Sharpe -- block resampling that preserves serial
    dependence, unlike an i.i.d. bootstrap.

References: Bailey & Lopez de Prado (2012, 2014); Politis & Romano (1994).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats

EULER_MASCHERONI = 0.5772156649015329


@dataclass
class PerformanceReport:
    # returns-based
    total_return: float
    cagr: float
    annual_volatility: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    max_drawdown_duration_days: float
    ulcer_index: float
    var_95: float
    cvar_95: float
    skew: float
    excess_kurtosis: float
    # trade-based
    n_trades: int
    win_rate: float
    profit_factor: float
    expectancy_r: float
    avg_win: float
    avg_loss: float
    payoff_ratio: float
    max_consecutive_losses: int
    avg_bars_held: float
    exposure: float
    turnover: float
    # significance
    psr: float
    deflated_sharpe: float
    sharpe_ci_low: float
    sharpe_ci_high: float
    n_trials_assumed: int

    def to_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


# --------------------------------------------------------------------------
# Return-series metrics
# --------------------------------------------------------------------------

def sharpe_ratio(returns: pd.Series, periods_per_year: int, rf: float = 0.0) -> float:
    excess = returns - rf / periods_per_year
    sd = excess.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, periods_per_year: int, target: float = 0.0) -> float:
    downside = returns[returns < target]
    dd = np.sqrt((downside ** 2).mean()) if len(downside) else 0.0
    if dd == 0 or np.isnan(dd):
        return 0.0
    return float((returns.mean() - target) / dd * np.sqrt(periods_per_year))


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def max_drawdown(equity: pd.Series) -> float:
    return float(drawdown_series(equity).min())


def max_drawdown_duration(equity: pd.Series) -> float:
    """Longest stretch, in days, spent below a prior equity peak."""
    dd = drawdown_series(equity)
    underwater = dd < 0
    if not underwater.any():
        return 0.0
    current = None
    best = pd.Timedelta(0)
    for ts, uw in underwater.items():
        if uw and current is None:
            current = ts
        elif not uw and current is not None:
            best = max(best, ts - current)
            current = None
    if current is not None:
        best = max(best, underwater.index[-1] - current)
    return float(best.total_seconds() / 86400)


def ulcer_index(equity: pd.Series) -> float:
    """RMS drawdown. Penalises depth *and* duration, unlike max drawdown."""
    dd = drawdown_series(equity) * 100
    return float(np.sqrt((dd ** 2).mean()))


def value_at_risk(returns: pd.Series, alpha: float = 0.05) -> tuple[float, float]:
    if returns.empty:
        return 0.0, 0.0
    var = float(np.quantile(returns, alpha))
    tail = returns[returns <= var]
    return var, float(tail.mean()) if len(tail) else var


# --------------------------------------------------------------------------
# Significance
# --------------------------------------------------------------------------

def probabilistic_sharpe_ratio(
    returns: pd.Series, periods_per_year: int, benchmark_sr: float = 0.0
) -> float:
    n = len(returns)
    if n < 10:
        return float("nan")
    sr = sharpe_ratio(returns, periods_per_year) / np.sqrt(periods_per_year)  # per-period
    bench = benchmark_sr / np.sqrt(periods_per_year)
    g3 = float(stats.skew(returns))
    g4 = float(stats.kurtosis(returns, fisher=False))
    denom = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))
    z = (sr - bench) * np.sqrt(n - 1) / denom
    return float(stats.norm.cdf(z))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """E[max SR] over n independent trials with Var(SR) across trials.

    This is the bar a strategy must clear to be interesting after searching.
    """
    if n_trials <= 1 or sr_variance <= 0:
        return 0.0
    g = EULER_MASCHERONI
    a = stats.norm.ppf(1 - 1 / n_trials)
    b = stats.norm.ppf(1 - 1 / (n_trials * np.e))
    return float(np.sqrt(sr_variance) * ((1 - g) * a + g * b))


def deflated_sharpe_ratio(
    returns: pd.Series,
    periods_per_year: int,
    n_trials: int,
    trial_sr_variance: float | None = None,
) -> float:
    """PSR against the expected maximum Sharpe of `n_trials` searches.

    If the variance of Sharpe across trials was not measured, fall back to the
    sampling variance of the Sharpe estimator itself -- conservative but far
    better than pretending only one strategy was ever tested.
    """
    if trial_sr_variance is None:
        n = max(len(returns), 2)
        sr = sharpe_ratio(returns, periods_per_year) / np.sqrt(periods_per_year)
        trial_sr_variance = (1 + 0.5 * sr ** 2) / (n - 1)
    bench_per_period = expected_max_sharpe(n_trials, trial_sr_variance)
    return probabilistic_sharpe_ratio(
        returns, periods_per_year, benchmark_sr=bench_per_period * np.sqrt(periods_per_year)
    )


def stationary_bootstrap_sharpe_ci(
    returns: pd.Series,
    periods_per_year: int,
    n_boot: int = 2_000,
    mean_block: int = 20,
    alpha: float = 0.05,
    seed: int = 7,
) -> tuple[float, float]:
    """Politis-Romano stationary bootstrap: geometric block lengths, so the
    resampled series keeps the autocorrelation structure of the original."""
    r = returns.to_numpy()
    n = len(r)
    if n < 30:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    stats_out = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(n, dtype=int)
        i = rng.integers(n)
        for k in range(n):
            idx[k] = i
            i = rng.integers(n) if rng.random() < p else (i + 1) % n
        sample = r[idx]
        sd = sample.std(ddof=1)
        stats_out[b] = sample.mean() / sd * np.sqrt(periods_per_year) if sd else 0.0
    return float(np.quantile(stats_out, alpha / 2)), float(np.quantile(stats_out, 1 - alpha / 2))


# --------------------------------------------------------------------------
# Trade-level metrics
# --------------------------------------------------------------------------

def trade_stats(trades_df: pd.DataFrame) -> dict[str, float]:
    if trades_df.empty:
        return dict(n_trades=0, win_rate=0.0, profit_factor=0.0, expectancy_r=0.0,
                    avg_win=0.0, avg_loss=0.0, payoff_ratio=0.0,
                    max_consecutive_losses=0, avg_bars_held=0.0)
    pnl = trades_df["pnl"]
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    gross_loss = abs(losses.sum())
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(abs(losses.mean())) if len(losses) else 0.0
    win_rate = float(len(wins) / len(pnl))
    return dict(
        n_trades=int(len(pnl)),
        win_rate=win_rate,
        profit_factor=float(wins.sum() / gross_loss) if gross_loss else float("inf"),
        # Expectancy in R multiples: what one unit of risk returns on average.
        expectancy_r=float(win_rate * (avg_win / avg_loss) - (1 - win_rate)) if avg_loss else 0.0,
        avg_win=avg_win,
        avg_loss=avg_loss,
        payoff_ratio=float(avg_win / avg_loss) if avg_loss else float("inf"),
        max_consecutive_losses=_max_streak(pnl <= 0),
        avg_bars_held=float(trades_df["bars_held"].mean()),
    )


def _max_streak(flags: pd.Series) -> int:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


# --------------------------------------------------------------------------
# Top-level report
# --------------------------------------------------------------------------

def build_report(
    result,
    periods_per_year: int = 252,
    n_trials: int = 1,
    bootstrap: bool = True,
) -> PerformanceReport:
    eq, ret = result.equity, result.returns
    years = max((eq.index[-1] - eq.index[0]).total_seconds() / (365.25 * 86400), 1e-9)
    total = float(eq.iloc[-1] / eq.iloc[0] - 1)
    mdd = max_drawdown(eq)
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1)
    var95, cvar95 = value_at_risk(ret)
    ts = trade_stats(result.trades_frame)

    lo, hi = (stationary_bootstrap_sharpe_ci(ret, periods_per_year)
              if bootstrap else (float("nan"), float("nan")))

    positions = result.positions
    turnover = float(positions.diff().abs().sum())

    return PerformanceReport(
        total_return=total,
        cagr=cagr,
        annual_volatility=float(ret.std(ddof=1) * np.sqrt(periods_per_year)),
        sharpe=sharpe_ratio(ret, periods_per_year),
        sortino=sortino_ratio(ret, periods_per_year),
        calmar=float(cagr / abs(mdd)) if mdd else 0.0,
        max_drawdown=mdd,
        max_drawdown_duration_days=max_drawdown_duration(eq),
        ulcer_index=ulcer_index(eq),
        var_95=var95,
        cvar_95=cvar95,
        skew=float(stats.skew(ret)),
        excess_kurtosis=float(stats.kurtosis(ret)),
        psr=probabilistic_sharpe_ratio(ret, periods_per_year),
        deflated_sharpe=deflated_sharpe_ratio(ret, periods_per_year, n_trials),
        sharpe_ci_low=lo,
        sharpe_ci_high=hi,
        n_trials_assumed=n_trials,
        exposure=float((positions.abs() > 1e-9).mean()),
        turnover=turnover,
        **ts,
    )
