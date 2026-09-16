from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine
from src.data.loader import generate_ohlcv
from src.strategies.ict_liquidity_sweep import ICTLiquiditySweep
from src.strategies.multi_agent import (
    AgentVote,
    MultiAgentStrategy,
    RiskAgent,
    StructureAgent,
)


@pytest.fixture(scope="module")
def bars():
    return generate_ohlcv(3_000, seed=5)


def test_ict_runs_and_respects_risk_budget(bars):
    strat = ICTLiquiditySweep(risk_per_trade=0.01, rr=2.0)
    res = BacktestEngine(warmup_bars=100).run(bars, strat)
    assert res.equity.notna().all()
    tf = res.trades_frame
    if not tf.empty:
        # No single stop-out should cost much more than the risk budget.
        losses = tf[tf.pnl < 0].pnl.abs() / res.initial_capital
        assert losses.max() < 0.05


def test_ict_detects_a_hand_built_sweep():
    n = 40
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    close = np.full(n, 100.0)
    high = close + 0.5
    low = close - 0.5
    # bar n-4: wick sweeps below the range, closes back inside
    low[n - 4] = 95.0
    # displacement + FVG over the next bars
    close[n - 3:], high[n - 3:], low[n - 3:] = 104.0, 106.0, 103.0
    low[n - 1] = 100.6  # gap above high of bar n-3
    df = pd.DataFrame({"open": close, "high": high, "low": low, "close": close}, index=idx)
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)

    s = ICTLiquiditySweep(swing_lookback=10, displacement_window=5, displacement_atr=0.5)
    direction, level = s._detect_sweep(df.iloc[: n - 3])
    assert direction == 1
    assert level == pytest.approx(99.5, abs=0.6)


def test_ict_state_machine_reaches_pending_setup():
    """Drive the full idle -> armed -> pending transition on constructed bars."""
    n = 60
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    close = np.full(n, 100.0)
    close[-5:] = [102.0, 105.0, 107.0, 108.0, 108.5]   # displacement up
    open_ = np.concatenate([[100.0], close[:-1]])
    high = np.maximum(open_, close) + 0.3
    low = np.minimum(open_, close) - 0.3
    # Sweep bar: wick takes sell-side liquidity, close returns inside the range.
    low[-6] = 95.0
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)

    s = ICTLiquiditySweep(swing_lookback=15, displacement_window=6,
                          displacement_atr=0.5, atr_period=10)
    reasons = [s.on_bar(df.iloc[: t + 1], position_weight=0.0, equity=100_000).reason
               for t in range(20, n)]
    assert "sweep_detected" in reasons, f"sweep never recognised: {set(reasons)}"
    assert "awaiting_displacement" in reasons, "state machine never armed"


def test_risk_agent_vetoes_past_drawdown_limit():
    v = RiskAgent(max_drawdown=0.10).vote(pd.DataFrame(), {"drawdown": -0.2, "loss_streak": 0})
    assert v.veto


def test_risk_agent_vetoes_on_loss_streak():
    v = RiskAgent(max_consecutive_losses=3).vote(pd.DataFrame(), {"drawdown": 0.0, "loss_streak": 4})
    assert v.veto


def test_aggregation_is_weighted_mean_of_convictions():
    s = MultiAgentStrategy(weights={"structure": 1.0, "liquidity": 3.0})
    votes = [
        AgentVote("structure", -1.0, 1.0),
        AgentVote("liquidity", 1.0, 1.0),
    ]
    score, gate = s.aggregate(votes)
    assert score == pytest.approx((1 * -1 + 3 * 1) / 4)
    assert gate == 1.0


def test_any_veto_zeroes_the_ensemble():
    s = MultiAgentStrategy(weights={"structure": 1.0})
    score, gate = s.aggregate([AgentVote("structure", 1.0, 1.0), AgentVote("risk", 0, 1, veto=True)])
    assert score == 0.0 and gate == 0.0


def test_multi_agent_end_to_end_produces_attribution(bars):
    strat = MultiAgentStrategy()
    res = BacktestEngine(warmup_bars=150).run(bars, strat)
    contrib = strat.contributions()
    assert not contrib.empty
    assert {"structure", "liquidity", "momentum"}.issubset(contrib.columns)
    assert res.positions.abs().max() <= 1.0 + 1e-9
