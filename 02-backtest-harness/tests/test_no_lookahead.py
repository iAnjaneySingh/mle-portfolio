"""The tests that make the numbers mean something."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine
from src.backtest.execution import CostModel
from src.backtest.types import Signal
from src.data.loader import generate_ohlcv
from src.strategies.base import Flat, Strategy


class SpyStrategy(Strategy):
    """Records the last timestamp it was shown on every call."""

    def __init__(self):
        self.seen_last = []

    def on_bar(self, history, position_weight, equity):
        self.seen_last.append(history.index[-1])
        return Signal(0.0)


class OracleStrategy(Strategy):
    """Tries to cheat by holding a reference to the full frame. The engine
    cannot stop that, but it CAN guarantee that what is passed in is a prefix --
    which is what this test checks."""

    def __init__(self, full: pd.DataFrame):
        self.full = full
        self.violations = 0

    def on_bar(self, history, position_weight, equity):
        if len(history) < len(self.full):
            if not history.index.equals(self.full.index[: len(history)]):
                self.violations += 1
        return Signal(0.0)


@pytest.fixture(scope="module")
def bars():
    return generate_ohlcv(1_500, seed=3)


def test_strategy_only_sees_a_prefix_of_history(bars):
    s = OracleStrategy(bars)
    BacktestEngine(warmup_bars=50).run(bars, s)
    assert s.violations == 0


def test_last_visible_bar_is_never_the_future(bars):
    s = SpyStrategy()
    BacktestEngine(warmup_bars=50).run(bars, s)
    assert s.seen_last[0] == bars.index[50]
    # strategy is never consulted on the final bar (nothing could fill)
    assert s.seen_last[-1] == bars.index[-2]


def test_fills_happen_at_next_bar_open(bars):
    class EnterOnce(Strategy):
        def __init__(self): self.fired = False
        def on_bar(self, history, position_weight, equity):
            if not self.fired:
                self.fired = True
                return Signal(1.0, reason="enter")
            return Signal(position_weight, reason="hold")

    engine = BacktestEngine(warmup_bars=50, cost_model=CostModel(0, 0, 0))
    res = engine.run(bars, EnterOnce())
    first = res.fills[0]
    # Signal emitted at close of bar 50 -> fill at open of bar 51.
    assert first.ts == bars.index[51]
    assert first.price == pytest.approx(float(bars["open"].iloc[51]))


def test_flat_strategy_preserves_capital_exactly(bars):
    res = BacktestEngine(100_000.0).run(bars, Flat())
    assert res.equity.iloc[-1] == pytest.approx(100_000.0)
    assert res.returns.abs().max() == 0.0


def test_costs_strictly_reduce_returns(bars):
    from src.strategies.base import BuyAndHold

    free = BacktestEngine(cost_model=CostModel(0, 0, 0)).run(bars, BuyAndHold())
    paid = BacktestEngine(cost_model=CostModel(5, 2, 0.5)).run(bars, BuyAndHold())
    assert paid.equity.iloc[-1] < free.equity.iloc[-1]


def test_stop_is_preferred_over_target_when_both_in_range():
    idx = pd.date_range("2024-01-01", periods=6, freq="1h", tz="UTC")
    bars = pd.DataFrame({
        "open":  [100, 100, 100, 100, 100, 100],
        "high":  [101, 101, 101, 120, 101, 101],   # bar 3 spans both levels
        "low":   [99, 99, 99, 80, 99, 99],
        "close": [100, 100, 100, 100, 100, 100],
    }, index=idx)

    class EnterWithBrackets(Strategy):
        def on_bar(self, history, position_weight, equity):
            if position_weight == 0:
                return Signal(1.0, stop_loss=90.0, take_profit=110.0, reason="bracket")
            return Signal(position_weight, stop_loss=90.0, take_profit=110.0)

    res = BacktestEngine(warmup_bars=1, cost_model=CostModel(0, 0, 0)).run(bars, EnterWithBrackets())
    assert res.trades, "expected a completed round trip"
    assert res.trades[0].exit_reason == "stop"


def test_engine_rejects_malformed_bars():
    idx = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
    bad = pd.DataFrame({"open": [1, 1, 1], "high": [0, 1, 1],
                        "low": [1, 1, 1], "close": [1, 1, 1]}, index=idx)
    with pytest.raises(ValueError):
        BacktestEngine().run(bad, Flat())


def test_duplicate_timestamps_rejected(bars):
    dup = pd.concat([bars.iloc[:10], bars.iloc[:10]])
    with pytest.raises(ValueError):
        BacktestEngine().run(dup.sort_index(), Flat())
