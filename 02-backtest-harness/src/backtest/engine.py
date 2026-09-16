"""Event-driven, bar-by-bar backtest engine.

The three invariants, each covered by a test in tests/test_no_lookahead.py:

  1. A strategy sees `bars.iloc[:t+1]` only. It is physically handed a slice,
     not the full frame, so it cannot index into the future by accident.
  2. Signals emitted on bar t fill at the OPEN of bar t+1. Filling at bar t's
     close is the single most common source of fake alpha in retail backtests.
  3. Intrabar stop/target resolution is pessimistic: if a bar's range contains
     both levels, the stop is assumed to have been hit first. Without tick data
     the ordering is unknowable, so the backtest takes the unfavourable branch.

Equity is marked to market on every close, which is what the return series --
and therefore every risk metric -- is computed from.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .execution import CostModel
from .types import Fill, Side, Signal, Trade

log = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("open", "high", "low", "close")


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    positions: pd.Series
    trades: list[Trade]
    fills: list[Fill]
    initial_capital: float
    metadata: dict = field(default_factory=dict)

    @property
    def trades_frame(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([t.__dict__ for t in self.trades])


class BacktestEngine:
    def __init__(
        self,
        initial_capital: float = 100_000.0,
        cost_model: CostModel | None = None,
        warmup_bars: int = 100,
        vol_lookback: int = 20,
        max_leverage: float = 1.0,
    ):
        self.initial_capital = initial_capital
        self.costs = cost_model or CostModel()
        self.warmup_bars = warmup_bars
        self.vol_lookback = vol_lookback
        self.max_leverage = max_leverage

    def run(self, bars: pd.DataFrame, strategy) -> BacktestResult:
        _validate(bars)
        bars = bars.sort_index()
        n = len(bars)

        realised_vol = (
            bars["close"].pct_change().rolling(self.vol_lookback).std().fillna(0.01)
        )

        cash = self.initial_capital
        qty = 0.0
        entry_price = 0.0
        entry_ts = bars.index[0]
        entry_bar = 0
        stop = target = None
        mae = mfe = 0.0
        pending: Signal | None = None

        equity = np.full(n, self.initial_capital, dtype=float)
        pos_weight = np.zeros(n, dtype=float)
        trades: list[Trade] = []
        fills: list[Fill] = []

        strategy.on_start(bars.iloc[: self.warmup_bars])

        for t in range(n):
            bar = bars.iloc[t]
            ts = bars.index[t]

            # ---- 1. Execute whatever was decided at the close of bar t-1 ----
            if pending is not None:
                mark = float(bar["open"])
                current_equity = cash + qty * mark
                desired_qty = self._target_qty(pending.target_weight, current_equity, mark)
                delta = desired_qty - qty
                if abs(delta * mark) > 1e-8:
                    price, slip = self.costs.fill_price(mark, delta, float(realised_vol.iloc[t]))
                    comm = self.costs.commission(delta, price)
                    cash -= delta * price + comm

                    closing = qty != 0 and np.sign(desired_qty) != np.sign(qty)
                    if closing or (qty != 0 and desired_qty == 0):
                        trades.append(_close_trade(
                            entry_ts, ts, entry_price, price, qty, t - entry_bar,
                            pending.reason or "signal_exit", mae, mfe))
                        mae = mfe = 0.0
                    opened = desired_qty != 0 and (qty == 0 or closing)
                    if opened:
                        entry_price, entry_ts, entry_bar = price, ts, t

                    fills.append(Fill(ts, price, delta, comm, abs(slip * delta),
                                      "entry" if qty == 0 else "rebalance"))
                    qty = desired_qty
                else:
                    opened = False

                # Brackets persist across "hold" signals. Only an explicit
                # stop/target updates them; a new entry or a flat clears them.
                # Without this, a strategy that returns Signal.hold() silently
                # drops its own stop loss on the very next bar.
                if pending.stop_loss is not None or pending.take_profit is not None:
                    stop, target = pending.stop_loss, pending.take_profit
                elif opened or qty == 0:
                    stop = target = None
                pending = None

            # ---- 2. Intrabar stop / target, pessimistic ordering ----
            if qty != 0 and (stop is not None or target is not None):
                hit, level = _resolve_intrabar(bar, qty, stop, target)
                if hit:
                    price, slip = self.costs.fill_price(level, -qty, float(realised_vol.iloc[t]))
                    comm = self.costs.commission(-qty, price)
                    cash -= -qty * price + comm
                    fills.append(Fill(ts, price, -qty, comm, abs(slip * qty), hit))
                    trades.append(_close_trade(
                        entry_ts, ts, entry_price, price, qty, t - entry_bar, hit, mae, mfe))
                    qty = 0.0
                    stop = target = None
                    mae = mfe = 0.0

            # ---- 3. Track excursions while in position ----
            if qty != 0 and entry_price:
                r_unit = abs(entry_price - stop) if stop else entry_price * 0.01
                if r_unit > 0:
                    if qty > 0:
                        mae = min(mae, (float(bar["low"]) - entry_price) / r_unit)
                        mfe = max(mfe, (float(bar["high"]) - entry_price) / r_unit)
                    else:
                        mae = min(mae, (entry_price - float(bar["high"])) / r_unit)
                        mfe = max(mfe, (entry_price - float(bar["low"])) / r_unit)

            # ---- 4. Mark to market ----
            close = float(bar["close"])
            equity[t] = cash + qty * close
            pos_weight[t] = (qty * close) / equity[t] if equity[t] else 0.0

            # ---- 5. Decide for the NEXT bar, using history through t only ----
            if t >= self.warmup_bars and t < n - 1:
                pending = strategy.on_bar(
                    bars.iloc[: t + 1],
                    position_weight=pos_weight[t],
                    equity=equity[t],
                )

        equity_s = pd.Series(equity, index=bars.index, name="equity")
        return BacktestResult(
            equity=equity_s,
            returns=equity_s.pct_change().fillna(0.0),
            positions=pd.Series(pos_weight, index=bars.index, name="position"),
            trades=trades,
            fills=fills,
            initial_capital=self.initial_capital,
            metadata={
                "strategy": type(strategy).__name__,
                "params": getattr(strategy, "params", {}),
                "bars": n,
                "warmup_bars": self.warmup_bars,
                "cost_model": self.costs.__dict__,
            },
        )

    def _target_qty(self, weight: float, equity: float, price: float) -> float:
        weight = float(np.clip(weight, -self.max_leverage, self.max_leverage))
        if price <= 0 or equity <= 0:
            return 0.0
        return weight * equity / price


def _resolve_intrabar(bar, qty, stop, target) -> tuple[str | None, float]:
    """Pessimistic: stop wins ties, because we cannot see intrabar sequence."""
    hi, lo = float(bar["high"]), float(bar["low"])
    if qty > 0:
        if stop is not None and lo <= stop:
            return "stop", stop
        if target is not None and hi >= target:
            return "target", target
    else:
        if stop is not None and hi >= stop:
            return "stop", stop
        if target is not None and lo <= target:
            return "target", target
    return None, 0.0


def _close_trade(entry_ts, exit_ts, entry_price, exit_price, qty, bars_held, reason, mae, mfe) -> Trade:
    side = Side.LONG if qty > 0 else Side.SHORT
    pnl = (exit_price - entry_price) * qty
    ret = pnl / (abs(qty) * entry_price) if entry_price and qty else 0.0
    return Trade(entry_ts, exit_ts, side, entry_price, exit_price, qty, pnl, ret,
                 bars_held, reason, mae, mfe)


def _validate(bars: pd.DataFrame) -> None:
    missing = set(REQUIRED_COLUMNS) - set(bars.columns)
    if missing:
        raise ValueError(f"bars missing columns: {sorted(missing)}")
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise TypeError("bars must be indexed by a DatetimeIndex")
    if not bars.index.is_monotonic_increasing:
        raise ValueError("bars index must be sorted ascending")
    if bars.index.has_duplicates:
        raise ValueError("bars index contains duplicate timestamps")
    bad = bars[(bars["high"] < bars["low"]) |
               (bars["high"] < bars["close"]) | (bars["low"] > bars["close"])]
    if len(bad):
        raise ValueError(f"{len(bad)} bars violate OHLC consistency")
