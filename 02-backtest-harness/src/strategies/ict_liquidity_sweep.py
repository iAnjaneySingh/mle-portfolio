"""ICT / Smart-Money-Concepts strategy, expressed as testable rules.

The discretionary version of this setup: price sweeps a pool of liquidity
resting beyond a recent swing, fails to continue, displaces back through
structure leaving a fair value gap, and you enter on the retracement into that
gap with a stop beyond the sweep wick.

Every one of those words has to become a number before a backtest means
anything, which is the entire point of this file:

  liquidity pool  -> extreme of the last `swing_lookback` bars
  sweep           -> a wick trades beyond that extreme, the close does not
  displacement    -> within `displacement_window` bars after the sweep, price
                     moves >= `displacement_atr` ATRs back through the level
  fair value gap  -> a three-bar imbalance created during that displacement
                     (bar[i-2].high < bar[i].low for a bullish FVG)
  entry           -> retracement into the FVG, within `fvg_expiry_bars`
  invalidation    -> stop beyond the sweep extreme, target at `rr` x risk

The setup is a three-state machine -- idle, armed (sweep seen, waiting for
displacement), pending (FVG formed, waiting for the retrace) -- because the
sweep bar and the displacement bar are by definition different bars. Collapsing
them into one condition is a bug that quietly produces zero trades.

Explicit rules are not the same as profitable rules. That is what the harness
is for: `run_backtest.py --strategy ict --sweep` reports the deflated Sharpe.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..backtest.types import Signal
from .base import Strategy


@dataclass
class ArmedSweep:
    direction: int      # +1 bullish (sell-side liquidity taken), -1 bearish
    level: float        # the swept extreme
    bar: int            # index of the sweep bar


@dataclass
class PendingSetup:
    direction: int
    fvg_low: float
    fvg_high: float
    invalidation: float
    created_at: int
    expires_at: int


def atr(history: pd.DataFrame, period: int = 14) -> float:
    h, l, c = history["high"], history["low"], history["close"]
    prev_close = c.shift()
    tr = np.maximum.reduce([
        (h - l).to_numpy(),
        (h - prev_close).abs().to_numpy(),
        (l - prev_close).abs().to_numpy(),
    ])
    if len(tr) < period:
        return float("nan")
    return float(np.nanmean(tr[-period:]))


class ICTLiquiditySweep(Strategy):
    def __init__(
        self,
        swing_lookback: int = 20,
        displacement_window: int = 5,
        displacement_atr: float = 1.0,
        fvg_expiry_bars: int = 12,
        risk_per_trade: float = 0.01,
        rr: float = 2.0,
        atr_period: int = 14,
        session_hours: tuple[int, int] | None = None,
    ):
        self.params = dict(
            swing_lookback=swing_lookback, displacement_window=displacement_window,
            displacement_atr=displacement_atr, fvg_expiry_bars=fvg_expiry_bars,
            risk_per_trade=risk_per_trade, rr=rr, atr_period=atr_period,
            session_hours=str(session_hours),
        )
        self.swing_lookback = swing_lookback
        self.displacement_window = displacement_window
        self.displacement_atr = displacement_atr
        self.fvg_expiry_bars = fvg_expiry_bars
        self.risk_per_trade = risk_per_trade
        self.rr = rr
        self.atr_period = atr_period
        self.session_hours = session_hours

        self.armed: ArmedSweep | None = None
        self.pending: PendingSetup | None = None

    # -- rule components ---------------------------------------------------
    def _detect_sweep(self, h: pd.DataFrame) -> tuple[int, float]:
        """(+1, level) if sell-side liquidity was swept, (-1, level) buy-side."""
        window = h.iloc[-(self.swing_lookback + 1):-1]
        if len(window) < self.swing_lookback:
            return 0, 0.0
        bar = h.iloc[-1]
        lo, hi = float(window["low"].min()), float(window["high"].max())
        if float(bar["low"]) < lo and float(bar["close"]) > lo:
            return 1, lo
        if float(bar["high"]) > hi and float(bar["close"]) < hi:
            return -1, hi
        return 0, 0.0

    def _displaced_since(self, h: pd.DataFrame, armed: ArmedSweep) -> bool:
        a = atr(h, self.atr_period)
        if not np.isfinite(a) or a <= 0:
            return False
        bars_since = len(h) - 1 - armed.bar
        if bars_since < 1:
            return False
        move = float(h["close"].iloc[-1]) - float(h["close"].iloc[armed.bar])
        return armed.direction * move >= self.displacement_atr * a

    def _find_fvg(self, h: pd.DataFrame, armed: ArmedSweep) -> tuple[float, float] | None:
        """Scan for a three-bar imbalance created since the sweep."""
        start = max(armed.bar, 2)
        best: tuple[float, float] | None = None
        for i in range(start, len(h)):
            a, c = h.iloc[i - 2], h.iloc[i]
            if armed.direction > 0 and float(c["low"]) > float(a["high"]):
                best = (float(a["high"]), float(c["low"]))
            elif armed.direction < 0 and float(c["high"]) < float(a["low"]):
                best = (float(c["high"]), float(a["low"]))
        return best

    def _in_session(self, ts: pd.Timestamp) -> bool:
        if self.session_hours is None:
            return True
        lo, hi = self.session_hours
        return lo <= ts.hour < hi

    # -- strategy interface ------------------------------------------------
    def on_bar(self, history: pd.DataFrame, position_weight: float, equity: float) -> Signal:
        i = len(history) - 1
        bar = history.iloc[-1]

        if position_weight != 0:
            return Signal.hold(position_weight)

        # Expire stale state.
        if self.armed and i - self.armed.bar > self.displacement_window:
            self.armed = None
        if self.pending and i > self.pending.expires_at:
            self.pending = None

        # State 3: waiting for the retrace into the gap.
        if self.pending is not None:
            p = self.pending
            if p.fvg_low <= float(bar["low"]) <= p.fvg_high or \
               p.fvg_low <= float(bar["close"]) <= p.fvg_high:
                entry = float(bar["close"])
                risk = abs(entry - p.invalidation)
                self.pending = None
                if risk <= 0:
                    return Signal(0.0, reason="invalid_risk")
                # Fixed-fractional sizing: a stop-out costs exactly
                # `risk_per_trade` of equity, regardless of instrument or vol.
                weight = float(np.clip(self.risk_per_trade * entry / risk, 0.0, 1.0))
                target = entry + p.direction * risk * self.rr
                return Signal(weight * p.direction, stop_loss=p.invalidation,
                              take_profit=target, reason="fvg_entry",
                              meta={"direction": p.direction, "risk_per_unit": risk})
            return Signal(0.0, reason="awaiting_retrace")

        # State 2: sweep seen, watching for displacement + FVG.
        if self.armed is not None:
            if self._displaced_since(history, self.armed):
                fvg = self._find_fvg(history, self.armed)
                if fvg:
                    lo = min(self.armed.level,
                             float(history["low"].iloc[self.armed.bar]))
                    hi = max(self.armed.level,
                             float(history["high"].iloc[self.armed.bar]))
                    invalidation = lo if self.armed.direction > 0 else hi
                    self.pending = PendingSetup(
                        self.armed.direction, fvg[0], fvg[1], invalidation,
                        i, i + self.fvg_expiry_bars)
                    self.armed = None
                    return Signal(0.0, reason="setup_armed")
            return Signal(0.0, reason="awaiting_displacement")

        # State 1: idle, hunting for a sweep.
        if self._in_session(history.index[-1]):
            direction, level = self._detect_sweep(history)
            if direction:
                self.armed = ArmedSweep(direction, level, i)
                return Signal(0.0, reason="sweep_detected")

        return Signal(0.0, reason="no_setup")
