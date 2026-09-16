"""Multi-agent ensemble with an explicit, auditable aggregation rule.

The failure mode of multi-agent trading architectures is that "the agents
deliberate and reach a decision" is untestable. Here each agent is a pure
function of history returning a signed conviction in [-1, 1] plus a confidence
in [0, 1], and the aggregator is one line of arithmetic you can argue about.

Agents:
  StructureAgent  -- trend/market-structure bias from swing highs and lows
  LiquidityAgent  -- wraps the ICT sweep+FVG setup as a conviction
  MomentumAgent   -- normalised momentum, the classic cross-sectional anomaly
  VolatilityAgent -- regime filter; suppresses everything in vol blowouts
  RiskAgent       -- hard veto on drawdown and consecutive-loss limits

Because every agent's output is logged per bar, `contributions()` gives a per-
agent attribution of the final PnL, which turns "the ensemble works" into a
claim you can decompose.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..backtest.types import Signal
from .base import Strategy
from .ict_liquidity_sweep import ICTLiquiditySweep, atr


@dataclass
class AgentVote:
    name: str
    conviction: float   # signed, [-1, 1]
    confidence: float   # [0, 1]
    veto: bool = False
    note: str = ""


class Agent(ABC):
    name: str = "agent"

    @abstractmethod
    def vote(self, history: pd.DataFrame, state: dict) -> AgentVote:
        ...


class StructureAgent(Agent):
    name = "structure"

    def __init__(self, lookback: int = 50, swing: int = 5):
        self.lookback, self.swing = lookback, swing

    def vote(self, history, state) -> AgentVote:
        h = history.iloc[-self.lookback:]
        highs = h["high"].rolling(self.swing * 2 + 1, center=True).max()
        lows = h["low"].rolling(self.swing * 2 + 1, center=True).min()
        swing_highs = h["high"][h["high"] == highs].dropna()
        swing_lows = h["low"][h["low"] == lows].dropna()
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return AgentVote(self.name, 0.0, 0.2, note="insufficient structure")
        hh = swing_highs.iloc[-1] > swing_highs.iloc[-2]
        hl = swing_lows.iloc[-1] > swing_lows.iloc[-2]
        if hh and hl:
            return AgentVote(self.name, 1.0, 0.8, note="HH+HL bullish structure")
        if not hh and not hl:
            return AgentVote(self.name, -1.0, 0.8, note="LH+LL bearish structure")
        return AgentVote(self.name, 0.0, 0.4, note="mixed structure")


class LiquidityAgent(Agent):
    name = "liquidity"

    def __init__(self, **kwargs):
        self.inner = ICTLiquiditySweep(**kwargs)

    def vote(self, history, state) -> AgentVote:
        sig = self.inner.on_bar(history, position_weight=0.0, equity=state.get("equity", 1.0))
        if sig.reason == "fvg_entry":
            direction = float(np.sign(sig.target_weight))
            return AgentVote(self.name, direction, 0.9, note="sweep+FVG entry")
        if self.inner.pending is not None:
            d = float(self.inner.pending.direction)
            return AgentVote(self.name, d * 0.4, 0.5, note="setup armed, awaiting retrace")
        return AgentVote(self.name, 0.0, 0.3, note="no setup")


class MomentumAgent(Agent):
    name = "momentum"

    def __init__(self, lookback: int = 60, skip: int = 5):
        self.lookback, self.skip = lookback, skip

    def vote(self, history, state) -> AgentVote:
        c = history["close"]
        if len(c) < self.lookback + self.skip:
            return AgentVote(self.name, 0.0, 0.1)
        # Skip the most recent bars: short-horizon reversal contaminates raw
        # momentum, which is why the academic factor is 12-2, not 12-0.
        past = c.iloc[-(self.lookback + self.skip)]
        recent = c.iloc[-self.skip]
        raw = (recent / past - 1)
        vol = c.pct_change().rolling(self.lookback).std().iloc[-1] * np.sqrt(self.lookback)
        z = raw / vol if vol and np.isfinite(vol) else 0.0
        return AgentVote(self.name, float(np.clip(z, -1, 1)), 0.6, note=f"z={z:.2f}")


class VolatilityAgent(Agent):
    name = "volatility"

    def __init__(self, period: int = 14, blowout_multiple: float = 2.5, lookback: int = 100):
        self.period, self.blowout, self.lookback = period, blowout_multiple, lookback

    def vote(self, history, state) -> AgentVote:
        a = atr(history, self.period)
        baseline = history["close"].diff().abs().rolling(self.lookback).mean().iloc[-1]
        if not np.isfinite(a) or not baseline:
            return AgentVote(self.name, 0.0, 0.1)
        ratio = a / baseline
        if ratio > self.blowout:
            return AgentVote(self.name, 0.0, 1.0, veto=True,
                             note=f"vol blowout {ratio:.1f}x")
        # Scale conviction down as vol rises: same risk budget, smaller size.
        return AgentVote(self.name, 0.0, float(np.clip(1.5 - ratio / 2, 0.1, 1.0)),
                         note=f"vol ratio {ratio:.2f}")


class RiskAgent(Agent):
    """Hard limits. This agent never expresses a view, only a veto."""

    name = "risk"

    def __init__(self, max_drawdown: float = 0.15, max_consecutive_losses: int = 5):
        self.max_drawdown = max_drawdown
        self.max_consecutive_losses = max_consecutive_losses

    def vote(self, history, state) -> AgentVote:
        dd = state.get("drawdown", 0.0)
        streak = state.get("loss_streak", 0)
        if dd <= -self.max_drawdown:
            return AgentVote(self.name, 0.0, 1.0, veto=True, note=f"drawdown {dd:.1%}")
        if streak >= self.max_consecutive_losses:
            return AgentVote(self.name, 0.0, 1.0, veto=True, note=f"{streak} losses in a row")
        return AgentVote(self.name, 0.0, 1.0, note="within limits")


class MultiAgentStrategy(Strategy):
    def __init__(
        self,
        agents: list[Agent] | None = None,
        weights: dict[str, float] | None = None,
        entry_threshold: float = 0.35,
        max_weight: float = 0.5,
        atr_stop_multiple: float = 2.0,
        rr: float = 2.0,
    ):
        self.agents = agents or [
            StructureAgent(), LiquidityAgent(), MomentumAgent(),
            VolatilityAgent(), RiskAgent(),
        ]
        self.weights = weights or {"structure": 1.0, "liquidity": 1.5, "momentum": 1.0}
        self.entry_threshold = entry_threshold
        self.max_weight = max_weight
        self.atr_stop_multiple = atr_stop_multiple
        self.rr = rr
        self.params = dict(
            entry_threshold=entry_threshold, max_weight=max_weight,
            atr_stop_multiple=atr_stop_multiple, rr=rr,
            agents=",".join(a.name for a in self.agents),
            weights=str(self.weights),
        )
        self.vote_log: list[dict] = []
        self._peak_equity = 0.0
        self._loss_streak = 0

    def record_trade_outcome(self, pnl: float) -> None:
        self._loss_streak = self._loss_streak + 1 if pnl <= 0 else 0

    def aggregate(self, votes: list[AgentVote]) -> tuple[float, float]:
        """Confidence-weighted mean conviction, scaled by the product of all
        non-directional agents' confidences (the regime/risk gates).

        Returns (score, gate). A veto zeroes the gate outright.
        """
        if any(v.veto for v in votes):
            return 0.0, 0.0
        directional = [v for v in votes if v.name in self.weights]
        gates = [v for v in votes if v.name not in self.weights]
        num = sum(self.weights[v.name] * v.confidence * v.conviction for v in directional)
        den = sum(self.weights[v.name] * v.confidence for v in directional) or 1.0
        gate = float(np.prod([v.confidence for v in gates])) if gates else 1.0
        return float(num / den), gate

    def on_bar(self, history: pd.DataFrame, position_weight: float, equity: float) -> Signal:
        self._peak_equity = max(self._peak_equity, equity)
        state = {
            "equity": equity,
            "drawdown": equity / self._peak_equity - 1 if self._peak_equity else 0.0,
            "loss_streak": self._loss_streak,
            "position_weight": position_weight,
        }

        votes = [a.vote(history, state) for a in self.agents]
        score, gate = self.aggregate(votes)
        self.vote_log.append({
            "ts": history.index[-1], "score": score, "gate": gate,
            **{v.name: v.conviction * v.confidence for v in votes},
        })

        if position_weight != 0:
            # Exit when conviction decays through half the entry threshold --
            # asymmetric thresholds avoid whipsawing around a single level.
            if abs(score) < self.entry_threshold / 2 or gate == 0.0:
                return Signal(0.0, reason="conviction_decay")
            return Signal.hold(position_weight)

        if abs(score) < self.entry_threshold or gate == 0.0:
            return Signal(0.0, reason="below_threshold")

        a = atr(history, 14)
        price = float(history["close"].iloc[-1])
        if not np.isfinite(a) or a <= 0:
            return Signal(0.0, reason="no_atr")

        direction = float(np.sign(score))
        weight = direction * min(abs(score) * gate, 1.0) * self.max_weight
        stop = price - direction * self.atr_stop_multiple * a
        target = price + direction * self.atr_stop_multiple * a * self.rr
        return Signal(weight, stop_loss=stop, take_profit=target,
                      reason="ensemble_entry",
                      meta={"score": score, "gate": gate,
                            "votes": {v.name: v.note for v in votes}})

    def contributions(self) -> pd.DataFrame:
        """Per-agent signed contribution over time, for attribution analysis."""
        return pd.DataFrame(self.vote_log).set_index("ts") if self.vote_log else pd.DataFrame()
