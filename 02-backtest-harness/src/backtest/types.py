from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class Signal:
    """What a strategy emits at the close of bar t.

    `target_weight` is a fraction of equity in [-1, 1]. Expressing intent as a
    target rather than a buy/sell order means position sizing, netting and
    rebalancing all live in the engine, where they can be tested once.
    """

    target_weight: float = 0.0
    stop_loss: float | None = None
    take_profit: float | None = None
    reason: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not -1.0 <= self.target_weight <= 1.0:
            raise ValueError(f"target_weight out of range: {self.target_weight}")

    @classmethod
    def hold(cls, current_weight: float, reason: str = "hold_position") -> "Signal":
        """Re-express the current position as a target, unchanged.

        Mark-to-market drift can push the realised weight marginally past 1.0
        between rebalances, so this clamps rather than raising -- the strategy
        is expressing "leave it alone", not requesting leverage.
        """
        return cls(target_weight=float(min(max(current_weight, -1.0), 1.0)), reason=reason)


@dataclass
class Fill:
    ts: pd.Timestamp
    price: float
    qty: float          # signed; positive = buy
    commission: float
    slippage: float
    kind: str           # entry | exit | stop | target | rebalance


@dataclass
class Trade:
    """A completed round trip, reconstructed from fills."""

    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    side: Side
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    return_pct: float
    bars_held: int
    exit_reason: str
    mae: float = 0.0    # maximum adverse excursion, in R
    mfe: float = 0.0    # maximum favourable excursion, in R
