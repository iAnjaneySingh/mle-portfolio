from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from ..backtest.types import Signal


class Strategy(ABC):
    """Interface enforced by the engine.

    `on_bar` receives history through the current bar and nothing beyond it.
    Anything a strategy needs about the future has to be learned, not indexed.
    """

    params: dict[str, Any] = {}

    def on_start(self, warmup_bars: pd.DataFrame) -> None:
        """Optional: fit state on the warmup window before trading begins."""

    @abstractmethod
    def on_bar(self, history: pd.DataFrame, position_weight: float, equity: float) -> Signal:
        ...


class BuyAndHold(Strategy):
    """Baseline. Every strategy's Sharpe should be reported next to this one --
    a 'profitable' strategy that underperforms holding the asset is not alpha."""

    params = {"weight": 1.0}

    def on_bar(self, history, position_weight, equity) -> Signal:
        return Signal(target_weight=1.0, reason="hold")


class Flat(Strategy):
    """Null strategy. Used in tests to prove the engine adds no drift of its own."""

    def on_bar(self, history, position_weight, equity) -> Signal:
        return Signal(target_weight=0.0, reason="flat")
