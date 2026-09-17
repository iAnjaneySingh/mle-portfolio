"""Pluggable strategy interface.

This is the seam where a real signal source plugs in — e.g. the conviction
score produced by a multi-agent trading system's debate/consensus step.
Anything that implements `generate_signal` can be backtested by this harness
without touching the engine or metrics code.
"""

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    @abstractmethod
    def generate_signal(self, date: pd.Timestamp, price_history: pd.Series) -> float:
        """Return a target position in [-1.0, 1.0] for `date`, given all prices
        up to and including `date`. 1.0 = fully long, -1.0 = fully short, 0 = flat.

        `price_history` is provided instead of just the latest price so
        strategies (including multi-agent ones) can compute their own
        rolling features — momentum, volatility regime, whatever the agents
        actually condition on.
        """
        raise NotImplementedError
