"""A simple moving-average-crossover strategy — a stand-in signal source so
the harness is runnable end to end without the full multi-agent system.

To plug in the real multi-agent conviction score instead: implement
`generate_signal` to call the agent debate/consensus step and return its
conviction in [-1, 1]. Nothing else in this repo needs to change.
"""

from backtest.strategy_interface import Strategy


class MovingAverageCrossoverStrategy(Strategy):
    def __init__(self, fast_window: int = 10, slow_window: int = 30):
        self.fast_window = fast_window
        self.slow_window = slow_window

    def generate_signal(self, date, price_history):
        if len(price_history) < self.slow_window:
            return 0.0  # not enough history yet — stay flat

        fast_ma = price_history.tail(self.fast_window).mean()
        slow_ma = price_history.tail(self.slow_window).mean()

        if slow_ma == 0:
            return 0.0

        # Signal strength scales with how far the fast MA has diverged from
        # the slow MA, clipped to [-1, 1] by the engine.
        divergence = (fast_ma - slow_ma) / slow_ma
        return divergence * 20  # scaling factor tuned for typical divergence magnitudes
