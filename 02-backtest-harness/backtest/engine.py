"""Core backtest simulation loop.

Deliberately simple and transparent: given a price series and a Strategy,
walk forward day by day, ask the strategy for a target position, apply it to
the *next* day's return (no lookahead), charge transaction costs on position
changes, and track the resulting equity curve.
"""

import pandas as pd

from .strategy_interface import Strategy


class BacktestEngine:
    def __init__(self, prices: pd.Series, strategy: Strategy,
                 transaction_cost_bps: float = 5.0, initial_capital: float = 100_000.0):
        """
        prices: pd.Series of closing prices, indexed by date, sorted ascending.
        transaction_cost_bps: cost per unit of position change, in basis points
            (5 bps = 0.05%), charged on every rebalance to make signal churn
            actually cost something — a common gap in naive backtests.
        """
        self.prices = prices.sort_index()
        self.strategy = strategy
        self.cost_rate = transaction_cost_bps / 10_000
        self.initial_capital = initial_capital

    def run(self) -> dict:
        dates = self.prices.index
        daily_returns = self.prices.pct_change().fillna(0.0)

        positions = []
        prev_position = 0.0

        for i, date in enumerate(dates):
            history = self.prices.iloc[: i + 1]
            target_position = self.strategy.generate_signal(date, history)
            target_position = max(-1.0, min(1.0, target_position))  # clip to valid range
            positions.append(target_position)

        position_series = pd.Series(positions, index=dates)

        # Position on day t earns the return realized on day t+1 (no lookahead:
        # you decide today's position using data through today's close, and it's
        # exposed to tomorrow's move).
        exposed_position = position_series.shift(1).fillna(0.0)
        strategy_returns = exposed_position * daily_returns

        turnover = position_series.diff().abs().fillna(position_series.abs())
        transaction_costs = turnover * self.cost_rate
        net_returns = strategy_returns - transaction_costs

        equity_curve = self.initial_capital * (1 + net_returns).cumprod()

        return {
            "equity_curve": equity_curve,
            "returns": net_returns,
            "positions": position_series,
            "turnover": turnover,
        }
