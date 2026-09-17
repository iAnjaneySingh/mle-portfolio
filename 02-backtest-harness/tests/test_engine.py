import pandas as pd

from backtest.engine import BacktestEngine
from backtest.strategy_interface import Strategy


class AlwaysLongStrategy(Strategy):
    def generate_signal(self, date, price_history):
        return 1.0


class AlwaysFlatStrategy(Strategy):
    def generate_signal(self, date, price_history):
        return 0.0


def _sample_prices():
    idx = pd.bdate_range("2023-01-01", periods=6)
    return pd.Series([100, 102, 101, 105, 110, 108], index=idx)


def test_always_long_matches_buy_and_hold_minus_costs():
    prices = _sample_prices()
    engine = BacktestEngine(prices, AlwaysLongStrategy(), transaction_cost_bps=0.0)
    result = engine.run()

    # With zero transaction costs and always fully long, strategy return should
    # equal the underlying asset's return (shifted by one day for no-lookahead).
    buy_hold_return = prices.iloc[-1] / prices.iloc[0] - 1
    strategy_return = result["equity_curve"].iloc[-1] / result["equity_curve"].iloc[0] - 1
    assert abs(strategy_return - buy_hold_return) < 1e-6


def test_always_flat_has_zero_returns():
    prices = _sample_prices()
    engine = BacktestEngine(prices, AlwaysFlatStrategy(), transaction_cost_bps=5.0)
    result = engine.run()
    assert (result["returns"] == 0).all()
    assert result["equity_curve"].iloc[-1] == result["equity_curve"].iloc[0]


def test_transaction_costs_reduce_returns_on_position_changes():
    prices = _sample_prices()

    class FlipFlopStrategy(Strategy):
        def generate_signal(self, date, price_history):
            return 1.0 if len(price_history) % 2 == 0 else -1.0

    engine_free = BacktestEngine(prices, FlipFlopStrategy(), transaction_cost_bps=0.0)
    engine_costly = BacktestEngine(prices, FlipFlopStrategy(), transaction_cost_bps=50.0)

    free_final = engine_free.run()["equity_curve"].iloc[-1]
    costly_final = engine_costly.run()["equity_curve"].iloc[-1]

    assert costly_final < free_final  # churn should cost money


def test_no_lookahead_first_day_position_has_no_effect():
    prices = _sample_prices()
    engine = BacktestEngine(prices, AlwaysLongStrategy(), transaction_cost_bps=0.0)
    result = engine.run()
    # first day's return must be exactly 0 since position is shifted forward one day
    assert result["returns"].iloc[0] == 0.0
