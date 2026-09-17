"""CLI entry point: load price data, run the backtest across a small
parameter grid, log every run to MLflow. Run `mlflow ui` afterward to
compare runs side by side.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.engine import BacktestEngine
from backtest.metrics import summarize
from backtest.mlflow_logger import log_backtest_run
from strategies.example_momentum_strategy import MovingAverageCrossoverStrategy


def load_prices(csv_path: str) -> pd.Series:
    df = pd.read_csv(csv_path, parse_dates=["date"])
    return df.set_index("date")["close"]


def main():
    prices = load_prices(Path(__file__).resolve().parent.parent / "data" / "sample_prices.csv")

    param_grid = [
        {"fast_window": 5, "slow_window": 20},
        {"fast_window": 10, "slow_window": 30},
        {"fast_window": 20, "slow_window": 60},
    ]

    for params in param_grid:
        strategy = MovingAverageCrossoverStrategy(**params)
        engine = BacktestEngine(prices, strategy, transaction_cost_bps=5.0)
        result = engine.run()

        metrics = summarize(result["equity_curve"], result["returns"])
        run_name = f"ma_crossover_fast{params['fast_window']}_slow{params['slow_window']}"

        run_id = log_backtest_run(
            run_name=run_name,
            params={**params, "transaction_cost_bps": 5.0, "initial_capital": 100_000},
            metrics=metrics,
            equity_curve=result["equity_curve"],
        )

        print(f"[{run_name}] run_id={run_id}")
        print(f"  Sharpe: {metrics['sharpe_ratio']:.2f}  "
              f"MaxDD: {metrics['max_drawdown']:.2%}  "
              f"WinRate: {metrics['win_rate']:.2%}  "
              f"TotalReturn: {metrics['total_return']:.2%}")


if __name__ == "__main__":
    main()
