"""Wraps MLflow so every backtest run is versioned: parameters, metrics, and
the equity curve itself, so runs are comparable later instead of living only
in a terminal scrollback."""

import tempfile
from pathlib import Path

import mlflow
import pandas as pd


def log_backtest_run(run_name: str, params: dict, metrics: dict,
                      equity_curve: pd.Series, experiment_name: str = "trading-backtests"):
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "equity_curve.csv"
            equity_curve.to_csv(csv_path, header=["equity"])
            mlflow.log_artifact(str(csv_path))

        run = mlflow.active_run()
        return run.info.run_id
