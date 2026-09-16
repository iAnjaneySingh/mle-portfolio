"""Every backtest run is an MLflow run.

Why this matters more than it sounds: a strategy repo without run tracking has
no defence against the researcher's own memory. Logging params, metrics, the
equity curve, the trade ledger and the *code+data hashes* means:

  * `n_trials` in the deflated Sharpe calculation can be read off the experiment
    instead of guessed, because every trial you ever ran is counted.
  * Two runs with identical params and data hashes must produce identical
    metrics; if they do not, the backtest is non-deterministic and that is a bug.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path

import mlflow
import pandas as pd

from .metrics import PerformanceReport, build_report


def data_fingerprint(bars: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(bars, index=True).values.tobytes())
    return h.hexdigest()[:16]


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "nogit"


def count_prior_trials(experiment: str) -> int:
    """How many backtests have been logged to this experiment so far.

    This is the honest `n_trials` for the multiple-testing correction.
    """
    try:
        exp = mlflow.get_experiment_by_name(experiment)
        if exp is None:
            return 1
        runs = mlflow.search_runs([exp.experiment_id], output_format="pandas")
        return max(len(runs), 1)
    except Exception:
        return 1


def log_backtest(
    result,
    bars: pd.DataFrame,
    strategy,
    experiment: str = "backtests",
    tracking_uri: str = "http://localhost:5000",
    periods_per_year: int = 252,
    n_trials: int | None = None,
    tags: dict | None = None,
) -> tuple[str, PerformanceReport]:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    trials = n_trials if n_trials is not None else count_prior_trials(experiment) + 1
    report = build_report(result, periods_per_year=periods_per_year, n_trials=trials)

    with mlflow.start_run() as run:
        params = {
            "strategy": type(strategy).__name__,
            "initial_capital": result.initial_capital,
            "periods_per_year": periods_per_year,
            "n_bars": len(bars),
            "start": str(bars.index[0]),
            "end": str(bars.index[-1]),
            **{f"p_{k}": v for k, v in getattr(strategy, "params", {}).items()},
            **{f"cost_{k}": v for k, v in result.metadata.get("cost_model", {}).items()},
        }
        mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
        mlflow.log_metrics({k: v for k, v in report.to_dict().items()
                            if pd.notna(v) and abs(v) != float("inf")})
        mlflow.set_tags({
            "data_fingerprint": data_fingerprint(bars),
            "git_sha": _git_sha(),
            "python": platform.python_version(),
            "n_trials_for_dsr": trials,
            **(tags or {}),
        })

        with tempfile.TemporaryDirectory() as d:
            dpath = Path(d)
            result.equity.to_frame().to_csv(dpath / "equity_curve.csv")
            result.returns.to_frame().to_csv(dpath / "returns.csv")
            tf = result.trades_frame
            if not tf.empty:
                tf.to_csv(dpath / "trades.csv", index=False)
            (dpath / "report.json").write_text(json.dumps(asdict(report), indent=2))
            _plot(result, dpath / "equity_curve.png")
            mlflow.log_artifacts(str(dpath))

        return run.info.run_id, report


def _plot(result, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .metrics import drawdown_series

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1, 1]})
    axes[0].plot(result.equity.index, result.equity.values, lw=1.2)
    axes[0].set_ylabel("equity"); axes[0].set_yscale("log"); axes[0].grid(alpha=.3)
    dd = drawdown_series(result.equity)
    axes[1].fill_between(dd.index, dd.values, 0, alpha=.4, color="crimson")
    axes[1].set_ylabel("drawdown"); axes[1].grid(alpha=.3)
    axes[2].plot(result.positions.index, result.positions.values, lw=.8, color="slategray")
    axes[2].set_ylabel("position"); axes[2].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
