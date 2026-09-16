"""CLI entry point.

  python run_backtest.py --strategy multi_agent --walk-forward
  python run_backtest.py --strategy ict --no-mlflow --sweep

The `--sweep` path is what makes the deflated Sharpe honest: it runs N parameter
combinations, logs every one, and then reports each result deflated by the
number of trials actually performed.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging

import pandas as pd

from src.backtest.engine import BacktestEngine
from src.backtest.execution import CostModel
from src.backtest.metrics import build_report
from src.backtest.walkforward import run_walk_forward
from src.data.loader import generate_ohlcv, load_csv
from src.strategies.base import BuyAndHold
from src.strategies.ict_liquidity_sweep import ICTLiquiditySweep
from src.strategies.multi_agent import MultiAgentStrategy

log = logging.getLogger(__name__)

BUILDERS = {
    "ict": lambda **kw: ICTLiquiditySweep(**kw),
    "multi_agent": lambda **kw: MultiAgentStrategy(**kw),
    "buy_hold": lambda **kw: BuyAndHold(),
}

SWEEP_GRID = {
    "ict": {"swing_lookback": [10, 20, 40], "displacement_atr": [0.8, 1.2, 1.8],
            "rr": [1.5, 2.0, 3.0]},
    "multi_agent": {"entry_threshold": [0.25, 0.35, 0.5], "atr_stop_multiple": [1.5, 2.0, 3.0],
                    "rr": [1.5, 2.0, 3.0]},
}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="multi_agent", choices=list(BUILDERS))
    ap.add_argument("--csv", default=None, help="OHLCV csv; synthetic data if omitted")
    ap.add_argument("--bars", type=int, default=20_000)
    ap.add_argument("--capital", type=float, default=100_000)
    ap.add_argument("--periods-per-year", type=int, default=24 * 252)
    ap.add_argument("--commission-bps", type=float, default=1.0)
    ap.add_argument("--walk-forward", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--tracking-uri", default="http://localhost:5000")
    ap.add_argument("--experiment", default="backtests")
    args = ap.parse_args()

    bars = load_csv(args.csv) if args.csv else generate_ohlcv(
        args.bars, periods_per_year=args.periods_per_year)
    engine = BacktestEngine(args.capital, CostModel(commission_bps=args.commission_bps))

    if args.walk_forward:
        df = run_walk_forward(
            bars,
            strategy_factory=lambda train: BUILDERS[args.strategy](),
            engine=engine,
            periods_per_year=args.periods_per_year,
        )
        cols = ["fold", "n_test", "sharpe", "max_drawdown", "win_rate", "profit_factor", "n_trades"]
        print(df[cols].to_string(index=False))
        print("\nsummary:", json.dumps(df.attrs.get("summary", {}), indent=2))
        return

    combos = _combos(args.strategy) if args.sweep else [{}]
    n_trials = len(combos)
    rows = []
    for i, params in enumerate(combos, 1):
        strategy = BUILDERS[args.strategy](**params)
        result = engine.run(bars, strategy)
        if args.no_mlflow:
            rep = build_report(result, args.periods_per_year, n_trials=n_trials,
                               bootstrap=not args.sweep)
        else:
            from src.backtest.tracking import log_backtest
            _, rep = log_backtest(result, bars, strategy, experiment=args.experiment,
                                  tracking_uri=args.tracking_uri,
                                  periods_per_year=args.periods_per_year, n_trials=n_trials)
        rows.append({**params, **rep.to_dict()})
        log.info("[%d/%d] sharpe=%.2f dsr=%.3f mdd=%.1f%% trades=%d",
                 i, n_trials, rep.sharpe, rep.deflated_sharpe, rep.max_drawdown * 100, rep.n_trades)

    out = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    show = [c for c in out.columns if c in set(SWEEP_GRID.get(args.strategy, {}))] + [
        "sharpe", "deflated_sharpe", "psr", "max_drawdown", "win_rate", "profit_factor", "n_trades"]
    print(out[show].head(15).to_string(index=False))
    if n_trials > 1:
        best = out.iloc[0]
        print(f"\nBest raw Sharpe {best.sharpe:.2f} across {n_trials} trials; "
              f"deflated Sharpe {best.deflated_sharpe:.3f}. "
              f"{'Survives' if best.deflated_sharpe > 0.95 else 'Does NOT survive'} "
              "the multiple-testing correction at 95%.")


def _combos(strategy: str) -> list[dict]:
    grid = SWEEP_GRID.get(strategy, {})
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*grid.values())]


if __name__ == "__main__":
    main()
