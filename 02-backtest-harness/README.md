# Trading Strategy Backtest Harness

A backtest engine with proper performance metrics (Sharpe, Sortino, max
drawdown, win rate, CAGR) and MLflow-versioned runs — the piece that turns
"I built a trading strategy" into "here's the evidence it works."

## Why this exists

A strategy architecture with no measured performance is not evidence of
anything. This harness plugs into *any* signal source — a moving-average
crossover (included, as a runnable example), or a multi-agent system's
conviction score — and produces the numbers a reviewer actually wants to see.

## What it handles

- **No-lookahead simulation**: a strategy's position on day *t* is only
  exposed to day *t+1*'s return, enforced by the engine, not left to the
  strategy author to get right
- **Transaction costs on turnover**: every position change is charged, so a
  strategy that churns its position looks worse than one that doesn't —
  matching real trading friction instead of a frictionless fantasy
- **Standard performance metrics**: Sharpe, Sortino, max drawdown, CAGR, win
  rate, total return — each independently unit-tested against hand-computed
  values, not just "trust the formula"
- **MLflow-versioned runs**: every backtest logs its parameters, metrics, and
  full equity curve as an MLflow run, so you can compare 50 parameter
  combinations later instead of scrolling terminal output

## Architecture

```
Strategy.generate_signal(date, price_history) -> position in [-1, 1]
                    |
              BacktestEngine
     (shift forward 1 day, apply txn costs)
                    |
         equity_curve, returns, positions
                    |
              metrics.summarize()
                    |
          mlflow_logger.log_backtest_run()
                    |
             mlflow ui  (compare runs)
```

## Run it

```bash
pip install -r requirements-dev.txt
python scripts/run_backtest.py
mlflow ui   # opens http://localhost:5000 — compare all logged runs
```

Sample output against the included synthetic price series:

```
[ma_crossover_fast5_slow20]   Sharpe: 0.08   MaxDD: -10.07%  WinRate: 45.60%  TotalReturn: 0.71%
[ma_crossover_fast10_slow30]  Sharpe: 0.07   MaxDD: -7.91%   WinRate: 46.60%  TotalReturn: 0.46%
[ma_crossover_fast20_slow60]  Sharpe: -0.18  MaxDD: -17.38%  WinRate: 42.00%  TotalReturn: -5.03%
```

(Honest framing: this is a synthetic random-walk price series, so weak/mixed
Sharpe ratios are the *expected* and correct result — a crossover strategy
has no real edge on noise. Swap in real price data and/or a real signal
source to get numbers that mean something.)

## Run tests

```bash
pytest tests/ -v
```

11 tests: 4 on the simulation engine (no-lookahead, transaction cost
correctness, buy-and-hold equivalence for an always-long strategy), 7 on the
metrics module (each checked against a hand-computed expected value).

## Plugging in a real strategy

Implement the `Strategy` interface:

```python
from backtest.strategy_interface import Strategy

class MultiAgentStrategy(Strategy):
    def generate_signal(self, date, price_history):
        return self.agent_system.get_conviction(date, price_history)  # in [-1, 1]
```

Pass an instance to `BacktestEngine`, run it, log it — nothing else in the
harness needs to change. This is the seam mentioned in the earlier review:
your multi-agent trading system's conviction output is exactly what this
interface expects.
