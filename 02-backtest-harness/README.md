# Backtest Harness for a Multi-Agent Trading System

An evaluation layer for a multi-agent trading architecture: an event-driven
backtest engine with realistic costs, a metrics suite that reports statistical
significance rather than just Sharpe, purged walk-forward validation, and MLflow
tracking of every run.

The premise is that a trading architecture without an evaluation harness is an
untested claim. This repo turns "the agents deliberate and reach a decision"
into a number with a confidence interval attached.

## The three things that make a backtest honest

**1. No lookahead, structurally.** A strategy's `on_bar` receives
`bars.iloc[:t+1]` — a slice, not the full frame. Signals emitted at the close of
bar *t* fill at the **open of bar t+1**. When a bar's range spans both a stop and
a target, the stop is assumed to have been hit first, because without tick data
the ordering is unknowable and the backtest should take the unfavourable branch.
Each of these has a test in `tests/test_no_lookahead.py` that fails loudly if it
regresses.

**2. Costs are a model, not a constant.** Per-side commission, a half-spread paid
on every fill, and square-root market impact scaled by participation rate
(`impact ≈ σ·√(Q/ADV)`). The engine is therefore *size-sensitive*: a strategy
that only works at $10k and dies at $10M shows it here instead of in production.

**3. Sharpe is reported with its multiple-testing correction.** `metrics.py`
implements the Probabilistic Sharpe Ratio and the **Deflated Sharpe Ratio**
(Bailey & López de Prado): PSR benchmarked against the *expected maximum* Sharpe
across N trials. The MLflow experiment knows how many backtests have been run, so
`n_trials` is read from the tracking store rather than wished into being 1. A
Sharpe of 1.8 found after 200 parameter combinations is not the same object as a
Sharpe of 1.8 found on the first try, and the DSR says so. Confidence intervals
come from a Politis–Romano **stationary bootstrap**, which preserves the serial
correlation an i.i.d. bootstrap destroys.

## Metrics

| Group | |
|---|---|
| Returns | total return, CAGR, annual vol, Sharpe, Sortino, Calmar |
| Risk | max drawdown, drawdown duration, Ulcer index, VaR/CVaR 95%, skew, excess kurtosis |
| Trades | count, win rate, profit factor, **expectancy in R**, payoff ratio, max consecutive losses, avg bars held, MAE/MFE per trade |
| Exposure | time in market, turnover |
| Significance | PSR, deflated Sharpe, bootstrap Sharpe CI, `n_trials` |

Expectancy in R is the one that matters most for a stop-based strategy: a 40%
win rate at 2R is a better business than a 70% win rate at 0.3R, and only
R-multiples make that comparable.

## The agents

`src/strategies/multi_agent.py` decomposes the system into pure functions, each
returning a signed conviction in [-1, 1] and a confidence in [0, 1]:

| Agent | Role |
|---|---|
| `StructureAgent` | HH/HL vs LH/LL market structure bias |
| `LiquidityAgent` | wraps the ICT sweep → displacement → FVG setup |
| `MomentumAgent` | 60-5 momentum, z-scored (skip window avoids short-horizon reversal) |
| `VolatilityAgent` | regime gate; hard veto on vol blowouts |
| `RiskAgent` | hard veto on drawdown and consecutive-loss limits |

Aggregation is a confidence-weighted mean of the directional agents, multiplied
by the product of the gate agents' confidences, with any veto zeroing the result.
That is one line of arithmetic you can argue about — which is the whole point.
`strategy.contributions()` returns per-agent signal history, so PnL can be
attributed rather than asserted.

`src/strategies/ict_liquidity_sweep.py` turns the discretionary SMC setup into a
three-state machine: idle → armed (sweep detected) → pending (FVG formed) →
entry on retracement. The sweep bar and the displacement bar are by definition
different bars; collapsing them into one condition is a bug that produces zero
trades and looks like "no setups occurred."

## Walk-forward with purging and embargo

`purged_walk_forward_splits` drops training bars whose forward label horizon
crosses into the test window, then adds an embargo gap for serial correlation
(López de Prado, AFML ch. 7). The fold summary reports `sharpe_consistency` —
the fraction of folds with positive Sharpe — because a strategy that is
spectacular in one fold and flat in four is a curve fit with good luck.

## Run it

```bash
pip install -r requirements.txt
make mlflow                                        # tracking server on :5000
python run_backtest.py --strategy multi_agent      # single run, logged
python run_backtest.py --strategy ict --sweep      # 27-point grid, DSR-corrected
python run_backtest.py --strategy multi_agent --walk-forward
python run_backtest.py --strategy buy_hold --no-mlflow   # the baseline to beat
pytest -q
```

Every run logs params, all metrics, the equity/drawdown/position plot, the full
trade ledger, a **hash of the input data**, and the git SHA. Two runs with the
same params and data hash must produce identical metrics; if they don't, the
backtest is non-deterministic, and that is a bug worth finding.

## What the harness currently says

On the synthetic regime-switching series (`generate_ohlcv`, 12k hourly bars,
fat-tailed innovations, occasional gaps), out of the box:

```
strategy   sharpe      mdd   trades  win_rate   PF   expectancy
buy_hold     0.25  -22.9%        -        -      -            -
ict          0.46   -5.4%       70     0.40   1.23        0.14R
multi_agent -2.62   -6.0%      570     0.42   0.69       -0.18R
```

The ensemble loses money on this data, and the harness says so plainly. That is
the harness working. The interesting follow-up is the attribution: the momentum
agent is the one dragging, which is unsurprising on a series with no persistent
trend, and it is a hypothesis you can now test instead of a hunch.

Point it at real bars with `--csv your_data.csv` before drawing any conclusion
about the strategy; the synthetic generator exists so the pipeline is
reproducible, not so the results are meaningful.

## Known limits

- Single instrument, single position. Multi-asset portfolio construction
  (correlation-aware sizing, risk parity) is the obvious next layer.
- Bar-level fills. Queue position and partial fills need tick data.
- Costs are calibrated to liquid futures/FX; equities need borrow and locate.
