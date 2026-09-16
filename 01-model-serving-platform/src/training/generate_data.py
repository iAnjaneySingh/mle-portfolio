"""Synthetic transaction generator so the whole stack runs with one command."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MERCHANTS = ["grocery", "fuel", "electronics", "travel", "gambling", "crypto", "dining"]
COUNTRIES = ["US", "GB", "IN", "DE", "NG", "BR"]


def generate(n: int = 60_000, n_accounts: int = 4_000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    accounts = [f"acct_{i:05d}" for i in range(n_accounts)]
    home = {a: rng.choice(COUNTRIES) for a in accounts}

    account_id = rng.choice(accounts, size=n)
    merchant = rng.choice(MERCHANTS, size=n, p=[.28, .18, .14, .12, .06, .06, .16])
    country = np.where(rng.random(n) < 0.12, rng.choice(COUNTRIES, n),
                       [home[a] for a in account_id])

    amount = np.exp(rng.normal(3.4, 1.15, n))
    amount_mean_30d = np.exp(rng.normal(3.4, 0.55, n))
    txn_count_24h = rng.poisson(3.2, n)
    seconds_since_prev = rng.exponential(9_000, n)

    start = pd.Timestamp("2025-01-01", tz="UTC")
    event_ts = start + pd.to_timedelta(np.sort(rng.uniform(0, 240 * 86400, n)), unit="s")

    df = pd.DataFrame({
        "account_id": account_id,
        "event_ts": event_ts,
        "amount": amount,
        "amount_mean_30d": amount_mean_30d,
        "txn_count_24h": txn_count_24h,
        "seconds_since_prev_txn": seconds_since_prev,
        "merchant_category": merchant,
        "country": country,
        "home_country": [home[a] for a in account_id],
    })

    # Label: a monotone function of risk drivers plus noise -> learnable signal.
    z = (
        -4.1
        + 0.55 * np.log1p(df.amount) / 2
        + 0.9 * (df.country != df.home_country)
        + 0.8 * df.merchant_category.isin(["gambling", "crypto"])
        + 0.25 * (df.amount / df.amount_mean_30d).clip(0, 20) / 5
        + 0.12 * df.txn_count_24h
    )
    p = 1 / (1 + np.exp(-z))
    df["is_fraud"] = (rng.random(n) < p).astype(int)

    # Inject covariate drift in the last 15% of the timeline so the monitoring
    # dashboard has something real to detect.
    tail = df.index[int(0.85 * n):]
    df.loc[tail, "amount"] *= 1.9
    df.loc[tail, "merchant_category"] = np.where(
        rng.random(len(tail)) < 0.35, "crypto", df.loc[tail, "merchant_category"]
    )
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/events.parquet")
    ap.add_argument("--rows", type=int, default=60_000)
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df = generate(args.rows)
    df.to_parquet(args.out, index=False)
    print(f"wrote {len(df):,} rows -> {args.out} (fraud rate {df.is_fraud.mean():.3%})")


if __name__ == "__main__":
    main()
