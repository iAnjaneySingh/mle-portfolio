"""Train/serve parity harness.

Runs the same events through both compilers and reports, per feature, the
fraction of rows that disagree beyond tolerance. This is a CI gate, not a
diagnostic: if parity drops below threshold the build fails, because a silently
diverging feature is worse than a broken one -- the broken one gets noticed.

Typical causes of failure, all of which this catches:
  * window boundary inclusivity ((t-w, t] vs [t-w, t))
  * Spark's `stddev` being the sample stddev while pandas defaults to ddof=1
    only for `.std()` and ddof=0 for `.agg('std')` in some versions
  * null handling in `count` (counts non-null; a null amount silently shrinks it)
  * the streaming side including the current event, the batch side not
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .definitions import FeatureGroup


@dataclass
class ParityResult:
    feature: str
    n_compared: int
    n_mismatched: int
    max_abs_diff: float
    max_rel_diff: float
    passed: bool

    @property
    def match_rate(self) -> float:
        return 1.0 - self.n_mismatched / self.n_compared if self.n_compared else 0.0


def compare(
    batch: pd.DataFrame,
    streaming: pd.DataFrame,
    group: FeatureGroup,
    key_cols: tuple[str, ...] = ("user_id", "event_ts"),
    rtol: float = 1e-6,
    atol: float = 1e-9,
    min_match_rate: float = 0.999,
) -> list[ParityResult]:
    merged = batch.merge(streaming, on=list(key_cols), suffixes=("_batch", "_stream"), how="inner")
    if merged.empty:
        raise ValueError("no overlapping rows between batch and streaming outputs")

    results = []
    for name in group.feature_names:
        b, s = f"{name}_batch", f"{name}_stream"
        if b not in merged or s not in merged:
            continue
        bv = merged[b].astype(float).to_numpy()
        sv = merged[s].astype(float).to_numpy()
        close = np.isclose(bv, sv, rtol=rtol, atol=atol, equal_nan=True)
        diff = np.abs(bv - sv)
        denom = np.where(np.abs(bv) > atol, np.abs(bv), 1.0)
        n_bad = int((~close).sum())
        results.append(ParityResult(
            feature=name,
            n_compared=len(merged),
            n_mismatched=n_bad,
            max_abs_diff=float(np.nanmax(diff)) if len(diff) else 0.0,
            max_rel_diff=float(np.nanmax(diff / denom)) if len(diff) else 0.0,
            passed=(1 - n_bad / len(merged)) >= min_match_rate,
        ))
    return results


def report(results: list[ParityResult]) -> str:
    lines = [f"{'feature':<24} {'match':>8} {'max_abs':>12} {'max_rel':>10}  status"]
    for r in results:
        lines.append(
            f"{r.feature:<24} {r.match_rate:>7.3%} {r.max_abs_diff:>12.6g} "
            f"{r.max_rel_diff:>10.4g}  {'PASS' if r.passed else 'FAIL'}"
        )
    failed = [r.feature for r in results if not r.passed]
    lines.append("")
    lines.append(f"{len(results) - len(failed)}/{len(results)} features in parity"
                 + (f" -- FAILING: {', '.join(failed)}" if failed else ""))
    return "\n".join(lines)
