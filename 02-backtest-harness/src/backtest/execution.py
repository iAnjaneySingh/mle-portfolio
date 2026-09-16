"""Cost model.

Backtests lie mainly through costs, so these are explicit and configurable:
  * commission  -- per-notional, both sides
  * spread      -- half-spread paid on every fill, crossing the book
  * slippage    -- square-root market impact, scaled by participation rate
  * fills       -- next bar's open, never the signal bar's close

The square-root law (impact ~ sigma * sqrt(Q / ADV)) is the standard empirical
form; with the default participation this is small, but it makes the backtest's
sensitivity to size explicit rather than assumed away.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CostModel:
    commission_bps: float = 1.0        # 0.01% per side
    half_spread_bps: float = 0.5
    impact_coefficient: float = 0.1    # multiplier on the sqrt-impact term
    adv_notional: float = 5e7          # average daily volume in currency terms

    def fill_price(self, reference_price: float, qty: float, volatility: float) -> tuple[float, float]:
        """Return (fill_price, slippage_cost_per_unit). qty is signed."""
        if qty == 0:
            return reference_price, 0.0
        direction = np.sign(qty)
        notional = abs(qty) * reference_price
        participation = min(notional / max(self.adv_notional, 1.0), 1.0)
        impact = self.impact_coefficient * volatility * np.sqrt(participation)
        half_spread = self.half_spread_bps / 1e4
        slip = reference_price * (half_spread + impact)
        return float(reference_price + direction * slip), float(slip)

    def commission(self, qty: float, price: float) -> float:
        return abs(qty) * price * self.commission_bps / 1e4
