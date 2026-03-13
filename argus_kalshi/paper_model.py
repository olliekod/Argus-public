from __future__ import annotations

import math
import random


def estimate_kalshi_taker_fee_usd(price_cents: int, quantity_centicx: int) -> float:
    """Estimate taker fees using Kalshi's 7% * p * (1-p) schedule.

    Fees are rounded up to the nearest cent per fill to stay conservative.
    """
    if price_cents <= 0 or quantity_centicx <= 0:
        return 0.0
    contracts = quantity_centicx / 100.0
    p = max(0.0, min(1.0, price_cents / 100.0))
    raw_fee_usd = 0.07 * contracts * p * (1.0 - p)
    if raw_fee_usd <= 0:
        return 0.0
    return math.ceil(raw_fee_usd * 100.0) / 100.0


def sample_paper_order_latency_s(min_ms: int, max_ms: int) -> float:
    """Return a non-negative simulated order latency in seconds."""
    lo = max(0, min_ms)
    hi = max(lo, max_ms)
    if hi <= 0:
        return 0.0
    return random.uniform(lo, hi) / 1000.0
