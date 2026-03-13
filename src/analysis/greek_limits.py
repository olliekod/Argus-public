"""
Greek Limits
=============

Enforces per-underlying and portfolio-level greek limits on proposed
allocations.  When limits are exceeded, allocation sizes (contracts/weights)
are reduced deterministically.

Unit Conventions (DOCUMENTED)
-----------------------------
- **delta**: shares-equivalent.  For a single option contract (100 shares
  multiplier), delta_shares_equiv = option_delta * 100 * num_contracts.
  For equity positions, delta_shares_equiv = number of shares.
- **vega**: USD per 1 volatility point (1% IV change).
  ``vega_usd_per_vol_point = bs_vega_per_1pct * num_contracts * 100``.
  (GreeksEngine.calculate_vega already returns vega per 1% IV change;
  we multiply by contracts * multiplier.)
- **gamma**: shares per $1 move in underlying.
  ``gamma_shares_per_dollar = bs_gamma * 100 * num_contracts``.
  (GreeksEngine.calculate_gamma returns gamma per share per $1.)

All limits are expressed in these units.

References
----------
- MASTER_PLAN.md §9 — Phase 5: Portfolio Risk Engine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("argus.greek_limits")

# Standard equity option multiplier
_OPTION_MULTIPLIER = 100


@dataclass
class GreekLimitsConfig:
    """Configuration for greek limits enforcement.

    Attributes
    ----------
    per_underlying : dict
        ``{underlying: {max_delta_shares, max_vega, max_gamma}}``.
        Limits are absolute values (applied symmetrically to long/short).
    portfolio_max_delta_shares : float
        Portfolio-wide absolute delta limit in shares-equivalent.
    portfolio_max_vega : float
        Portfolio-wide absolute vega limit in USD per vol point.
    portfolio_max_gamma : float
        Portfolio-wide absolute gamma limit in shares per dollar.
    enforce_existing_positions : bool
        If True, existing position greeks count toward limits.
    """
    per_underlying: Dict[str, Dict[str, float]] = field(default_factory=dict)
    portfolio_max_delta_shares: float = float("inf")
    portfolio_max_vega: float = float("inf")
    portfolio_max_gamma: float = float("inf")
    enforce_existing_positions: bool = True


def compute_proposed_greeks(
    normalized_allocs: List[Any],
    existing_greeks: Dict[str, float],
    enforce_existing: bool = True,
) -> Dict[str, float]:
    """Compute total proposed greeks (existing + new).

    Parameters
    ----------
    normalized_allocs : list
        NormalizedAllocation objects with ``delta_shares_equiv``, ``vega``,
        ``gamma`` attributes.
    existing_greeks : dict
        ``{delta: float, gamma: float, vega: float}`` from current positions.
    enforce_existing : bool
        Whether to include existing position greeks in totals.

    Returns
    -------
    dict
        ``{delta: float, gamma: float, vega: float}`` total.
    """
    total_delta = 0.0
    total_gamma = 0.0
    total_vega = 0.0

    for alloc in normalized_allocs:
        total_delta += getattr(alloc, "delta_shares_equiv", 0.0)
        total_gamma += getattr(alloc, "gamma", 0.0)
        total_vega += getattr(alloc, "vega", 0.0)

    if enforce_existing:
        total_delta += existing_greeks.get("delta", 0.0)
        total_gamma += existing_greeks.get("gamma", 0.0)
        total_vega += existing_greeks.get("vega", 0.0)

    return {
        "delta": total_delta,
        "gamma": total_gamma,
        "vega": total_vega,
    }


def compute_greeks_for_underlying(
    normalized_allocs: List[Any],
    underlying: str,
    existing_greeks_for_underlying: Optional[Dict[str, float]] = None,
    enforce_existing: bool = True,
) -> Dict[str, float]:
    """Compute greeks for a specific underlying.

    Parameters
    ----------
    normalized_allocs : list
        All NormalizedAllocation objects.
    underlying : str
        Underlying to filter by.
    existing_greeks_for_underlying : dict, optional
        Existing position greeks for this underlying.
    enforce_existing : bool
        Whether to include existing greeks.

    Returns
    -------
    dict
        ``{delta: float, gamma: float, vega: float}`` for this underlying.
    """
    total = {"delta": 0.0, "gamma": 0.0, "vega": 0.0}

    for alloc in normalized_allocs:
        if getattr(alloc, "underlying", "") != underlying:
            continue
        total["delta"] += getattr(alloc, "delta_shares_equiv", 0.0)
        total["gamma"] += getattr(alloc, "gamma", 0.0)
        total["vega"] += getattr(alloc, "vega", 0.0)

    if enforce_existing and existing_greeks_for_underlying:
        total["delta"] += existing_greeks_for_underlying.get("delta", 0.0)
        total["gamma"] += existing_greeks_for_underlying.get("gamma", 0.0)
        total["vega"] += existing_greeks_for_underlying.get("vega", 0.0)

    return total


def compute_scale_factor_for_limit(
    current_value: float,
    limit: float,
) -> float:
    """Compute a scale factor to bring current_value within limit.

    Both current_value and limit are treated as absolute values.
    Returns 1.0 if within limit, otherwise limit / |current_value|.

    Never returns > 1.0 (monotone: never increases exposure).
    """
    abs_current = abs(current_value)
    if abs_current <= 0 or abs_current <= limit:
        return 1.0
    return min(1.0, limit / abs_current)
