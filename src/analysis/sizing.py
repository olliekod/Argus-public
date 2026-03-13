"""
Position Sizing Module
=======================

Implements the sizing stack for Phase 5 (strategy allocation):

- **Forecast** dataclass — standardized signal object per strategy/instrument.
- **Fractional Kelly** sizing — conservative Kelly criterion with caps.
- **Vol-target overlay** — scales positions to target portfolio volatility.
- **Options spread sizing** — converts risk budget to contract count.
- **Estimation error shrinkage** — shrinks expected return toward zero by
  confidence.

References
----------
- MASTER_PLAN.md §9.3 — Sizing stack.
- Kelly, J.L. (1956). A New Interpretation of Information Rate.
  *Bell System Technical Journal*.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger("argus.sizing")

# Hard per-play cap from MASTER_PLAN §8.5
_DEFAULT_PER_PLAY_CAP = 0.07  # 7% of equity


@dataclass
class Forecast:
    """Standardized forecast object per strategy/instrument.

    Attributes
    ----------
    strategy_id : str
        Unique identifier for the strategy.
    instrument : str
        Instrument/symbol being traded.
    mu : float
        Expected return (from walk-forward or replay).
    sigma : float
        Volatility (rolling realized or from backtest).
    edge_score : float
        Composite score from StrategyEvaluator.
    cost : float
        Estimated cost per trade (round trip).
    confidence : float
        Confidence in [0, 1] (from regime coverage, sample size).
    meta : dict
        Additional metadata (e.g., regime, DSR, etc.).
    """
    strategy_id: str
    instrument: str
    mu: float
    sigma: float
    edge_score: float = 0.0
    cost: float = 0.0
    confidence: float = 1.0
    meta: Dict[str, Any] = field(default_factory=dict)


def shrink_mu(mu: float, confidence: float) -> float:
    """Shrink expected return toward zero by confidence.

    Parameters
    ----------
    mu : float
        Raw expected return.
    confidence : float
        Confidence in [0, 1].

    Returns
    -------
    float
        Shrunk expected return: ``confidence * mu``.
    """
    c = max(0.0, min(1.0, confidence))
    return c * mu


def fractional_kelly_size(
    forecast: Forecast,
    kelly_fraction: float = 0.25,
    per_play_cap: float = _DEFAULT_PER_PLAY_CAP,
    min_edge_over_cost: float = 0.0,
    apply_shrinkage: bool = True,
) -> float:
    """Compute fractional Kelly position size.

    Full Kelly:   f* = mu / sigma^2
    Fractional:   f  = c * f* = c * mu / sigma^2
    Capped at per_play_cap.

    Parameters
    ----------
    forecast : Forecast
        Forecast with mu, sigma, cost, confidence.
    kelly_fraction : float
        Kelly fraction c in [0, 1] (default 0.25 = quarter-Kelly).
    per_play_cap : float
        Maximum position as fraction of equity (default 0.07 = 7%).
    min_edge_over_cost : float
        Minimum edge over cost to size (default 0.0).
    apply_shrinkage : bool
        Whether to apply confidence shrinkage to mu (default True).

    Returns
    -------
    float
        Position size as fraction of equity, in [-per_play_cap, per_play_cap].
        Returns 0 if edge <= cost.
    """
    mu = forecast.mu
    if apply_shrinkage:
        mu = shrink_mu(mu, forecast.confidence)

    # Skip if edge doesn't cover costs
    if mu <= forecast.cost + min_edge_over_cost:
        return 0.0

    sigma = forecast.sigma
    if sigma <= 0:
        logger.warning(
            "Zero or negative sigma for %s/%s; returning 0 size",
            forecast.strategy_id, forecast.instrument,
        )
        return 0.0

    # Full Kelly: f* = (mu - cost) / sigma^2  (net edge)
    f_star = (mu - forecast.cost) / (sigma ** 2)

    # Fractional Kelly
    f = kelly_fraction * f_star

    # Cap
    cap = abs(per_play_cap)
    f = max(-cap, min(cap, f))

    return round(f, 8)


def vol_target_overlay(
    weight: float,
    forecast_sigma: float,
    target_vol_annual: float = 0.10,
) -> float:
    """Apply vol-target overlay to a position weight.

    Scales the weight so that the position contributes approximately
    ``target_vol_annual`` to portfolio volatility.

    Parameters
    ----------
    weight : float
        Raw position weight (from Kelly or other sizing).
    forecast_sigma : float
        Annualized volatility of the strategy/instrument.
    target_vol_annual : float
        Target annualized volatility (default 0.10 = 10%).

    Returns
    -------
    float
        Adjusted weight.
    """
    if forecast_sigma <= 0:
        return 0.0

    scale = target_vol_annual / forecast_sigma
    return round(weight * scale, 8)


def contracts_from_risk_budget(
    risk_budget_usd: float,
    max_loss_per_contract: float,
) -> int:
    """Compute number of option spread contracts from a risk budget.

    Parameters
    ----------
    risk_budget_usd : float
        Dollar risk budget for this position (from portfolio-level
        Kelly share * equity * cap).
    max_loss_per_contract : float
        Maximum loss per contract (spread width * multiplier - credit).

    Returns
    -------
    int
        Number of contracts (floored to integer). Returns 0 if
        max_loss_per_contract <= 0.
    """
    if max_loss_per_contract <= 0:
        return 0
    if risk_budget_usd <= 0:
        return 0

    contracts = int(math.floor(risk_budget_usd / max_loss_per_contract))
    return max(contracts, 0)


def size_position(
    forecast: Forecast,
    equity: float,
    kelly_fraction: float = 0.25,
    per_play_cap: float = _DEFAULT_PER_PLAY_CAP,
    vol_target_annual: Optional[float] = None,
    min_edge_over_cost: float = 0.0,
    apply_shrinkage: bool = True,
) -> Dict[str, Any]:
    """Full sizing pipeline: Kelly -> vol-target -> cap.

    Parameters
    ----------
    forecast : Forecast
        Strategy forecast.
    equity : float
        Current portfolio equity in dollars.
    kelly_fraction : float
        Kelly fraction (default 0.25).
    per_play_cap : float
        Maximum position as fraction of equity (default 0.07).
    vol_target_annual : float, optional
        If set, applies vol-target overlay.
    min_edge_over_cost : float
        Minimum edge over cost.
    apply_shrinkage : bool
        Whether to apply confidence shrinkage.

    Returns
    -------
    dict
        ``weight``: final position weight (fraction of equity),
        ``dollar_risk``: weight * equity,
        ``kelly_raw``: pre-cap Kelly weight,
        ``vol_adjusted``: whether vol overlay was applied.
    """
    w = fractional_kelly_size(
        forecast,
        kelly_fraction=kelly_fraction,
        per_play_cap=per_play_cap,
        min_edge_over_cost=min_edge_over_cost,
        apply_shrinkage=apply_shrinkage,
    )

    kelly_raw = w
    vol_adjusted = False

    if vol_target_annual is not None and vol_target_annual > 0:
        w = vol_target_overlay(w, forecast.sigma, vol_target_annual)
        vol_adjusted = True

    # Re-apply cap after vol overlay
    cap = abs(per_play_cap)
    w = max(-cap, min(cap, w))

    return {
        "weight": round(w, 8),
        "dollar_risk": round(w * equity, 2),
        "kelly_raw": round(kelly_raw, 8),
        "vol_adjusted": vol_adjusted,
        "strategy_id": forecast.strategy_id,
        "instrument": forecast.instrument,
    }
