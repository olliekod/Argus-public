# Created by Oliver Meihls

# Tail-Risk / Heston Scenario Layer
#
# THIN wrapper around ``gpu_engine`` functions used **only for risk capping**.
# This module does NOT rank spreads or decide attractiveness — it only answers:
# "should we size down / reject because tail risk is too high?"
#
# The primary metric is **probability of touch**: the probability that the
# underlying price touches the short strike before expiry, estimated via
# Heston stochastic volatility Monte Carlo (first passage).
#
# If full Heston calibration is not available, conservative default parameters
# are used from config and the result is treated as a stress scenario.
#
# Strict Separation
# - This module does NOT use Heston to rank spreads or decide attractiveness.
# - The Heston "alpha product" (trade_calculator, etf_options_detector) remains separate.
# - Phase 4C MC/bootstrap is trade/PnL resampling for robustness — unrelated.
#
# References
# - MASTER_PLAN.md §9 — Phase 5: Portfolio Risk Engine (tail-risk layer).

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger("argus.tail_risk_scenario")


@dataclass
class TailRiskConfig:
    # Configuration for tail-risk scenario evaluation.
    #
    # Attributes
    # enabled_for_options : bool
    # Whether to run tail-risk checks on options allocations.
    # max_prob_touch : float
    # Maximum acceptable probability of touching the short strike
    # before expiry (as a fraction, e.g. 0.35 = 35%).
    # stress_iv_bump : float
    # Additional IV bump for stress scenario (e.g. 0.20 = +20 vol pts).
    # max_stress_loss_pct : float
    # Maximum acceptable stress-scenario loss as fraction of equity.
    # mc_simulations : int
    # Number of Monte Carlo paths for touch probability estimation.
    # mc_steps_per_year : int
    # Time discretization for Heston MC (steps per year).
    # seed : int
    # Random seed for deterministic MC (0 = no seeding).
    # default_heston_kappa : float
    # Default Heston mean-reversion speed (used when calibration unavailable).
    # default_heston_sigma_v : float
    # Default Heston vol-of-vol.
    # default_heston_rho : float
    # Default Heston price-vol correlation.
    enabled_for_options: bool = True
    max_prob_touch: float = 0.35
    stress_iv_bump: float = 0.20
    max_stress_loss_pct: float = 0.05
    mc_simulations: int = 100_000
    mc_steps_per_year: int = 252
    seed: int = 42
    default_heston_kappa: float = 2.0
    default_heston_sigma_v: float = 0.5
    default_heston_rho: float = -0.7


@dataclass
class TailRiskResult:
    # Result of tail-risk evaluation for a single allocation.
    #
    # Attributes
    # prob_touch : float
    # Estimated probability of touching the short strike (0-1).
    # prob_touch_stressed : float
    # Same under stressed IV (IV + stress_iv_bump).
    # max_loss_usd : float
    # Maximum loss in USD for the proposed contracts.
    # stress_loss_usd : float
    # Loss under stress scenario.
    # allowed_contracts : int
    # Maximum contracts allowed by tail-risk constraint.
    # capped : bool
    # Whether the proposed contracts were reduced.
    # reason : str
    # Human-readable reason string.
    prob_touch: float = 0.0
    prob_touch_stressed: float = 0.0
    max_loss_usd: float = 0.0
    stress_loss_usd: float = 0.0
    allowed_contracts: int = 0
    capped: bool = False
    reason: str = ""


def evaluate_tail_risk(
    *,
    S: float,
    short_strike: float,
    long_strike: float,
    T: float,
    credit: float,
    v0: float,
    proposed_contracts: int,
    equity_usd: float,
    config: TailRiskConfig,
    heston_kappa: Optional[float] = None,
    heston_sigma_v: Optional[float] = None,
    heston_rho: Optional[float] = None,
) -> TailRiskResult:
    # Evaluate tail-risk for a single option spread allocation.
    #
    # This is the ONLY entry point for tail-risk evaluation.  It computes
    # probability of touch via Heston MC and determines the maximum
    # allowed contract count.
    #
    # Parameters
    # S : float
    # Current underlying price.
    # short_strike : float
    # Short strike price.
    # long_strike : float
    # Long strike price.
    # T : float
    # Time to expiry in years.
    # credit : float
    # Net credit received per share.
    # v0 : float
    # Initial variance (IV^2).  If 0, uses a conservative default.
    # proposed_contracts : int
    # Number of contracts proposed by allocator.
    # equity_usd : float
    # Current portfolio equity.
    # config : TailRiskConfig
    # Tail-risk configuration.
    # heston_kappa : float, optional
    # Heston kappa override.
    # heston_sigma_v : float, optional
    # Heston sigma_v override.
    # heston_rho : float, optional
    # Heston rho override.
    #
    # Returns
    # TailRiskResult
    if not config.enabled_for_options:
        return TailRiskResult(
            allowed_contracts=proposed_contracts,
            reason="tail_risk_disabled",
        )

    if proposed_contracts <= 0:
        return TailRiskResult(
            allowed_contracts=0,
            reason="no_contracts_proposed",
        )

    if T <= 0 or S <= 0:
        return TailRiskResult(
            allowed_contracts=0,
            capped=True,
            reason="invalid_inputs_T_or_S",
        )

    # Use defaults if calibration not available
    kappa = heston_kappa if heston_kappa is not None else config.default_heston_kappa
    sigma_v = heston_sigma_v if heston_sigma_v is not None else config.default_heston_sigma_v
    rho = heston_rho if heston_rho is not None else config.default_heston_rho

    if v0 <= 0:
        # Conservative: assume 60% IV
        v0 = 0.60 ** 2

    # Compute prob_touch using GPU engine (Heston MC)
    prob_touch = _compute_prob_touch(
        S=S,
        short_strike=short_strike,
        long_strike=long_strike,
        credit=credit,
        T=T,
        v0=v0,
        config=config,
        kappa=kappa,
        sigma_v=sigma_v,
        rho=rho,
    )

    # Stressed prob_touch (bump IV)
    stressed_v0 = (math.sqrt(v0) + config.stress_iv_bump) ** 2
    prob_touch_stressed = _compute_prob_touch(
        S=S,
        short_strike=short_strike,
        long_strike=long_strike,
        credit=credit,
        T=T,
        v0=stressed_v0,
        config=config,
        kappa=kappa,
        sigma_v=sigma_v,
        rho=rho,
    )

    # Max loss per contract
    spread_width = abs(short_strike - long_strike)
    if spread_width < 1e-9:
        # Degenerate spread (same strike) — reject as untestable
        return TailRiskResult(
            prob_touch=prob_touch,
            prob_touch_stressed=prob_touch_stressed,
            allowed_contracts=0,
            capped=True,
            reason="degenerate_spread_zero_width",
        )
    max_loss_per_contract = (spread_width - credit) * _OPTION_MULTIPLIER
    if max_loss_per_contract <= 0:
        max_loss_per_contract = spread_width * _OPTION_MULTIPLIER

    # Total max loss
    max_loss_usd = max_loss_per_contract * proposed_contracts

    # Stress loss (prob-weighted expected shortfall approximation)
    stress_loss_usd = max_loss_per_contract * proposed_contracts * prob_touch_stressed

    # Determine allowed contracts
    allowed = proposed_contracts

    # Check 1: prob_touch exceeds threshold
    if prob_touch > config.max_prob_touch:
        # Scale down proportionally
        if prob_touch > 0:
            scale = config.max_prob_touch / prob_touch
            allowed = max(0, int(proposed_contracts * scale))

    # Check 2: stress loss exceeds equity fraction
    max_stress_loss_usd = equity_usd * config.max_stress_loss_pct
    if stress_loss_usd > max_stress_loss_usd and max_loss_per_contract > 0:
        max_contracts_by_stress = int(max_stress_loss_usd / (max_loss_per_contract * max(prob_touch_stressed, 0.01)))
        allowed = min(allowed, max(0, max_contracts_by_stress))

    capped = allowed < proposed_contracts
    reason_parts = []
    if prob_touch > config.max_prob_touch:
        reason_parts.append(
            f"prob_touch={prob_touch:.4f}>{config.max_prob_touch:.4f}"
        )
    if stress_loss_usd > max_stress_loss_usd:
        reason_parts.append(
            f"stress_loss=${stress_loss_usd:.0f}>max${max_stress_loss_usd:.0f}"
        )
    reason = "; ".join(reason_parts) if reason_parts else "within_limits"

    return TailRiskResult(
        prob_touch=prob_touch,
        prob_touch_stressed=prob_touch_stressed,
        max_loss_usd=max_loss_usd,
        stress_loss_usd=stress_loss_usd,
        allowed_contracts=allowed,
        capped=capped,
        reason=reason,
    )


# Standard equity option multiplier
_OPTION_MULTIPLIER = 100


def _compute_prob_touch(
    *,
    S: float,
    short_strike: float,
    long_strike: float,
    credit: float,
    T: float,
    v0: float,
    config: TailRiskConfig,
    kappa: float,
    sigma_v: float,
    rho: float,
) -> float:
    # Compute probability of touching the short strike via Heston MC.
    #
    # Falls back to analytical approximation if GPU engine is unavailable.
    try:
        from .gpu_engine import get_gpu_engine
        engine = get_gpu_engine(force_cpu=True)

        steps = max(10, int(T * config.mc_steps_per_year))

        # Seed for determinism
        if config.seed > 0:
            try:
                import torch
                torch.manual_seed(config.seed)
            except ImportError:
                pass

        result = engine.monte_carlo_pop_heston(
            S=S,
            short_strike=short_strike,
            long_strike=long_strike,
            credit=credit,
            T=T,
            v0=v0,
            simulations=config.mc_simulations,
            steps=steps,
        )

        # prob_of_touch_stop is the touch probability (0-100 scale)
        prob_touch = result.get("prob_of_touch_stop", 0.0) / 100.0
        return max(0.0, min(1.0, prob_touch))

    except Exception as exc:
        logger.warning("GPU engine unavailable for tail-risk, using analytical: %s", exc)
        return _analytical_prob_touch(S, short_strike, T, math.sqrt(v0))


def _analytical_prob_touch(
    S: float,
    barrier: float,
    T: float,
    sigma: float,
) -> float:
    # Analytical first-passage approximation for GBM (conservative fallback).
    #
    # Handles both downside barriers (put spreads, barrier < S) and upside
    # barriers (call spreads, barrier > S) using the reflection principle.
    #
    # For downside: P(min S_t <= B | S_0 = S)
    # For upside:   P(max S_t >= B | S_0 = S)
    if S <= 0 or sigma <= 0 or T <= 0:
        return 0.0

    if barrier > S:
        # Upside barrier (call spread): P(max S_t >= B)
        # Use symmetry: compute P(max S_t >= B) using reflection for upper barrier
        try:
            from scipy.stats import norm as scipy_norm
            r = 0.045
            log_ratio = math.log(barrier / S)
            d1 = (-log_ratio + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
            d2 = (-log_ratio + (r - 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
            exponent = 2.0 * r / (sigma ** 2)
            p = scipy_norm.cdf(d1) + ((barrier / S) ** exponent) * scipy_norm.cdf(d2)
            return max(0.0, min(1.0, p))
        except ImportError:
            return 0.5
    elif S <= barrier:
        return 1.0  # Already touching

    # Downside barrier (put spread): P(min S_t <= B)
    try:
        from scipy.stats import norm as scipy_norm
        r = 0.045

        log_ratio = math.log(barrier / S)
        d1 = (log_ratio + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = (log_ratio + (r - 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))

        exponent = 2.0 * r / (sigma ** 2)
        p = scipy_norm.cdf(d1) + ((barrier / S) ** exponent) * scipy_norm.cdf(d2)
        return max(0.0, min(1.0, p))
    except ImportError:
        # Last resort: conservative fixed estimate
        return 0.5
