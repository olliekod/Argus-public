"""
Drawdown Containment
=====================

Computes a throttle factor [min_throttle, 1.0] based on current portfolio
drawdown.  The throttle is applied as a scalar multiplier on all normalized
allocation exposures.

Two modes are supported:

- **linear**: throttle = max(min_throttle, 1 - k * (dd - threshold))
- **step**: throttle = throttle_scale once drawdown exceeds threshold

Hysteresis is provided via ``recovery_threshold_pct``: once throttled,
the throttle is only fully released when drawdown recovers to
``recovery_threshold_pct`` (which should be < ``threshold_pct``).

References
----------
- MASTER_PLAN.md §9 — Phase 5: Portfolio Risk Engine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("argus.drawdown_containment")


@dataclass
class DrawdownConfig:
    """Configuration for drawdown-based throttling.

    Attributes
    ----------
    threshold_pct : float
        Drawdown fraction at which throttling begins (e.g. 0.10 = 10%).
    throttle_mode : str
        ``"linear"`` or ``"step"``.
    throttle_scale : float
        Used in step mode: throttle drops to this value when dd > threshold.
    k : float
        Used in linear mode: slope of throttle decay.
        throttle = max(min_throttle, 1 - k * (dd - threshold)).
    recovery_threshold_pct : float
        Drawdown level at which full exposure is restored (hysteresis).
        Must be < threshold_pct.
    min_throttle : float
        Floor for the throttle factor (e.g. 0.1 = never below 10%).
    """
    threshold_pct: float = 0.10
    throttle_mode: str = "linear"
    throttle_scale: float = 0.5
    k: float = 5.0
    recovery_threshold_pct: float = 0.05
    min_throttle: float = 0.1


def compute_drawdown_throttle(
    current_drawdown_pct: float,
    config: DrawdownConfig,
    *,
    was_throttled: bool = False,
) -> float:
    """Compute the drawdown throttle factor.

    Parameters
    ----------
    current_drawdown_pct : float
        Current drawdown as a fraction (0.0 = no drawdown, 0.15 = 15%).
    config : DrawdownConfig
        Throttle configuration.
    was_throttled : bool
        Whether the portfolio was previously in a throttled state.
        Used for hysteresis: if True, throttling continues until
        drawdown recovers below ``recovery_threshold_pct``.

    Returns
    -------
    float
        Throttle factor in [config.min_throttle, 1.0].
        1.0 means no throttling.
    """
    dd = max(0.0, current_drawdown_pct)
    threshold = config.threshold_pct
    recovery = config.recovery_threshold_pct
    min_t = max(0.0, min(1.0, config.min_throttle))

    # Hysteresis: if we were throttled and haven't recovered, stay throttled
    if was_throttled and dd > recovery:
        # Still in throttled zone — compute the actual throttle value
        pass  # fall through to mode computation
    elif dd <= threshold:
        # Not in drawdown zone (or recovered past hysteresis)
        return 1.0

    # Compute throttle based on mode
    if config.throttle_mode == "step":
        throttle = config.throttle_scale
    elif config.throttle_mode == "linear":
        excess = dd - threshold
        throttle = 1.0 - config.k * excess
    else:
        logger.warning(
            "Unknown throttle_mode '%s', defaulting to linear",
            config.throttle_mode,
        )
        excess = dd - threshold
        throttle = 1.0 - config.k * excess

    # Clamp to [min_throttle, 1.0]
    throttle = max(min_t, min(1.0, throttle))

    logger.info(
        "Drawdown throttle: dd=%.4f threshold=%.4f mode=%s -> throttle=%.4f",
        dd, threshold, config.throttle_mode, throttle,
    )
    return throttle
