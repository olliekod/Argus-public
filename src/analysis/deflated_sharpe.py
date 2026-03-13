"""
Deflated Sharpe Ratio (DSR)
============================

Implements the Deflated Sharpe Ratio from Bailey & López de Prado (2014),
"The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest
Overfitting and Non-Normality."

DSR corrects for:
1. Selection bias — best of N trials inflates observed Sharpe.
2. Non-normality — skewness and kurtosis of returns.

The DSR is a probability (0–1).  A strategy should only be deployed if
DSR >= threshold (default 0.95).

References
----------
- Bailey, D.H. & López de Prado, M. (2014). The Deflated Sharpe Ratio.
  *Journal of Portfolio Management*, 40(5), 94–107.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Sequence

logger = logging.getLogger("argus.deflated_sharpe")

# Euler–Mascheroni constant
_EULER_MASCHERONI = 0.5772156649015329


def _normal_cdf(x: float) -> float:
    """Standard normal CDF using the complementary error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _normal_ppf(p: float) -> float:
    """Inverse standard normal CDF (percent-point function).

    Uses a rational approximation (Abramowitz & Stegun 26.2.23)
    for p in (0, 1).
    """
    if p <= 0.0:
        return -10.0
    if p >= 1.0:
        return 10.0
    if p == 0.5:
        return 0.0

    if p < 0.5:
        sign = -1.0
        q = p
    else:
        sign = 1.0
        q = 1.0 - p

    t = math.sqrt(-2.0 * math.log(q))
    # Rational approximation coefficients
    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308
    numerator = c0 + c1 * t + c2 * t * t
    denominator = 1.0 + d1 * t + d2 * t * t + d3 * t * t * t
    return sign * (t - numerator / denominator)


def compute_sharpe_stats(returns: Sequence[float]) -> dict:
    """Compute Sharpe ratio and higher moments from a return series.

    Parameters
    ----------
    returns : sequence of float
        Per-period returns (not annualized).

    Returns
    -------
    dict
        Keys: ``sharpe``, ``mean``, ``std``, ``skew``, ``kurtosis``,
        ``n_obs``.  Kurtosis is *excess* kurtosis (normal = 0).
    """
    n = len(returns)
    if n < 2:
        return {
            "sharpe": 0.0, "mean": 0.0, "std": 0.0,
            "skew": 0.0, "kurtosis": 0.0, "n_obs": n,
        }

    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / (n - 1)
    std = math.sqrt(var) if var > 0 else 0.0

    if std == 0:
        return {
            "sharpe": 0.0, "mean": mean, "std": 0.0,
            "skew": 0.0, "kurtosis": 0.0, "n_obs": n,
        }

    sharpe = mean / std

    # Skewness (sample, using n-1 consistent with variance)
    m3 = sum((r - mean) ** 3 for r in returns) / (n - 1)
    skew = m3 / (std ** 3)

    # Excess kurtosis (sample, using n-1 consistent with variance)
    m4 = sum((r - mean) ** 4 for r in returns) / (n - 1)
    kurtosis = (m4 / (std ** 4)) - 3.0

    return {
        "sharpe": sharpe,
        "mean": mean,
        "std": std,
        "skew": skew,
        "kurtosis": kurtosis,
        "n_obs": n,
    }


def threshold_sharpe_ratio(
    sharpe_variance: float,
    n_trials: int,
) -> float:
    """Compute the threshold Sharpe ratio SR_0 (False Strategy Theorem).

    This is the expected maximum Sharpe among ``n_trials`` unskilled
    strategies.

    Parameters
    ----------
    sharpe_variance : float
        Cross-sectional variance of Sharpe ratios across trials, i.e.
        Var[SR_n].
    n_trials : int
        Number of independent trials (or effective number of trials
        after clustering).

    Returns
    -------
    float
        SR_0, the threshold Sharpe ratio.
    """
    if n_trials <= 0:
        return 0.0
    if sharpe_variance <= 0:
        return 0.0

    std_sr = math.sqrt(sharpe_variance)
    gamma = _EULER_MASCHERONI
    e = math.e
    n = max(n_trials, 1)

    # Inverse normal CDF arguments — clamp to avoid domain issues
    p1 = max(1e-15, min(1.0 - 1e-15, 1.0 - 1.0 / n))
    p2 = max(1e-15, min(1.0 - 1e-15, 1.0 - 1.0 / (n * e)))

    sr_0 = std_sr * (
        (1.0 - gamma) * _normal_ppf(p1) + gamma * _normal_ppf(p2)
    )
    return sr_0


def compute_deflated_sharpe_ratio(
    observed_sharpe: float,
    threshold_sr: float,
    n_obs: int,
    skewness: float = 0.0,
    kurtosis: float = 0.0,
) -> float:
    """Compute the Deflated Sharpe Ratio.

    Parameters
    ----------
    observed_sharpe : float
        Non-annualized Sharpe ratio of the best strategy.
    threshold_sr : float
        Threshold Sharpe from :func:`threshold_sharpe_ratio`.
    n_obs : int
        Number of return observations (sample length T).
    skewness : float
        Skewness of returns (default 0 = normal).
    kurtosis : float
        Excess kurtosis of returns (default 0 = normal).

    Returns
    -------
    float
        DSR value in [0, 1].  Deploy only if DSR >= threshold (e.g. 0.95).
    """
    if n_obs < 2:
        return 0.0

    t_minus_1 = n_obs - 1

    # Standard error of the Sharpe ratio per Bailey & López de Prado (2014):
    # SE(SR) = sqrt((1 - skew*SR + ((raw_kurt-1)/4)*SR^2) / (T-1))
    # where SR is the *observed* Sharpe ratio and raw_kurt is the raw
    # kurtosis (not excess).  Since our `kurtosis` parameter is excess
    # kurtosis (= raw - 3), we convert: (raw_kurt - 1) = (kurtosis + 2).
    denom_sq = (
        1.0
        - skewness * observed_sharpe
        + ((kurtosis + 2.0) / 4.0) * (observed_sharpe ** 2)
    )

    # Guard against negative or zero denominator
    if denom_sq <= 0:
        logger.warning(
            "DSR denominator non-positive (%.6f); skew=%.4f, kurt=%.4f, sr=%.4f",
            denom_sq, skewness, kurtosis, observed_sharpe,
        )
        return 0.0

    denom = math.sqrt(denom_sq)

    # Test statistic
    z = ((observed_sharpe - threshold_sr) * math.sqrt(t_minus_1)) / denom

    dsr = _normal_cdf(z)
    return dsr


def deflated_sharpe_ratio(
    returns: Sequence[float],
    n_trials: int,
    all_sharpes: Optional[Sequence[float]] = None,
) -> dict:
    """End-to-end DSR computation from raw returns.

    Parameters
    ----------
    returns : sequence of float
        Per-period return series of the *best* strategy.
    n_trials : int
        Total number of independent strategy trials (or effective N).
    all_sharpes : sequence of float, optional
        Sharpe ratios of all trials.  If provided, cross-sectional
        variance is computed from these.  Otherwise, assumes unit
        variance as a conservative default.

    Returns
    -------
    dict
        ``dsr``, ``observed_sharpe``, ``threshold_sr``, ``n_obs``,
        ``n_trials``, ``skew``, ``kurtosis``, ``sharpe_variance``.
    """
    stats = compute_sharpe_stats(returns)

    # Cross-sectional variance of Sharpe ratios
    if all_sharpes and len(all_sharpes) >= 2:
        mean_sr = sum(all_sharpes) / len(all_sharpes)
        sr_var = sum((s - mean_sr) ** 2 for s in all_sharpes) / (len(all_sharpes) - 1)
    else:
        # Conservative default: assume unit variance
        sr_var = 1.0

    sr_0 = threshold_sharpe_ratio(sr_var, n_trials)

    dsr = compute_deflated_sharpe_ratio(
        observed_sharpe=stats["sharpe"],
        threshold_sr=sr_0,
        n_obs=stats["n_obs"],
        skewness=stats["skew"],
        kurtosis=stats["kurtosis"],
    )

    return {
        "dsr": round(dsr, 6),
        "observed_sharpe": round(stats["sharpe"], 6),
        "threshold_sr": round(sr_0, 6),
        "n_obs": stats["n_obs"],
        "n_trials": n_trials,
        "skew": round(stats["skew"], 6),
        "kurtosis": round(stats["kurtosis"], 6),
        "sharpe_variance": round(sr_var, 6),
    }
