"""
Reality Check / SPA Test
=========================

Implements White's (2000) Reality Check for data snooping, with the
Stationary Bootstrap of Politis & Romano (1994).

Tests whether the best strategy outperforms a benchmark after correcting
for multiple testing (data snooping).

References
----------
- White, H. (2000). A Reality Check for Data Snooping. *Econometrica*.
- Hansen, P.R. (2005). A Test for Superior Predictive Ability.
  *Journal of Business & Economic Statistics*.
- Politis, D.N. & Romano, J.P. (1994). The Stationary Bootstrap.
  *JASA*, 89(428), 1303–1313.
"""

from __future__ import annotations

import logging
import math
import random
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger("argus.reality_check")


def _stationary_bootstrap_indices(
    n: int,
    block_size: float,
    rng: random.Random,
) -> List[int]:
    """Generate bootstrap indices using the stationary bootstrap.

    The stationary bootstrap draws block starts randomly and extends
    each block with probability ``1 - 1/block_size`` at each step,
    producing random-length blocks with expected length ``block_size``.

    Parameters
    ----------
    n : int
        Length of the original series.
    block_size : float
        Expected block length (geometric distribution parameter).
    rng : random.Random
        Random number generator.

    Returns
    -------
    list of int
        Bootstrap index sequence of length ``n``.
    """
    if n == 0:
        return []

    p = 1.0 / max(block_size, 1.0)  # probability of starting new block
    indices = []
    pos = rng.randint(0, n - 1)

    for _ in range(n):
        indices.append(pos)
        if rng.random() < p:
            # Start a new block
            pos = rng.randint(0, n - 1)
        else:
            # Continue current block (wrap around)
            pos = (pos + 1) % n

    return indices


def _compute_hac_variance(series: Sequence[float], bandwidth: int = 0) -> float:
    """Compute HAC (Newey-West) variance of the mean of a series.

    Parameters
    ----------
    series : sequence of float
        Time series.
    bandwidth : int
        Number of lags for Newey-West correction.  If 0, uses no
        correction (simple variance of the mean).

    Returns
    -------
    float
        Estimated variance of the sample mean.
    """
    n = len(series)
    if n < 2:
        return 1.0

    mean = sum(series) / n
    demeaned = [x - mean for x in series]

    # Autocovariance at lag 0
    gamma_0 = sum(d * d for d in demeaned) / n

    if bandwidth == 0:
        return max(gamma_0 / n, 1e-15)

    # Newey-West HAC
    total = gamma_0
    for lag in range(1, min(bandwidth + 1, n)):
        weight = 1.0 - lag / (bandwidth + 1.0)
        gamma_lag = sum(demeaned[i] * demeaned[i - lag] for i in range(lag, n)) / n
        total += 2.0 * weight * gamma_lag

    return max(total / n, 1e-15)


def reality_check(
    strategy_returns: Dict[str, Sequence[float]],
    benchmark_returns: Optional[Sequence[float]] = None,
    n_bootstrap: int = 1000,
    block_size: float = 10.0,
    seed: Optional[int] = 42,
    hac_bandwidth: int = 0,
) -> dict:
    """Run White's Reality Check for data snooping.

    Tests H0: no strategy beats the benchmark, after correcting for
    having tested multiple strategies.

    Parameters
    ----------
    strategy_returns : dict[str, sequence of float]
        Mapping from strategy name to per-period return series.
        All series must have the same length.
    benchmark_returns : sequence of float, optional
        Per-period benchmark returns.  If None, assumes zero
        (i.e., tests whether any strategy has positive returns).
    n_bootstrap : int
        Number of bootstrap replications (default 1000).
    block_size : float
        Expected block length for stationary bootstrap (default 10).
    seed : int, optional
        Random seed for reproducibility.
    hac_bandwidth : int
        Bandwidth for HAC variance estimation (0 = no correction).

    Returns
    -------
    dict
        ``p_value``, ``test_statistic``, ``best_strategy``,
        ``best_mean_excess``, ``n_strategies``, ``n_obs``.
    """
    if not strategy_returns:
        return {
            "p_value": 1.0,
            "test_statistic": 0.0,
            "best_strategy": None,
            "best_mean_excess": 0.0,
            "n_strategies": 0,
            "n_obs": 0,
        }

    # Validate all series have same length
    names = list(strategy_returns.keys())
    T = len(strategy_returns[names[0]])
    for name in names:
        if len(strategy_returns[name]) != T:
            raise ValueError(
                f"All return series must have same length; "
                f"'{names[0]}' has {T}, '{name}' has {len(strategy_returns[name])}"
            )

    if T < 2:
        return {
            "p_value": 1.0,
            "test_statistic": 0.0,
            "best_strategy": names[0] if names else None,
            "best_mean_excess": 0.0,
            "n_strategies": len(names),
            "n_obs": T,
        }

    # Compute excess returns over benchmark
    bench = list(benchmark_returns) if benchmark_returns is not None else [0.0] * T
    if len(bench) != T:
        raise ValueError(
            f"Benchmark length ({len(bench)}) must match strategy length ({T})"
        )

    # d_k,t = strategy return - benchmark return
    excess: Dict[str, List[float]] = {}
    for name in names:
        rets = strategy_returns[name]
        excess[name] = [float(rets[t]) - float(bench[t]) for t in range(T)]

    # Mean excess returns and variances
    mean_excess: Dict[str, float] = {}
    var_excess: Dict[str, float] = {}
    for name in names:
        mean_excess[name] = sum(excess[name]) / T
        var_excess[name] = _compute_hac_variance(excess[name], hac_bandwidth)

    # Test statistic: V_n = max_k (sqrt(T) * d_bar_k / sqrt(V_k))
    best_name = None
    best_stat = -math.inf
    for name in names:
        v_k = var_excess[name]
        if v_k <= 0:
            continue
        stat_k = math.sqrt(T) * mean_excess[name] / math.sqrt(v_k)
        if stat_k > best_stat:
            best_stat = stat_k
            best_name = name

    if best_name is None:
        return {
            "p_value": 1.0,
            "test_statistic": 0.0,
            "best_strategy": None,
            "best_mean_excess": 0.0,
            "n_strategies": len(names),
            "n_obs": T,
        }

    # Bootstrap
    rng = random.Random(seed)
    count_exceed = 0

    for _ in range(n_bootstrap):
        indices = _stationary_bootstrap_indices(T, block_size, rng)

        # Compute bootstrap test statistic under the null
        # Under H0, we center the excess returns (subtract mean)
        boot_max = -math.inf
        for name in names:
            exc = excess[name]
            mu = mean_excess[name]
            v_k = var_excess[name]
            if v_k <= 0:
                continue

            # Bootstrap mean of centered excess returns
            boot_mean = sum(exc[idx] - mu for idx in indices) / T
            boot_stat = math.sqrt(T) * boot_mean / math.sqrt(v_k)
            if boot_stat > boot_max:
                boot_max = boot_stat

        if boot_max >= best_stat:
            count_exceed += 1

    p_value = count_exceed / n_bootstrap

    return {
        "p_value": round(p_value, 6),
        "test_statistic": round(best_stat, 6),
        "best_strategy": best_name,
        "best_mean_excess": round(mean_excess.get(best_name, 0.0), 8),
        "n_strategies": len(names),
        "n_obs": T,
    }
