"""
Correlation / Cluster Exposure Control
========================================

Enforces per-underlying and per-cluster exposure caps based on:

1. **Underlying caps** — always enforced regardless of returns data.
2. **Strategy-level correlation** — computed from realized returns when
   available; strategies with corr > threshold are grouped into clusters.
3. **Cluster caps** — aggregate exposure within a cluster is capped.

The correlation matrix is computed from strategy-level realized returns
(post-cost if available).  If returns data is insufficient, underlying
caps are still enforced and a ClampReason is logged.

Determinism: clustering uses single-linkage with a fixed distance threshold
(1 - corr_threshold).  Ordering is stable via sorted strategy IDs.

References
----------
- MASTER_PLAN.md §9 — Phase 5: Portfolio Risk Engine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("argus.correlation_exposure")


@dataclass
class CorrelationConfig:
    """Configuration for correlation-based exposure control.

    Attributes
    ----------
    rolling_days : int
        Number of trading days for rolling correlation window.
    min_obs : int
        Minimum number of overlapping return observations required.
    estimator : str
        Correlation estimator.  Currently only ``"pearson"`` is supported.
    nan_policy : str
        How to handle NaN returns: ``"skip"`` (pairwise complete) or
        ``"conservative"`` (treat missing as correlated).
    max_exposure_per_underlying_usd : float
        Maximum notional exposure per underlying in USD.
        Set to 0 to disable underlying caps.
    max_exposure_per_cluster_usd : float
        Maximum aggregate notional exposure per correlation cluster in USD.
    max_correlated_pair_exposure_usd : float
        Maximum combined exposure for any pair with corr > threshold.
    cluster_method : str
        Clustering method.  ``"threshold"`` (simple) or ``"hierarchical"``.
    corr_threshold_for_cluster : float
        Correlation threshold above which strategies are grouped.
    """
    rolling_days: int = 60
    min_obs: int = 45
    estimator: str = "pearson"
    nan_policy: str = "skip"
    max_exposure_per_underlying_usd: float = 0.0
    max_exposure_per_cluster_usd: float = 0.0
    max_correlated_pair_exposure_usd: float = 0.0
    cluster_method: str = "threshold"
    corr_threshold_for_cluster: float = 0.8


def get_strategy_returns_for_correlation(
    strategy_return_series: Dict[str, Dict[str, float]],
    strategy_ids: List[str],
    as_of_ts_ms: int,
    rolling_days: int = 60,
) -> Dict[str, List[float]]:
    """Extract aligned return vectors for the given strategies.

    Only uses data points with date keys that parse to dates ≤ as_of_ts_ms.
    Returns a dict mapping strategy_id to a list of floats (returns aligned
    by date across all requested strategies).

    Parameters
    ----------
    strategy_return_series : dict
        ``{strategy_id: {date_str: return}}``.
    strategy_ids : list of str
        Which strategies to include.
    as_of_ts_ms : int
        Cutoff timestamp; only returns with date ≤ this are included.
    rolling_days : int
        Maximum number of recent days to include.

    Returns
    -------
    dict
        ``{strategy_id: [return_values]}`` aligned by common dates.
    """
    from datetime import datetime, timezone

    as_of_dt = datetime.fromtimestamp(as_of_ts_ms / 1000.0, tz=timezone.utc)

    # Collect all date keys per strategy (only those ≤ as_of)
    all_dates_per_strat: Dict[str, Set[str]] = {}
    for sid in sorted(strategy_ids):
        series = strategy_return_series.get(sid, {})
        valid_dates: Set[str] = set()
        for date_str in series:
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                )
                if dt <= as_of_dt:
                    valid_dates.add(date_str)
            except ValueError:
                continue
        all_dates_per_strat[sid] = valid_dates

    # Find common dates across all strategies
    if not all_dates_per_strat:
        return {}

    common_dates = set.intersection(*all_dates_per_strat.values()) if all_dates_per_strat else set()

    # Sort dates descending, take most recent rolling_days
    sorted_dates = sorted(common_dates, reverse=True)[:rolling_days]
    sorted_dates.sort()  # chronological order

    if not sorted_dates:
        return {}

    result: Dict[str, List[float]] = {}
    for sid in sorted(strategy_ids):
        series = strategy_return_series.get(sid, {})
        result[sid] = [series.get(d, 0.0) for d in sorted_dates]

    return result


def compute_correlation_matrix(
    aligned_returns: Dict[str, List[float]],
    min_obs: int = 45,
    estimator: str = "pearson",
) -> Dict[Tuple[str, str], float]:
    """Compute pairwise correlation from aligned return vectors.

    Parameters
    ----------
    aligned_returns : dict
        ``{strategy_id: [returns]}``, all lists same length.
    min_obs : int
        Minimum observations required; pairs below this get corr=NaN.
    estimator : str
        ``"pearson"`` only for now.

    Returns
    -------
    dict
        ``{(sid_a, sid_b): corr}`` for all ordered pairs where sid_a < sid_b.
    """
    sids = sorted(aligned_returns.keys())
    result: Dict[Tuple[str, str], float] = {}

    for i, sid_a in enumerate(sids):
        for j in range(i + 1, len(sids)):
            sid_b = sids[j]
            va = aligned_returns[sid_a]
            vb = aligned_returns[sid_b]

            n = min(len(va), len(vb))
            if n < min_obs:
                result[(sid_a, sid_b)] = float("nan")
                continue

            # Pearson correlation
            mean_a = sum(va[:n]) / n
            mean_b = sum(vb[:n]) / n
            cov = sum((va[k] - mean_a) * (vb[k] - mean_b) for k in range(n)) / n
            std_a = (sum((va[k] - mean_a) ** 2 for k in range(n)) / n) ** 0.5
            std_b = (sum((vb[k] - mean_b) ** 2 for k in range(n)) / n) ** 0.5

            if std_a < 1e-12 or std_b < 1e-12:
                result[(sid_a, sid_b)] = 0.0
            else:
                result[(sid_a, sid_b)] = cov / (std_a * std_b)

    return result


def build_clusters(
    strategy_ids: List[str],
    corr_matrix: Dict[Tuple[str, str], float],
    threshold: float = 0.8,
) -> List[List[str]]:
    """Build correlation clusters using simple threshold-based union-find.

    Strategies with pairwise correlation ≥ threshold are grouped together.
    Ordering within clusters and across clusters is deterministic (sorted).

    Parameters
    ----------
    strategy_ids : list of str
        All strategy IDs to consider.
    corr_matrix : dict
        ``{(sid_a, sid_b): corr}`` from ``compute_correlation_matrix``.
    threshold : float
        Correlation threshold for grouping.

    Returns
    -------
    list of list of str
        Each inner list is a cluster, sorted.  Clusters are sorted by
        first member.
    """
    sids = sorted(strategy_ids)
    parent: Dict[str, str] = {s: s for s in sids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # Deterministic: smaller ID becomes root
            if ra > rb:
                ra, rb = rb, ra
            parent[rb] = ra

    for (sid_a, sid_b), corr in sorted(corr_matrix.items()):
        import math
        if math.isnan(corr):
            continue
        if corr >= threshold:
            union(sid_a, sid_b)

    # Group by root
    clusters_map: Dict[str, List[str]] = {}
    for s in sids:
        root = find(s)
        clusters_map.setdefault(root, []).append(s)

    # Sort clusters deterministically
    clusters = [sorted(members) for members in clusters_map.values()]
    clusters.sort(key=lambda c: c[0])
    return clusters


def compute_underlying_exposures(
    allocations: List[Any],
) -> Dict[str, float]:
    """Sum absolute notional_usd per underlying from normalized allocations.

    Parameters
    ----------
    allocations : list
        NormalizedAllocation objects (must have ``underlying`` and
        ``notional_usd`` attributes).

    Returns
    -------
    dict
        ``{underlying: total_abs_notional_usd}``.
    """
    exposures: Dict[str, float] = {}
    for alloc in allocations:
        und = getattr(alloc, "underlying", "UNKNOWN")
        notional = abs(getattr(alloc, "notional_usd", 0.0))
        exposures[und] = exposures.get(und, 0.0) + notional
    return exposures
