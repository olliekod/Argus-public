"""
Risk Attribution
=================

Produces a structured RiskAttribution report from clamped allocations and
clamp reasons.  Output is a JSON-serializable artifact.

References
----------
- MASTER_PLAN.md §9 — Phase 5: Portfolio Risk Engine.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("argus.risk_attribution")


@dataclass
class StrategyAttribution:
    """Risk attribution for a single strategy.

    Attributes
    ----------
    strategy_id : str
    instrument : str
    notional_usd : float
    max_loss_usd : float
    delta_contribution : float
    vega_contribution : float
    gamma_contribution : float
    pct_of_total_risk_budget : float
    contracts : int
    weight : float
    """
    strategy_id: str = ""
    instrument: str = ""
    notional_usd: float = 0.0
    max_loss_usd: float = 0.0
    delta_contribution: float = 0.0
    vega_contribution: float = 0.0
    gamma_contribution: float = 0.0
    pct_of_total_risk_budget: float = 0.0
    contracts: int = 0
    weight: float = 0.0


@dataclass
class ClampSummary:
    """Aggregated clamp summary."""
    total_clamps: int = 0
    by_constraint_id: Dict[str, int] = field(default_factory=dict)
    biggest_reductions: List[Dict[str, Any]] = field(default_factory=list)
    kills: int = 0
    warns: int = 0
    infos: int = 0


@dataclass
class PortfolioRiskSummary:
    """Portfolio-level risk summary."""
    total_delta: float = 0.0
    total_vega: float = 0.0
    total_gamma: float = 0.0
    total_max_loss_usd: float = 0.0
    total_notional_usd: float = 0.0
    drawdown_throttle_factor: float = 1.0
    cluster_exposures: Dict[str, float] = field(default_factory=dict)
    equity_usd: float = 0.0
    active_strategies: int = 0


@dataclass
class RiskAttribution:
    """Full risk attribution report.

    Attributes
    ----------
    per_strategy : list of StrategyAttribution
    portfolio_summary : PortfolioRiskSummary
    clamp_summary : ClampSummary
    config_hash : str
        Hash of the risk engine config for reproducibility.
    as_of_ts_ms : int
    """
    per_strategy: List[StrategyAttribution] = field(default_factory=list)
    portfolio_summary: PortfolioRiskSummary = field(
        default_factory=PortfolioRiskSummary
    )
    clamp_summary: ClampSummary = field(default_factory=ClampSummary)
    config_hash: str = ""
    as_of_ts_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict with sorted keys."""
        return _to_sorted_dict(asdict(self))


def build_risk_attribution(
    clamped_allocations: List[Any],
    clamp_reasons: List[Any],
    *,
    equity_usd: float = 0.0,
    drawdown_throttle_factor: float = 1.0,
    cluster_exposures: Optional[Dict[str, float]] = None,
    config_hash: str = "",
    as_of_ts_ms: int = 0,
) -> RiskAttribution:
    """Build a RiskAttribution from clamped allocations and reasons.

    Parameters
    ----------
    clamped_allocations : list
        List of Allocation objects (from allocation_engine).
    clamp_reasons : list
        List of ClampReason objects.
    equity_usd : float
        Portfolio equity.
    drawdown_throttle_factor : float
        Applied drawdown throttle.
    cluster_exposures : dict, optional
        ``{cluster_label: exposure_usd}``.
    config_hash : str
        Hash of risk engine config.
    as_of_ts_ms : int
        Timestamp.

    Returns
    -------
    RiskAttribution
    """
    # Per-strategy attribution
    per_strategy: List[StrategyAttribution] = []
    total_max_loss = 0.0
    total_notional = 0.0
    total_delta = 0.0
    total_vega = 0.0
    total_gamma = 0.0

    for alloc in clamped_allocations:
        notional = abs(getattr(alloc, "dollar_risk", 0.0))
        max_loss = notional  # conservative: full dollar_risk as max_loss
        contracts = getattr(alloc, "contracts", 0)
        weight = getattr(alloc, "weight", 0.0)

        # Check if allocation has greek info from normalized form
        delta_c = getattr(alloc, "_delta_contrib", 0.0)
        vega_c = getattr(alloc, "_vega_contrib", 0.0)
        gamma_c = getattr(alloc, "_gamma_contrib", 0.0)

        total_max_loss += max_loss
        total_notional += notional
        total_delta += delta_c
        total_vega += vega_c
        total_gamma += gamma_c

        per_strategy.append(StrategyAttribution(
            strategy_id=getattr(alloc, "strategy_id", ""),
            instrument=getattr(alloc, "instrument", ""),
            notional_usd=round(notional, 2),
            max_loss_usd=round(max_loss, 2),
            delta_contribution=round(delta_c, 4),
            vega_contribution=round(vega_c, 4),
            gamma_contribution=round(gamma_c, 6),
            contracts=contracts,
            weight=round(weight, 8),
        ))

    # Compute pct_of_total_risk_budget
    for attr in per_strategy:
        if total_max_loss > 0:
            attr.pct_of_total_risk_budget = round(
                attr.max_loss_usd / total_max_loss, 4
            )

    # Sort by strategy_id for determinism
    per_strategy.sort(key=lambda a: a.strategy_id)

    # Clamp summary
    clamp_summary = _build_clamp_summary(clamp_reasons)

    # Portfolio summary
    active = sum(1 for a in clamped_allocations if getattr(a, "weight", 0) != 0)
    portfolio_summary = PortfolioRiskSummary(
        total_delta=round(total_delta, 4),
        total_vega=round(total_vega, 4),
        total_gamma=round(total_gamma, 6),
        total_max_loss_usd=round(total_max_loss, 2),
        total_notional_usd=round(total_notional, 2),
        drawdown_throttle_factor=round(drawdown_throttle_factor, 4),
        cluster_exposures=dict(cluster_exposures or {}),
        equity_usd=equity_usd,
        active_strategies=active,
    )

    return RiskAttribution(
        per_strategy=per_strategy,
        portfolio_summary=portfolio_summary,
        clamp_summary=clamp_summary,
        config_hash=config_hash,
        as_of_ts_ms=as_of_ts_ms,
    )


def persist_risk_attribution(
    attribution: RiskAttribution,
    output_path: str,
) -> str:
    """Write risk attribution to JSON file.

    Parameters
    ----------
    attribution : RiskAttribution
    output_path : str

    Returns
    -------
    str
        The resolved output path.
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(attribution.to_dict(), f, indent=2, sort_keys=True)
    logger.info("Risk attribution written to %s", p)
    return str(p)


def _build_clamp_summary(clamp_reasons: List[Any]) -> ClampSummary:
    """Aggregate clamp reasons into a summary."""
    by_constraint: Dict[str, int] = defaultdict(int)
    kills = 0
    warns = 0
    infos = 0
    reductions: List[Dict[str, Any]] = []

    for cr in clamp_reasons:
        cid = getattr(cr, "constraint_id", "unknown")
        by_constraint[cid] += 1
        severity = getattr(cr, "severity", "info")
        if severity == "kill":
            kills += 1
        elif severity == "warn":
            warns += 1
        else:
            infos += 1

        # Track reductions for biggest_reductions
        before = getattr(cr, "before", {})
        after = getattr(cr, "after", {})
        before_w = before.get("weight", before.get("contracts", 0))
        after_w = after.get("weight", after.get("contracts", 0))
        if isinstance(before_w, (int, float)) and isinstance(after_w, (int, float)):
            reduction = abs(before_w) - abs(after_w)
            if reduction > 0:
                reductions.append({
                    "allocation_id": getattr(cr, "allocation_id", ""),
                    "constraint_id": cid,
                    "reduction": round(reduction, 6),
                    "reason": getattr(cr, "reason", ""),
                })

    # Sort reductions by magnitude descending, take top 10
    reductions.sort(key=lambda r: r["reduction"], reverse=True)
    biggest = reductions[:10]

    return ClampSummary(
        total_clamps=len(clamp_reasons),
        by_constraint_id=dict(sorted(by_constraint.items())),
        biggest_reductions=biggest,
        kills=kills,
        warns=warns,
        infos=infos,
    )


def _to_sorted_dict(obj: Any) -> Any:
    """Recursively sort dict keys for deterministic JSON output."""
    if isinstance(obj, dict):
        return {k: _to_sorted_dict(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        return [_to_sorted_dict(item) for item in obj]
    return obj
