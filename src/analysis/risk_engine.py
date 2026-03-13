"""
Risk Engine
============

Core orchestrator for Phase 5 Portfolio Risk Engine.

The RiskEngine receives proposed allocations from AllocationEngine,
clamps them through a fixed sequence of constraints, and returns:
1. Clamped allocations
2. Structured ClampReason entries
3. RiskAttribution artifact

Constraint ordering (stable, explicit):
    (1) Aggregate exposure cap
    (2) Drawdown throttle
    (3) Correlation / cluster caps
    (4) Greek limits
    (5) Tail-risk (Heston/PoP) for options only

Invariants:
- **Deterministic**: same inputs → same outputs (no unseeded randomness).
- **No lookahead**: only uses data ≤ portfolio_state.as_of_ts_ms.
- **Monotone**: clamp never increases exposure.
- **Idempotent**: clamp(clamp(x)) == clamp(x).
- **Stable ordering**: allocations sorted by allocation_id before clamping.

Unit Normalization
------------------
Proposed allocations are converted to NormalizedAllocation (canonical
USD/greeks units) before clamping, then converted back to Allocation.

References
----------
- MASTER_PLAN.md §9 — Phase 5: Portfolio Risk Engine.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .allocation_engine import Allocation
from .correlation_exposure import (
    CorrelationConfig,
    build_clusters,
    compute_correlation_matrix,
    compute_underlying_exposures,
    get_strategy_returns_for_correlation,
)
from .drawdown_containment import DrawdownConfig, compute_drawdown_throttle
from .greek_limits import (
    GreekLimitsConfig,
    compute_greeks_for_underlying,
    compute_proposed_greeks,
    compute_scale_factor_for_limit,
)
from .portfolio_state import PortfolioState
from .risk_attribution import (
    RiskAttribution,
    build_risk_attribution,
)
from .tail_risk_scenario import TailRiskConfig, evaluate_tail_risk

logger = logging.getLogger("argus.risk_engine")

# Option contract multiplier
_OPTION_MULTIPLIER = 100


# ═══════════════════════════════════════════════════════════════════════════
#  Data models
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ClampReason:
    """Structured record of a single clamping action.

    Attributes
    ----------
    constraint_id : str
        Which constraint generated this clamp (e.g. "drawdown_throttle").
    allocation_id : str
        Identifier for the allocation that was clamped.
    before : dict
        State before clamping (e.g. {"weight": 0.07, "contracts": 5}).
    after : dict
        State after clamping.
    reason : str
        Human-readable explanation.
    severity : str
        ``"info"`` | ``"warn"`` | ``"kill"``.
    ts_ms : int
        Timestamp of the clamp action.
    """
    constraint_id: str
    allocation_id: str
    before: Dict[str, Any]
    after: Dict[str, Any]
    reason: str
    severity: str = "info"
    ts_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "constraint_id": self.constraint_id,
            "allocation_id": self.allocation_id,
            "before": self.before,
            "after": self.after,
            "reason": self.reason,
            "severity": self.severity,
            "ts_ms": self.ts_ms,
        }


@dataclass
class NormalizedAllocation:
    """Canonical internal representation of an allocation in USD/greek units.

    All constraints operate on this representation.  After clamping, it is
    converted back to an Allocation.

    Unit Conventions
    ----------------
    - notional_usd: absolute USD exposure (weight * equity).
    - max_loss_usd: worst-case loss in USD.
    - delta_shares_equiv: delta in shares (option_delta * multiplier * contracts).
    - delta_usd: delta_shares_equiv * underlying_price.
    - vega: USD per 1 vol point (vega_per_1pct * multiplier * contracts).
    - gamma: shares per $1 move (bs_gamma * multiplier * contracts).
    """
    allocation_id: str
    strategy_id: str
    underlying: str
    instrument_type: str = "equity"
    proposed_weight: float = 0.0
    proposed_contracts: int = 0
    notional_usd: float = 0.0
    max_loss_usd: float = 0.0
    delta_shares_equiv: float = 0.0
    delta_usd: float = 0.0
    vega: float = 0.0
    gamma: float = 0.0
    # Option-specific fields for tail risk
    spot: float = 0.0
    short_strike: float = 0.0
    long_strike: float = 0.0
    T: float = 0.0
    credit: float = 0.0
    v0: float = 0.0
    # Original allocation reference fields
    kelly_raw: float = 0.0
    vol_adjusted: bool = False
    dollar_risk: float = 0.0
    # Mutable: current weight/contracts after clamping
    current_weight: float = 0.0
    current_contracts: int = 0

    @property
    def is_option(self) -> bool:
        return self.instrument_type in ("option_spread", "option_single")


@dataclass
class RiskEngineConfig:
    """Full configuration for the risk engine.

    Attributes
    ----------
    enabled : bool
        Master switch for the risk engine.
    aggregate_exposure_cap : float
        Maximum sum of absolute weights.
    drawdown : DrawdownConfig
    correlation : CorrelationConfig
    greek_limits : GreekLimitsConfig
    tail_risk : TailRiskConfig
    risk_attribution_output_path : str
    enforce_idempotence_check : bool
    enforce_monotone_check : bool
    """
    enabled: bool = True
    aggregate_exposure_cap: float = 1.0
    drawdown: DrawdownConfig = field(default_factory=DrawdownConfig)
    correlation: CorrelationConfig = field(default_factory=CorrelationConfig)
    greek_limits: GreekLimitsConfig = field(default_factory=GreekLimitsConfig)
    tail_risk: TailRiskConfig = field(default_factory=TailRiskConfig)
    risk_attribution_output_path: str = "logs/risk_attribution.json"
    enforce_idempotence_check: bool = True
    enforce_monotone_check: bool = True

    def config_hash(self) -> str:
        """Compute a deterministic hash of this config for versioning."""
        serializable = _config_to_sorted_dict(self)
        raw = json.dumps(serializable, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ═══════════════════════════════════════════════════════════════════════════
#  RiskEngine
# ═══════════════════════════════════════════════════════════════════════════


class RiskEngine:
    """Portfolio risk engine that clamps proposed allocations.

    Usage::

        engine = RiskEngine()
        clamped, reasons, attribution = engine.clamp(
            proposed_allocations=allocations,
            portfolio_state=state,
            risk_config=config,
        )
    """

    def __init__(self) -> None:
        self._was_throttled: bool = False

    def clamp(
        self,
        proposed_allocations: List[Allocation],
        portfolio_state: PortfolioState,
        risk_config: RiskEngineConfig,
    ) -> Tuple[List[Allocation], List[ClampReason], RiskAttribution]:
        """Apply all risk constraints to proposed allocations.

        Parameters
        ----------
        proposed_allocations : list of Allocation
            Raw allocations from AllocationEngine.
        portfolio_state : PortfolioState
            Current portfolio snapshot.
        risk_config : RiskEngineConfig
            Risk engine configuration.

        Returns
        -------
        tuple
            (clamped_allocations, clamp_reasons, risk_attribution)
        """
        if portfolio_state.as_of_ts_ms <= 0:
            logger.warning(
                "as_of_ts_ms is 0; using wall-clock time (breaks determinism)"
            )
        ts_ms = portfolio_state.as_of_ts_ms or int(time.time() * 1000)
        equity = portfolio_state.equity_usd
        reasons: List[ClampReason] = []

        # Record initial total exposure for monotone check
        initial_total_exposure = sum(
            abs(a.weight) for a in proposed_allocations
        )

        # Sort allocations by stable key for deterministic ordering.
        # Include weight as tiebreaker for identical strategy+instrument pairs.
        sorted_allocs = sorted(
            proposed_allocations,
            key=lambda a: (a.strategy_id, a.instrument, a.weight),
        )

        # ── Step 0: Normalize to canonical units ──────────────────────
        normalized = self._normalize(sorted_allocs, portfolio_state)

        # ── Constraint (1): Aggregate exposure cap ────────────────────
        normalized, new_reasons = self._apply_aggregate_cap(
            normalized, risk_config.aggregate_exposure_cap, equity, ts_ms,
        )
        reasons.extend(new_reasons)

        # ── Constraint (2): Drawdown throttle ─────────────────────────
        normalized, new_reasons, throttle_factor = self._apply_drawdown_throttle(
            normalized, portfolio_state, risk_config.drawdown,
            risk_config.aggregate_exposure_cap, ts_ms,
        )
        reasons.extend(new_reasons)

        # ── Constraint (3): Correlation / cluster caps ────────────────
        normalized, new_reasons, cluster_exposures = self._apply_correlation_caps(
            normalized, portfolio_state, risk_config.correlation, equity, ts_ms,
        )
        reasons.extend(new_reasons)

        # ── Constraint (4): Greek limits ──────────────────────────────
        normalized, new_reasons = self._apply_greek_limits(
            normalized, portfolio_state, risk_config.greek_limits, ts_ms,
        )
        reasons.extend(new_reasons)

        # ── Constraint (5): Tail-risk (options only) ──────────────────
        normalized, new_reasons = self._apply_tail_risk(
            normalized, portfolio_state, risk_config.tail_risk, ts_ms,
        )
        reasons.extend(new_reasons)

        # ── Convert back to Allocation objects ────────────────────────
        clamped = self._denormalize(normalized, equity)

        # ── Monotone check ────────────────────────────────────────────
        if risk_config.enforce_monotone_check:
            final_total_exposure = sum(abs(a.weight) for a in clamped)
            if final_total_exposure > initial_total_exposure + 1e-10:
                logger.error(
                    "MONOTONE VIOLATION: initial_exposure=%.8f "
                    "final_exposure=%.8f",
                    initial_total_exposure, final_total_exposure,
                )
                raise RuntimeError(
                    f"Monotone violation: exposure increased from "
                    f"{initial_total_exposure:.8f} to {final_total_exposure:.8f}"
                )

        # ── Idempotence check ─────────────────────────────────────────
        if risk_config.enforce_idempotence_check:
            self._check_idempotence(
                clamped, portfolio_state, risk_config,
            )

        # ── Build attribution ─────────────────────────────────────────
        # Attach greek contributions to allocations for attribution
        _attach_greek_contributions(clamped, normalized)

        attribution = build_risk_attribution(
            clamped_allocations=clamped,
            clamp_reasons=reasons,
            equity_usd=equity,
            drawdown_throttle_factor=throttle_factor,
            cluster_exposures=cluster_exposures,
            config_hash=risk_config.config_hash(),
            as_of_ts_ms=ts_ms,
        )

        logger.info(
            "RiskEngine clamp complete: %d allocations, %d clamp_reasons, "
            "throttle=%.4f, config_hash=%s",
            len(clamped), len(reasons), throttle_factor,
            risk_config.config_hash(),
        )

        return clamped, reasons, attribution

    # ──────────────────────────────────────────────────────────────────
    #  Normalization
    # ──────────────────────────────────────────────────────────────────

    def _normalize(
        self,
        allocations: List[Allocation],
        state: PortfolioState,
    ) -> List[NormalizedAllocation]:
        """Convert Allocation objects to NormalizedAllocation."""
        equity = state.equity_usd
        result: List[NormalizedAllocation] = []

        for alloc in allocations:
            alloc_id = f"{alloc.strategy_id}:{alloc.instrument}"

            # Determine instrument type: check position records first,
            # fall back to contracts > 0 heuristic.
            instrument_type = "equity"
            for pos in state.current_positions:
                if (pos.strategy_id == alloc.strategy_id
                        and pos.underlying == alloc.instrument
                        and pos.instrument_type in ("option_spread", "option_single")):
                    instrument_type = pos.instrument_type
                    break
            else:
                if alloc.contracts > 0:
                    instrument_type = "option_spread"
            is_option = instrument_type in ("option_spread", "option_single")

            notional = abs(alloc.weight * equity)
            max_loss = abs(alloc.dollar_risk)

            # Default greek values (will be overridden if greeks available)
            delta_se = 0.0
            delta_usd = 0.0
            vega = 0.0
            gamma = 0.0

            # For options, try to compute greeks from position data
            spot = 0.0
            short_strike = 0.0
            long_strike = 0.0
            T = 0.0
            credit = 0.0
            v0 = 0.0

            if is_option:
                # Look for position metadata in portfolio state
                for pos in state.current_positions:
                    if (pos.strategy_id == alloc.strategy_id
                            and pos.underlying == alloc.instrument):
                        delta_se = pos.greeks.get("delta", 0.0) * pos.qty
                        vega = pos.greeks.get("vega", 0.0) * abs(pos.qty)
                        gamma = pos.greeks.get("gamma", 0.0) * abs(pos.qty)
                        spot = pos.meta.get("spot", 0.0)
                        short_strike = pos.meta.get("short_strike", 0.0)
                        long_strike = pos.meta.get("long_strike", 0.0)
                        T = pos.meta.get("T", 0.0)
                        credit = pos.meta.get("credit", 0.0)
                        v0 = pos.meta.get("v0", 0.0)
                        break

                if spot > 0:
                    delta_usd = delta_se * spot
            else:
                # Equity: delta = weight * equity / price (approx = weight)
                delta_se = alloc.weight * equity
                delta_usd = delta_se

            na = NormalizedAllocation(
                allocation_id=alloc_id,
                strategy_id=alloc.strategy_id,
                underlying=alloc.instrument,
                instrument_type=instrument_type,
                proposed_weight=alloc.weight,
                proposed_contracts=alloc.contracts,
                notional_usd=notional,
                max_loss_usd=max_loss,
                delta_shares_equiv=delta_se,
                delta_usd=delta_usd,
                vega=vega,
                gamma=gamma,
                spot=spot,
                short_strike=short_strike,
                long_strike=long_strike,
                T=T,
                credit=credit,
                v0=v0,
                kelly_raw=alloc.kelly_raw,
                vol_adjusted=alloc.vol_adjusted,
                dollar_risk=alloc.dollar_risk,
                current_weight=alloc.weight,
                current_contracts=alloc.contracts,
            )
            result.append(na)

        return result

    def _denormalize(
        self,
        normalized: List[NormalizedAllocation],
        equity: float,
    ) -> List[Allocation]:
        """Convert NormalizedAllocation back to Allocation objects."""
        result: List[Allocation] = []
        for na in normalized:
            # Preserve original dollar_risk (risk budget), scaled proportionally
            # if the weight was clamped down.
            if abs(na.proposed_weight) > 1e-12:
                scale = na.current_weight / na.proposed_weight
            else:
                scale = 0.0
            dollar_risk = round(na.dollar_risk * scale, 2)
            result.append(Allocation(
                strategy_id=na.strategy_id,
                instrument=na.underlying,
                weight=round(na.current_weight, 8),
                dollar_risk=dollar_risk,
                kelly_raw=na.kelly_raw,
                vol_adjusted=na.vol_adjusted,
                contracts=na.current_contracts,
            ))
        return result

    # ──────────────────────────────────────────────────────────────────
    #  Constraint (1): Aggregate exposure cap
    # ──────────────────────────────────────────────────────────────────

    def _apply_aggregate_cap(
        self,
        normalized: List[NormalizedAllocation],
        cap: float,
        equity: float,
        ts_ms: int,
    ) -> Tuple[List[NormalizedAllocation], List[ClampReason]]:
        """Scale down if sum of abs weights exceeds aggregate cap."""
        reasons: List[ClampReason] = []
        total_abs = sum(abs(na.current_weight) for na in normalized)

        if total_abs <= 0 or total_abs <= cap:
            return normalized, reasons

        scale = cap / total_abs
        logger.info(
            "Aggregate cap: total=%.6f > cap=%.6f, scale=%.6f",
            total_abs, cap, scale,
        )

        for na in normalized:
            before_w = na.current_weight
            na.current_weight = before_w * scale
            na.notional_usd = abs(na.current_weight * equity)
            na.delta_shares_equiv *= scale
            na.delta_usd *= scale
            na.vega *= scale
            na.gamma *= scale

            if abs(before_w - na.current_weight) > 1e-10:
                reasons.append(ClampReason(
                    constraint_id="aggregate_cap",
                    allocation_id=na.allocation_id,
                    before={"weight": round(before_w, 8)},
                    after={"weight": round(na.current_weight, 8)},
                    reason=f"Aggregate exposure {total_abs:.6f} > cap {cap:.6f}; scaled by {scale:.6f}",
                    severity="info",
                    ts_ms=ts_ms,
                ))

        return normalized, reasons

    # ──────────────────────────────────────────────────────────────────
    #  Constraint (2): Drawdown throttle
    # ──────────────────────────────────────────────────────────────────

    def _apply_drawdown_throttle(
        self,
        normalized: List[NormalizedAllocation],
        state: PortfolioState,
        dd_config: DrawdownConfig,
        aggregate_exposure_cap: float,
        ts_ms: int,
    ) -> Tuple[List[NormalizedAllocation], List[ClampReason], float]:
        """Apply drawdown-based throttling as a tighter aggregate cap.

        For idempotence, the throttle is applied as a **cap** on total
        exposure rather than a per-allocation multiplier.  The effective
        cap is ``aggregate_exposure_cap * throttle``.  If total exposure
        is already within this cap, no change is made.  This guarantees
        clamp(clamp(x)) == clamp(x) because the cap-check is absolute.
        """
        reasons: List[ClampReason] = []
        throttle = compute_drawdown_throttle(
            state.current_drawdown_pct, dd_config,
            was_throttled=self._was_throttled,
        )

        # Update hysteresis state
        self._was_throttled = throttle < 1.0

        if throttle >= 1.0:
            return normalized, reasons, 1.0

        # Effective cap: aggregate cap reduced by the throttle factor
        effective_cap = aggregate_exposure_cap * throttle
        total_abs = sum(abs(na.current_weight) for na in normalized)

        if total_abs <= 0 or total_abs <= effective_cap + 1e-12:
            # Already within the drawdown-adjusted cap — idempotent no-op
            return normalized, reasons, throttle

        scale = effective_cap / total_abs
        equity = state.equity_usd
        logger.info(
            "Drawdown throttle: dd=%.4f throttle=%.4f effective_cap=%.6f "
            "total=%.6f scale=%.6f",
            state.current_drawdown_pct, throttle, effective_cap,
            total_abs, scale,
        )

        for na in normalized:
            before_w = na.current_weight
            na.current_weight = before_w * scale
            na.notional_usd = abs(na.current_weight * equity)
            na.delta_shares_equiv *= scale
            na.delta_usd *= scale
            na.vega *= scale
            na.gamma *= scale

            if na.is_option and abs(before_w) > 1e-12:
                new_contracts = max(0, int(na.current_contracts * scale))
                na.current_contracts = new_contracts

            if abs(before_w - na.current_weight) > 1e-10:
                reasons.append(ClampReason(
                    constraint_id="drawdown_throttle",
                    allocation_id=na.allocation_id,
                    before={"weight": round(before_w, 8)},
                    after={"weight": round(na.current_weight, 8)},
                    reason=(
                        f"Drawdown {state.current_drawdown_pct:.4f} triggered "
                        f"throttle={throttle:.4f}; effective_cap={effective_cap:.6f}"
                    ),
                    severity="warn",
                    ts_ms=ts_ms,
                ))

        return normalized, reasons, throttle

    # ──────────────────────────────────────────────────────────────────
    #  Constraint (3): Correlation / cluster caps
    # ──────────────────────────────────────────────────────────────────

    def _apply_correlation_caps(
        self,
        normalized: List[NormalizedAllocation],
        state: PortfolioState,
        corr_config: CorrelationConfig,
        equity: float,
        ts_ms: int,
    ) -> Tuple[List[NormalizedAllocation], List[ClampReason], Dict[str, float]]:
        """Apply underlying and cluster exposure caps."""
        reasons: List[ClampReason] = []
        cluster_exposures: Dict[str, float] = {}

        # ── Per-underlying caps ───────────────────────────────────────
        if corr_config.max_exposure_per_underlying_usd > 0:
            underlyings = sorted(set(na.underlying for na in normalized))
            for underlying in underlyings:
                und_allocs = [
                    na for na in normalized if na.underlying == underlying
                ]
                total_notional = sum(abs(na.notional_usd) for na in und_allocs)
                cap = corr_config.max_exposure_per_underlying_usd

                if total_notional > cap and total_notional > 0:
                    scale = cap / total_notional
                    for na in und_allocs:
                        before_w = na.current_weight
                        na.current_weight *= scale
                        na.notional_usd = abs(na.current_weight * equity)
                        na.delta_shares_equiv *= scale
                        na.delta_usd *= scale
                        na.vega *= scale
                        na.gamma *= scale

                        if na.is_option:
                            new_c = max(0, int(na.current_contracts * scale))
                            na.current_contracts = new_c

                        if abs(before_w - na.current_weight) > 1e-10:
                            reasons.append(ClampReason(
                                constraint_id="underlying_cap",
                                allocation_id=na.allocation_id,
                                before={"weight": round(before_w, 8)},
                                after={"weight": round(na.current_weight, 8)},
                                reason=f"Underlying {underlying} exposure ${total_notional:.0f} > cap ${cap:.0f}",
                                severity="warn",
                                ts_ms=ts_ms,
                            ))

        # ── Strategy-level correlation clusters ───────────────────────
        strategy_ids = sorted(set(na.strategy_id for na in normalized))
        has_returns = bool(state.strategy_return_series)

        if has_returns and corr_config.max_exposure_per_cluster_usd > 0:
            aligned = get_strategy_returns_for_correlation(
                state.strategy_return_series,
                strategy_ids,
                state.as_of_ts_ms,
                corr_config.rolling_days,
            )

            n_obs = min(len(v) for v in aligned.values()) if aligned else 0
            if n_obs >= corr_config.min_obs:
                corr_matrix = compute_correlation_matrix(
                    aligned,
                    min_obs=corr_config.min_obs,
                    estimator=corr_config.estimator,
                )
                clusters = build_clusters(
                    strategy_ids, corr_matrix,
                    threshold=corr_config.corr_threshold_for_cluster,
                )

                cluster_cap = corr_config.max_exposure_per_cluster_usd
                for i, cluster in enumerate(clusters):
                    cluster_label = f"cluster_{i}_{cluster[0]}"
                    cluster_allocs = [
                        na for na in normalized if na.strategy_id in cluster
                    ]
                    cluster_notional = sum(
                        abs(na.notional_usd) for na in cluster_allocs
                    )
                    cluster_exposures[cluster_label] = round(cluster_notional, 2)

                    if cluster_notional > cluster_cap and cluster_notional > 0:
                        scale = cluster_cap / cluster_notional
                        for na in cluster_allocs:
                            before_w = na.current_weight
                            na.current_weight *= scale
                            na.notional_usd = abs(na.current_weight * equity)
                            na.delta_shares_equiv *= scale
                            na.delta_usd *= scale
                            na.vega *= scale
                            na.gamma *= scale

                            if na.is_option:
                                new_c = max(
                                    0, int(na.current_contracts * scale)
                                )
                                na.current_contracts = new_c

                            if abs(before_w - na.current_weight) > 1e-10:
                                reasons.append(ClampReason(
                                    constraint_id="cluster_cap",
                                    allocation_id=na.allocation_id,
                                    before={"weight": round(before_w, 8)},
                                    after={"weight": round(na.current_weight, 8)},
                                    reason=f"Cluster {cluster_label} exposure ${cluster_notional:.0f} > cap ${cluster_cap:.0f}",
                                    severity="warn",
                                    ts_ms=ts_ms,
                                ))
            else:
                # Insufficient data — log reason
                for sid in strategy_ids:
                    reasons.append(ClampReason(
                        constraint_id="correlation_data",
                        allocation_id=sid,
                        before={},
                        after={},
                        reason=f"insufficient_returns_data (n_obs={n_obs} < min_obs={corr_config.min_obs})",
                        severity="info",
                        ts_ms=ts_ms,
                    ))
        elif not has_returns and corr_config.max_exposure_per_cluster_usd > 0:
            for sid in strategy_ids:
                reasons.append(ClampReason(
                    constraint_id="correlation_data",
                    allocation_id=sid,
                    before={},
                    after={},
                    reason="insufficient_returns_data (no strategy_return_series)",
                    severity="info",
                    ts_ms=ts_ms,
                ))

        return normalized, reasons, cluster_exposures

    # ──────────────────────────────────────────────────────────────────
    #  Constraint (4): Greek limits
    # ──────────────────────────────────────────────────────────────────

    def _apply_greek_limits(
        self,
        normalized: List[NormalizedAllocation],
        state: PortfolioState,
        gl_config: GreekLimitsConfig,
        ts_ms: int,
    ) -> Tuple[List[NormalizedAllocation], List[ClampReason]]:
        """Apply per-underlying and portfolio greek limits."""
        reasons: List[ClampReason] = []
        existing_greeks = state.total_position_greeks()
        equity = state.equity_usd

        # ── Per-underlying limits ─────────────────────────────────────
        underlyings_with_limits = sorted(gl_config.per_underlying.keys())
        for underlying in underlyings_with_limits:
            limits = gl_config.per_underlying[underlying]
            und_allocs = [
                na for na in normalized if na.underlying == underlying
            ]
            if not und_allocs:
                continue

            # Get existing greeks for this underlying
            existing_und = {"delta": 0.0, "gamma": 0.0, "vega": 0.0}
            if gl_config.enforce_existing_positions:
                for pos in state.current_positions:
                    if pos.underlying == underlying:
                        existing_und["delta"] += pos.greeks.get("delta", 0.0) * pos.qty
                        existing_und["gamma"] += pos.greeks.get("gamma", 0.0) * abs(pos.qty)
                        existing_und["vega"] += pos.greeks.get("vega", 0.0) * abs(pos.qty)

            proposed_greeks = compute_greeks_for_underlying(
                und_allocs, underlying, existing_und,
                enforce_existing=gl_config.enforce_existing_positions,
            )

            # Compute minimum scale factor across all greek types
            min_scale = 1.0
            limiting_greek = ""

            for greek_name, limit_key in [
                ("delta", "max_delta_shares"),
                ("vega", "max_vega"),
                ("gamma", "max_gamma"),
            ]:
                limit = limits.get(limit_key, float("inf"))
                if limit < float("inf"):
                    scale = compute_scale_factor_for_limit(
                        proposed_greeks[greek_name], limit
                    )
                    if scale < min_scale:
                        min_scale = scale
                        limiting_greek = greek_name

            if min_scale < 1.0:
                for na in und_allocs:
                    before_w = na.current_weight
                    na.current_weight *= min_scale
                    na.notional_usd = abs(na.current_weight * equity)
                    na.delta_shares_equiv *= min_scale
                    na.delta_usd *= min_scale
                    na.vega *= min_scale
                    na.gamma *= min_scale

                    if na.is_option:
                        new_c = max(0, int(na.current_contracts * min_scale))
                        na.current_contracts = new_c

                    if abs(before_w - na.current_weight) > 1e-10:
                        reasons.append(ClampReason(
                            constraint_id=f"greek_limit_{underlying}",
                            allocation_id=na.allocation_id,
                            before={"weight": round(before_w, 8)},
                            after={"weight": round(na.current_weight, 8)},
                            reason=f"{underlying} {limiting_greek} limit exceeded; scale={min_scale:.6f}",
                            severity="warn",
                            ts_ms=ts_ms,
                        ))

        # ── Portfolio-level limits ────────────────────────────────────
        total_greeks = compute_proposed_greeks(
            normalized, existing_greeks,
            enforce_existing=gl_config.enforce_existing_positions,
        )

        portfolio_min_scale = 1.0
        limiting_greek = ""

        for greek_name, limit in [
            ("delta", gl_config.portfolio_max_delta_shares),
            ("vega", gl_config.portfolio_max_vega),
            ("gamma", gl_config.portfolio_max_gamma),
        ]:
            if limit < float("inf"):
                scale = compute_scale_factor_for_limit(
                    total_greeks[greek_name], limit
                )
                if scale < portfolio_min_scale:
                    portfolio_min_scale = scale
                    limiting_greek = greek_name

        if portfolio_min_scale < 1.0:
            for na in normalized:
                before_w = na.current_weight
                na.current_weight *= portfolio_min_scale
                na.notional_usd = abs(na.current_weight * equity)
                na.delta_shares_equiv *= portfolio_min_scale
                na.delta_usd *= portfolio_min_scale
                na.vega *= portfolio_min_scale
                na.gamma *= portfolio_min_scale

                if na.is_option:
                    new_c = max(0, int(na.current_contracts * portfolio_min_scale))
                    na.current_contracts = new_c

                if abs(before_w - na.current_weight) > 1e-10:
                    reasons.append(ClampReason(
                        constraint_id="greek_limit_portfolio",
                        allocation_id=na.allocation_id,
                        before={"weight": round(before_w, 8)},
                        after={"weight": round(na.current_weight, 8)},
                        reason=f"Portfolio {limiting_greek} limit exceeded; scale={portfolio_min_scale:.6f}",
                        severity="warn",
                        ts_ms=ts_ms,
                    ))

        return normalized, reasons

    # ──────────────────────────────────────────────────────────────────
    #  Constraint (5): Tail-risk (options only)
    # ──────────────────────────────────────────────────────────────────

    def _apply_tail_risk(
        self,
        normalized: List[NormalizedAllocation],
        state: PortfolioState,
        tr_config: TailRiskConfig,
        ts_ms: int,
    ) -> Tuple[List[NormalizedAllocation], List[ClampReason]]:
        """Apply Heston/PoP-based tail risk caps to options allocations."""
        reasons: List[ClampReason] = []

        if not tr_config.enabled_for_options:
            return normalized, reasons

        equity = state.equity_usd

        for na in normalized:
            if not na.is_option or na.current_contracts <= 0:
                continue

            # Need spot and strikes for tail risk eval
            if na.spot <= 0 or na.short_strike <= 0 or na.long_strike <= 0:
                # Cannot validate risk — conservative fallback
                if na.v0 <= 0 and na.spot <= 0:
                    # Completely missing data: clamp to zero
                    before_c = na.current_contracts
                    before_w = na.current_weight
                    na.current_contracts = 0
                    na.current_weight = 0.0
                    na.notional_usd = 0.0
                    reasons.append(ClampReason(
                        constraint_id="tail_risk_missing_data",
                        allocation_id=na.allocation_id,
                        before={"contracts": before_c, "weight": round(before_w, 8)},
                        after={"contracts": 0, "weight": 0.0},
                        reason="Cannot validate tail risk: missing spot/strikes/v0; clamped to 0",
                        severity="kill",
                        ts_ms=ts_ms,
                    ))
                continue

            result = evaluate_tail_risk(
                S=na.spot,
                short_strike=na.short_strike,
                long_strike=na.long_strike,
                T=na.T,
                credit=na.credit,
                v0=na.v0,
                proposed_contracts=na.current_contracts,
                equity_usd=equity,
                config=tr_config,
            )

            if result.capped:
                before_c = na.current_contracts
                before_w = na.current_weight
                na.current_contracts = result.allowed_contracts

                # Scale weight proportionally
                if before_c > 0:
                    scale = result.allowed_contracts / before_c
                    na.current_weight *= scale
                    na.notional_usd = abs(na.current_weight * equity)
                    na.delta_shares_equiv *= scale
                    na.delta_usd *= scale
                    na.vega *= scale
                    na.gamma *= scale

                severity = "kill" if result.allowed_contracts == 0 else "warn"
                reasons.append(ClampReason(
                    constraint_id="tail_risk",
                    allocation_id=na.allocation_id,
                    before={"contracts": before_c, "weight": round(before_w, 8)},
                    after={
                        "contracts": result.allowed_contracts,
                        "weight": round(na.current_weight, 8),
                    },
                    reason=f"Tail risk: {result.reason} (prob_touch={result.prob_touch:.4f})",
                    severity=severity,
                    ts_ms=ts_ms,
                ))

        return normalized, reasons

    # ──────────────────────────────────────────────────────────────────
    #  Idempotence check
    # ──────────────────────────────────────────────────────────────────

    def _check_idempotence(
        self,
        clamped: List[Allocation],
        state: PortfolioState,
        config: RiskEngineConfig,
    ) -> None:
        """Verify clamp(clamp(x)) == clamp(x).

        Runs a second pass and compares outputs.  Only logs warnings
        (does not raise) to avoid disrupting production flow.
        """
        # Disable checks during the second pass to avoid infinite recursion
        config_copy = RiskEngineConfig(
            enabled=config.enabled,
            aggregate_exposure_cap=config.aggregate_exposure_cap,
            drawdown=config.drawdown,
            correlation=config.correlation,
            greek_limits=config.greek_limits,
            tail_risk=config.tail_risk,
            risk_attribution_output_path=config.risk_attribution_output_path,
            enforce_idempotence_check=False,
            enforce_monotone_check=False,
        )

        second_clamped, _, _ = self.clamp(clamped, state, config_copy)

        # Compare weights and contracts
        for orig, second in zip(
            sorted(clamped, key=lambda a: f"{a.strategy_id}:{a.instrument}"),
            sorted(second_clamped, key=lambda a: f"{a.strategy_id}:{a.instrument}"),
        ):
            if abs(orig.weight - second.weight) > 1e-8:
                logger.error(
                    "IDEMPOTENCE VIOLATION for %s: first=%.8f second=%.8f",
                    orig.strategy_id, orig.weight, second.weight,
                )
            if orig.contracts != second.contracts:
                logger.error(
                    "IDEMPOTENCE VIOLATION (contracts) for %s: "
                    "first=%d second=%d",
                    orig.strategy_id, orig.contracts, second.contracts,
                )


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _attach_greek_contributions(
    allocations: List[Allocation],
    normalized: List[NormalizedAllocation],
) -> None:
    """Attach greek contribution values to Allocation objects for attribution.

    Sets private attributes ``_delta_contrib``, ``_vega_contrib``,
    ``_gamma_contrib`` on each Allocation.
    """
    # Build lookup
    norm_by_id: Dict[str, NormalizedAllocation] = {}
    for na in normalized:
        norm_by_id[na.allocation_id] = na

    for alloc in allocations:
        key = f"{alloc.strategy_id}:{alloc.instrument}"
        na = norm_by_id.get(key)
        if na:
            object.__setattr__(alloc, "_delta_contrib", na.delta_shares_equiv)
            object.__setattr__(alloc, "_vega_contrib", na.vega)
            object.__setattr__(alloc, "_gamma_contrib", na.gamma)
        else:
            object.__setattr__(alloc, "_delta_contrib", 0.0)
            object.__setattr__(alloc, "_vega_contrib", 0.0)
            object.__setattr__(alloc, "_gamma_contrib", 0.0)


def _config_to_sorted_dict(config: RiskEngineConfig) -> Dict[str, Any]:
    """Convert RiskEngineConfig to a sorted dict for hashing."""
    result: Dict[str, Any] = {}
    result["enabled"] = config.enabled
    result["aggregate_exposure_cap"] = config.aggregate_exposure_cap
    result["enforce_idempotence_check"] = config.enforce_idempotence_check
    result["enforce_monotone_check"] = config.enforce_monotone_check
    result["risk_attribution_output_path"] = config.risk_attribution_output_path

    # Drawdown
    dd = config.drawdown
    result["drawdown"] = {
        "k": dd.k,
        "min_throttle": dd.min_throttle,
        "recovery_threshold_pct": dd.recovery_threshold_pct,
        "threshold_pct": dd.threshold_pct,
        "throttle_mode": dd.throttle_mode,
        "throttle_scale": dd.throttle_scale,
    }

    # Correlation
    corr = config.correlation
    result["correlation"] = {
        "cluster_method": corr.cluster_method,
        "corr_threshold_for_cluster": corr.corr_threshold_for_cluster,
        "estimator": corr.estimator,
        "max_correlated_pair_exposure_usd": corr.max_correlated_pair_exposure_usd,
        "max_exposure_per_cluster_usd": corr.max_exposure_per_cluster_usd,
        "max_exposure_per_underlying_usd": corr.max_exposure_per_underlying_usd,
        "min_obs": corr.min_obs,
        "nan_policy": corr.nan_policy,
        "rolling_days": corr.rolling_days,
    }

    # Greek limits
    gl = config.greek_limits
    result["greek_limits"] = {
        "enforce_existing_positions": gl.enforce_existing_positions,
        "per_underlying": dict(sorted(gl.per_underlying.items())),
        "portfolio_max_delta_shares": gl.portfolio_max_delta_shares,
        "portfolio_max_gamma": gl.portfolio_max_gamma,
        "portfolio_max_vega": gl.portfolio_max_vega,
    }

    # Tail risk
    tr = config.tail_risk
    result["tail_risk"] = {
        "default_heston_kappa": tr.default_heston_kappa,
        "default_heston_rho": tr.default_heston_rho,
        "default_heston_sigma_v": tr.default_heston_sigma_v,
        "enabled_for_options": tr.enabled_for_options,
        "max_prob_touch": tr.max_prob_touch,
        "max_stress_loss_pct": tr.max_stress_loss_pct,
        "mc_simulations": tr.mc_simulations,
        "mc_steps_per_year": tr.mc_steps_per_year,
        "seed": tr.seed,
        "stress_iv_bump": tr.stress_iv_bump,
    }

    return dict(sorted(result.items()))
