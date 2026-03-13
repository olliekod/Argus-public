"""
Allocation Engine
==================

Consumes a set of candidate strategies with forecasts and outputs target
exposures (position weights) subject to risk constraints.

Architecture
------------
1. **Input:** List of :class:`Forecast` objects from the candidate set.
2. **Sizing:** Fractional Kelly with confidence shrinkage (per-strategy).
3. **Overlay:** Optional vol-target overlay.
4. **Caps:** Per-play cap (7% of equity), aggregate cap.
5. **Output:** Dict mapping strategy_id -> target exposure.

References
----------
- MASTER_PLAN.md §8.5 — Per-play cap.
- MASTER_PLAN.md §9.3 — Sizing stack.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Tuple

from .sizing import (
    Forecast,
    fractional_kelly_size,
    size_position,
    contracts_from_risk_budget,
)

if TYPE_CHECKING:
    from .portfolio_state import PortfolioState
    from .risk_engine import ClampReason, RiskAttribution, RiskEngineConfig

logger = logging.getLogger("argus.allocation_engine")

# Hard per-play cap from MASTER_PLAN §8.5
_DEFAULT_PER_PLAY_CAP = 0.07  # 7% of equity
_DEFAULT_AGGREGATE_CAP = 1.0  # 100% of equity (sum of abs weights)


@dataclass
class AllocationConfig:
    """Configuration for the allocation engine.

    Attributes
    ----------
    kelly_fraction : float
        Kelly fraction c in [0, 1] (default 0.25 = quarter-Kelly).
    per_play_cap : float
        Maximum position as fraction of equity (default 0.07 = 7%).
    vol_target_annual : float, optional
        Annualized volatility target for vol overlay (None = skip).
    min_edge_over_cost : float
        Minimum edge over cost to size (default 0.0).
    aggregate_exposure_cap : float
        Maximum sum of absolute weights (default 1.0 = 100% of equity).
    apply_shrinkage : bool
        Whether to apply confidence shrinkage to mu (default True).
    """
    kelly_fraction: float = 0.25
    per_play_cap: float = _DEFAULT_PER_PLAY_CAP
    vol_target_annual: Optional[float] = None
    min_edge_over_cost: float = 0.0
    aggregate_exposure_cap: float = _DEFAULT_AGGREGATE_CAP
    apply_shrinkage: bool = True


@dataclass
class Allocation:
    """Output allocation for a single strategy/instrument.

    Attributes
    ----------
    strategy_id : str
        Strategy identifier.
    instrument : str
        Instrument/symbol.
    weight : float
        Target weight as fraction of equity.
    dollar_risk : float
        Dollar risk (weight * equity).
    kelly_raw : float
        Pre-cap Kelly weight.
    vol_adjusted : bool
        Whether vol overlay was applied.
    contracts : int
        For options, number of contracts (0 for equity).
    """
    strategy_id: str
    instrument: str
    weight: float = 0.0
    dollar_risk: float = 0.0
    kelly_raw: float = 0.0
    vol_adjusted: bool = False
    contracts: int = 0


class AllocationEngine:
    """Consumes forecasts, outputs target exposures under risk constraints.

    Usage::

        engine = AllocationEngine(
            config=AllocationConfig(kelly_fraction=0.25, per_play_cap=0.07),
            equity=100_000.0,
        )
        allocations = engine.allocate(forecasts)
        for a in allocations:
            print(f"{a.strategy_id}: {a.weight:.4f} ({a.dollar_risk:.2f})")
    """

    def __init__(
        self,
        config: Optional[AllocationConfig] = None,
        equity: float = 10_000.0,
    ) -> None:
        self._config = config or AllocationConfig()
        self._equity = equity

    @property
    def config(self) -> AllocationConfig:
        return self._config

    @property
    def equity(self) -> float:
        return self._equity

    def update_equity(self, equity: float) -> None:
        """Update the portfolio equity level."""
        self._equity = equity

    def allocate(
        self,
        forecasts: List[Forecast],
        max_loss_per_contract: Optional[Dict[str, float]] = None,
    ) -> List[Allocation]:
        """Compute target allocations for a set of forecasts.

        Parameters
        ----------
        forecasts : list of Forecast
            One forecast per strategy/instrument pair.
        max_loss_per_contract : dict[str, float], optional
            For options strategies, maps strategy_id to max loss per
            contract.  Used to compute contract counts.

        Returns
        -------
        list of Allocation
            Target allocations, one per forecast (zero-weight entries
            included for transparency).
        """
        cfg = self._config
        max_loss = max_loss_per_contract or {}

        raw_allocations: List[Allocation] = []

        for forecast in forecasts:
            result = size_position(
                forecast=forecast,
                equity=self._equity,
                kelly_fraction=cfg.kelly_fraction,
                per_play_cap=cfg.per_play_cap,
                vol_target_annual=cfg.vol_target_annual,
                min_edge_over_cost=cfg.min_edge_over_cost,
                apply_shrinkage=cfg.apply_shrinkage,
            )

            # Compute contracts for options strategies
            n_contracts = 0
            sid = forecast.strategy_id
            if sid in max_loss and result["dollar_risk"] > 0:
                n_contracts = contracts_from_risk_budget(
                    risk_budget_usd=abs(result["dollar_risk"]),
                    max_loss_per_contract=max_loss[sid],
                )

            alloc = Allocation(
                strategy_id=sid,
                instrument=forecast.instrument,
                weight=result["weight"],
                dollar_risk=result["dollar_risk"],
                kelly_raw=result["kelly_raw"],
                vol_adjusted=result["vol_adjusted"],
                contracts=n_contracts,
            )
            raw_allocations.append(alloc)

        # Apply aggregate exposure cap
        allocations = self._apply_aggregate_cap(raw_allocations)

        # Log summary
        total_exposure = sum(abs(a.weight) for a in allocations)
        active = sum(1 for a in allocations if a.weight != 0)
        logger.info(
            "Allocation: %d active/%d total, gross_exposure=%.4f, equity=%.2f",
            active, len(allocations), total_exposure, self._equity,
        )

        return allocations

    def _apply_aggregate_cap(
        self,
        allocations: List[Allocation],
    ) -> List[Allocation]:
        """Scale down allocations if aggregate exposure exceeds cap.

        All weights are scaled proportionally so that the sum of
        absolute weights equals the aggregate cap.
        """
        cap = self._config.aggregate_exposure_cap
        total_abs = sum(abs(a.weight) for a in allocations)

        if total_abs <= 0 or total_abs <= cap:
            return allocations

        scale = cap / total_abs
        logger.info(
            "Aggregate exposure %.4f exceeds cap %.4f; scaling by %.4f",
            total_abs, cap, scale,
        )

        result = []
        per_cap = self._config.per_play_cap
        for a in allocations:
            new_weight = a.weight * scale
            # Re-enforce per-play cap after scaling
            new_weight = max(-per_cap, min(per_cap, new_weight))
            new_dollar = round(new_weight * self._equity, 2)
            result.append(Allocation(
                strategy_id=a.strategy_id,
                instrument=a.instrument,
                weight=round(new_weight, 8),
                dollar_risk=new_dollar,
                kelly_raw=a.kelly_raw,
                vol_adjusted=a.vol_adjusted,
                contracts=a.contracts,
            ))

        return result

    def allocate_with_risk_engine(
        self,
        forecasts: List[Forecast],
        portfolio_state: "PortfolioState",
        risk_config: "RiskEngineConfig",
        max_loss_per_contract: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[Allocation], List["ClampReason"], "RiskAttribution"]:
        """Compute allocations then clamp through the risk engine.

        Parameters
        ----------
        forecasts : list of Forecast
            One forecast per strategy/instrument pair.
        portfolio_state : PortfolioState
            Current portfolio snapshot.
        risk_config : RiskEngineConfig
            Risk engine configuration.
        max_loss_per_contract : dict, optional
            For options strategies, maps strategy_id to max loss per contract.

        Returns
        -------
        tuple
            (clamped_allocations, clamp_reasons, risk_attribution)
        """
        from .risk_engine import RiskEngine

        # Step 1: Compute proposed allocations as before
        proposed = self.allocate(forecasts, max_loss_per_contract)

        # Step 2: Clamp through risk engine
        engine = RiskEngine()
        clamped, reasons, attribution = engine.clamp(
            proposed_allocations=proposed,
            portfolio_state=portfolio_state,
            risk_config=risk_config,
        )

        # Log summary
        active = sum(1 for a in clamped if a.weight != 0)
        logger.info(
            "AllocationEngine + RiskEngine: %d active/%d total, "
            "%d clamp_reasons",
            active, len(clamped), len(reasons),
        )

        return clamped, reasons, attribution

    def summary(self, allocations: List[Allocation]) -> Dict[str, Any]:
        """Generate a summary dict for logging/persistence."""
        active = [a for a in allocations if a.weight != 0]
        return {
            "equity": self._equity,
            "config": {
                "kelly_fraction": self._config.kelly_fraction,
                "per_play_cap": self._config.per_play_cap,
                "vol_target_annual": self._config.vol_target_annual,
                "aggregate_exposure_cap": self._config.aggregate_exposure_cap,
            },
            "n_total": len(allocations),
            "n_active": len(active),
            "gross_exposure": round(sum(abs(a.weight) for a in allocations), 6),
            "net_exposure": round(sum(a.weight for a in allocations), 6),
            "allocations": [
                {
                    "strategy_id": a.strategy_id,
                    "instrument": a.instrument,
                    "weight": a.weight,
                    "dollar_risk": a.dollar_risk,
                    "contracts": a.contracts,
                }
                for a in active
            ],
        }
