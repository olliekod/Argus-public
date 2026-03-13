"""
Regime-Conditional Strategy Router
==================================

Routes to the appropriate sub-strategy based on current regime.

Routing rules:
- VOL_SPIKE or VOL_HIGH + IV available → HighVolCreditStrategy
- VRP > min_vrp + vol not VOL_SPIKE + trend not TREND_DOWN → VRPCreditSpreadStrategy
- Overnight entry window + fwd_return > threshold → OvernightSessionStrategy

Priority: HighVol first (when vol elevated), then VRP (when VRP positive and vol OK),
then Overnight (when at session transition). Only one strategy emits per bar.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Type

from src.analysis.replay_harness import ReplayStrategy, TradeIntent
from src.core.outcome_engine import BarData, OutcomeResult

logger = logging.getLogger("argus.strategies.regime_conditional")


def _resolve_strategy_class(name: str) -> Type[ReplayStrategy]:
    if name == "vrp":
        from src.strategies.vrp_credit_spread import VRPCreditSpreadStrategy
        return VRPCreditSpreadStrategy
    if name == "high_vol":
        from src.strategies.high_vol_credit import HighVolCreditStrategy
        return HighVolCreditStrategy
    if name == "overnight_session":
        from src.strategies.overnight_session import OvernightSessionStrategy
        return OvernightSessionStrategy
    raise ValueError(f"Unknown strategy: {name!r}")


class RegimeConditionalStrategy(ReplayStrategy):
    """Routes to VRP, HighVol, or OvernightSession based on regime."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        cfg = params or {}
        self._vrp_params = cfg.get("vrp_params", {"min_vrp": 0.05})
        self._high_vol_params = cfg.get("high_vol_params", {"min_iv": 0.12})
        self._overnight_params = cfg.get("overnight_params", {"fwd_return_threshold": 0.005})

        self._vrp = _resolve_strategy_class("vrp")(self._vrp_params)
        self._high_vol = _resolve_strategy_class("high_vol")(self._high_vol_params)
        self._overnight = _resolve_strategy_class("overnight_session")(self._overnight_params)

        self._active_strategy: Optional[ReplayStrategy] = None
        self._routed_counts: Dict[str, int] = {"vrp": 0, "high_vol": 0, "overnight": 0}

    @property
    def strategy_id(self) -> str:
        return "REGIME_CONDITIONAL_V1"

    def on_bar(
        self,
        bar: BarData,
        sim_ts_ms: int,
        session_regime: str,
        visible_outcomes: Dict[int, OutcomeResult],
        *,
        visible_regimes: Optional[Dict[str, Dict[str, Any]]] = None,
        visible_snapshots: Optional[List[Any]] = None,
    ) -> None:
        self._active_strategy = None
        self._vrp.on_bar(bar, sim_ts_ms, session_regime, visible_outcomes,
                         visible_regimes=visible_regimes, visible_snapshots=visible_snapshots)
        self._high_vol.on_bar(bar, sim_ts_ms, session_regime, visible_outcomes,
                              visible_regimes=visible_regimes, visible_snapshots=visible_snapshots)
        self._overnight.on_bar(bar, sim_ts_ms, session_regime, visible_outcomes,
                               visible_regimes=visible_regimes, visible_snapshots=visible_snapshots)

    def generate_intents(self, sim_ts_ms: int) -> List[TradeIntent]:
        visible_regimes = getattr(self._vrp, "visible_regimes", {})
        symbol_regime = visible_regimes.get("SPY", {})
        vol = symbol_regime.get("vol_regime", "UNKNOWN")
        trend = symbol_regime.get("trend_regime", "UNKNOWN")

        # 1. High vol: VOL_SPIKE or VOL_HIGH → HighVolCredit
        if vol in ("VOL_SPIKE", "VOL_HIGH"):
            intents = self._high_vol.generate_intents(sim_ts_ms)
            if intents:
                self._routed_counts["high_vol"] += 1
                self._active_strategy = self._high_vol
                return intents

        # 2. VRP: vrp > min and vol not VOL_SPIKE and trend not TREND_DOWN
        if vol != "VOL_SPIKE" and trend != "TREND_DOWN":
            intents = self._vrp.generate_intents(sim_ts_ms)
            if intents:
                self._routed_counts["vrp"] += 1
                self._active_strategy = self._vrp
                return intents

        # 3. Overnight: session-transition momentum
        intents = self._overnight.generate_intents(sim_ts_ms)
        if intents:
            self._routed_counts["overnight"] += 1
            self._active_strategy = self._overnight
            return intents

        return []

    def finalize(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"routed_counts": dict(self._routed_counts)}
        if hasattr(self._vrp, "finalize"):
            result["vrp"] = self._vrp.finalize()
        if hasattr(self._high_vol, "finalize"):
            result["high_vol"] = self._high_vol.finalize()
        if hasattr(self._overnight, "finalize"):
            result["overnight"] = self._overnight.finalize()
        return result
