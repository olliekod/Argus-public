"""
High-Volatility Credit Strategy
===============================

Replay strategy that sells premium when vol regime is elevated (VOL_SPIKE or VOL_HIGH).
Thrives in high IV environments where VRP would block.

Config keys (passed via thresholds dict):
  min_iv: float       Minimum IV to trade (default 0.15 = 15%)
  allowed_vol_regimes: list  Vol regimes to trade in (default VOL_SPIKE, VOL_HIGH)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.analysis.replay_harness import ReplayStrategy, TradeIntent
from src.core.outcome_engine import BarData, OutcomeResult
from src.strategies.vrp_credit_spread import _select_iv_from_snapshots

logger = logging.getLogger("argus.strategies.high_vol_credit")


class HighVolCreditStrategy(ReplayStrategy):
    """Sell credit when vol regime is elevated (VOL_SPIKE, VOL_HIGH).

    Does not require VRP > 0; instead seeks high IV environments.
    """

    def __init__(self, thresholds: Optional[Dict[str, Any]] = None):
        self._thresholds = thresholds or {
            "min_iv": 0.12,
            "allowed_vol_regimes": ["VOL_SPIKE", "VOL_HIGH"],
        }
        self.last_iv: Optional[float] = None
        self._logged_gating = False

    @property
    def strategy_id(self) -> str:
        return "HIGH_VOL_CREDIT_V1"

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
        self.visible_regimes = visible_regimes or {}
        iv = _select_iv_from_snapshots(visible_snapshots or [])
        if iv is not None:
            self.last_iv = iv

    def generate_intents(self, sim_ts_ms: int) -> List[TradeIntent]:
        intents: List[TradeIntent] = []
        if self.last_iv is None:
            return intents

        symbol_regime = self.visible_regimes.get("SPY", {})
        vol = symbol_regime.get("vol_regime", "UNKNOWN")
        trend = symbol_regime.get("trend_regime", "UNKNOWN")

        allowed = self._thresholds.get("allowed_vol_regimes", ["VOL_SPIKE", "VOL_HIGH"])
        min_iv = float(self._thresholds.get("min_iv", 0.15))

        if vol not in allowed:
            return intents
        if trend == "TREND_DOWN":
            return intents
        if self.last_iv < min_iv:
            if not self._logged_gating:
                self._logged_gating = True
                logger.warning(
                    "HighVolCredit: vol=%s but iv=%.4f < min_iv=%.2f",
                    vol, self.last_iv, min_iv,
                )
            return intents

        intents.append(TradeIntent(
            symbol="SPY",
            side="SELL",
            quantity=1,
            intent_type="OPEN",
            tag="HIGH_VOL_CREDIT",
            meta={
                "iv": self.last_iv,
                "vol_regime": vol,
                "trend_regime": trend,
            },
        ))
        return intents
