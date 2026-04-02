# Created by Oliver Meihls

# Market-level risk regime scaffold.
#
# Disabled by default (empty risk basket). Emits MarketRegimeEvent on
# `regimes.market` only when enabled via `risk_basket_symbols`.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .bus import EventBus
from .events import TOPIC_REGIMES_MARKET, TOPIC_REGIMES_SYMBOL
from .regimes import MarketRegimeEvent, SymbolRegimeEvent, compute_config_hash, canonical_metrics_json


@dataclass
class MarketRegimeConfig:
    risk_basket_symbols: List[str]


class MarketRegimeDetector:
    def __init__(self, bus: EventBus, risk_basket_symbols: Optional[List[str]] = None) -> None:
        self._bus = bus
        self._basket = [s.upper() for s in (risk_basket_symbols or [])]
        self._enabled = len(self._basket) > 0
        self._config_hash = compute_config_hash({"risk_basket_symbols": self._basket})
        self._latest: Dict[str, SymbolRegimeEvent] = {}
        bus.subscribe(TOPIC_REGIMES_SYMBOL, self._on_symbol_regime)

    def _on_symbol_regime(self, event: SymbolRegimeEvent) -> None:
        if not self._enabled:
            return
        sym = event.symbol.upper()
        if sym not in self._basket:
            return
        self._latest[sym] = event

        missing = [s for s in self._basket if s not in self._latest]
        if missing:
            self._emit(event, "UNKNOWN", 0.0, {"missing_symbols": missing})
            return

        risk_votes = []
        metrics = {}

        if "SPY" in self._latest:
            spy = self._latest["SPY"]
            metrics["spy_trend"] = spy.trend_regime
            metrics["spy_vol"] = spy.vol_regime
            if spy.trend_regime == "TREND_UP" and spy.vol_regime in {"VOL_LOW", "VOL_NORMAL"}:
                risk_votes.append(1)
            elif spy.trend_regime == "TREND_DOWN" or spy.vol_regime == "VOL_SPIKE":
                risk_votes.append(-1)
            else:
                risk_votes.append(0)

        if "DIA" in self._latest:
            dia = self._latest["DIA"]
            metrics["dia_trend"] = dia.trend_regime
            metrics["dia_vol"] = dia.vol_regime
            if dia.trend_regime == "TREND_UP" and dia.vol_regime in {"VOL_LOW", "VOL_NORMAL"}:
                risk_votes.append(1)
            elif dia.trend_regime == "TREND_DOWN" or dia.vol_regime == "VOL_SPIKE":
                risk_votes.append(-1)
            else:
                risk_votes.append(0)

        for defensive in ("TLT", "GLD"):
            if defensive in self._latest:
                d = self._latest[defensive]
                metrics[f"{defensive.lower()}_trend"] = d.trend_regime
                if d.trend_regime == "TREND_UP":
                    risk_votes.append(-1)
                elif d.trend_regime == "TREND_DOWN":
                    risk_votes.append(1)
                else:
                    risk_votes.append(0)

        if not risk_votes:
            self._emit(event, "UNKNOWN", 0.0, {"reason": "no_supported_signals"})
            return

        score = sum(risk_votes)
        if score > 0:
            risk = "RISK_ON"
        elif score < 0:
            risk = "RISK_OFF"
        else:
            risk = "NEUTRAL"
        conf = min(1.0, abs(score) / max(1, len(risk_votes)))
        metrics["score"] = score
        self._emit(event, risk, conf, metrics)

    def _emit(self, event: SymbolRegimeEvent, risk: str, confidence: float, metrics: Dict[str, object]) -> None:
        out = MarketRegimeEvent(
            market="GLOBAL",
            timeframe=event.timeframe,
            timestamp_ms=event.timestamp_ms,
            session_regime="UNKNOWN",
            risk_regime=risk,
            confidence=confidence,
            data_quality_flags=event.data_quality_flags,
            config_hash=self._config_hash,
            metrics_json=canonical_metrics_json(metrics),
        )
        self._bus.publish(TOPIC_REGIMES_MARKET, out)
