"""
Strategy Base Classes
=====================

Provides the abstract base class for all strategy modules.
Strategies are bar-clocked, deterministic, and emit SignalEvents.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ..core.events import BarEvent, TOPIC_MARKET_BARS, TOPIC_SIGNALS_RAW
from ..core.regimes import SymbolRegimeEvent, MarketRegimeEvent
from ..core.signals import (
    SignalEvent,
    compute_config_hash,
    compute_signal_id,
    DIRECTION_LONG,
    DIRECTION_SHORT,
    SIGNAL_TYPE_ENTRY,
)

logger = logging.getLogger("argus.strategies.base")


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    Strategies must be:
    - **Bar-clocked**: Only evaluate when a BarEvent arrives
    - **Deterministic**: Same bars + regimes → same signals
    - **Config-versioned**: Include config_hash in all signals
    
    Subclasses must implement:
    - `strategy_id` property
    - `default_config` property
    - `evaluate()` method
    """
    
    def __init__(
        self,
        bus,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._bus = bus
        self._config = config or self.default_config
        self._config_hash = compute_config_hash(self._config)
        
        # Per-symbol state (subclasses can extend)
        self._symbol_states: Dict[str, Dict[str, Any]] = {}
        
        # Latest regime events (cached for evaluation)
        self._symbol_regimes: Dict[str, SymbolRegimeEvent] = {}
        self._market_regimes: Dict[str, MarketRegimeEvent] = {}
        
        # Telemetry
        self._bars_processed = 0
        self._signals_emitted = 0
        self._signals_suppressed = 0
        
        # Subscribe
        self._bus.subscribe(TOPIC_MARKET_BARS, self._on_bar)
        self._bus.subscribe("regimes.symbol", self._on_symbol_regime)
        self._bus.subscribe("regimes.market", self._on_market_regime)
        
        logger.info(
            "%s initialized — config_hash=%s",
            self.strategy_id, self._config_hash
        )
    
    # ─── Abstract Properties ─────────────────────────────────────────────────
    
    @property
    @abstractmethod
    def strategy_id(self) -> str:
        """Unique identifier for this strategy (e.g., 'FVG_BREAKOUT_V1')."""
        ...
    
    @property
    @abstractmethod
    def default_config(self) -> Dict[str, Any]:
        """Default configuration thresholds for this strategy."""
        ...
    
    # ─── Abstract Methods ────────────────────────────────────────────────────
    
    @abstractmethod
    def evaluate(
        self,
        bar: BarEvent,
        symbol_regime: Optional[SymbolRegimeEvent],
        market_regime: Optional[MarketRegimeEvent],
    ) -> Optional[SignalEvent]:
        """
        Evaluate the strategy for a given bar.
        
        Returns a SignalEvent if conditions are met, None otherwise.
        Must be deterministic.
        """
        ...
    
    # ─── Event Handlers ──────────────────────────────────────────────────────
    
    def _on_bar(self, bar: BarEvent) -> None:
        """Process incoming bar event."""
        self._bars_processed += 1
        
        # Get cached regimes
        symbol_regime = self._symbol_regimes.get(bar.symbol)
        market = self._get_market_for_symbol(bar.symbol)
        market_regime = self._market_regimes.get(market)
        
        # Evaluate strategy
        signal = self.evaluate(bar, symbol_regime, market_regime)
        
        if signal is not None:
            self._emit_signal(signal)
    
    def _on_symbol_regime(self, event: SymbolRegimeEvent) -> None:
        """Cache latest symbol regime."""
        self._symbol_regimes[event.symbol] = event
    
    def _on_market_regime(self, event: MarketRegimeEvent) -> None:
        """Cache latest market regime."""
        self._market_regimes[event.market] = event
    
    # ─── Signal Emission ─────────────────────────────────────────────────────
    
    def _emit_signal(self, signal: SignalEvent) -> None:
        """Publish signal to the bus."""
        self._signals_emitted += 1
        self._bus.publish(TOPIC_SIGNALS_RAW, signal)
        logger.debug(
            "[%s] Signal emitted: %s %s @ %d",
            self.strategy_id, signal.direction, signal.symbol, signal.timestamp_ms
        )
    
    # ─── Helper Methods ──────────────────────────────────────────────────────
    
    def _get_market_for_symbol(self, symbol: str) -> str:
        """Determine market for a symbol."""
        from ..core.regimes import get_market_for_symbol
        return get_market_for_symbol(symbol)
    
    def _create_signal(
        self,
        bar: BarEvent,
        direction: str,
        signal_type: str = SIGNAL_TYPE_ENTRY,
        entry_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        tp_price: Optional[float] = None,
        confidence: float = 1.0,
        data_quality_flags: int = 0,
        regime_snapshot: Optional[Dict[str, str]] = None,
        features_snapshot: Optional[Dict[str, float]] = None,
        explain: str = "",
    ) -> SignalEvent:
        """Create a SignalEvent with proper attribution."""
        timestamp_ms = int(bar.timestamp * 1000)
        
        return SignalEvent(
            timestamp_ms=timestamp_ms,
            strategy_id=self.strategy_id,
            config_hash=self._config_hash,
            symbol=bar.symbol,
            direction=direction,
            signal_type=signal_type,
            timeframe=bar.bar_duration,
            entry_type="MARKET" if entry_price is None else "LIMIT",
            entry_price=entry_price,
            stop_price=stop_price,
            tp_price=tp_price,
            confidence=confidence,
            data_quality_flags=data_quality_flags,
            regime_snapshot=regime_snapshot or {},
            features_snapshot=features_snapshot or {},
            explain=explain,
            idempotency_key=compute_signal_id(
                self.strategy_id,
                self._config_hash,
                bar.symbol,
                timestamp_ms,
            ),
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Return strategy telemetry."""
        return {
            "strategy_id": self.strategy_id,
            "config_hash": self._config_hash,
            "bars_processed": self._bars_processed,
            "signals_emitted": self._signals_emitted,
            "signals_suppressed": self._signals_suppressed,
            "cached_symbol_regimes": len(self._symbol_regimes),
            "cached_market_regimes": len(self._market_regimes),
        }
