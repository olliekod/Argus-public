# Created by Oliver Meihls

# Base Detector Class
#
# Abstract base class for all opportunity detectors.

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, Optional
import logging

from ..core.events import (
    BarEvent,
    SignalEvent,
    Priority,
    TOPIC_MARKET_BARS,
    TOPIC_SIGNALS,
)

# Instrument allowlist — only these may appear as SignalEvent.symbol
TRADEABLE_INSTRUMENTS = frozenset({"IBIT", "BITO", "SPY", "QQQ", "IWM"})


class BaseDetector(ABC):
    # Abstract base class for opportunity detectors.
    #
    # All detectors must implement:
    # - analyze(): Check for opportunities
    # - calculate_edge(): Calculate net edge after costs
    #
    # Bus integration (optional):
    # - Call ``attach_bus(bus)`` to subscribe to ``market.bars`` and
    # auto-publish ``SignalEvent`` on ``signals.detections``.
    # - Override ``on_bar(event)`` for bar-driven analysis.

    def __init__(self, config: Dict[str, Any], db):
        # Initialize detector.
        #
        # Args:
        # config: Detector-specific configuration (from thresholds.yaml)
        # db: Database instance for logging
        self.config = config
        self.db = db
        self.enabled = config.get('enabled', True)
        self.name = self.__class__.__name__
        self.logger = logging.getLogger(f'argus.detectors.{self.name}')

        # Cost assumptions
        self.slippage_bps = config.get('slippage_bps', 5)
        self.fee_bps = config.get('fee_bps', 5)

        # Event bus reference (set by attach_bus)
        self._event_bus = None
        self._activity_callback = None

    # ── bus wiring ──────────────────────────────────────────

    def attach_bus(self, bus) -> None:
        # Subscribe this detector to ``market.bars`` on *bus*.
        self._event_bus = bus
        bus.subscribe(TOPIC_MARKET_BARS, self._bus_on_bar)
        self.logger.info("Attached to event bus (market.bars)")

    def set_activity_callback(self, callback) -> None:
        # Provide a callback for activity tracking (detector_name, event_ts, kind).
        self._activity_callback = callback

    def _bus_on_bar(self, event: BarEvent) -> None:
        # Internal handler invoked by the bus worker thread.
        if self._activity_callback:
            self._activity_callback(self.name, event.event_ts, "bar")
        try:
            self.on_bar(event)
        except Exception:
            self.logger.exception("on_bar error for %s", event.symbol)

    def on_bar(self, event: BarEvent) -> None:
        # Override in subclass for bar-driven analysis.
        #
        # Default implementation is a no-op so detectors that rely
        # exclusively on the existing ``analyze()`` path keep working.

        pass

    def _publish_signal(
        self,
        symbol: str,
        signal_type: str,
        priority: Priority,
        data: Dict[str, Any],
    ) -> None:
        # Convenience: publish a :class:`SignalEvent` if bus is attached.
        #
        # Enforces the instrument allowlist — signals for non-tradeable
        # assets are rejected with a warning.
        if symbol not in TRADEABLE_INSTRUMENTS:
            self.logger.warning(
                "Signal for non-tradeable %s suppressed (allowlist: %s)",
                symbol,
                TRADEABLE_INSTRUMENTS,
            )
            return
        if self._event_bus is None:
            return
        import time
        sig = SignalEvent(
            detector=self.name,
            symbol=symbol,
            signal_type=signal_type,
            priority=priority,
            timestamp=time.time(),
            data=data,
        )
        self._event_bus.publish(TOPIC_SIGNALS, sig)
    
    @abstractmethod
    async def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict]:
        # Analyze market data for opportunities.
        #
        # Args:
        # market_data: Current market state
        #
        # Returns:
        # Detection dict if opportunity found, None otherwise
        pass
    
    @abstractmethod
    def calculate_edge(self, detection: Dict) -> float:
        # Calculate net edge after trading costs.
        #
        # Args:
        # detection: Detection data
        #
        # Returns:
        # Net edge in basis points
        pass
    
    def should_trigger_entry(self, detection: Dict) -> bool:
        # Determine if this detection should trigger a simulated trade.
        #
        # Args:
        # detection: Detection data
        #
        # Returns:
        # True if should trigger entry
        min_edge = self.config.get('min_edge_after_fees_bps', 10)
        return detection.get('net_edge_bps', 0) >= min_edge
    
    def calculate_position_size(self, capital: float) -> float:
        # Calculate position size based on config.
        #
        # Args:
        # capital: Total capital
        #
        # Returns:
        # Position size in USD
        size_percent = self.config.get('position_size_percent', 10)
        max_leverage = self.config.get('max_leverage', 3)
        
        base_size = capital * (size_percent / 100)
        return base_size * max_leverage
    
    def calculate_stops(
        self,
        entry_price: float,
        is_long: bool
    ) -> Dict[str, float]:
        # Calculate stop loss and take profit levels.
        #
        # Args:
        # entry_price: Entry price
        # is_long: True for long, False for short
        #
        # Returns:
        # Dict with stop_loss and take_profit
        stop_percent = self.config.get('stop_loss_percent', 1.5) / 100
        tp_percent = self.config.get('take_profit_percent', 1.0) / 100
        
        if is_long:
            stop_loss = entry_price * (1 - stop_percent)
            take_profit = entry_price * (1 + tp_percent)
        else:
            stop_loss = entry_price * (1 + stop_percent)
            take_profit = entry_price * (1 - tp_percent)
        
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
        }
    
    async def log_detection(self, detection: Dict) -> int:
        # Log detection to database.
        #
        # Args:
        # detection: Detection data
        #
        # Returns:
        # Database ID of inserted record
        detection['timestamp'] = datetime.now(timezone.utc).isoformat()
        detection_id = await self.db.insert_detection(detection)
        self.logger.info(
            f"Detection logged: {detection.get('opportunity_type')} - "
            f"{detection.get('asset')} - Edge: {detection.get('net_edge_bps', 0):.1f} bps"
        )
        return detection_id
    
    def create_detection(
        self,
        opportunity_type: str,
        asset: str,
        exchange: str,
        detection_data: Dict,
        **kwargs
    ) -> Dict:
        # Create a standardized detection dict.
        #
        # Args:
        # opportunity_type: Type of opportunity
        # asset: Asset symbol
        # exchange: Exchange name
        # detection_data: Type-specific data
        # **kwargs: Additional fields
        #
        # Returns:
        # Detection dict ready for database
        # Calculate edge
        raw_edge_bps = kwargs.get('estimated_edge_bps', 0)
        net_edge_bps = self.calculate_edge({'raw_edge_bps': raw_edge_bps})
        
        detection = {
            'opportunity_type': opportunity_type,
            'asset': asset,
            'exchange': exchange,
            'detection_data': detection_data,
            'estimated_edge_bps': raw_edge_bps,
            'estimated_slippage_bps': self.slippage_bps,
            'estimated_fees_bps': self.fee_bps,
            'net_edge_bps': net_edge_bps,
            'alert_tier': kwargs.get('alert_tier', 2),
            **kwargs
        }
        
        # Check if should trigger
        detection['would_trigger_entry'] = self.should_trigger_entry(detection)
        
        return detection
