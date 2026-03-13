"""
Volatility Regime Detector
==========================

Detects sudden volatility expansion or compression.
"""

from typing import Any, Dict, List, Optional

from .base_detector import BaseDetector
from ..core.events import BarEvent
from ..core.utils import calculate_volatility


class VolatilityDetector(BaseDetector):
    """
    Detector for volatility regime changes.
    
    Monitors realized volatility and detects:
    - Expansion: Vol spikes to 2x+ normal
    - Compression: Vol drops to 0.5x normal
    
    Provides context for other strategies rather than
    standalone trades.
    """
    
    def __init__(self, config: Dict[str, Any], db):
        super().__init__(config, db)
        
        # Windows
        self.short_window = config.get('short_window_hours', 1)
        self.long_window = config.get('long_window_hours', 24)
        
        # Thresholds
        self.expansion_threshold = config.get('expansion_threshold', 2.0)
        self.compression_threshold = config.get('compression_threshold', 0.5)
        self.min_observations = config.get('min_observations', 60)
        
        # Price cache for volatility calculation
        self._price_history: Dict[str, List[Dict]] = {}
        
        # Current regime
        self._current_regime: Dict[str, str] = {}  # 'expansion', 'normal', 'compression'
        
        self.logger.info(
            f"VolatilityDetector initialized: expansion>{self.expansion_threshold}x, "
            f"compression<{self.compression_threshold}x"
        )
    
    def update_price(self, symbol: str, price: float, timestamp: str) -> None:
        """Add price point to history."""
        if symbol not in self._price_history:
            self._price_history[symbol] = []
        
        self._price_history[symbol].append({
            'price': price,
            'timestamp': timestamp
        })
        
        # Keep 48 hours of 1-minute data
        max_entries = 48 * 60
        if len(self._price_history[symbol]) > max_entries:
            self._price_history[symbol] = self._price_history[symbol][-max_entries:]
    
    async def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict]:
        """
        Analyze for volatility regime changes.
        
        Args:
            market_data: Price update data
            
        Returns:
            Detection if regime change detected
        """
        if not self.enabled:
            return None
        
        symbol = market_data.get('symbol', '')
        price = market_data.get('last_price', 0) or market_data.get('mark_price', 0)
        timestamp = market_data.get('timestamp', '')
        exchange = market_data.get('exchange', 'unknown')
        
        if price == 0:
            return None
        
        # Update price history
        self.update_price(symbol, price, timestamp)
        
        # Check if enough data
        history = self._price_history.get(symbol, [])
        
        long_window_minutes = self.long_window * 60
        if len(history) < long_window_minutes:
            return None
        
        # Calculate short and long volatility
        short_prices = [h['price'] for h in history[-(self.short_window * 60):]]
        long_prices = [h['price'] for h in history[-(self.long_window * 60):]]
        
        if len(short_prices) < self.min_observations:
            return None
        
        short_vol = calculate_volatility(short_prices, period_minutes=1, annualize=True)
        long_vol = calculate_volatility(long_prices, period_minutes=1, annualize=True)
        
        if long_vol == 0:
            return None
        
        # Calculate ratio
        vol_ratio = short_vol / long_vol
        
        # Determine regime
        if vol_ratio >= self.expansion_threshold:
            new_regime = 'expansion'
        elif vol_ratio <= self.compression_threshold:
            new_regime = 'compression'
        else:
            new_regime = 'normal'
        
        # Check for regime change
        old_regime = self._current_regime.get(symbol, 'normal')
        
        if new_regime == old_regime:
            return None  # No change
        
        # Regime changed!
        self._current_regime[symbol] = new_regime
        
        detection = self.create_detection(
            opportunity_type='volatility',
            asset=symbol,
            exchange=exchange,
            detection_data={
                'regime': new_regime,
                'previous_regime': old_regime,
                'short_vol': short_vol,
                'long_vol': long_vol,
                'vol_ratio': vol_ratio,
                'short_window_hours': self.short_window,
                'long_window_hours': self.long_window,
            },
            current_price=price,
            volatility_1h=short_vol,
            volatility_24h=long_vol,
            estimated_edge_bps=0,  # Informational only
            alert_tier=3,  # Background info
            notes=f"Volatility regime: {old_regime} → {new_regime}"
        )
        
        # Don't trigger trades - informational only
        detection['would_trigger_entry'] = False
        
        await self.log_detection(detection)
        
        return detection
    
    def calculate_edge(self, detection: Dict) -> float:
        """Not applicable - informational detector."""
        return 0
    
    def get_current_regime(self, symbol: str) -> str:
        """Get current volatility regime for a symbol."""
        return self._current_regime.get(symbol, 'unknown')
    
    # ── bar-driven path (via event bus) ───────────────────

    def on_bar(self, event: BarEvent) -> None:
        """Ingest a 1-minute bar as a price update for vol calculation."""
        self.update_price(event.symbol, event.close, str(event.timestamp))

    def get_volatility(self, symbol: str) -> Dict[str, float]:
        """Get current volatility readings."""
        history = self._price_history.get(symbol, [])
        
        if len(history) < self.min_observations:
            return {'short': 0, 'long': 0, 'ratio': 0}
        
        short_prices = [h['price'] for h in history[-(self.short_window * 60):]]
        long_prices = [h['price'] for h in history[-(self.long_window * 60):]]
        
        short_vol = calculate_volatility(short_prices, period_minutes=1, annualize=True)
        long_vol = calculate_volatility(long_prices, period_minutes=1, annualize=True)
        
        return {
            'short': short_vol,
            'long': long_vol,
            'ratio': short_vol / long_vol if long_vol > 0 else 0
        }
