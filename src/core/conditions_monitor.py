"""
Market Conditions Monitor
=========================

Synthesizes all data sources into a single "warmth" score (1-10)
for proactive opportunity alerts.

Inputs:
- BTC IV (Deribit) → elevated = opportunity
- Funding rate (Bybit) → extremes = contrarian signal
- BTC price momentum (Bybit) → fear/relief context

Output:
- Composite score 1-10
- Warmth label: cooling, neutral, warming, prime
- Specific implications for IBIT/BITO trading
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConditionsSnapshot:
    """Point-in-time market conditions."""
    timestamp: datetime
    score: int  # 1-10
    label: str  # cooling, neutral, warming, prime
    
    # Components
    btc_iv: float  # Current ATM IV %
    btc_iv_percentile: float  # IV percentile (0-100)
    funding_rate: float  # Current funding rate
    funding_zscore: float  # How extreme is funding
    btc_price: float
    btc_change_24h: float
    btc_change_5d: float
    
    # Market status
    market_open: bool
    
    # Derived signals
    iv_signal: str  # low, normal, elevated, high
    funding_signal: str  # shorts_pay, neutral, longs_pay
    momentum_signal: str  # bearish, neutral, bullish
    
    # Implication
    implication: str


class ConditionsMonitor:
    """
    Synthesizes market data into actionable conditions score.
    
    Runs every 30 minutes and alerts on threshold crossings.
    """
    
    # Warmth thresholds
    COOLING_MAX = 3
    NEUTRAL_MAX = 5
    WARMING_MAX = 7
    PRIME_MIN = 8
    
    # Alert thresholds (score must cross these to trigger)
    ALERT_THRESHOLDS = [4, 6, 8]
    
    def __init__(
        self,
        config: Dict[str, Any] = None,
        on_alert: Optional[Callable] = None,
    ):
        """
        Initialize conditions monitor.
        
        Args:
            config: Configuration dict from thresholds.yaml
            on_alert: Callback for sending alerts
        """
        config = config or {}
        
        self.poll_interval = config.get('poll_interval_minutes', 30) * 60
        self.alert_callback = on_alert
        
        # IV thresholds (for IBIT put spread strategy)
        self.iv_low = config.get('iv_low_threshold', 40)
        self.iv_elevated = config.get('iv_elevated_threshold', 55)
        self.iv_high = config.get('iv_high_threshold', 70)
        
        # Funding thresholds
        self.funding_extreme = config.get('funding_extreme_threshold', 0.03)  # 0.03%
        
        # State
        self._last_score: int = 5
        self._last_snapshot: Optional[ConditionsSnapshot] = None
        self._running = False
        
        # Data sources (set by orchestrator)
        self._get_btc_iv: Optional[Callable] = None
        self._get_funding: Optional[Callable] = None
        self._get_btc_price: Optional[Callable] = None
        self._get_risk_flow: Optional[Callable] = None
        
        logger.info("Conditions Monitor initialized")
    
    def set_data_sources(
        self,
        get_btc_iv: Optional[Callable] = None,
        get_funding: Optional[Callable] = None,
        get_btc_price: Optional[Callable] = None,
        get_risk_flow: Optional[Callable] = None,
    ):
        """Set callbacks for fetching data from connectors."""
        self._get_btc_iv = get_btc_iv
        self._get_funding = get_funding
        self._get_btc_price = get_btc_price
        self._get_risk_flow = get_risk_flow
    
    def _is_market_open(self) -> bool:
        """Check if US stock market is currently open."""
        eastern = ZoneInfo("America/New_York")
        now = datetime.now(eastern)

        # Market hours: 9:30 AM - 4:00 PM ET (weekdays only)
        if now.weekday() >= 5:  # Saturday or Sunday
            return False

        if now.hour < 9 or now.hour >= 16:
            return False
        if now.hour == 9 and now.minute < 30:
            return False

        return True
    
    def _calculate_iv_signal(self, iv: float) -> Tuple[str, int]:
        """Calculate IV signal and score contribution."""
        if iv >= self.iv_high:
            return "high", 3  # Great for selling premium
        elif iv >= self.iv_elevated:
            return "elevated", 2
        elif iv >= self.iv_low:
            return "normal", 1
        else:
            return "low", 0  # Bad for selling premium
    
    def _calculate_funding_signal(self, rate: float) -> Tuple[str, int]:
        """Calculate funding signal and score contribution."""
        if rate <= -self.funding_extreme:
            return "shorts_pay", 2  # Contrarian bullish
        elif rate >= self.funding_extreme:
            return "longs_pay", 2  # Crowded longs, potential for squeeze
        else:
            return "neutral", 1
    
    def _calculate_momentum_signal(self, change_24h: float) -> Tuple[str, int]:
        """Calculate momentum signal and score contribution."""
        if change_24h <= -3:
            return "bearish", 2  # Good for put spreads (collecting premium on fear)
        elif change_24h >= 3:
            return "bullish", 1  # Less ideal, but IV often elevated
        else:
            return "neutral", 1
    
    def _get_warmth_label(self, score: int) -> str:
        """Convert score to label."""
        if score <= self.COOLING_MAX:
            return "cooling"
        elif score <= self.NEUTRAL_MAX:
            return "neutral"
        elif score <= self.WARMING_MAX:
            return "warming"
        else:
            return "prime"
    
    def _generate_implication(
        self,
        score: int,
        iv_signal: str,
        funding_signal: str,
        momentum_signal: str,
        market_open: bool,
    ) -> str:
        """Generate human-readable implication."""
        if score >= 8:
            if market_open:
                return "Prime conditions for put spreads. Check IBIT/BITO now."
            else:
                return "Prime conditions building. Watch for entries at market open."
        elif score >= 6:
            return "Conditions improving. Monitor for further warming."
        elif score >= 4:
            return "Neutral conditions. No immediate action needed."
        else:
            reasons = []
            if iv_signal == "low":
                reasons.append("IV too low for premium selling")
            if momentum_signal == "bullish":
                reasons.append("Market trending up, less fear premium")
            return "Conditions unfavorable. " + (", ".join(reasons) if reasons else "Wait for better setup.")
    
    async def calculate_conditions(self) -> ConditionsSnapshot:
        """Calculate current market conditions."""
        now = datetime.now(timezone.utc)
        market_open = self._is_market_open()
        
        # Gather data from sources
        btc_iv = 50.0  # Default
        if self._get_btc_iv:
            try:
                iv_data = await self._get_btc_iv()
                btc_iv = iv_data.get('atm_iv', 50.0) if iv_data else 50.0
            except Exception as e:
                logger.warning(f"Failed to get BTC IV: {e}")
        
        funding_rate = 0.01  # Default (0.01%)
        if self._get_funding:
            try:
                funding_data = await self._get_funding()
                funding_rate = funding_data.get('rate', 0.01) * 100 if funding_data else 0.01
            except Exception as e:
                logger.warning(f"Failed to get funding: {e}")
        
        btc_price = 95000.0  # Default
        btc_change = 0.0
        btc_change_5d = 0.0
        if self._get_btc_price:
            try:
                price_data = await self._get_btc_price()
                if price_data:
                    btc_price = price_data.get('price', 95000)
                    btc_change = price_data.get('change_24h_pct', 0)
                    btc_change_5d = price_data.get('change_5d_pct', 0)
            except Exception as e:
                logger.warning(f"Failed to get BTC price: {e}")
        
        # Calculate signals
        iv_signal, iv_score = self._calculate_iv_signal(btc_iv)
        funding_signal, funding_score = self._calculate_funding_signal(funding_rate)
        momentum_signal, momentum_score = self._calculate_momentum_signal(btc_change)
        # Market open bonus
        market_bonus = 1 if market_open else 0
        
        # Calculate total score (1-10 scale)
        raw_score = iv_score + funding_score + momentum_score + market_bonus
        score = min(10, max(1, raw_score))
        
        # Get label and implication
        label = self._get_warmth_label(score)
        implication = self._generate_implication(
            score, iv_signal, funding_signal, momentum_signal, market_open
        )
        
        # Calculate IV percentile (simplified - based on thresholds)
        if btc_iv >= self.iv_high:
            iv_percentile = 90
        elif btc_iv >= self.iv_elevated:
            iv_percentile = 70
        elif btc_iv >= self.iv_low:
            iv_percentile = 50
        else:
            iv_percentile = 25
        
        # Calculate funding z-score (simplified)
        funding_zscore = funding_rate / 0.01  # Normalize to typical 0.01%
        
        snapshot = ConditionsSnapshot(
            timestamp=now,
            score=score,
            label=label,
            btc_iv=btc_iv,
            btc_iv_percentile=iv_percentile,
            funding_rate=funding_rate,
            funding_zscore=funding_zscore,
            btc_price=btc_price,
            btc_change_24h=btc_change,
            btc_change_5d=btc_change_5d,
            market_open=market_open,
            iv_signal=iv_signal,
            funding_signal=funding_signal,
            momentum_signal=momentum_signal,
            implication=implication,
        )
        
        self._last_snapshot = snapshot
        return snapshot
    
    async def check_and_alert(self) -> Optional[ConditionsSnapshot]:
        """Check conditions and send alert if threshold crossed."""
        snapshot = await self.calculate_conditions()
        
        old_score = self._last_score
        new_score = snapshot.score
        
        # Check for significant state changes
        should_alert = False
        
        if self._last_snapshot:
            old_label = self._last_snapshot.label
            new_label = snapshot.label
            
            # 1. Always alert on entering PRIME conditions
            if new_label == "prime" and old_label != "prime":
                should_alert = True
                
            # 2. Alert on WARMING (but not if just oscillating prime<->warming)
            elif new_label == "warming" and old_label in ["neutral", "cooling"]:
                should_alert = True
                
            # 3. Suppress cooling/neutral alerts to reduce noise
            # (User requested "High-Signal" only)
            
        self._last_score = new_score
        
        if should_alert and self.alert_callback:
            try:
                await self.alert_callback(snapshot)
            except Exception as e:
                logger.error(f"Failed to send conditions alert: {e}")
        
        return snapshot
    
    async def get_current_conditions(self) -> Dict[str, Any]:
        """Get current conditions as dict (for /status command)."""
        if self._last_snapshot is None:
            await self.calculate_conditions()
        
        s = self._last_snapshot
        eastern = ZoneInfo("America/New_York")
        market_time = datetime.now(eastern).strftime("%H:%M:%S %Z")
        updated_et = s.timestamp.replace(tzinfo=timezone.utc).astimezone(eastern)
        res = {
            'score': s.score,
            'warmth_label': s.label,
            'btc_iv': f"{s.btc_iv:.0f}",
            'iv_rank': f"{s.btc_iv_percentile:.0f}",
            'funding': f"{s.funding_rate:+.3f}%",
            'market_open': s.market_open,
            'implication': s.implication,
            'btc_price': s.btc_price,
            'btc_change': s.btc_change_24h,
            'btc_change_5d': s.btc_change_5d,
            'market_time_et': market_time,
            'last_updated': s.timestamp.isoformat(),
            'last_updated_et': updated_et.strftime("%Y-%m-%d %H:%M:%S %Z"),
            'risk_flow': "N/A",
        }
        if self._get_risk_flow:
            try:
                val = self._get_risk_flow()
                if val is not None:
                    res['risk_flow'] = "risk-on" if val > 0 else ("risk-off" if val < 0 else "neutral")
            except Exception:
                pass
        return res
    
    async def start_monitoring(self) -> None:
        """Start the monitoring loop."""
        self._running = True
        logger.info(f"Starting conditions monitoring (interval: {self.poll_interval}s)")
        
        # Initial check
        await self.check_and_alert()
        
        while self._running:
            await asyncio.sleep(self.poll_interval)
            if self._running:
                try:
                    await self.check_and_alert()
                except Exception as e:
                    logger.error(f"Conditions check error: {e}")
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring loop."""
        self._running = False
        logger.info("Conditions monitoring stopped")


# Self-test
async def test_conditions_monitor():
    """Test the conditions monitor."""
    print("Conditions Monitor Test")
    print("=" * 40)
    
    monitor = ConditionsMonitor()
    
    # Test with mock data sources
    async def mock_iv():
        return {'atm_iv': 65.0}
    
    async def mock_funding():
        return {'rate': -0.0005}  # -0.05% (shorts paying)
    
    async def mock_btc_price():
        return {'price': 97500, 'change_24h_pct': -2.1}
    
    monitor.set_data_sources(
        get_btc_iv=mock_iv,
        get_funding=mock_funding,
        get_btc_price=mock_btc_price,
    )
    
    snapshot = await monitor.calculate_conditions()
    
    print(f"Score: {snapshot.score}/10 ({snapshot.label})")
    print(f"BTC IV: {snapshot.btc_iv}% ({snapshot.iv_signal})")
    print(f"Funding: {snapshot.funding_rate:+.3f}% ({snapshot.funding_signal})")
    print(f"BTC Change: {snapshot.btc_change_24h:+.1f}%")
    print(f"BTC: ${snapshot.btc_price:,.0f} ({snapshot.btc_change_24h:+.1f}%)")
    print(f"Market: {'OPEN' if snapshot.market_open else 'CLOSED'}")
    print()
    print(f"Implication: {snapshot.implication}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_conditions_monitor())
