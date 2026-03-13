"""
Sentiment Collector
===================

Aggregates sentiment data from multiple sources:
1. Crypto Fear & Greed Index (alternative.me - FREE)
2. Bitcoin/Crypto social trends
3. Options sentiment (put/call ratio)

Used to enhance trading signals.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from dataclasses import dataclass
import asyncio

try:
    import aiohttp
except ImportError:
    raise ImportError("Required: pip install aiohttp")

logger = logging.getLogger(__name__)


@dataclass
class SentimentSnapshot:
    """Current sentiment state."""
    timestamp: str
    
    # Fear & Greed (0-100)
    fear_greed_value: int
    fear_greed_label: str  # 'Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed'
    
    # Trend (compared to yesterday)
    fear_greed_trend: str  # 'rising', 'falling', 'stable'
    fear_greed_change: int
    
    # Signal interpretation
    signal: str  # 'bullish', 'bearish', 'neutral'
    signal_strength: int  # 1-5


class SentimentCollector:
    """
    Collects and interprets market sentiment.
    
    Primary source: Crypto Fear & Greed Index
    - 0-25: Extreme Fear (potential buying opportunity)
    - 25-45: Fear
    - 45-55: Neutral
    - 55-75: Greed
    - 75-100: Extreme Greed (potential selling/caution)
    
    For PUT SPREAD selling:
    - Extreme Fear = GOOD (high IV, oversold)
    - Extreme Greed = CAUTION (complacent, low IV)
    """
    
    # Fear & Greed API (free, no auth required)
    FEAR_GREED_URL = "https://api.alternative.me/fng/"
    
    # Thresholds
    EXTREME_FEAR = 25
    FEAR = 45
    GREED = 55
    EXTREME_GREED = 75
    
    def __init__(self):
        """Initialize sentiment collector."""
        self._cache: Optional[SentimentSnapshot] = None
        self._cache_time: Optional[datetime] = None
        self._cache_ttl = timedelta(minutes=30)  # Cache for 30 min
        
        logger.info("Sentiment Collector initialized")
    
    async def get_fear_greed(self, limit: int = 2) -> Dict:
        """
        Fetch Fear & Greed Index from alternative.me.
        
        Args:
            limit: Number of days to fetch (1 = today, 2 = today + yesterday)
            
        Returns:
            Raw API response
        """
        url = f"{self.FEAR_GREED_URL}?limit={limit}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.error(f"Fear & Greed API error: {response.status}")
                        return None
        except Exception as e:
            logger.error(f"Failed to fetch Fear & Greed: {e}")
            return None
    
    async def get_sentiment(self, force_refresh: bool = False) -> Optional[SentimentSnapshot]:
        """
        Get current sentiment snapshot.
        
        Uses caching to avoid hammering API.
        
        Args:
            force_refresh: Bypass cache
            
        Returns:
            SentimentSnapshot or None
        """
        # Check cache
        if not force_refresh and self._cache and self._cache_time:
            if datetime.now() - self._cache_time < self._cache_ttl:
                return self._cache
        
        # Fetch fresh data
        data = await self.get_fear_greed(limit=2)
        
        if not data or 'data' not in data:
            return self._cache  # Return stale cache if available
        
        entries = data['data']
        if not entries:
            return None
        
        # Current value
        current = entries[0]
        current_value = int(current['value'])
        current_label = current['value_classification']
        
        # Yesterday for trend
        if len(entries) > 1:
            yesterday = entries[1]
            yesterday_value = int(yesterday['value'])
            change = current_value - yesterday_value
            
            if change > 5:
                trend = 'rising'
            elif change < -5:
                trend = 'falling'
            else:
                trend = 'stable'
        else:
            change = 0
            trend = 'unknown'
        
        # Interpret for PUT SPREAD strategy
        signal, strength = self._interpret_for_puts(current_value, trend)
        
        snapshot = SentimentSnapshot(
            timestamp=datetime.now().isoformat(),
            fear_greed_value=current_value,
            fear_greed_label=current_label,
            fear_greed_trend=trend,
            fear_greed_change=change,
            signal=signal,
            signal_strength=strength,
        )
        
        # Cache
        self._cache = snapshot
        self._cache_time = datetime.now()
        
        logger.info(f"Sentiment: {current_value} ({current_label}), signal: {signal}")
        
        return snapshot
    
    def _interpret_for_puts(self, value: int, trend: str) -> tuple:
        """
        Interpret Fear & Greed for PUT SPREAD selling strategy.
        
        For selling puts/put spreads:
        - Extreme Fear = BULLISH (good entry, high IV)
        - Fear + Rising = BULLISH (recovery starting)
        - Neutral = NEUTRAL
        - Greed = BEARISH (cautious)
        - Extreme Greed = BEARISH (stay out, low premium)
        
        Returns:
            (signal, strength) where signal is 'bullish'/'bearish'/'neutral'
            and strength is 1-5
        """
        if value <= self.EXTREME_FEAR:
            # Extreme fear = great for selling puts
            if trend == 'rising':
                return 'bullish', 5  # Recovery starting, perfect
            else:
                return 'bullish', 4  # Still fearful, good IV
        
        elif value <= self.FEAR:
            # Fear = good
            if trend == 'rising':
                return 'bullish', 4
            else:
                return 'bullish', 3
        
        elif value <= self.GREED:
            # Neutral
            return 'neutral', 2
        
        elif value <= self.EXTREME_GREED:
            # Greed = cautious
            return 'bearish', 2
        
        else:
            # Extreme greed = stay out
            return 'bearish', 4
    
    def format_telegram(self, snapshot: SentimentSnapshot) -> str:
        """Format sentiment for Telegram notification."""
        # Signal emoji
        if snapshot.signal == 'bullish':
            emoji = '[+]' if snapshot.signal_strength >= 4 else '[.]'
        elif snapshot.signal == 'bearish':
            emoji = '[-]' if snapshot.signal_strength >= 3 else '[.]'
        else:
            emoji = '[=]'
        
        # Trend arrow
        if snapshot.fear_greed_trend == 'rising':
            arrow = '^'
        elif snapshot.fear_greed_trend == 'falling':
            arrow = 'v'
        else:
            arrow = '-'
        
        return f"""
SENTIMENT UPDATE

Fear & Greed: {snapshot.fear_greed_value}/100
Label: {snapshot.fear_greed_label}
Trend: {arrow} ({snapshot.fear_greed_change:+d} vs yesterday)

Signal: {emoji} {snapshot.signal.upper()}
Strength: {'*' * snapshot.signal_strength}{'.' * (5 - snapshot.signal_strength)}
"""


async def test_sentiment():
    """Test sentiment collector."""
    collector = SentimentCollector()
    
    print("Fetching sentiment...")
    snapshot = await collector.get_sentiment()
    
    if snapshot:
        print(collector.format_telegram(snapshot))
    else:
        print("Failed to fetch sentiment")


if __name__ == "__main__":
    asyncio.run(test_sentiment())
