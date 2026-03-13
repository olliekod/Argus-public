"""
Gap Risk Tracker
================

Tracks BTC price gaps during market-closed hours (nights, weekends, holidays).
Alerts when significant moves occur that could impact IBIT/BITO at market open.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# US Eastern timezone for market hours
ET = ZoneInfo("America/New_York")


@dataclass
class GapSnapshot:
    """Market close snapshot."""
    timestamp: str
    btc_price: float
    ibit_price: Optional[float]
    bito_price: Optional[float]
    market_date: str  # YYYY-MM-DD of trading day


@dataclass
class GapRiskStatus:
    """Current gap risk assessment."""
    btc_close_price: float
    btc_current_price: float
    gap_percent: float
    gap_direction: str  # 'up', 'down', 'flat'
    risk_level: str  # 'info', 'elevated', 'high', 'extreme'
    hours_since_close: float
    message: str


class GapRiskTracker:
    """
    Tracks BTC price gaps during off-market hours.
    
    Market hours: 9:30 AM - 4:00 PM ET, Monday-Friday
    Off-hours: 4:00 PM - 9:30 AM ET weekdays, all weekend
    
    Thresholds (higher risk tolerance):
    - 5%:  Info (log only)
    - 8%:  Elevated (Telegram FYI)
    - 12%: High (immediate alert)
    - 15%: Extreme (circuit breaker suggested)
    """
    
    def __init__(self, db, config: Dict = None):
        """
        Initialize gap risk tracker.
        
        Args:
            db: Database instance
            config: Gap risk configuration from thresholds.yaml
        """
        self.db = db
        config = config or {}
        
        # Thresholds (adjusted for higher risk tolerance)
        self.info_threshold = config.get('info_threshold_pct', 5)
        self.elevated_threshold = config.get('elevated_threshold_pct', 8)
        self.high_threshold = config.get('high_threshold_pct', 12)
        self.extreme_threshold = config.get('extreme_threshold_pct', 15)
        
        # Alert schedule
        self.sunday_alert_hour = config.get('sunday_alert_hour', 20)  # 8 PM ET
        self.monday_alert_hour = config.get('monday_alert_hour', 8)   # 8 AM ET
        
        # Position adjustment thresholds
        self.reduce_size_above = config.get('reduce_size_above_pct', 12)
        self.skip_trades_above = config.get('skip_trades_above_pct', 15)
        
        # Cache
        self._last_close_snapshot: Optional[GapSnapshot] = None
        
        logger.info(
            f"GapRiskTracker initialized: thresholds {self.info_threshold}/"
            f"{self.elevated_threshold}/{self.high_threshold}/{self.extreme_threshold}%"
        )
    
    async def initialize(self) -> None:
        """Create database table and load last snapshot."""
        await self._create_table()
        await self._load_last_snapshot()

    async def get_status(self) -> Dict:
        """Get current gap risk status for reporting."""
        current_price = await self._get_latest_btc_price()
        if current_price is None:
            return {
                'level': 'unknown',
                'message': 'BTC price unavailable',
            }
        status = self.calculate_gap(current_price)
        if not status:
            return {
                'level': 'low',
                'message': 'No market close snapshot available',
            }
        return {
            'level': status.risk_level,
            'gap_percent': status.gap_percent,
            'gap_direction': status.gap_direction,
            'hours_since_close': status.hours_since_close,
            'message': status.message,
        }
    
    async def _create_table(self) -> None:
        """Create market_close_snapshots table if not exists."""
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS market_close_snapshots (
                id INTEGER PRIMARY KEY,
                timestamp TEXT NOT NULL,
                btc_price REAL NOT NULL,
                ibit_price REAL,
                bito_price REAL,
                market_date TEXT NOT NULL,
                snapshot_type TEXT DEFAULT 'close'
            )
        """)
        
        # Create index for fast lookups
        await self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_close_snapshots_date 
            ON market_close_snapshots(market_date DESC)
        """)

    async def _get_latest_btc_price(self) -> Optional[float]:
        """Fetch the latest BTC price snapshot from the database."""
        row = await self.db.fetch_one("""
            SELECT price
            FROM price_snapshots
            WHERE exchange = 'bybit' AND asset = 'BTCUSDT' AND price_type = 'spot'
            ORDER BY timestamp DESC
            LIMIT 1
        """)
        return row[0] if row else None
    
    async def _load_last_snapshot(self) -> None:
        """Load most recent market close snapshot from database."""
        result = await self.db.fetch_one("""
            SELECT timestamp, btc_price, ibit_price, bito_price, market_date
            FROM market_close_snapshots
            WHERE snapshot_type = 'close'
            ORDER BY timestamp DESC
            LIMIT 1
        """)
        
        if result:
            self._last_close_snapshot = GapSnapshot(
                timestamp=result[0],
                btc_price=result[1],
                ibit_price=result[2],
                bito_price=result[3],
                market_date=result[4],
            )
            logger.info(
                f"Loaded last close snapshot: {self._last_close_snapshot.market_date}, "
                f"BTC=${self._last_close_snapshot.btc_price:,.0f}"
            )
    
    def is_market_open(self) -> bool:
        """Check if US stock market is currently open."""
        now = datetime.now(ET)
        
        # Weekend check
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Time check (9:30 AM - 4:00 PM ET)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    
    def get_market_status(self) -> Dict:
        """Get current market status with next open/close times."""
        now = datetime.now(ET)
        is_open = self.is_market_open()
        
        if is_open:
            # Market is open, next event is close
            next_event = "close"
            next_time = now.replace(hour=16, minute=0, second=0, microsecond=0)
        else:
            # Market closed, calculate next open
            next_event = "open"
            
            if now.weekday() == 4 and now.hour >= 16:
                # Friday after close -> Monday
                days_until = 3
            elif now.weekday() == 5:
                # Saturday -> Monday
                days_until = 2
            elif now.weekday() == 6:
                # Sunday -> Monday
                days_until = 1
            elif now.hour >= 16:
                # Weekday after close -> next day
                days_until = 1
            else:
                # Weekday before open -> today
                days_until = 0
            
            next_time = (now + timedelta(days=days_until)).replace(
                hour=9, minute=30, second=0, microsecond=0
            )
        
        return {
            'is_open': is_open,
            'next_event': next_event,
            'next_time': next_time.isoformat(),
            'current_time_et': now.isoformat(),
        }
    
    async def snapshot_market_close(
        self, 
        btc_price: float,
        ibit_price: Optional[float] = None,
        bito_price: Optional[float] = None,
    ) -> GapSnapshot:
        """
        Take a snapshot at market close (4:00 PM ET).
        
        Should be called automatically by orchestrator at 4 PM ET on weekdays.
        
        Args:
            btc_price: Current BTC price
            ibit_price: IBIT closing price (if available)
            bito_price: BITO closing price (if available)
            
        Returns:
            Created snapshot
        """
        now = datetime.now(ET)
        market_date = now.strftime('%Y-%m-%d')
        
        snapshot = GapSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            btc_price=btc_price,
            ibit_price=ibit_price,
            bito_price=bito_price,
            market_date=market_date,
        )
        
        # Save to database
        await self.db.execute("""
            INSERT INTO market_close_snapshots 
            (timestamp, btc_price, ibit_price, bito_price, market_date, snapshot_type)
            VALUES (?, ?, ?, ?, ?, 'close')
        """, (
            snapshot.timestamp,
            snapshot.btc_price,
            snapshot.ibit_price,
            snapshot.bito_price,
            snapshot.market_date,
        ))
        
        self._last_close_snapshot = snapshot
        
        logger.info(
            f"Market close snapshot saved: {market_date}, "
            f"BTC=${btc_price:,.0f}, IBIT=${ibit_price or 0:.2f}"
        )
        
        return snapshot
    
    def calculate_gap(self, current_btc_price: float) -> Optional[GapRiskStatus]:
        """
        Calculate the current gap from last market close.
        
        Args:
            current_btc_price: Current BTC price
            
        Returns:
            GapRiskStatus with gap analysis, or None if no snapshot exists
        """
        if not self._last_close_snapshot:
            logger.warning("No market close snapshot available")
            return None
        
        close_price = self._last_close_snapshot.btc_price
        gap_percent = ((current_btc_price - close_price) / close_price) * 100
        
        # Determine direction
        if gap_percent > 0.5:
            direction = 'up'
        elif gap_percent < -0.5:
            direction = 'down'
        else:
            direction = 'flat'
        
        # Determine risk level
        abs_gap = abs(gap_percent)
        if abs_gap >= self.extreme_threshold:
            risk_level = 'extreme'
        elif abs_gap >= self.high_threshold:
            risk_level = 'high'
        elif abs_gap >= self.elevated_threshold:
            risk_level = 'elevated'
        elif abs_gap >= self.info_threshold:
            risk_level = 'info'
        else:
            risk_level = 'minimal'
        
        # Calculate hours since close
        try:
            close_time = datetime.fromisoformat(self._last_close_snapshot.timestamp)
            if close_time.tzinfo is None:
                close_time = close_time.replace(tzinfo=timezone.utc)
            hours_since = (datetime.now(timezone.utc) - close_time).total_seconds() / 3600
        except:
            hours_since = 0
        
        # Build message
        message = self._build_message(gap_percent, direction, risk_level, hours_since)
        
        return GapRiskStatus(
            btc_close_price=close_price,
            btc_current_price=current_btc_price,
            gap_percent=gap_percent,
            gap_direction=direction,
            risk_level=risk_level,
            hours_since_close=hours_since,
            message=message,
        )
    
    def _build_message(
        self, 
        gap_percent: float, 
        direction: str, 
        risk_level: str,
        hours: float
    ) -> str:
        """Build human-readable gap message."""
        arrow = "📈" if direction == 'up' else "📉" if direction == 'down' else "➡️"
        
        if risk_level == 'extreme':
            emoji = "🔴"
            advice = "Consider skipping trades at Monday open. Very high gap risk."
        elif risk_level == 'high':
            emoji = "🚨"
            advice = "Reduce position size by 50%. Elevated gap risk."
        elif risk_level == 'elevated':
            emoji = "⚠️"
            advice = "Monitor closely. Some gap risk present."
        elif risk_level == 'info':
            emoji = "📊"
            advice = "Minor movement. Normal variance."
        else:
            emoji = "✅"
            advice = "No significant gap."
        
        period = "overnight" if hours < 16 else "weekend"
        
        return f"""{emoji} GAP RISK UPDATE

{arrow} BTC moved {gap_percent:+.1f}% {period}
• Close: ${self._last_close_snapshot.btc_price:,.0f}
• Now: ${self._last_close_snapshot.btc_price * (1 + gap_percent/100):,.0f}
• Risk Level: {risk_level.upper()}

💡 {advice}"""
    
    def should_alert(self, status: GapRiskStatus) -> Tuple[bool, str]:
        """
        Determine if an alert should be sent.
        
        Returns:
            (should_alert, alert_tier) where tier is 'info', 'warning', 'high', 'extreme'
        """
        if status.risk_level == 'minimal':
            return False, ''
        
        if status.risk_level == 'info':
            return False, 'info'  # Log only, no Telegram
        
        return True, status.risk_level
    
    def should_reduce_size(self, status: GapRiskStatus) -> bool:
        """Check if position size should be reduced based on gap."""
        return abs(status.gap_percent) >= self.reduce_size_above
    
    def should_skip_trades(self, status: GapRiskStatus) -> bool:
        """Check if trades should be skipped based on gap."""
        return abs(status.gap_percent) >= self.skip_trades_above
    
    def get_position_adjustment(self, status: GapRiskStatus) -> Dict:
        """Get recommended position adjustment based on gap."""
        if self.should_skip_trades(status):
            return {
                'action': 'skip',
                'size_multiplier': 0,
                'reason': f"Gap of {status.gap_percent:+.1f}% exceeds {self.skip_trades_above}% threshold"
            }
        elif self.should_reduce_size(status):
            return {
                'action': 'reduce',
                'size_multiplier': 0.5,
                'reason': f"Gap of {status.gap_percent:+.1f}% exceeds {self.reduce_size_above}% threshold"
            }
        else:
            return {
                'action': 'normal',
                'size_multiplier': 1.0,
                'reason': "Gap within acceptable range"
            }
    
    def format_telegram_alert(self, status: GapRiskStatus) -> str:
        """Format gap status for Telegram."""
        return status.message
    
    async def get_gap_history(self, days: int = 30) -> list:
        """Get historical gap data for analysis."""
        results = await self.db.fetch_all("""
            SELECT timestamp, btc_price, market_date
            FROM market_close_snapshots
            WHERE snapshot_type = 'close'
            ORDER BY timestamp DESC
            LIMIT ?
        """, (days,))
        
        return [
            {
                'timestamp': r[0],
                'btc_price': r[1],
                'market_date': r[2],
            }
            for r in results
        ]


async def test_gap_tracker():
    """Test the gap risk tracker."""
    print("Gap Risk Tracker Test")
    print("=" * 40)
    
    # Mock database for testing
    class MockDB:
        async def execute(self, *args): pass
        async def fetch_one(self, *args): return None
        async def fetch_all(self, *args): return []
    
    tracker = GapRiskTracker(MockDB())
    
    # Test market status
    status = tracker.get_market_status()
    print(f"Market open: {status['is_open']}")
    print(f"Next event: {status['next_event']} at {status['next_time']}")
    
    # Create a mock snapshot
    tracker._last_close_snapshot = GapSnapshot(
        timestamp=datetime.now(timezone.utc).isoformat(),
        btc_price=90000,
        ibit_price=52.50,
        bito_price=18.30,
        market_date="2026-01-30",
    )
    
    # Test gap calculation at different levels
    test_prices = [
        (90000, "No change"),
        (85500, "-5% (info)"),
        (82800, "-8% (elevated)"),
        (79200, "-12% (high)"),
        (76500, "-15% (extreme)"),
        (95000, "+5.5% (info/up)"),
    ]
    
    for price, desc in test_prices:
        gap = tracker.calculate_gap(price)
        if gap:
            should_alert, tier = tracker.should_alert(gap)
            adj = tracker.get_position_adjustment(gap)
            print(f"\n{desc}:")
            print(f"  Gap: {gap.gap_percent:+.1f}%")
            print(f"  Risk: {gap.risk_level}")
            print(f"  Alert: {should_alert} ({tier})")
            print(f"  Position: {adj['action']} (x{adj['size_multiplier']})")


if __name__ == "__main__":
    asyncio.run(test_gap_tracker())
