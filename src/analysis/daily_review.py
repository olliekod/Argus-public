"""
Daily Review
============

Generates 4 PM ET daily summary including:
- Paper trading P&L
- Open positions
- Month-to-date performance
- Top 5 performers (90-day)
- IV Regime Forecast
- Market conditions summary
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional
import logging

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # Python < 3.9

logger = logging.getLogger(__name__)


@dataclass
class TopPerformer:
    """Top performing paper trader."""
    trader_id: str
    strategy: str
    realized_pnl: float
    return_pct: float
    trades: int
    win_rate: float


@dataclass
class IVRegimeForecast:
    """IV regime analysis and forecast."""
    current_iv: float
    iv_percentile: float  # 0-100
    days_elevated: int  # Days IV has been above threshold
    regime: str  # 'low', 'normal', 'elevated', 'high'
    forecast: str  # Human-readable forecast message
    historical_insight: str  # What historically happens at this level


@dataclass
class DailyReviewData:
    """Daily review data structure."""
    date: date
    
    # Paper trading
    trades_today: int
    trades_opened_today: int
    trades_closed_today: int
    pnl_today: float
    pnl_today_pct: float
    open_positions: List[Dict]
    open_positions_count: int
    
    # Month-to-date
    trades_mtd: int
    pnl_mtd: float
    pnl_mtd_pct: float
    win_rate_mtd: float

    # Year-to-date
    trades_ytd: int
    pnl_ytd: float
    pnl_ytd_pct: float
    win_rate_ytd: float
    
    # Account
    account_value: float
    starting_balance: float
    
    # Conditions
    conditions_score: int
    conditions_label: str
    btc_iv: float
    
    # Risk
    gap_risk_level: str
    
    # NEW: Top performers (90-day)
    top_performers: List[TopPerformer] = field(default_factory=list)
    
    # NEW: IV Regime Forecast
    iv_forecast: Optional[IVRegimeForecast] = None
    
    # NEW: GPU Performance
    gpu_stats: str = "Offline"


class DailyReview:
    """
    Generates 4 PM ET daily summary.
    
    Pulls data from paper trader and conditions monitor.
    Enhanced with top performers and IV regime forecast.
    """
    
    REVIEW_HOUR_ET = 16  # 4 PM ET
    ET_TIMEZONE = ZoneInfo("America/New_York")  # Proper DST handling
    
    # IV regime thresholds
    IV_LOW = 40
    IV_ELEVATED = 55
    IV_HIGH = 70
    
    def __init__(
        self,
        starting_balance: float = 5000.0,
        on_send: Optional[Callable] = None,
    ):
        """
        Initialize daily review.
        
        Args:
            starting_balance: Starting paper trading balance
            on_send: Callback to send the review (e.g., Telegram)
        """
        self.starting_balance = starting_balance
        self.send_callback = on_send
        
        # Data source callbacks (set by orchestrator)
        self._get_trades: Optional[Callable] = None
        self._get_positions: Optional[Callable] = None
        self._get_conditions: Optional[Callable] = None
        self._get_gap_risk: Optional[Callable] = None
        self._get_trade_stats: Optional[Callable] = None
        self._get_top_performers: Optional[Callable] = None  # NEW
        self._get_iv_history: Optional[Callable] = None  # NEW
        
        self._running = False
        self._last_review_date: Optional[date] = None
        
        # Track IV history for regime forecast
        self._iv_history: List[Dict] = []
        
        logger.info("Daily Review initialized")
    
    def set_data_sources(
        self,
        get_trades: Optional[Callable] = None,
        get_positions: Optional[Callable] = None,
        get_conditions: Optional[Callable] = None,
        get_gap_risk: Optional[Callable] = None,
        get_trade_stats: Optional[Callable] = None,
        get_top_performers: Optional[Callable] = None,
        get_iv_history: Optional[Callable] = None,
    ):
        """Set callbacks for fetching data."""
        self._get_trades = get_trades
        self._get_positions = get_positions
        self._get_conditions = get_conditions
        self._get_gap_risk = get_gap_risk
        self._get_trade_stats = get_trade_stats
        self._get_top_performers = get_top_performers
        self._get_iv_history = get_iv_history
    
    def _is_market_day(self, d: date) -> bool:
        """Check if date is a market day (weekday)."""
        return d.weekday() < 5
    
    def _should_run_review(self) -> bool:
        """Check if it's time for daily review (DST-aware)."""
        # Get current time in ET with proper DST handling
        now_utc = datetime.now(timezone.utc).replace(tzinfo=ZoneInfo("UTC"))
        now_et = now_utc.astimezone(self.ET_TIMEZONE)
        today = now_et.date()
        
        # Check if we already ran today
        if self._last_review_date == today:
            return False
        
        # Check if it's a market day
        if not self._is_market_day(today):
            return False
        
        # Run at or after 4 PM ET
        return now_et.hour >= self.REVIEW_HOUR_ET
    
    def _calculate_iv_regime_forecast(self, current_iv: float) -> IVRegimeForecast:
        """
        Calculate IV regime and forecast.
        
        Based on current IV level and historical patterns.
        """
        # Determine regime
        if current_iv >= self.IV_HIGH:
            regime = "high"
            percentile = 90
        elif current_iv >= self.IV_ELEVATED:
            regime = "elevated"
            percentile = 70
        elif current_iv >= self.IV_LOW:
            regime = "normal"
            percentile = 50
        else:
            regime = "low"
            percentile = 25
        
        # Count days elevated (from IV history if available)
        days_elevated = 0
        if self._iv_history:
            for entry in reversed(self._iv_history[-14:]):  # Last 14 days
                if entry.get('iv', 0) >= self.IV_ELEVATED:
                    days_elevated += 1
                else:
                    break
        
        # Generate forecast based on regime and duration
        if regime == "high":
            forecast = f"IV at {current_iv:.0f}% is very high. Historically drops 15-25% within 3-7 days."
            if days_elevated >= 5:
                forecast += " Extended duration suggests imminent compression."
            historical_insight = "At 90th+ percentile, win rates for premium selling historically peak at 78%."
        elif regime == "elevated":
            forecast = f"IV at {current_iv:.0f}% is elevated. Good conditions for premium selling."
            if days_elevated >= 3:
                forecast += f" Elevated for {days_elevated} days - watch for mean reversion."
            historical_insight = "Elevated IV typically persists 3-5 days before normalizing."
        elif regime == "normal":
            forecast = f"IV at {current_iv:.0f}% is normal. Standard opportunity detection active."
            historical_insight = "Normal IV requires tighter strike selection for adequate premium."
        else:
            forecast = f"IV at {current_iv:.0f}% is low. Premium selling less attractive."
            historical_insight = "Low IV periods historically last 2-4 weeks before volatility events."
        
        return IVRegimeForecast(
            current_iv=current_iv,
            iv_percentile=percentile,
            days_elevated=days_elevated,
            regime=regime,
            forecast=forecast,
            historical_insight=historical_insight,
        )
    
    async def generate_review(self) -> DailyReviewData:
        """Generate daily review data."""
        today = date.today()
        month_start = today.replace(day=1)
        
        # Get trades
        trades_today = []
        trades_mtd = []
        trades_ytd = []
        trades_opened_today = 0
        trades_closed_today = 0
        win_rate_ytd = 0.0
        pnl_ytd = 0.0
        stats: Dict[str, Any] = {}
        
        if self._get_trade_stats:
            try:
                stats = await self._get_trade_stats()
                trades_today = [None] * int(stats.get('trades_today', 0))
                trades_mtd = [None] * int(stats.get('trades_mtd', 0))
                trades_ytd = [None] * int(stats.get('trades_ytd', 0))
                trades_opened_today = int(stats.get('opened_today', 0))
                trades_closed_today = int(stats.get('trades_today', 0))
                pnl_today = float(stats.get('today_pnl', 0))
                pnl_mtd = float(stats.get('month_pnl', 0))
                pnl_ytd = float(stats.get('year_pnl', 0))
                win_rate_mtd = float(stats.get('win_rate_mtd', 0))
                win_rate_ytd = float(stats.get('win_rate_ytd', 0))
            except Exception as e:
                logger.warning(f"Failed to get trade stats: {e}")
                pnl_today = 0.0
                pnl_mtd = 0.0
                win_rate_mtd = 0.0
                win_rate_ytd = 0.0
        elif self._get_trades:
            try:
                all_trades = await self._get_trades()
                normalized = []
                for t in all_trades:
                    trade_date = t.get('date')
                    if isinstance(trade_date, str):
                        try:
                            trade_date = date.fromisoformat(trade_date)
                        except ValueError:
                            trade_date = None
                    if trade_date:
                        normalized.append({**t, 'date': trade_date})
                trades_today = [t for t in normalized if t.get('date') == today]
                trades_mtd = [t for t in normalized if t.get('date') >= month_start]
                trades_ytd = [t for t in normalized if t.get('date') >= date(today.year, 1, 1)]
            except Exception as e:
                logger.warning(f"Failed to get trades: {e}")
        
        # Calculate P&L
        if not self._get_trade_stats:
            pnl_today = sum(t.get('realized_pnl', 0) for t in trades_today)
            pnl_mtd = sum(t.get('realized_pnl', 0) for t in trades_mtd)
            pnl_ytd = sum(t.get('realized_pnl', 0) for t in trades_ytd)
        
        # Win rate
        if not self._get_trade_stats:
            wins_mtd = sum(1 for t in trades_mtd if t.get('realized_pnl', 0) > 0)
            win_rate_mtd = (wins_mtd / len(trades_mtd) * 100) if trades_mtd else 0
            wins_ytd = sum(1 for t in trades_ytd if t.get('realized_pnl', 0) > 0)
            win_rate_ytd = (wins_ytd / len(trades_ytd) * 100) if trades_ytd else 0
        
        # Account value (starting + MTD P&L + unrealized)
        account_value = self.starting_balance + pnl_ytd
        
        # Get open positions
        positions = []
        open_positions_count = 0
        if self._get_positions:
            try:
                positions = await self._get_positions() or []
                open_positions_count = len(positions)
                # Add unrealized P&L to account value
                unrealized = sum(p.get('unrealized_pnl', 0) for p in positions)
                account_value += unrealized
            except Exception as e:
                logger.warning(f"Failed to get positions: {e}")
        if self._get_trade_stats and stats:
            open_positions_count = int(stats.get('open_positions', open_positions_count))
            trades_opened_today = int(stats.get('opened_today', trades_opened_today))
            trades_closed_today = int(stats.get('trades_today', trades_closed_today))
        
        # P&L percentages
        pnl_today_pct = (pnl_today / self.starting_balance * 100) if self.starting_balance else 0
        pnl_mtd_pct = (pnl_mtd / self.starting_balance * 100) if self.starting_balance else 0
        pnl_ytd_pct = (pnl_ytd / self.starting_balance * 100) if self.starting_balance else 0
        
        # Get conditions
        conditions_score = 5
        conditions_label = "neutral"
        btc_iv = 50.0
        
        if self._get_conditions:
            try:
                cond = await self._get_conditions()
                if cond:
                    conditions_score = cond.get('score', 5)
                    conditions_label = cond.get('warmth_label', 'neutral')
                    btc_iv = float(cond.get('btc_iv', 50))
            except Exception as e:
                logger.warning(f"Failed to get conditions: {e}")
        
        # Get gap risk
        gap_risk_level = "low"
        if self._get_gap_risk:
            try:
                risk = await self._get_gap_risk()
                gap_risk_level = risk.get('level', 'low') if risk else 'low'
            except Exception as e:
                logger.warning(f"Failed to get gap risk: {e}")
        
        # NEW: Get top performers (90-day)
        top_performers = []
        if self._get_top_performers:
            try:
                performers_data = await self._get_top_performers(days=90, top_n=5)
                if performers_data:
                    for p in performers_data:
                        top_performers.append(TopPerformer(
                            trader_id=p.get('trader_id', 'Unknown'),
                            strategy=p.get('strategy', 'Unknown'),
                            realized_pnl=p.get('realized_pnl', 0),
                            return_pct=p.get('return_pct', 0),
                            trades=p.get('total_trades', 0),
                            win_rate=p.get('win_rate', 0),
                        ))
            except Exception as e:
                logger.warning(f"Failed to get top performers: {e}")
        
        # NEW: Calculate IV regime forecast
        iv_forecast = self._calculate_iv_regime_forecast(btc_iv)
        
        # NEW: Get GPU Performance
        gpu_stats = "Offline"
        try:
            from src.analysis.gpu_engine import get_gpu_engine
            gpu = get_gpu_engine()
            gpu_stats = gpu.format_status()
        except Exception:
            pass
            
        # Update IV history for tracking
        self._iv_history.append({'date': today.isoformat(), 'iv': btc_iv})
        # Keep only last 30 days
        if len(self._iv_history) > 30:
            self._iv_history = self._iv_history[-30:]
        
        return DailyReviewData(
            date=today,
            trades_today=len(trades_today),
            trades_opened_today=trades_opened_today,
            trades_closed_today=trades_closed_today or len(trades_today),
            pnl_today=pnl_today,
            pnl_today_pct=pnl_today_pct,
            open_positions=positions,
            open_positions_count=open_positions_count,
            trades_mtd=len(trades_mtd),
            pnl_mtd=pnl_mtd,
            pnl_mtd_pct=pnl_mtd_pct,
            win_rate_mtd=win_rate_mtd,
            trades_ytd=len(trades_ytd),
            pnl_ytd=pnl_ytd,
            pnl_ytd_pct=pnl_ytd_pct,
            win_rate_ytd=win_rate_ytd,
            account_value=account_value,
            starting_balance=self.starting_balance,
            conditions_score=conditions_score,
            conditions_label=conditions_label,
            btc_iv=btc_iv,
            gap_risk_level=gap_risk_level,
            top_performers=top_performers,
            iv_forecast=iv_forecast,
            gpu_stats=gpu_stats,
        )
    
    def format_review(self, data: DailyReviewData) -> str:
        """Format review data as Telegram message."""
        # Emojis based on performance
        today_emoji = "🟢" if data.pnl_today >= 0 else "🔴"
        mtd_emoji = "🟢" if data.pnl_mtd >= 0 else "🔴"
        
        lines = [
            f"📊 <b>DAILY REVIEW — {data.date.strftime('%b %d, %Y')}</b>",
            "",
            "<b>PAPER TRADING:</b>",
            f"{today_emoji} Today: ${data.pnl_today:+.2f} ({data.pnl_today_pct:+.1f}%)",
            f"   Opened: {data.trades_opened_today} | Closed: {data.trades_closed_today}",
            f"{mtd_emoji} Month-to-Date: ${data.pnl_mtd:+.2f} ({data.pnl_mtd_pct:+.1f}%)",
            f"   Trades: {data.trades_mtd} | Win rate: {data.win_rate_mtd:.0f}%",
            f"{mtd_emoji} Year-to-Date: ${data.pnl_ytd:+.2f} ({data.pnl_ytd_pct:+.1f}%)",
            f"   Trades: {data.trades_ytd} | Win rate: {data.win_rate_ytd:.0f}%",
            "",
            f"💰 Account value: ${data.account_value:,.2f}",
        ]
        
        # Open positions
        if data.open_positions_count:
            lines.append("")
            lines.append(f"<b>OPEN POSITIONS:</b> {data.open_positions_count}")
            for pos in data.open_positions[:3]:  # Show max 3
                pnl = pos.get('unrealized_pnl', 0)
                pnl_emoji = "🟢" if pnl >= 0 else "🔴"
                lines.append(f"  {pnl_emoji} {pos.get('symbol', '?')} | ${pnl:+.2f}")
        
        # NEW: Top 5 Performers (90-day)
        if data.top_performers:
            lines.append("")
            lines.append("<b>🏆 TOP PERFORMERS (90-Day):</b>")
            for i, perf in enumerate(data.top_performers[:5], 1):
                medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i-1]
                lines.append(
                    f"  {medal} {perf.trader_id}: ${perf.realized_pnl:+,.0f} "
                    f"({perf.return_pct:+.1f}%) | {perf.strategy} | "
                    f"{perf.trades} trades, {perf.win_rate:.0f}% win"
                )
        
        # NEW: IV Regime Forecast
        if data.iv_forecast:
            iv = data.iv_forecast
            regime_emoji = {
                'low': '📉',
                'normal': '📊',
                'elevated': '📈',
                'high': '🔥',
            }.get(iv.regime, '📊')
            
            lines.extend([
                "",
                f"<b>{regime_emoji} IV REGIME FORECAST:</b>",
                f"  {iv.forecast}",
                f"  <i>💡 {iv.historical_insight}</i>",
            ])
        
        # Conditions
        warmth_emoji = {
            'cooling': '❄️',
            'neutral': '➖',
            'warming': '🔥',
            'prime': '🎯',
        }.get(data.conditions_label, '🌡️')
        
        lines.extend([
            "",
            "<b>CONDITIONS:</b>",
            f"  {warmth_emoji} Score: {data.conditions_score}/10 ({data.conditions_label})",
            f"  📈 BTC IV: {data.btc_iv:.0f}%",
            f"  ⚠️ Gap risk: {data.gap_risk_level.upper()}",
        ])
        
        # Footer
        lines.extend([
            "",
            f"<i>End of day review • Next review tomorrow 4 PM ET</i>",
        ])
        
        # GPU Performance Section
        lines.append("")
        lines.append(data.gpu_stats)
        
        return "\n".join(lines)
    
    async def run_review(self) -> Optional[str]:
        """Generate and send daily review."""
        try:
            data = await self.generate_review()
            message = self.format_review(data)
            
            if self.send_callback:
                await self.send_callback(message)
            
            self._last_review_date = data.date
            logger.info(f"Daily review sent for {data.date}")
            
            return message
        except Exception as e:
            logger.error(f"Failed to generate daily review: {e}")
            return None
    
    async def start_monitoring(self) -> None:
        """Start monitoring for review time."""
        self._running = True
        logger.info("Daily review monitoring started")
        
        while self._running:
            if self._should_run_review():
                await self.run_review()
            
            # Check every 5 minutes
            await asyncio.sleep(300)
    
    def stop_monitoring(self) -> None:
        """Stop monitoring."""
        self._running = False
        logger.info("Daily review monitoring stopped")


# For manual/immediate review
async def get_pnl_summary(daily_review: DailyReview) -> Dict[str, Any]:
    """Get P&L summary for /pnl command."""
    data = await daily_review.generate_review()
    return {
        'today_pnl': data.pnl_today,
        'today_pct': data.pnl_today_pct,
        'month_pnl': data.pnl_mtd,
        'month_pct': data.pnl_mtd_pct,
        'year_pnl': data.pnl_ytd,
        'year_pct': data.pnl_ytd_pct,
        'trades_today': data.trades_today,
        'trades_mtd': data.trades_mtd,
        'win_rate_mtd': data.win_rate_mtd,
        'opened_today': data.trades_opened_today,
        'open_positions': data.open_positions_count,
        'account_value': data.account_value,
    }


# Self-test
if __name__ == "__main__":
    async def test():
        print("Daily Review Test")
        print("=" * 40)
        
        review = DailyReview(starting_balance=5000)
        
        # Mock data sources
        async def mock_trades():
            return [
                {'date': date.today(), 'realized_pnl': 42.50},
                {'date': date.today() - timedelta(days=5), 'realized_pnl': -15.00},
                {'date': date.today() - timedelta(days=10), 'realized_pnl': 28.00},
            ]
        
        async def mock_positions():
            return [
                {'symbol': 'IBIT', 'strikes': '$48/$44', 'unrealized_pnl': 12.50},
            ]
        
        async def mock_conditions():
            return {'score': 7, 'warmth_label': 'warming', 'btc_iv': '68'}
        
        review.set_data_sources(
            get_trades=mock_trades,
            get_positions=mock_positions,
            get_conditions=mock_conditions,
        )
        
        data = await review.generate_review()
        message = review.format_review(data)
        print(message)
    
    asyncio.run(test())
