"""
Performance Tracker
===================

Analyzes paper trading performance with detailed metrics.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import statistics

logger = logging.getLogger(__name__)


@dataclass
class PerformanceBucket:
    """Performance metrics for a specific IV Rank bucket."""
    iv_rank_range: str
    trade_count: int
    win_rate: float
    avg_pnl: float
    total_pnl: float


class PerformanceTracker:
    """
    Analyzes paper trading performance.
    
    Metrics:
    - Win rate, total P&L
    - Performance by IV Rank bucket
    - Drawdown analysis
    - Time-based patterns
    """
    
    # IV Rank buckets for analysis
    IV_BUCKETS = [
        (50, 60, "50-60%"),
        (60, 70, "60-70%"),
        (70, 80, "70-80%"),
        (80, 100, "80%+"),
    ]
    
    def __init__(self, db):
        """
        Initialize performance tracker.
        
        Args:
            db: Database instance
        """
        self.db = db
        logger.info("Performance Tracker initialized")
    
    async def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary.
        
        Returns:
            Dict with all performance metrics
        """
        # Basic stats
        cursor = await self.db._connection.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN status != 'OPEN' THEN 1 ELSE 0 END) as closed,
                SUM(CASE WHEN pnl_dollars > 0 THEN 1 ELSE 0 END) as winners,
                SUM(CASE WHEN pnl_dollars < 0 THEN 1 ELSE 0 END) as losers,
                SUM(pnl_dollars) as total_pnl,
                AVG(pnl_dollars) as avg_pnl,
                AVG(pnl_percent) as avg_pnl_pct,
                MIN(pnl_dollars) as worst,
                MAX(pnl_dollars) as best
            FROM paper_trades
        """)
        row = await cursor.fetchone()
        
        if not row or row[0] == 0:
            return self._empty_summary()
        
        total = row[0]
        closed = row[1] or 0
        winners = row[2] or 0
        losers = row[3] or 0
        
        summary = {
            'total_trades': total,
            'closed_trades': closed,
            'open_trades': total - closed,
            'winners': winners,
            'losers': losers,
            'win_rate': (winners / closed * 100) if closed > 0 else 0,
            'total_pnl': row[4] or 0,
            'avg_pnl': row[5] or 0,
            'avg_pnl_pct': row[6] or 0,
            'best_trade': row[8] or 0,
            'worst_trade': row[7] or 0,
        }
        
        # Add IV bucket analysis
        summary['by_iv_rank'] = await self._analyze_by_iv_rank()
        
        # Add drawdown
        summary['max_drawdown'] = await self._calculate_max_drawdown()
        
        # Add by exit reason
        summary['by_exit_reason'] = await self._analyze_by_exit_reason()
        
        return summary
    
    def _empty_summary(self) -> Dict[str, Any]:
        """Return empty summary when no trades."""
        return {
            'total_trades': 0,
            'closed_trades': 0,
            'open_trades': 0,
            'winners': 0,
            'losers': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'avg_pnl': 0,
            'avg_pnl_pct': 0,
            'best_trade': 0,
            'worst_trade': 0,
            'by_iv_rank': [],
            'max_drawdown': 0,
            'by_exit_reason': {},
        }
    
    async def _analyze_by_iv_rank(self) -> List[PerformanceBucket]:
        """Analyze performance by IV Rank bucket."""
        buckets = []
        
        for low, high, label in self.IV_BUCKETS:
            cursor = await self.db._connection.execute("""
                SELECT
                    COUNT(*) as cnt,
                    SUM(CASE WHEN pnl_dollars > 0 THEN 1 ELSE 0 END) as winners,
                    AVG(pnl_dollars) as avg_pnl,
                    SUM(pnl_dollars) as total_pnl
                FROM paper_trades
                WHERE iv_rank_at_entry >= ? AND iv_rank_at_entry < ?
                  AND status != 'OPEN'
            """, (low, high))
            row = await cursor.fetchone()
            
            if row and row[0] > 0:
                buckets.append(PerformanceBucket(
                    iv_rank_range=label,
                    trade_count=row[0],
                    win_rate=(row[1] / row[0] * 100) if row[0] > 0 else 0,
                    avg_pnl=row[2] or 0,
                    total_pnl=row[3] or 0,
                ))
        
        return buckets
    
    async def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from cumulative P&L."""
        cursor = await self.db._connection.execute("""
            SELECT pnl_dollars, exit_timestamp
            FROM paper_trades
            WHERE status != 'OPEN'
            ORDER BY exit_timestamp
        """)
        rows = await cursor.fetchall()
        
        if not rows:
            return 0.0
        
        cumulative = 0.0
        peak = 0.0
        max_drawdown = 0.0
        
        for row in rows:
            pnl = row[0] or 0
            cumulative += pnl
            
            if cumulative > peak:
                peak = cumulative
            
            drawdown = peak - cumulative
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return max_drawdown
    
    async def _analyze_by_exit_reason(self) -> Dict[str, Dict]:
        """Analyze performance by exit reason."""
        cursor = await self.db._connection.execute("""
            SELECT
                exit_reason,
                COUNT(*) as cnt,
                AVG(pnl_dollars) as avg_pnl,
                SUM(pnl_dollars) as total_pnl
            FROM paper_trades
            WHERE status != 'OPEN' AND exit_reason IS NOT NULL
            GROUP BY exit_reason
        """)
        rows = await cursor.fetchall()
        
        result = {}
        for row in rows:
            result[row[0] or 'unknown'] = {
                'count': row[1],
                'avg_pnl': row[2] or 0,
                'total_pnl': row[3] or 0,
            }
        
        return result
    
    async def get_equity_curve(self) -> List[Dict]:
        """
        Get equity curve data for plotting.
        
        Returns:
            List of {timestamp, cumulative_pnl} dicts
        """
        cursor = await self.db._connection.execute("""
            SELECT exit_timestamp, pnl_dollars
            FROM paper_trades
            WHERE status != 'OPEN'
            ORDER BY exit_timestamp
        """)
        rows = await cursor.fetchall()
        
        curve = []
        cumulative = 0.0
        
        for row in rows:
            cumulative += row[1] or 0
            curve.append({
                'timestamp': row[0],
                'cumulative_pnl': cumulative,
            })
        
        return curve
    
    def format_report(self, summary: Dict) -> str:
        """Format performance summary as readable report."""
        lines = [
            "=" * 50,
            "📊 PAPER TRADING PERFORMANCE REPORT",
            "=" * 50,
            "",
            "OVERVIEW",
            "-" * 30,
            f"  Total Trades: {summary['total_trades']}",
            f"  Open: {summary['open_trades']}",
            f"  Closed: {summary['closed_trades']}",
            "",
            "PERFORMANCE",
            "-" * 30,
            f"  Win Rate: {summary['win_rate']:.1f}%",
            f"  Winners: {summary['winners']}",
            f"  Losers: {summary['losers']}",
            "",
            "P&L",
            "-" * 30,
            f"  Total: ${summary['total_pnl']:.2f}",
            f"  Average: ${summary['avg_pnl']:.2f} ({summary['avg_pnl_pct']:.1f}%)",
            f"  Best: ${summary['best_trade']:.2f}",
            f"  Worst: ${summary['worst_trade']:.2f}",
            f"  Max Drawdown: ${summary['max_drawdown']:.2f}",
            "",
        ]
        
        # IV Rank buckets
        if summary.get('by_iv_rank'):
            lines.extend([
                "PERFORMANCE BY IV RANK",
                "-" * 30,
            ])
            for bucket in summary['by_iv_rank']:
                lines.append(
                    f"  {bucket.iv_rank_range}: "
                    f"{bucket.trade_count} trades, "
                    f"{bucket.win_rate:.0f}% win, "
                    f"${bucket.total_pnl:.2f}"
                )
            lines.append("")
        
        # Exit reasons
        if summary.get('by_exit_reason'):
            lines.extend([
                "BY EXIT REASON",
                "-" * 30,
            ])
            for reason, data in summary['by_exit_reason'].items():
                lines.append(
                    f"  {reason}: {data['count']} trades, "
                    f"avg ${data['avg_pnl']:.2f}"
                )
            lines.append("")
        
        lines.append("=" * 50)
        
        return "\n".join(lines)
    
    def format_telegram_summary(self, summary: Dict) -> str:
        """Format summary for Telegram."""
        return f"""
PAPER TRADING SUMMARY

Trades: {summary['closed_trades']} closed, {summary['open_trades']} open
Win Rate: {summary['win_rate']:.1f}%
Total P&L: ${summary['total_pnl']:.2f}
Avg P&L: ${summary['avg_pnl']:.2f} ({summary['avg_pnl_pct']:.1f}%)
Max Drawdown: ${summary['max_drawdown']:.2f}
"""
    
    async def load_trades(self) -> None:
        """Load trades from database (for initialization)."""
        # Trades are loaded on-demand in other methods
        pass
    
    async def get_summary_stats(self) -> Dict:
        """Get summary stats compatible with paper_performance.py."""
        summary = await self.get_summary()
        
        # Add additional fields expected by script
        account_size = 5000.0
        
        return {
            'total_trades': summary['total_trades'],
            'open_trades': summary['open_trades'],
            'closed_trades': summary['closed_trades'],
            'winners': summary['winners'],
            'losers': summary['losers'],
            'win_rate': summary['win_rate'],
            'total_pnl': summary['total_pnl'],
            'total_return_pct': (summary['total_pnl'] / account_size * 100) if account_size else 0,
            'avg_return_pct': summary['avg_pnl_pct'],
            'best_trade_pct': (summary['best_trade'] / account_size * 100) if account_size else 0,
            'worst_trade_pct': (summary['worst_trade'] / account_size * 100) if account_size else 0,
            'max_drawdown_pct': (summary['max_drawdown'] / account_size * 100) if account_size else 0,
        }
    
    async def get_open_trades(self) -> List[Dict]:
        """Get all open trades."""
        cursor = await self.db._connection.execute("""
            SELECT *
            FROM paper_trades
            WHERE status = 'OPEN'
            ORDER BY entry_timestamp DESC
        """)
        rows = await cursor.fetchall()
        
        # Get column names
        columns = [desc[0] for desc in cursor.description]
        
        return [dict(zip(columns, row)) for row in rows]
    
    async def get_closed_trades(self, limit: int = 10) -> List[Dict]:
        """Get recent closed trades."""
        cursor = await self.db._connection.execute("""
            SELECT *
            FROM paper_trades
            WHERE status != 'OPEN'
            ORDER BY exit_timestamp DESC
            LIMIT ?
        """, (limit,))
        rows = await cursor.fetchall()
        
        columns = [desc[0] for desc in cursor.description]
        
        return [dict(zip(columns, row)) for row in rows]
    
    async def get_analysis_by_iv_bucket(self) -> Dict:
        """Get performance analysis by IV bucket."""
        buckets = await self._analyze_by_iv_rank()
        
        return {
            bucket.iv_rank_range: {
                'count': bucket.trade_count,
                'win_rate': bucket.win_rate,
                'avg_pnl': bucket.avg_pnl,
                'total_pnl': bucket.total_pnl,
            }
            for bucket in buckets
        }
    
    async def get_first_trade_date(self) -> Optional[str]:
        """Get timestamp of first trade."""
        cursor = await self.db._connection.execute("""
            SELECT MIN(entry_timestamp)
            FROM paper_trades
        """)
        row = await cursor.fetchone()
        
        return row[0] if row else None

