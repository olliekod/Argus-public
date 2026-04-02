# Created by Oliver Meihls

# Virtual trade logging and tracking for ETF put spreads.
# Logs entries when signals fire, tracks open positions, calculates P&L.

import asyncio
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from enum import Enum
import json

logger = logging.getLogger(__name__)


class TradeStatus(Enum):
    # Status of a paper trade.
    OPEN = "OPEN"
    CLOSED_PROFIT = "CLOSED_PROFIT"
    CLOSED_LOSS = "CLOSED_LOSS"
    CLOSED_TIME = "CLOSED_TIME"  # Closed due to time exit
    EXPIRED = "EXPIRED"


@dataclass
class PaperTrade:
    # Represents a paper trade.
    id: Optional[int]
    entry_timestamp: str
    symbol: str
    
    # Trade structure
    expiration: str
    short_strike: float
    long_strike: float
    spread_width: float
    
    # Entry
    entry_credit: float
    quantity: int
    
    # Greeks at entry
    entry_delta: float
    entry_theta: float
    entry_pop: float
    
    # Status
    status: TradeStatus
    
    # Exit (filled when closed)
    exit_timestamp: Optional[str] = None
    exit_debit: Optional[float] = None
    exit_reason: Optional[str] = None
    
    # P&L
    pnl_dollars: Optional[float] = None
    pnl_percent: Optional[float] = None
    
    # Market conditions at entry
    proxy_iv_at_entry: Optional[float] = None
    underlying_price_at_entry: Optional[float] = None
    iv_rank_at_entry: Optional[float] = None
    
    # Notes
    notes: Optional[str] = None


class PaperTrader:
    # Paper trading simulator for ETF put spreads.
    #
    # Features:
    # - Log virtual trades when signals fire
    # - Track open positions
    # - Auto-close at 50% profit target
    # - Close at time exit (e.g., 5 DTE)
    # - Calculate cumulative P&L
    
    # Exit parameters
    PROFIT_TARGET_PCT = 0.50  # Close at 50% of max credit
    TIME_EXIT_DTE = 5         # Close at 5 DTE if not profitable
    
    def __init__(self, db):
        # Initialize paper trader.
        #
        # Args:
        # db: Database instance
        self.db = db
        self._open_trades: List[PaperTrade] = []
        self._initialized = False
        logger.info("Paper Trader initialized")
    
    async def initialize(self) -> None:
        # Create paper trades table if needed and load open trades.
        if self._initialized:
            return
        
        await self._create_table()
        await self._load_open_trades()
        self._initialized = True
    
    async def _create_table(self) -> None:
        # Create paper_trades table.
        await self.db._connection.execute("""
            CREATE TABLE IF NOT EXISTS paper_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                expiration TEXT NOT NULL,
                short_strike REAL NOT NULL,
                long_strike REAL NOT NULL,
                spread_width REAL NOT NULL,
                entry_credit REAL NOT NULL,
                quantity INTEGER NOT NULL,
                entry_delta REAL,
                entry_theta REAL,
                entry_pop REAL,
                status TEXT NOT NULL,
                exit_timestamp TEXT,
                exit_debit REAL,
                exit_reason TEXT,
                pnl_dollars REAL,
                pnl_percent REAL,
                proxy_iv_at_entry REAL,
                underlying_price_at_entry REAL,
                iv_rank_at_entry REAL,
                notes TEXT
            )
        """)
        await self.db._connection.commit()
        logger.debug("Paper trades table ready")
    
    async def _load_open_trades(self) -> None:
        # Load open trades from database.
        cursor = await self.db._connection.execute("""
            SELECT * FROM paper_trades WHERE status = 'OPEN'
        """)
        rows = await cursor.fetchall()
        
        self._open_trades = []
        for row in rows:
            trade = self._row_to_trade(row)
            self._open_trades.append(trade)
        
        logger.info(f"Loaded {len(self._open_trades)} open paper trades")
    
    def _row_to_trade(self, row) -> PaperTrade:
        # Convert database row to PaperTrade object.
        return PaperTrade(
            id=row[0],
            entry_timestamp=row[1],
            symbol=row[2],
            expiration=row[3],
            short_strike=row[4],
            long_strike=row[5],
            spread_width=row[6],
            entry_credit=row[7],
            quantity=row[8],
            entry_delta=row[9],
            entry_theta=row[10],
            entry_pop=row[11],
            status=TradeStatus(row[12]),
            exit_timestamp=row[13],
            exit_debit=row[14],
            exit_reason=row[15],
            pnl_dollars=row[16],
            pnl_percent=row[17],
            proxy_iv_at_entry=row[18],
            underlying_price_at_entry=row[19],
            iv_rank_at_entry=row[20],
            notes=row[21],
        )
    
    async def open_trade(
        self,
        recommendation,  # TradeRecommendation
        proxy_iv: float = 0,
    ) -> PaperTrade:
        # Open a new paper trade based on recommendation.
        #
        # Args:
        # recommendation: TradeRecommendation from TradeCalculator
        # proxy_iv: Current volatility proxy IV (BTC, etc.)
        #
        # Returns:
        # Created PaperTrade
        await self.initialize()
        
        trade = PaperTrade(
            id=None,
            entry_timestamp=datetime.now().isoformat(),
            symbol=recommendation.symbol,
            expiration=recommendation.expiration,
            short_strike=recommendation.short_strike,
            long_strike=recommendation.long_strike,
            spread_width=recommendation.spread_width,
            entry_credit=recommendation.net_credit,
            quantity=recommendation.num_contracts,
            entry_delta=recommendation.net_delta,
            entry_theta=recommendation.net_theta,
            entry_pop=recommendation.probability_of_profit,
            status=TradeStatus.OPEN,
            proxy_iv_at_entry=proxy_iv,
            underlying_price_at_entry=recommendation.underlying_price,
            iv_rank_at_entry=recommendation.iv_rank,
            notes=f"Auto-opened from signal. PoP: {recommendation.probability_of_profit:.0f}%",
        )
        
        # Insert into database
        cursor = await self.db._connection.execute("""
            INSERT INTO paper_trades (
                entry_timestamp, symbol, expiration, short_strike, long_strike,
                spread_width, entry_credit, quantity, entry_delta, entry_theta,
                entry_pop, status, proxy_iv_at_entry, underlying_price_at_entry,
                iv_rank_at_entry, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade.entry_timestamp, trade.symbol, trade.expiration,
            trade.short_strike, trade.long_strike, trade.spread_width,
            trade.entry_credit, trade.quantity, trade.entry_delta,
            trade.entry_theta, trade.entry_pop, trade.status.value,
            trade.proxy_iv_at_entry, trade.underlying_price_at_entry,
            trade.iv_rank_at_entry, trade.notes,
        ))
        await self.db._connection.commit()
        
        trade.id = cursor.lastrowid
        self._open_trades.append(trade)
        
        logger.info(
            f"Opened paper trade #{trade.id}: "
            f"{trade.symbol} ${trade.short_strike}/${trade.long_strike} for ${trade.entry_credit:.2f}"
        )
        
        return trade
    
    async def close_trade(
        self,
        trade_id: int,
        exit_debit: float,
        reason: str = "manual",
    ) -> Optional[PaperTrade]:
        # Close a paper trade.
        #
        # Args:
        # trade_id: ID of trade to close
        # exit_debit: Cost to close the spread (buy back)
        # reason: Reason for closing
        #
        # Returns:
        # Updated PaperTrade or None
        await self.initialize()
        
        # Find trade
        trade = next((t for t in self._open_trades if t.id == trade_id), None)
        if not trade:
            logger.warning(f"Trade #{trade_id} not found")
            return None
        
        # Calculate P&L
        # Credit received - Debit paid = Profit per share
        # Multiply by 100 (shares per contract) and quantity
        pnl_per_share = trade.entry_credit - exit_debit
        pnl_dollars = pnl_per_share * 100 * trade.quantity
        pnl_percent = (pnl_per_share / trade.entry_credit) * 100 if trade.entry_credit > 0 else 0
        
        # Determine status
        if pnl_dollars > 0:
            status = TradeStatus.CLOSED_PROFIT
        elif reason == "time_exit":
            status = TradeStatus.CLOSED_TIME
        elif reason == "expired":
            status = TradeStatus.EXPIRED
        else:
            status = TradeStatus.CLOSED_LOSS
        
        # Update trade
        trade.exit_timestamp = datetime.now().isoformat()
        trade.exit_debit = exit_debit
        trade.exit_reason = reason
        trade.pnl_dollars = pnl_dollars
        trade.pnl_percent = pnl_percent
        trade.status = status
        
        # Update database
        await self.db._connection.execute("""
            UPDATE paper_trades SET
                status = ?,
                exit_timestamp = ?,
                exit_debit = ?,
                exit_reason = ?,
                pnl_dollars = ?,
                pnl_percent = ?
            WHERE id = ?
        """, (
            trade.status.value, trade.exit_timestamp, trade.exit_debit,
            trade.exit_reason, trade.pnl_dollars, trade.pnl_percent, trade.id,
        ))
        await self.db._connection.commit()
        
        # Remove from open trades
        self._open_trades = [t for t in self._open_trades if t.id != trade_id]
        
        logger.info(
            f"Closed paper trade #{trade.id}: "
            f"P&L ${pnl_dollars:.2f} ({pnl_percent:.1f}%), reason: {reason}"
        )
        
        return trade
    
    async def check_exit_conditions(
        self,
        current_price: float,
        current_spread_value: Optional[float] = None,
    ) -> List[PaperTrade]:
        # Check if any open trades should be closed.
        #
        # Args:
        # current_price: Current underlying price
        # current_spread_value: Current value of spread (if available)
        #
        # Returns:
        # List of trades that were closed
        await self.initialize()
        
        closed_trades = []
        today = datetime.now().date()
        
        for trade in self._open_trades.copy():
            should_close = False
            exit_debit = None
            reason = None
            
            # Check expiration
            try:
                exp_date = datetime.strptime(trade.expiration, "%Y-%m-%d").date()
                dte = (exp_date - today).days
            except:
                dte = 999
            
            # 1. Check if expired
            if dte <= 0:
                should_close = True
                # At expiration, if price > short strike, spread is worthless (full profit)
                if current_price > trade.short_strike:
                    exit_debit = 0
                    reason = "expired_profit"
                else:
                    # Price below short strike - max loss
                    exit_debit = trade.spread_width
                    reason = "expired_loss"
            
            # 2. Check 50% profit target (if we have current spread value)
            elif current_spread_value is not None:
                profit_target = trade.entry_credit * self.PROFIT_TARGET_PCT
                if current_spread_value <= profit_target:
                    should_close = True
                    exit_debit = current_spread_value
                    reason = "profit_target"
            
            # 3. Check time exit (close at 5 DTE)
            elif dte <= self.TIME_EXIT_DTE:
                should_close = True
                # Estimate spread value based on price
                if current_price > trade.short_strike:
                    # OTM, spread is worth less
                    exit_debit = trade.entry_credit * 0.3  # Rough estimate
                else:
                    # ITM, spread is worth more
                    exit_debit = trade.entry_credit * 0.7
                reason = "time_exit"
            
            if should_close and exit_debit is not None:
                closed = await self.close_trade(trade.id, exit_debit, reason)
                if closed:
                    closed_trades.append(closed)
        
        return closed_trades
    
    async def get_open_trades(self) -> List[PaperTrade]:
        # Get all open trades.
        await self.initialize()
        return self._open_trades.copy()
    
    async def get_all_trades(self, limit: int = 50) -> List[PaperTrade]:
        # Get all trades (open and closed).
        await self.initialize()
        
        cursor = await self.db._connection.execute("""
            SELECT * FROM paper_trades
            ORDER BY entry_timestamp DESC
            LIMIT ?
        """, (limit,))
        rows = await cursor.fetchall()
        
        return [self._row_to_trade(row) for row in rows]
    
    async def get_statistics(self) -> Dict[str, Any]:
        # Get paper trading statistics.
        #
        # Returns:
        # Dict with win rate, total P&L, etc.
        await self.initialize()
        
        cursor = await self.db._connection.execute("""
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN pnl_dollars > 0 THEN 1 ELSE 0 END) as winners,
                SUM(CASE WHEN pnl_dollars < 0 THEN 1 ELSE 0 END) as losers,
                SUM(CASE WHEN status = 'OPEN' THEN 1 ELSE 0 END) as open_trades,
                SUM(pnl_dollars) as total_pnl,
                AVG(pnl_dollars) as avg_pnl,
                AVG(pnl_percent) as avg_pnl_pct,
                MIN(pnl_dollars) as worst_trade,
                MAX(pnl_dollars) as best_trade,
                AVG(entry_pop) as avg_pop
            FROM paper_trades
        """)
        row = await cursor.fetchone()
        
        if not row or row[0] == 0:
            return {
                'total_trades': 0,
                'open_trades': 0,
                'closed_trades': 0,
                'winners': 0,
                'losers': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_pnl': 0,
                'avg_pnl_pct': 0,
                'best_trade': 0,
                'worst_trade': 0,
                'avg_pop': 0,
            }
        
        total = row[0]
        winners = row[1] or 0
        losers = row[2] or 0
        open_trades = row[3] or 0
        closed = winners + losers
        
        return {
            'total_trades': total,
            'open_trades': open_trades,
            'closed_trades': closed,
            'winners': winners,
            'losers': losers,
            'win_rate': (winners / closed * 100) if closed > 0 else 0,
            'total_pnl': row[4] or 0,
            'avg_pnl': row[5] or 0,
            'avg_pnl_pct': row[6] or 0,
            'best_trade': row[8] or 0,
            'worst_trade': row[7] or 0,
            'avg_pop': row[9] or 0,
        }
    
    def format_statistics(self, stats: Dict) -> str:
        # Format statistics for display.
        return f"""
📊 PAPER TRADING STATISTICS
{'='*40}

Trades:
  Total: {stats['total_trades']}
  Open: {stats['open_trades']}
  Closed: {stats['closed_trades']}

Performance:
  Win Rate: {stats['win_rate']:.1f}%
  Winners: {stats['winners']}
  Losers: {stats['losers']}

P&L:
  Total: ${stats['total_pnl']:.2f}
  Average: ${stats['avg_pnl']:.2f} ({stats['avg_pnl_pct']:.1f}%)
  Best: ${stats['best_trade']:.2f}
  Worst: ${stats['worst_trade']:.2f}

Strategy:
  Avg Entry PoP: {stats['avg_pop']:.0f}%
"""


# Test function
async def test_paper_trader():
    # Test the paper trader.
    print("=" * 60)
    print("PAPER TRADER TEST")
    print("=" * 60)
    
    # Would need database connection - this is just structure test
    print("\nPaperTrader class created successfully.")
    print("Use with database: trader = PaperTrader(db)")
    print("Then: await trader.open_trade(recommendation)")


if __name__ == "__main__":
    asyncio.run(test_paper_trader())
