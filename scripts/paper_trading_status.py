"""
Paper Trading Status Script
===========================

Check paper trading performance and open positions.
Run: python scripts\paper_trading_status.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.database import Database
from src.analysis.paper_trader import PaperTrader
from src.analysis.performance_tracker import PerformanceTracker


async def main():
    """Show paper trading status."""
    print("=" * 60)
    print("PAPER TRADING STATUS")
    print("=" * 60)
    
    # Connect to database
    db = Database("data/argus.db")
    await db.connect()
    
    # Initialize components
    trader = PaperTrader(db)
    await trader.initialize()
    tracker = PerformanceTracker(db)
    
    # Get open trades
    open_trades = await trader.get_open_trades()
    print(f"\n📂 OPEN TRADES: {len(open_trades)}")
    print("-" * 40)
    
    if open_trades:
        for trade in open_trades:
            print(f"  #{trade.id}: IBIT ${trade.short_strike}/${trade.long_strike}")
            print(f"       Exp: {trade.expiration}, Credit: ${trade.entry_credit:.2f}")
            print(f"       PoP: {trade.entry_pop:.0f}%, Qty: {trade.quantity}")
            print()
    else:
        print("  No open trades.")
    
    # Get all trades
    all_trades = await trader.get_all_trades(limit=10)
    print(f"\n📜 RECENT TRADES (last 10):")
    print("-" * 40)
    
    if all_trades:
        for trade in all_trades:
            status_emoji = {
                'OPEN': '🟢',
                'CLOSED_PROFIT': '💰',
                'CLOSED_LOSS': '❌',
                'CLOSED_TIME': '⏰',
                'EXPIRED': '📅',
            }.get(trade.status.value, '⚪')
            
            pnl_str = f"${trade.pnl_dollars:.2f}" if trade.pnl_dollars else "N/A"
            print(f"  {status_emoji} #{trade.id}: ${trade.short_strike}/${trade.long_strike} "
                  f"-> P&L: {pnl_str}")
    else:
        print("  No trades in database.")
    
    # Get statistics
    summary = await tracker.get_summary()
    print(f"\n📊 PERFORMANCE SUMMARY:")
    print("-" * 40)
    print(f"  Total Trades: {summary['total_trades']}")
    print(f"  Closed: {summary['closed_trades']}")
    print(f"  Win Rate: {summary['win_rate']:.1f}%")
    print(f"  Total P&L: ${summary['total_pnl']:.2f}")
    print(f"  Avg P&L: ${summary['avg_pnl']:.2f}")
    print(f"  Max Drawdown: ${summary['max_drawdown']:.2f}")
    
    # IV Rank analysis
    if summary.get('by_iv_rank'):
        print(f"\n📈 BY IV RANK:")
        print("-" * 40)
        for bucket in summary['by_iv_rank']:
            print(f"  {bucket.iv_rank_range}: "
                  f"{bucket.trade_count} trades, "
                  f"{bucket.win_rate:.0f}% win, "
                  f"${bucket.total_pnl:.2f}")
    
    await db.close()
    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
