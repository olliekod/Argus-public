"""
Paper Trading Performance Report
================================

Shows complete performance of paper trading since inception.
Run: python scripts/paper_performance.py
"""

import sys
import asyncio
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.database import Database
from src.analysis.performance_tracker import PerformanceTracker

# Default database path
DB_PATH = "data/argus.db"


async def run_report():
    """Generate full paper trading performance report."""
    print()
    print("=" * 70)
    print("PAPER TRADING PERFORMANCE REPORT")
    print("=" * 70)
    
    # Initialize database with path
    db = Database(DB_PATH)
    await db.connect()
    
    tracker = PerformanceTracker(db)
    
    # Get all trades
    await tracker.load_trades()
    
    # Summary
    stats = await tracker.get_summary_stats()
    
    print()
    print("SUMMARY")
    print("-" * 50)
    
    if stats['total_trades'] == 0:
        print("  No paper trades yet.")
        print()
        print("  Paper trading will log trades when the IBIT detector")
        print("  fires signals. This happens when:")
        print("    - BTC IV > threshold (currently 25%)")
        print("    - IBIT drops > threshold (currently 0.5%)")
        print("    - During US market hours (9:30 AM - 4:00 PM EST)")
        print()
        print("  Keep Argus running to accumulate paper trades.")
        await db.close()
        return
    
    print(f"  Total Trades:    {stats['total_trades']}")
    print(f"  Open Trades:     {stats['open_trades']}")
    print(f"  Closed Trades:   {stats['closed_trades']}")
    print()
    
    if stats['closed_trades'] > 0:
        print(f"  Winners:         {stats['winners']}")
        print(f"  Losers:          {stats['losers']}")
        print(f"  Win Rate:        {stats['win_rate']:.1f}%")
        print()
        print("P&L")
        print("-" * 50)
        print(f"  Total P&L:       ${stats['total_pnl']:+.2f}")
        print(f"  Total Return:    {stats['total_return_pct']:+.2f}%")
        print(f"  Avg Return:      {stats['avg_return_pct']:+.2f}%")
        print(f"  Best Trade:      {stats['best_trade_pct']:+.2f}%")
        print(f"  Worst Trade:     {stats['worst_trade_pct']:+.2f}%")
        print(f"  Max Drawdown:    {stats['max_drawdown_pct']:.2f}%")
    
    # Open positions
    open_trades = await tracker.get_open_trades()
    
    if open_trades:
        print()
        print("OPEN POSITIONS")
        print("-" * 50)
        for trade in open_trades:
            days_held = (datetime.now() - datetime.fromisoformat(trade['entry_time'])).days
            print(f"  #{trade['id']}: {trade['expiration']} "
                  f"${trade['short_strike']}/{trade['long_strike']} "
                  f"x{trade['num_contracts']} | "
                  f"{days_held}d held | "
                  f"Credit: ${trade['entry_credit']:.2f}")
    
    # Recent closed trades
    closed_trades = await tracker.get_closed_trades(limit=10)
    
    if closed_trades:
        print()
        print("RECENT CLOSED TRADES (last 10)")
        print("-" * 50)
        for trade in closed_trades:
            status = "[+]" if trade['pnl'] > 0 else "[-]"
            print(f"  {status} {trade['exit_time'][:10]}: "
                  f"${trade['short_strike']}/{trade['long_strike']} | "
                  f"P&L: ${trade['pnl']:+.2f} ({trade['return_pct']:+.2f}%) | "
                  f"{trade['exit_reason']}")
    
    # By IV bucket
    iv_analysis = await tracker.get_analysis_by_iv_bucket()
    
    if iv_analysis:
        print()
        print("PERFORMANCE BY IV BUCKET")
        print("-" * 50)
        for bucket, data in iv_analysis.items():
            print(f"  {bucket}: {data['count']} trades, "
                  f"{data['win_rate']:.0f}% win, "
                  f"${data['total_pnl']:+.2f}")
    
    # Time analysis
    print()
    print("TIME ANALYSIS")
    print("-" * 50)
    
    first_trade = await tracker.get_first_trade_date()
    if first_trade:
        days_since = (datetime.now() - datetime.fromisoformat(first_trade)).days
        trades_per_week = stats['total_trades'] / max(1, days_since / 7)
        print(f"  First Trade:     {first_trade[:10]}")
        print(f"  Days Active:     {days_since}")
        print(f"  Trades/Week:     {trades_per_week:.1f}")
    
    print()
    print("=" * 70)
    
    await db.close()


def main():
    """Entry point."""
    asyncio.run(run_report())


if __name__ == "__main__":
    main()
