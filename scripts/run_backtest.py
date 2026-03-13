"""
Run Strategy Backtest
=====================

Backtests the IBIT put spread strategy using BITO as proxy.
Run: python scripts/run_backtest.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.backtester import StrategyBacktester


def main():
    """Run backtest with current strategy parameters."""
    print("=" * 60)
    print("📈 IBIT PUT SPREAD STRATEGY BACKTEST")
    print("    Using BITO as proxy for IBIT")
    print("    Account Size: $5,000")
    print("=" * 60)
    
    # Current parameters (looser for more trades)
    params = {
        'iv_threshold': 0.40,         # 40% IV threshold
        'price_drop_trigger': -0.02,  # 2% drop trigger
        'target_delta': 0.18,
        'spread_width_pct': 0.05,
        'profit_target': 0.50,
        'time_exit_dte': 5,
        'entry_dte': 14,
    }
    
    backtester = StrategyBacktester(
        symbol="BITO", 
        params=params,
        account_size=5000.0,
    )
    
    # Backtest last 2 years
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
    
    print(f"\nPeriod: {start_date} to {end_date}")
    print("Fetching data and running simulation...\n")
    
    result = backtester.run_backtest(start_date, end_date)
    print(backtester.format_report(result))
    
    # Sample trades with % returns
    if result.trades:
        print("\n📋 SAMPLE TRADES:")
        print("-" * 60)
        for trade in result.trades[:10]:
            win = "✅" if trade.total_pnl > 0 else "❌"
            print(f"  {win} {trade.entry_date}: "
                  f"${trade.short_strike:.0f}/${trade.long_strike:.0f} x{trade.num_contracts} "
                  f"→ {trade.account_return_pct:+.2f}% ({trade.exit_reason})")
        
        # Monthly breakdown
        print("\n📅 MONTHLY RETURNS:")
        print("-" * 60)
        monthly = {}
        for trade in result.trades:
            month = trade.entry_date[:7]
            monthly[month] = monthly.get(month, 0) + trade.account_return_pct
        
        for month, ret in sorted(monthly.items()):
            bar = "█" * int(abs(ret) * 2) if abs(ret) > 0.5 else "▪"
            print(f"  {month}: {ret:+6.2f}% {bar}")
    
    print("\n" + "=" * 60)
    print("💡 To optimize parameters: python scripts/optimize_params.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
