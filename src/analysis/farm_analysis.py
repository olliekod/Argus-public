"""
Paper Trader Farm Analysis Utility
==================================

Standalone script to analyze the performance of the 400,000 parallel paper traders.
Identifies top-performing parameter combinations and strategy clusters.
"""

import asyncio
import os
import sys
import pandas as pd
import sqlite3
from typing import List, Dict

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

async def get_top_traders(db_path: str, top_n: int = 20, days: int = 30):
    """Fetch top performing traders from the database."""
    conn = sqlite3.connect(db_path)
    
    query = f"""
    SELECT 
        trader_id,
        strategy_type,
        COUNT(*) as total_trades,
        SUM(realized_pnl) as total_pnl,
        AVG(realized_pnl) as avg_pnl,
        SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate
    FROM paper_trades
    WHERE timestamp >= date('now', '-{days} days')
    GROUP BY trader_id
    HAVING total_trades >= 3
    ORDER BY total_pnl DESC
    LIMIT {top_n}
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def analyze_farm():
    """Run analysis and print report."""
    db_path = "data/argus.db"
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return

    print("=" * 60)
    print("🚜 ARGUS PAPER TRADER FARM ANALYSIS")
    print("=" * 60)
    
    try:
        loop = asyncio.get_event_loop()
        df = loop.run_until_complete(get_top_traders(db_path))
        
        if df.empty:
            print("\nNo trades found to analyze yet. Check back after a few days of market data!")
            return
            
        print(f"\nTOP {len(df)} PERFORMERS (Last 30 Days):")
        print("-" * 60)
        print(df.to_string(index=False))
        
        # Analyze strategy distribution
        print("\n\nSTRATEGY PERFORMANCE:")
        print("-" * 60)
        conn = sqlite3.connect(db_path)
        strat_query = """
        SELECT 
            strategy_type,
            COUNT(*) as total_trades,
            SUM(realized_pnl) as total_pnl,
            AVG(realized_pnl) as avg_pnl
        FROM paper_trades
        GROUP BY strategy_type
        """
        strat_df = pd.read_sql_query(strat_query, conn)
        conn.close()
        print(strat_df.to_string(index=False))
        
    except Exception as e:
        print(f"Error running analysis: {e}")

if __name__ == "__main__":
    analyze_farm()
