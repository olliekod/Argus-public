import sqlite3
import os
from datetime import datetime, timezone

DB_PATH = "data/argus.db"

def check_crypto_quality():
    if not os.path.exists(DB_PATH):
        print(f"Error: {DB_PATH} not found")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    print("Checking Crypto Data Quality (Last 30 Minutes)...")
    print("-" * 60)

    # 1. Check Bybit Quotes (Spot/Perp)
    query_bybit = """
    SELECT symbol, count(*) as count, max(timestamp) as last_ts
    FROM market_bars
    WHERE source = 'bybit'
    GROUP BY symbol
    ORDER BY count DESC
    """
    cursor.execute(query_bybit)
    rows = cursor.fetchall()
    print("\nBybit Market Bars:")
    for row in rows:
        print(f"- {row['symbol']}: {row['count']} bars, last: {row['last_ts']}")

    # 2. Check Deribit metrics
    query_metrics = """
    SELECT source, metric, symbol, count(*) as count, max(timestamp) as last_ts
    FROM market_metrics
    WHERE source IN ('deribit', 'bybit')
    GROUP BY 1, 2, 3
    ORDER BY last_ts DESC
    LIMIT 20
    """
    cursor.execute(query_metrics)
    rows = cursor.fetchall()
    print("\nDeribit/Bybit Metrics:")
    for row in rows:
        print(f"- {row['source']} | {row['metric']} | {row['symbol']}: {row['count']} samples, last: {row['last_ts']}")

    # 3. Check Option Chain Snapshots (Deribit provides these usually via Tastytrade or direct)
    # Actually, DeribitClient fetches public options data.
    query_snaps = """
    SELECT provider, symbol, count(*) as count, max(timestamp_ms) as last_ts_ms
    FROM option_chain_snapshots
    WHERE provider = 'deribit'
    GROUP BY symbol
    """
    try:
        cursor.execute(query_snaps)
        rows = cursor.fetchall()
        print("\nDeribit Option Chain Snapshots:")
        for row in rows:
            last_ts = datetime.fromtimestamp(row['last_ts_ms']/1000, tz=timezone.utc).isoformat()
            print(f"- {row['symbol']}: {row['count']} snapshots, last: {last_ts}")
    except sqlite3.OperationalError:
        print("\nNo option_chain_snapshots table or error querying it.")

    conn.close()

if __name__ == "__main__":
    check_crypto_quality()
