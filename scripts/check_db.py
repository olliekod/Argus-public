"""
Database Verification Script
============================

Test that the SQLite database is working and view stored data.
"""

import asyncio
import sqlite3
from pathlib import Path
from datetime import datetime


def check_database():
    """Check database and display stored data."""
    db_path = Path("data/argus.db")
    
    if not db_path.exists():
        print("❌ Database file not found at data/argus.db")
        print("   Run Argus first to create the database.")
        return False
    
    print(f"✅ Database found: {db_path}")
    print(f"   Size: {db_path.stat().st_size / 1024:.1f} KB")
    print()
    
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # List all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = cursor.fetchall()
    
    print("=" * 60)
    print("DATABASE TABLES")
    print("=" * 60)
    
    for table in tables:
        table_name = table['name']
        cursor.execute(f"SELECT COUNT(*) as count FROM {table_name}")
        count = cursor.fetchone()['count']
        print(f"  {table_name}: {count} records")
    
    print()
    print("=" * 60)
    print("RECENT DETECTIONS (last 10)")
    print("=" * 60)
    
    cursor.execute("""
        SELECT timestamp, opportunity_type, asset, exchange, net_edge_bps, alert_tier
        FROM detections
        ORDER BY timestamp DESC
        LIMIT 10
    """)
    detections = cursor.fetchall()
    
    if detections:
        for d in detections:
            edge = d['net_edge_bps'] or 0
            print(f"  {d['timestamp'][:19]} | {d['opportunity_type']:<15} | {d['asset']:<10} | {edge:>6.1f} bps")
    else:
        print("  No detections yet - Argus is still gathering initial data")
    
    print()
    print("=" * 60)
    print("RECENT SYSTEM HEALTH (last 5)")
    print("=" * 60)
    
    cursor.execute("""
        SELECT timestamp, component, status
        FROM system_health
        ORDER BY timestamp DESC
        LIMIT 5
    """)
    health = cursor.fetchall()
    
    if health:
        for h in health:
            print(f"  {h['timestamp'][:19]} | {h['component']:<20} | {h['status']}")
    else:
        print("  No health checks recorded yet")
    
    print()
    print("=" * 60)
    print("FUNDING RATES (last 10)")
    print("=" * 60)
    
    cursor.execute("""
        SELECT timestamp, exchange, asset, funding_rate
        FROM funding_rates
        ORDER BY timestamp DESC
        LIMIT 10
    """)
    funding = cursor.fetchall()
    
    if funding:
        for f in funding:
            rate = f['funding_rate'] * 100 if f['funding_rate'] else 0
            print(f"  {f['timestamp'][:19]} | {f['exchange']:<10} | {f['asset']:<6} | {rate:>8.4f}%")
    else:
        print("  No funding rates logged yet")
    
    print()
    print("=" * 60)
    print("PRICE SNAPSHOTS (last 10)")
    print("=" * 60)
    
    cursor.execute("""
        SELECT timestamp, exchange, asset, price_type, price
        FROM price_snapshots
        ORDER BY timestamp DESC
        LIMIT 10
    """)
    prices = cursor.fetchall()
    
    if prices:
        for p in prices:
            print(f"  {p['timestamp'][:19]} | {p['exchange']:<10} | {p['asset']:<6} | {p['price_type']:<6} | ${p['price']:,.2f}")
    else:
        print("  No price snapshots logged yet")
    
    print()
    print("=" * 60)
    print("OPTIONS IV (last 10)")
    print("=" * 60)
    
    cursor.execute("""
        SELECT timestamp, asset, implied_volatility, underlying_price
        FROM options_iv
        ORDER BY timestamp DESC
        LIMIT 10
    """)
    iv_data = cursor.fetchall()
    
    if iv_data:
        for iv in iv_data:
            vol = iv['implied_volatility'] * 100 if iv['implied_volatility'] and iv['implied_volatility'] < 10 else (iv['implied_volatility'] or 0)
            print(f"  {iv['timestamp'][:19]} | {iv['asset']:<6} | IV: {vol:>6.1f}% | ${iv['underlying_price'] or 0:,.0f}")
    else:
        print("  No options IV data logged yet")
    
    conn.close()
    
    print()
    print("=" * 60)
    print("✅ Database verification complete!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    check_database()
