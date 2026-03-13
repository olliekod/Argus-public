import sqlite3
import os

db_path = "data/argus.db"
if not os.path.exists(db_path):
    print(f"Database {db_path} not found")
    exit(1)

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

print("Checking market_bars table for IBIT and BITO (by source)...")
symbols = ["IBIT", "BITO"]
for symbol in symbols:
    cursor.execute("SELECT source, COUNT(*), MAX(timestamp) FROM market_bars WHERE symbol = ? GROUP BY source", (symbol,))
    rows = cursor.fetchall()
    if not rows:
        print(f"{symbol}: No rows found")
    for row in rows:
        print(f"{symbol} (source={row[0]}): Count={row[1]}, Last={row[2]}")

conn.close()
