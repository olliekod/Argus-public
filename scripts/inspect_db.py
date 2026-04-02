# Created by Oliver Meihls

import sqlite3
import os

db_path = "data/argus.db"
if not os.path.exists(db_path):
    print(f"Error: {db_path} not found")
    exit(1)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

print("Tables in argus.db:")
for table in tables:
    print(f"- {table[0]}")
    # Get schema for interesting tables
    if "outcome" in table[0].lower() or "perf" in table[0].lower() or "trade" in table[0].lower() or "attrib" in table[0].lower():
        cursor.execute(f"PRAGMA table_info({table[0]})")
        columns = cursor.fetchall()
        for col in columns:
            print(f"  * {col[1]} ({col[2]})")

conn.close()
