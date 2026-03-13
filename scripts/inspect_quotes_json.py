"""Dump a sample quotes_json from option_chain_snapshots with missing atm_iv."""
import sqlite3
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

db_path = sys.argv[1] if len(sys.argv) > 1 else "data/argus.db"
conn = sqlite3.connect(db_path)
row = conn.execute(
    "SELECT quotes_json FROM option_chain_snapshots "
    "WHERE (atm_iv IS NULL OR atm_iv <= 0) AND quotes_json != '' LIMIT 1"
).fetchone()
if row:
    s = row[0][:5000]
    print(s)
else:
    print("No rows")
conn.close()
