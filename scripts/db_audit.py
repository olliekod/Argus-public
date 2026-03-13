import sqlite3
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

DB_PATH = "data/argus.db"

def format_table(headers: List[str], rows: List[List[Any]]) -> str:
    """Pretty-print a table to console."""
    if not rows:
        return "No data found."
    
    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val)))
    
    # Format header
    header_str = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep_str = "-+-".join("-" * w for w in widths)
    
    output = [header_str, sep_str]
    for row in rows:
        output.append(" | ".join(str(val).ljust(widths[i]) for i, val in enumerate(row)))
    
    return "\n".join(output)

def parse_iso(ts_str: str) -> datetime:
    """Safe parse of ISO timestamp, handling various formats."""
    # SQLite might store with/without Z or +00:00 or as naive string
    try:
        t_str = ts_str
        if t_str.endswith('Z'):
            t_str = t_str[:-1] + '+00:00'
        
        # Handle cases where microsecond precision varies or space vs T
        t_str = t_str.replace(' ', 'T')
        
        dt = datetime.fromisoformat(t_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        # Last ditch effort: try to parse and force UTC
        try:
            from dateutil import parser
            return parser.isoparse(ts_str).replace(tzinfo=timezone.utc)
        except (ImportError, ValueError):
            # If dateutil is missing or also fails, try a manual split
            try:
                dt = datetime.fromisoformat(ts_str.split('.')[0])
                return dt.replace(tzinfo=timezone.utc)
            except:
                raise ValueError(f"Could not parse timestamp: {ts_str}")

def get_duration_info(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Calculate the total duration of data in the DB across all tables."""
    tables = ["market_bars", "market_metrics", "detections"]
    min_ts = None
    max_ts = None
    
    for table in tables:
        try:
            row = conn.execute(f"SELECT min(timestamp), max(timestamp) FROM {table}").fetchone()
            if row and row[0]:
                t_min = parse_iso(row[0])
                t_max = parse_iso(row[1])
                if min_ts is None or t_min < min_ts: min_ts = t_min
                if max_ts is None or t_max > max_ts: max_ts = t_max
        except sqlite3.OperationalError:
            continue
            
    if not min_ts or not max_ts:
        return {"hours": 0, "minutes": 0, "seconds": 0, "total_seconds": 0}
        
    delta = max_ts - min_ts
    total_sec = delta.total_seconds()
    return {
        "hours": total_sec / 3600,
        "minutes": total_sec / 60,
        "seconds": total_sec,
        "total_seconds": total_sec,
        "start": min_ts,
        "end": max_ts
    }

def audit_bars(conn: sqlite3.Connection):
    """Analyze bar continuity and gaps."""
    print("\n>>> 1. Bar Continuity Analysis")
    rows = conn.execute("SELECT DISTINCT symbol FROM market_bars").fetchall()
    symbols = [r[0] for r in rows]
    
    table_rows = []
    gap_reports = []
    
    for sym in symbols:
        data = conn.execute(
            "SELECT timestamp FROM market_bars WHERE symbol = ? ORDER BY timestamp ASC", 
            (sym,)
        ).fetchall()
        
        if not data: continue
        
        timestamps = [parse_iso(r[0]) for r in data]
        first, last = timestamps[0], timestamps[-1]
        
        # Expected count based on 1-min intervals
        duration_min = int((last - first).total_seconds() / 60) + 1
        actual_count = len(timestamps)
        missing = duration_min - actual_count
        
        # Find gaps
        max_gap = 0
        gaps_found = 0
        for i in range(1, len(timestamps)):
            diff = (timestamps[i] - timestamps[i-1]).total_seconds() / 60
            if diff > 1.1: # More than 1 minute (buffer for floating point)
                gaps_found += 1
                max_gap = max(max_gap, int(diff - 1))
        
        table_rows.append([
            sym, actual_count, duration_min, missing, f"{max_gap}m"
        ])
        
        if missing > 0:
            gap_reports.append(f"  [!] {sym}: {missing} missing bars. Largest gap: {max_gap} mins.")

    print(format_table(["Symbol", "Actual", "Expected", "Missing", "Max Gap"], table_rows))
    if gap_reports:
        print("\nGap Details:")
        for report in gap_reports:
            print(report)

def audit_metrics(conn: sqlite3.Connection, duration: Dict[str, Any]):
    """Analyze metric ingestion rates."""
    print("\n>>> 2. Metric Ingestion Cadence")
    sql = """
        SELECT metric, source, symbol, count(*) as count 
        FROM market_metrics 
        GROUP BY 1, 2, 3 
        ORDER BY count DESC
    """
    rows = conn.execute(sql).fetchall()
    
    table_rows = []
    total_rows = 0
    hrs = max(duration["hours"], 0.01) # Avoid div by zero
    
    for r in rows:
        rate = r['count'] / hrs
        total_rows += r['count']
        table_rows.append([
            r['metric'], r['source'], r['symbol'], r['count'], f"{rate:.1f}/hr"
        ])
        
    print(format_table(["Metric", "Src", "Sym", "Count", "Rate"], table_rows))
    print(f"\nTotal metrics: {total_rows} ({total_rows / hrs:.1f}/hr average)")

def audit_detections(conn: sqlite3.Connection, duration: Dict[str, Any]):
    """Analyze detection volume."""
    print("\n>>> 3. Signal Detection Analysis")
    sql = """
        SELECT opportunity_type, asset, count(*) as count 
        FROM detections 
        GROUP BY 1, 2 
        ORDER BY count DESC
    """
    try:
        rows = conn.execute(sql).fetchall()
        table_rows = []
        mins = max(duration["minutes"], 0.01)
        
        for r in rows:
            rate = r['count'] / mins
            table_rows.append([
                r['opportunity_type'], r['asset'], r['count'], f"{rate:.2f}/min"
            ])
            
        print(format_table(["Type", "Asset", "Count", "Rate"], table_rows))
    except sqlite3.OperationalError as e:
        print(f"Skipping detections: {e}")

def audit_storage(db_path: str, duration: Dict[str, Any]):
    """Analyze DB size and growth projections."""
    print("\n>>> 4. Storage & Growth Diagnostics")
    size_bytes = os.path.getsize(db_path)
    size_mb = size_bytes / (1024 * 1024)
    
    hrs = max(duration["hours"], 0.01)
    rate_mb = size_mb / hrs
    
    print(f"Current Size: {size_mb:.2f} MB")
    print(f"Growth Rate:  {rate_mb:.2f} MB/hour")
    print(f"Projected 24h: {(size_mb + rate_mb * 24):.2f} MB")
    print(f"Projected 7d:  {(size_mb + rate_mb * 24 * 7):.2f} MB")

def audit_freshness(conn: sqlite3.Connection):
    """Check newest timestamps for data staleness."""
    print("\n>>> 5. Data Freshness (Age)")
    tables = ["price_snapshots", "market_bars", "market_metrics", "detections", "system_health"]
    now = datetime.now(timezone.utc)
    
    table_rows = []
    for table in tables:
        try:
            row = conn.execute(f"SELECT max(timestamp) FROM {table}").fetchone()
            if row and row[0]:
                ts = parse_iso(row[0])
                age = (now - ts).total_seconds()
                
                status = "OK"
                if age > 300: status = "STALE (>5m)"
                if age > 3600: status = "DEAD (>1h)"
                
                age_str = f"{int(age // 60)}m {int(age % 60)}s" if age > 60 else f"{int(age)}s"
                table_rows.append([table, ts.strftime("%H:%M:%S"), age_str, status])
            else:
                table_rows.append([table, "N/A", "-", "EMPTY"])
        except sqlite3.OperationalError:
            continue

    print(format_table(["Table", "Newest", "Age", "Status"], table_rows))

def run_audit():
    if not os.path.exists(DB_PATH):
        print(f"Error: Database file not found at {DB_PATH}")
        return

    print("=" * 70)
    print(f"ARGUS SOAK TEST AUDIT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        duration = get_duration_info(conn)
        if duration["total_seconds"] > 0:
            print(f"Data Span: {duration['total_seconds']/3600:.2f} hours")
            print(f"Range:     {duration['start'].strftime('%H:%M:%S')} -> {duration['end'].strftime('%H:%M:%S')} UTC")
        
        audit_bars(conn)
        audit_metrics(conn, duration)
        audit_detections(conn, duration)
        audit_storage(DB_PATH, duration)
        audit_freshness(conn)
        
    except Exception as e:
        import traceback
        print(f"Fatal audit error: {e}")
        traceback.print_exc()
    finally:
        conn.close()
        print("\n" + "=" * 70)

if __name__ == "__main__":
    run_audit()
