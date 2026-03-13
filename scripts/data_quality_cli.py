#!/usr/bin/env python3
"""
Data Quality CLI
================

Command-line tool to generate data quality reports.

Usage:
    python -m scripts.data_quality_cli --providers alpaca yahoo --symbols IBIT BITO --days 7
    python -m scripts.data_quality_cli --output reports/quality.json
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import load_all_config
from src.core.database import Database
from src.analysis.data_quality import DataQualityReport


async def main():
    parser = argparse.ArgumentParser(
        description="Generate data quality report for Argus data sources"
    )
    parser.add_argument(
        "--providers",
        nargs="*",
        default=["alpaca", "yahoo"],
        help="Data providers to analyze (default: alpaca yahoo)",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=["IBIT", "BITO"],
        help="Symbols to analyze (default: IBIT BITO)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days to analyze (default: 7)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (optional, prints to stdout if not specified)",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="config",
        help="Path to config directory (default: config)",
    )
    
    args = parser.parse_args()
    
    # Load config and initialize database
    full_config = load_all_config(args.config_dir)
    config = full_config
    db_path = config.get("database", {}).get("path", "data/argus.db")
    db = Database(db_path)
    await db.connect()
    
    try:
        # Generate report
        report = DataQualityReport(db)
        result = await report.generate_report(
            providers=args.providers,
            symbols=args.symbols,
            days=args.days,
        )
    finally:
        await db.close()
    
    # Format output
    output = json.dumps(result, indent=2)
    
    # Write or print
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output)
        print(f"Report written to {args.output}")
    else:
        print(output)
    
    # Summary
    print("\n--- Summary ---")
    for m in result.get("metrics", []):
        status = "✓" if m.get("gap_rate_pct", 0) < 5 else "⚠" if m.get("gap_rate_pct", 0) < 20 else "✗"
        print(
            f"{status} {m['provider']}/{m['symbol']}: "
            f"{m['actual_bars']}/{m['expected_bars']} bars "
            f"({m.get('gap_rate_pct', 0):.1f}% missing), "
            f"staleness p50={m.get('staleness_p50_s', '-')}s"
        )


if __name__ == "__main__":
    asyncio.run(main())
