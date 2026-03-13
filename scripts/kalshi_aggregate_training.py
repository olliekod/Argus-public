#!/usr/bin/env python3
"""
kalshi_aggregate_training.py — Merge Kalshi training run archives into a classifier-ready dataset.

Reads all logs/training_data/run_*.jsonl files, deduplicates by order_id,
extracts classifier features, and writes:
  - logs/training_data/aggregate_TIMESTAMP.jsonl  (full records, one per line)
  - logs/training_data/labeled_TIMESTAMP.csv      (flat CSV with feature columns)

Usage:
  python scripts/kalshi_aggregate_training.py
  python scripts/kalshi_aggregate_training.py --input-dir logs/training_data --output-dir logs/training_data
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Columns written to the classifier CSV.  Records missing a column get "".
FEATURE_COLS = [
    "order_id",
    "family",
    "side",
    "fill_price_cents",
    "edge_at_entry",
    "tts_at_entry",
    "drift_at_entry",
    "flow_at_entry",
    "obi_at_entry",
    "strike_distance_pct",
    "near_money",
    "outcome",  # target: "win" | "loss" | "scratch"
]


def _ts() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")


def aggregate(input_dir: Path, output_dir: Path) -> dict:
    """
    Merge all run_*.jsonl files in input_dir, deduplicate by order_id,
    and write aggregate JSONL + labeled CSV to output_dir.

    Returns a summary dict with keys:
      total_files, total_records_raw, unique_records,
      aggregate_path (str or None), csv_path (str or None)
    """
    run_files = sorted(input_dir.glob("run_*.jsonl"))
    if not run_files:
        return {
            "total_files": 0,
            "total_records_raw": 0,
            "unique_records": 0,
            "aggregate_path": None,
            "csv_path": None,
        }

    seen: dict[str, dict] = {}  # dedup key → record
    total_raw = 0
    _nokey_counter = 0

    for path in run_files:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            total_raw += 1
            # Use order_id, fill_id, or id as dedup key
            oid = rec.get("order_id") or rec.get("fill_id") or rec.get("id")
            if oid:
                if oid not in seen:
                    seen[oid] = rec
            else:
                # No dedup key — include with synthetic key to avoid silent drops
                _nokey_counter += 1
                seen[f"_nokey_{_nokey_counter}"] = rec

    records = list(seen.values())
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = _ts()

    # Full aggregate JSONL
    agg_path = output_dir / f"aggregate_{stamp}.jsonl"
    with agg_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Classifier-ready CSV (feature columns only; missing → empty string)
    csv_path = output_dir / f"labeled_{stamp}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FEATURE_COLS, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow({col: rec.get(col, "") for col in FEATURE_COLS})

    return {
        "total_files": len(run_files),
        "total_records_raw": total_raw,
        "unique_records": len(records),
        "aggregate_path": str(agg_path),
        "csv_path": str(csv_path),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Aggregate Kalshi training run archives into a classifier-ready dataset."
    )
    ap.add_argument(
        "--input-dir",
        default=str(ROOT / "logs" / "training_data"),
        help="Directory containing run_*.jsonl files (default: logs/training_data)",
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "logs" / "training_data"),
        help="Output directory for aggregate files (default: logs/training_data)",
    )
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 1

    summary = aggregate(input_dir=input_dir, output_dir=output_dir)

    if summary["total_files"] == 0:
        print("No run_*.jsonl files found. Run at least one cycle first.")
        return 0

    print(f"Files processed  : {summary['total_files']}")
    print(f"Raw records      : {summary['total_records_raw']}")
    print(f"Unique records   : {summary['unique_records']}")
    print(f"Aggregate JSONL  : {summary['aggregate_path']}")
    print(f"Classifier CSV   : {summary['csv_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
