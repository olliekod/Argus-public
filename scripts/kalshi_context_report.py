# Created by Oliver Meihls

# Compact causal-slice report for Argus Kalshi paper logs.
#
# Reads logs/paper_trades.jsonl and analyzes settlement outcomes grouped by
# decision_context fields written at fill time.
#
# This script is JSONL-only and does not write to the DB.

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _load_settlements(path: Path, since_ts: Optional[float]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if obj.get("type") != "settlement":
                continue
            ts = obj.get("timestamp")
            if since_ts is not None and isinstance(ts, (int, float)) and ts < since_ts:
                continue
            rows.append(obj)
    return rows


def _agg(rows: Iterable[Dict[str, Any]], key_fn) -> List[Tuple[str, int, float, float, float]]:
    stats: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])  # n, pnl_sum, wins
    for r in rows:
        key = key_fn(r)
        pnl = float(r.get("pnl_usd") or 0.0)
        won = 1.0 if bool(r.get("won")) else 0.0
        s = stats[key]
        s[0] += 1.0
        s[1] += pnl
        s[2] += won
    out: List[Tuple[str, int, float, float, float]] = []
    for key, (n_f, pnl_sum, wins) in stats.items():
        n = int(n_f)
        avg = pnl_sum / n if n else 0.0
        wr = (100.0 * wins / n) if n else 0.0
        out.append((key, n, pnl_sum, avg, wr))
    return out


def _print_table(title: str, rows: List[Tuple[str, int, float, float, float]], min_samples: int, top: int, reverse: bool) -> None:
    filt = [r for r in rows if r[1] >= min_samples]
    filt.sort(key=lambda x: x[2], reverse=reverse)
    print()
    print(title)
    if not filt:
        print("  (no rows)")
        return
    for key, n, pnl_sum, avg, wr in filt[:top]:
        print(f"  {key:45} n={n:6d} pnl={pnl_sum:10.2f} avg={avg:8.3f} wr={wr:6.2f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description="Causal-slice settlement report from paper_trades.jsonl")
    ap.add_argument("--log", default="logs/paper_trades.jsonl", help="Path to paper_trades.jsonl")
    ap.add_argument("--hours", type=float, default=3.0, help="Lookback window in hours")
    ap.add_argument("--min-samples", type=int, default=300, help="Minimum samples per slice")
    ap.add_argument("--top", type=int, default=15, help="Top rows to print per section")
    args = ap.parse_args()

    path = Path(args.log)
    if not path.exists():
        raise SystemExit(f"log not found: {path}")

    max_ts: Optional[float] = None
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            ts = obj.get("timestamp")
            if isinstance(ts, (int, float)):
                if max_ts is None or ts > max_ts:
                    max_ts = ts
    if max_ts is None:
        raise SystemExit("no timestamped records found")

    since_ts = max_ts - (args.hours * 3600.0)
    settlements = _load_settlements(path, since_ts=since_ts)
    total = len(settlements)
    if total == 0:
        raise SystemExit("no settlement records in selected window")

    total_pnl = sum(float(r.get("pnl_usd") or 0.0) for r in settlements)
    total_wins = sum(1 for r in settlements if bool(r.get("won")))
    ctx_present = sum(1 for r in settlements if isinstance(r.get("decision_context"), dict) and r.get("decision_context"))

    print(f"window_hours={args.hours:.2f} settlements={total} pnl={total_pnl:.2f} wr={100.0*total_wins/total:.2f}%")
    print(f"context_coverage={ctx_present}/{total} ({100.0*ctx_present/total:.2f}%)")

    def ctx_get(r: Dict[str, Any], k: str, default: str = "na") -> str:
        ctx = r.get("decision_context") or {}
        if not isinstance(ctx, dict):
            return default
        v = ctx.get(k, default)
        return str(v)

    by_family_side = _agg(
        settlements,
        lambda r: f"{r.get('family','UNK')}|{r.get('side','?')}",
    )
    by_edge_bucket = _agg(settlements, lambda r: f"eb={ctx_get(r, 'eb')}")
    by_price_bucket = _agg(settlements, lambda r: f"pb={ctx_get(r, 'pb')}")
    by_tts_bucket = _agg(settlements, lambda r: f"tb={ctx_get(r, 'tb')}")
    by_edge_price = _agg(settlements, lambda r: f"eb={ctx_get(r, 'eb')}|pb={ctx_get(r, 'pb')}")
    by_fam_edge_price = _agg(
        settlements,
        lambda r: f"{r.get('family','UNK')}|{r.get('side','?')}|eb={ctx_get(r, 'eb')}|pb={ctx_get(r, 'pb')}",
    )

    _print_table("Best Family|Side", by_family_side, args.min_samples, args.top, reverse=True)
    _print_table("Worst Family|Side", by_family_side, args.min_samples, args.top, reverse=False)
    _print_table("Best Edge Bucket", by_edge_bucket, args.min_samples, args.top, reverse=True)
    _print_table("Worst Edge Bucket", by_edge_bucket, args.min_samples, args.top, reverse=False)
    _print_table("Best Price Bucket", by_price_bucket, args.min_samples, args.top, reverse=True)
    _print_table("Worst Price Bucket", by_price_bucket, args.min_samples, args.top, reverse=False)
    _print_table("Best TTS Bucket", by_tts_bucket, args.min_samples, args.top, reverse=True)
    _print_table("Worst TTS Bucket", by_tts_bucket, args.min_samples, args.top, reverse=False)
    _print_table("Best Edge|Price", by_edge_price, args.min_samples, args.top, reverse=True)
    _print_table("Worst Edge|Price", by_edge_price, args.min_samples, args.top, reverse=False)
    _print_table("Best Family|Side|Edge|Price", by_fam_edge_price, args.min_samples, args.top, reverse=True)
    _print_table("Worst Family|Side|Edge|Price", by_fam_edge_price, args.min_samples, args.top, reverse=False)


if __name__ == "__main__":
    main()
