# Created by Oliver Meihls

# Promotion gate for selecting stable live-candidate contexts.
#
# Gate conditions:
# - positive expectancy (avg pnl > 0)
# - minimum samples
# - positive expectancy in each of last N settlement cycles
# - concentration risk below threshold within candidate slice
#
# Candidate key:
# family|side|edge_bucket|price_bucket
# from settlement decision_context.

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _ctx_get(rec: Dict[str, Any], key: str, default: str = "na") -> str:
    ctx = rec.get("decision_context") or {}
    if not isinstance(ctx, dict):
        return default
    return str(ctx.get(key, default))


def _candidate_key(rec: Dict[str, Any]) -> str:
    family = str(rec.get("family") or "UNK")
    side = str(rec.get("side") or "?")
    return f"{family}|{side}|eb={_ctx_get(rec, 'eb')}|pb={_ctx_get(rec, 'pb')}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Promotion gate for Kalshi context candidates")
    ap.add_argument("--log", default="logs/paper_trades.jsonl", help="Path to paper_trades.jsonl")
    ap.add_argument("--hours", type=float, default=8.0, help="Lookback window in hours")
    ap.add_argument("--cycle-minutes", type=int, default=30, help="Cycle size in minutes for stability gating")
    ap.add_argument("--min-samples", type=int, default=400, help="Minimum settlements per candidate")
    ap.add_argument("--min-cycle-samples", type=int, default=50, help="Minimum per-cycle samples to count cycle")
    ap.add_argument("--min-cycles", type=int, default=3, help="Required consecutive positive cycles")
    ap.add_argument("--max-concentration-share", type=float, default=0.35, help="Max market-side qty share in candidate")
    ap.add_argument("--top", type=int, default=15, help="Rows to print")
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
    cycle_s = max(60, int(args.cycle_minutes) * 60)

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            if obj.get("type") != "settlement":
                continue
            ts = obj.get("timestamp")
            if not isinstance(ts, (int, float)) or ts < since_ts:
                continue
            rows.append(obj)
    if not rows:
        raise SystemExit("no settlements in selected window")

    agg: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        key = _candidate_key(r)
        item = agg.get(key)
        if item is None:
            item = {
                "n": 0,
                "pnl": 0.0,
                "wins": 0,
                "qty": 0.0,
                "market_side_qty": defaultdict(float),
                "cycles": defaultdict(lambda: {"n": 0, "pnl": 0.0}),
            }
            agg[key] = item
        pnl = float(r.get("pnl_usd") or 0.0)
        qty = float(r.get("quantity_contracts") or 0.0)
        won = 1 if bool(r.get("won")) else 0
        ts = float(r.get("timestamp") or 0.0)
        mside = f"{r.get('market_ticker')}|{r.get('side')}"
        cyc = int(ts // cycle_s)

        item["n"] += 1
        item["pnl"] += pnl
        item["wins"] += won
        item["qty"] += qty
        item["market_side_qty"][mside] += qty
        item["cycles"][cyc]["n"] += 1
        item["cycles"][cyc]["pnl"] += pnl

    passing: List[Tuple[str, int, float, float, float, int, float]] = []
    failing: List[Tuple[str, int, float, float, float, int, float, str]] = []
    for key, it in agg.items():
        n = int(it["n"])
        pnl = float(it["pnl"])
        avg = pnl / n if n else 0.0
        wr = 100.0 * float(it["wins"]) / n if n else 0.0
        qty = float(it["qty"])
        top_share = 0.0
        if qty > 0:
            top_share = max(float(v) for v in it["market_side_qty"].values()) / qty

        cycles = sorted(it["cycles"].items(), key=lambda x: x[0], reverse=True)
        good_cycles = 0
        for _, cyc in cycles:
            if int(cyc["n"]) < int(args.min_cycle_samples):
                continue
            if float(cyc["pnl"]) <= 0.0:
                break
            good_cycles += 1
            if good_cycles >= int(args.min_cycles):
                break

        reason = None
        if n < int(args.min_samples):
            reason = "min_samples"
        elif avg <= 0.0:
            reason = "non_positive_expectancy"
        elif top_share > float(args.max_concentration_share):
            reason = "concentration_too_high"
        elif good_cycles < int(args.min_cycles):
            reason = "insufficient_positive_cycles"

        if reason is None:
            passing.append((key, n, pnl, avg, wr, good_cycles, top_share))
        else:
            failing.append((key, n, pnl, avg, wr, good_cycles, top_share, reason))

    passing.sort(key=lambda x: x[3], reverse=True)
    failing.sort(key=lambda x: x[3])

    total = len(rows)
    total_pnl = sum(float(r.get("pnl_usd") or 0.0) for r in rows)
    total_wr = 100.0 * sum(1 for r in rows if bool(r.get("won"))) / max(1, total)
    print(f"window_hours={args.hours:.2f} settlements={total} pnl={total_pnl:.2f} wr={total_wr:.2f}%")
    print(
        f"gate=min_samples:{args.min_samples} min_cycles:{args.min_cycles} "
        f"min_cycle_samples:{args.min_cycle_samples} max_concentration_share:{args.max_concentration_share:.2f}"
    )

    print("\nPROMOTABLE CANDIDATES")
    if not passing:
        print("  (none)")
    else:
        for row in passing[: args.top]:
            key, n, pnl, avg, wr, cyc, share = row
            print(
                f"  {key:55} n={n:6d} pnl={pnl:10.2f} avg={avg:8.3f} "
                f"wr={wr:6.2f}% cycles={cyc} top_share={share:5.2f}"
            )

    print("\nDO-NOT-PROMOTE (WORST)")
    if not failing:
        print("  (none)")
    else:
        for row in failing[: args.top]:
            key, n, pnl, avg, wr, cyc, share, reason = row
            print(
                f"  {key:55} n={n:6d} pnl={pnl:10.2f} avg={avg:8.3f} "
                f"wr={wr:6.2f}% cycles={cyc} top_share={share:5.2f} reason={reason}"
            )


if __name__ == "__main__":
    main()
