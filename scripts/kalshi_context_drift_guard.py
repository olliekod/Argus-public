# Created by Oliver Meihls

# Context drift guard: compare current-window performance of promoted (core)
# context keys against their policy baseline, flag drift, and optionally
# auto-demote drifted keys.
#
# Usage:
# python scripts/kalshi_context_drift_guard.py
# python scripts/kalshi_context_drift_guard.py --policy config/kalshi_context_policy.json --log logs/paper_trades.jsonl
# python scripts/kalshi_context_drift_guard.py --auto-demote --hours 2 --drift-threshold -0.25

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _ctx_get(rec: Dict[str, Any], key: str, default: str = "na") -> str:
    ctx = rec.get("decision_context") or {}
    if not isinstance(ctx, dict):
        return default
    return str(ctx.get(key, default))


def _context_key(rec: Dict[str, Any]) -> str:
    # Build context key matching kalshi_apply_promotion format.
    family = str(rec.get("family") or "UNK")
    side_code = _ctx_get(rec, "sd", "na")
    side = "yes" if side_code == "y" else ("no" if side_code == "n" else side_code)
    eb = _ctx_get(rec, "eb")
    pb = _ctx_get(rec, "pb")
    sdb = _ctx_get(rec, "sdb")
    nm = _ctx_get(rec, "nm", "na")
    return f"{family}|{side}|{eb}|{pb}|{sdb}|{nm}"


def _load_policy(policy_path: Path) -> Dict[str, Any]:
    with policy_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_window_settlements(
    log_path: Path,
    hours: float,
) -> List[Dict[str, Any]]:
    # Load settlement records from the last N hours of the log.
    max_ts: Optional[float] = None
    with log_path.open("r", encoding="utf-8", errors="ignore") as fh:
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
        return []

    since_ts = max_ts - (hours * 3600.0)
    rows: List[Dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as fh:
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

    return rows


def _aggregate_by_key(
    rows: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    # Aggregate settlements by context key.
    agg: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "total_pnl": 0.0, "wins": 0}
    )
    for r in rows:
        key = _context_key(r)
        item = agg[key]
        pnl = float(r.get("pnl_usd") or 0.0)
        won = 1 if bool(r.get("won")) else 0
        item["count"] += 1
        item["total_pnl"] += pnl
        item["wins"] += won
    return dict(agg)


def _detect_drift(
    policy_keys: Dict[str, Any],
    current_agg: Dict[str, Dict[str, Any]],
    drift_threshold: float,
) -> List[Dict[str, Any]]:
    # Compare core keys in policy against current window performance.
    #
    # Returns list of drift alert dicts for keys whose current avg_pnl
    # falls below drift_threshold.
    alerts: List[Dict[str, Any]] = []
    for key, info in policy_keys.items():
        if info.get("lane") != "core":
            continue

        baseline_avg = float(info.get("avg_pnl", 0.0))
        current = current_agg.get(key)

        if current is None:
            # No data in current window -- not a drift, just absent
            continue

        count = current["count"]
        if count == 0:
            continue

        current_avg = current["total_pnl"] / count
        current_wr = current["wins"] / count

        if current_avg < drift_threshold:
            alerts.append({
                "key": key,
                "baseline_avg_pnl": round(baseline_avg, 4),
                "current_avg_pnl": round(current_avg, 4),
                "current_count": count,
                "current_win_rate": round(current_wr, 4),
                "current_total_pnl": round(current["total_pnl"], 4),
                "delta": round(current_avg - baseline_avg, 4),
            })

    alerts.sort(key=lambda x: x["current_avg_pnl"])
    return alerts


def _apply_auto_demote(
    policy: Dict[str, Any],
    alerts: List[Dict[str, Any]],
    policy_path: Path,
) -> int:
    # Demote drifted core keys to explore in the policy file.
    #
    # Returns number of keys demoted.
    demoted = 0
    keys = policy.get("keys", {})
    for alert in alerts:
        key = alert["key"]
        if key in keys and keys[key].get("lane") == "core":
            keys[key]["lane"] = "explore"
            keys[key]["weight"] = 0.5
            demoted += 1

    if demoted > 0:
        policy["timestamp"] = datetime.now(timezone.utc).isoformat()
        # Update summary counts
        core_count = sum(1 for v in keys.values() if v["lane"] == "core")
        explore_count = sum(1 for v in keys.values() if v["lane"] == "explore")
        unknown_count = sum(1 for v in keys.values() if v["lane"] == "unknown")
        policy["summary"]["core_count"] = core_count
        policy["summary"]["explore_count"] = explore_count
        policy["summary"]["unknown_count"] = unknown_count

        with policy_path.open("w", encoding="utf-8") as fh:
            json.dump(policy, fh, indent=2, ensure_ascii=False)
            fh.write("\n")

    return demoted


def _write_drift_report(
    output_dir: Path,
    alerts: List[Dict[str, Any]],
    demoted: int,
    hours: float,
    drift_threshold: float,
    auto_demote: bool,
) -> Path:
    # Write drift report JSON to output_dir. Returns the report path.
    os.makedirs(output_dir, exist_ok=True)
    now = datetime.now(timezone.utc)
    filename = f"drift_report_{now.strftime('%Y%m%d_%H%M%S')}.json"
    report_path = output_dir / filename

    report = {
        "timestamp": now.isoformat(),
        "window_hours": hours,
        "drift_threshold": drift_threshold,
        "auto_demote": auto_demote,
        "demoted_count": demoted,
        "alert_count": len(alerts),
        "alerts": alerts,
    }

    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    return report_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare current window vs policy baseline for promoted contexts and flag drift."
    )
    ap.add_argument(
        "--policy",
        default="config/kalshi_context_policy.json",
        help="Path to context policy JSON (default: config/kalshi_context_policy.json)",
    )
    ap.add_argument(
        "--log",
        default="logs/paper_trades.jsonl",
        help="Path to paper_trades.jsonl (default: logs/paper_trades.jsonl)",
    )
    ap.add_argument(
        "--hours",
        type=float,
        default=2.0,
        help="Comparison window in hours (default: 2.0)",
    )
    ap.add_argument(
        "--drift-threshold",
        type=float,
        default=-0.25,
        help="Avg PnL threshold below which drift is flagged (default: -0.25)",
    )
    ap.add_argument(
        "--auto-demote",
        action="store_true",
        default=False,
        help="Automatically demote drifted core keys to explore in the policy file",
    )
    ap.add_argument(
        "--output-dir",
        default="logs/analysis",
        help="Directory for drift report output (default: logs/analysis)",
    )
    args = ap.parse_args()

    policy_path = Path(args.policy)
    log_path = Path(args.log)
    output_dir = Path(args.output_dir)

    # --- Load policy ---
    if not policy_path.exists():
        print(f"Policy file not found: {policy_path}")
        print("Run kalshi_apply_promotion.py first to generate a policy.")
        raise SystemExit(0)

    policy = _load_policy(policy_path)
    policy_keys = policy.get("keys", {})
    core_keys = {k: v for k, v in policy_keys.items() if v.get("lane") == "core"}

    if not core_keys:
        print("No core keys in policy. Nothing to monitor for drift.")
        raise SystemExit(0)

    print(f"Policy loaded: {len(core_keys)} core keys to monitor")

    # --- Load current window settlements ---
    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        print("No settlements to compare. Exiting.")
        raise SystemExit(0)

    rows = _load_window_settlements(log_path, args.hours)
    if not rows:
        print(f"No settlement records in last {args.hours:.1f}h window.")
        raise SystemExit(0)

    print(f"Loaded {len(rows)} settlements from last {args.hours:.1f}h window")
    print(f"Drift threshold: avg_pnl < {args.drift_threshold}")
    print()

    # --- Aggregate and detect drift ---
    current_agg = _aggregate_by_key(rows)
    alerts = _detect_drift(policy_keys, current_agg, args.drift_threshold)

    # --- Print results ---
    if not alerts:
        print("No drift detected. All core keys performing above threshold.")
    else:
        print(f"DRIFT ALERTS: {len(alerts)} core key(s) below threshold")
        print("-" * 90)
        for a in alerts:
            print(
                f"  {a['key']:55s}  baseline={a['baseline_avg_pnl']:+.4f}  "
                f"current={a['current_avg_pnl']:+.4f}  n={a['current_count']:4d}  "
                f"delta={a['delta']:+.4f}"
            )
        print()

    # --- Auto-demote if requested ---
    demoted = 0
    if args.auto_demote and alerts:
        demoted = _apply_auto_demote(policy, alerts, policy_path)
        print(f"Auto-demoted {demoted} key(s) from core -> explore in {policy_path}")
    elif args.auto_demote and not alerts:
        print("Auto-demote enabled but no drift detected; policy unchanged.")

    # --- Write drift report ---
    report_path = _write_drift_report(
        output_dir, alerts, demoted, args.hours, args.drift_threshold, args.auto_demote
    )
    print(f"Drift report written to: {report_path}")

    # --- Summary line ---
    print()
    core_in_window = sum(1 for k in core_keys if k in current_agg)
    print(
        f"Summary: {core_in_window}/{len(core_keys)} core keys had data in window, "
        f"{len(alerts)} drifted, {demoted} demoted"
    )


if __name__ == "__main__":
    main()
