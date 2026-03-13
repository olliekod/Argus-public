"""
Cross-runpack rollup: aggregate and rank multiple runpack outputs.

Reads metrics_summary.json from each runpack directory, compares runs side-by-side,
and ranks them by a robust score (PnL quality adjusted for drawdown and concentration).

Usage:
    python scripts/kalshi_runpack_rollup.py
    python scripts/kalshi_runpack_rollup.py --dir logs/analysis --hours 8
    python scripts/kalshi_runpack_rollup.py logs/analysis/runpack_20260308_1600_abc12345 logs/analysis/runpack_...
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_metrics(runpack_dir: Path) -> Optional[Dict[str, Any]]:
    """Load metrics_summary.json from a runpack dir. Falls back to manifest for older packs."""
    ms_path = runpack_dir / "metrics_summary.json"
    if ms_path.exists():
        try:
            return json.loads(ms_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    # Fallback: build minimal metrics from manifest (older runpacks without metrics_summary)
    mf_path = runpack_dir / "manifest.json"
    if not mf_path.exists():
        return None
    try:
        manifest = json.loads(mf_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return {
        "run_id": manifest.get("run_id", runpack_dir.name),
        "generated_at_utc": manifest.get("generated_at_utc", ""),
        "git_hash": manifest.get("git_hash", "unknown"),
        "config_hash": manifest.get("config_hash", "unknown"),
        "windows": [
            {
                "hours": o["window_hours"],
                "settlements": o["settlements"],
                "total_pnl_usd": 0.0,
                "avg_pnl_usd": 0.0,
                "win_rate_pct": 0.0,
                "max_drawdown_usd": 0.0,
                "top_concentration_share": 0.0,
                "market_side_hhi": 0.0,
                "context_coverage_pct": 0.0,
                "edge_retention_poor_keys": 0,
                "promotable_contexts": 0,
            }
            for o in manifest.get("outputs", [])
        ],
    }


def _robust_score(w: Dict[str, Any]) -> float:
    """
    Robust score: avg PnL quality penalized for drawdown and concentration.

    Formula:
      robust = avg_pnl * (1 - dd_penalty) * (1 - conc_penalty)

    dd_penalty  = min(1, drawdown / max(0.01, |total_pnl|))   — how much DD relative to gains
    conc_penalty = max(0, top_conc - 0.30)                    — penalty above 30% concentration
    """
    avg = float(w.get("avg_pnl_usd", 0.0))
    total_pnl = float(w.get("total_pnl_usd", 0.0))
    dd = float(w.get("max_drawdown_usd", 0.0))
    conc = float(w.get("top_concentration_share", 1.0))
    dd_penalty = min(1.0, dd / max(0.01, abs(total_pnl) if total_pnl != 0 else 0.01))
    conc_penalty = max(0.0, conc - 0.30)
    return avg * (1.0 - 0.5 * dd_penalty) * (1.0 - conc_penalty)


def _window_comparability(w: Dict[str, Any], target_hours: float) -> bool:
    """Return True if the window is close enough to the target to be comparable."""
    actual = float(w.get("hours", 0))
    return abs(actual - target_hours) <= target_hours * 0.5


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-runpack rollup")
    ap.add_argument("runpacks", nargs="*", help="Runpack directories to include (default: auto-discover)")
    ap.add_argument(
        "--dir",
        default="logs/analysis",
        help="Scan this directory for runpack_* subdirs (default: logs/analysis)",
    )
    ap.add_argument("--out", default="logs/analysis/rollup", help="Output directory (default: logs/analysis/rollup)")
    ap.add_argument("--hours", type=float, default=8.0, help="Target window to compare (default: 8h)")
    ap.add_argument(
        "--min-settlements",
        type=int,
        default=50,
        help="Skip runpacks with fewer settlements than this (default: 50)",
    )
    args = ap.parse_args()

    # Collect runpack directories
    dirs: List[Path] = []
    if args.runpacks:
        for p in args.runpacks:
            d = Path(p)
            if d.is_dir():
                dirs.append(d)
            else:
                print(f"[rollup] skip {p}: not a directory")
    else:
        base = Path(args.dir)
        if base.is_dir():
            dirs = sorted(d for d in base.iterdir() if d.is_dir() and d.name.startswith("runpack_"))

    if not dirs:
        raise SystemExit(
            f"No runpack directories found. Pass dirs as arguments or use --dir (scanned: {args.dir})"
        )

    rows: List[Dict[str, Any]] = []
    for d in dirs:
        metrics = _load_metrics(d)
        if not metrics:
            print(f"[rollup] skip {d.name}: no metrics_summary.json or manifest.json")
            continue

        windows = metrics.get("windows", [])
        if not windows:
            print(f"[rollup] skip {d.name}: no windows in metrics")
            continue

        # Find the closest window to --hours
        comparable = [w for w in windows if _window_comparability(w, args.hours)]
        if not comparable:
            print(f"[rollup] skip {d.name}: no window within 50% of {args.hours}h")
            continue
        closest = min(comparable, key=lambda w: abs(float(w.get("hours", 0)) - args.hours))

        if int(closest.get("settlements", 0)) < args.min_settlements:
            print(
                f"[rollup] skip {d.name}: only {closest.get('settlements')} settlements "
                f"(< {args.min_settlements})"
            )
            continue

        rows.append({
            "run_id": metrics.get("run_id", d.name),
            "generated_at_utc": metrics.get("generated_at_utc", ""),
            "git_hash": metrics.get("git_hash", "unknown"),
            "config_hash": metrics.get("config_hash", "unknown"),
            "runpack_dir": str(d),
            "window_hours": float(closest.get("hours", args.hours)),
            "settlements": int(closest.get("settlements", 0)),
            "total_pnl_usd": float(closest.get("total_pnl_usd", 0.0)),
            "avg_pnl_usd": float(closest.get("avg_pnl_usd", 0.0)),
            "win_rate_pct": float(closest.get("win_rate_pct", 0.0)),
            "max_drawdown_usd": float(closest.get("max_drawdown_usd", 0.0)),
            "top_concentration_share": float(closest.get("top_concentration_share", 0.0)),
            "market_side_hhi": float(closest.get("market_side_hhi", 0.0)),
            "context_coverage_pct": float(closest.get("context_coverage_pct", 0.0)),
            "edge_retention_poor_keys": int(closest.get("edge_retention_poor_keys", 0)),
            "promotable_contexts": int(closest.get("promotable_contexts", 0)),
            "robust_score": _robust_score(closest),
        })

    if not rows:
        raise SystemExit("No qualifying runpacks found after filtering.")

    # Sort by robust_score descending
    rows.sort(key=lambda r: r["robust_score"], reverse=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # CSV
    csv_path = out_dir / f"rollup_summary_{ts}.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # JSON
    json_path = out_dir / f"rollup_summary_{ts}.json"
    json_path.write_text(
        json.dumps(
            {"generated_at_utc": ts, "target_hours": args.hours, "run_count": len(rows), "runs": rows},
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    # Markdown report
    md_path = out_dir / f"rollup_report_{ts}.md"
    positive_runs = sum(1 for r in rows if r["total_pnl_usd"] > 0)
    md_lines = [
        f"# Runpack Rollup — {ts}",
        f"",
        f"**Target window:** {args.hours}h &nbsp;|&nbsp; "
        f"**Runs included:** {len(rows)} &nbsp;|&nbsp; "
        f"**Positive PnL:** {positive_runs}/{len(rows)} ({100*positive_runs//len(rows)}%)",
        "",
        "## Runs Ranked by Robust Score",
        "",
        "| # | Run ID | n | Total PnL | Avg PnL | WR% | Max DD | Top Conc | HHI | Promo | Robust |",
        "|---|--------|---|-----------|---------|-----|--------|----------|-----|-------|--------|",
    ]
    for i, r in enumerate(rows, 1):
        md_lines.append(
            f"| {i} | `{r['run_id']}` | {r['settlements']} | "
            f"{r['total_pnl_usd']:+.2f} | {r['avg_pnl_usd']:+.4f} | "
            f"{r['win_rate_pct']:.1f} | {r['max_drawdown_usd']:.2f} | "
            f"{r['top_concentration_share']:.1%} | {r['market_side_hhi']:.4f} | "
            f"{r['promotable_contexts']} | {r['robust_score']:.4f} |"
        )

    md_lines.extend([
        "",
        "## Metric Definitions",
        "",
        "| Metric | Meaning |",
        "|--------|---------|",
        "| **Robust Score** | `avg_pnl × (1 − dd_penalty) × (1 − conc_penalty)` — PnL quality adjusted for risk |",
        "| **Max DD** | Peak-to-trough drawdown on cumulative PnL (lower is better) |",
        "| **Top Conc** | Largest single market-side share of total fill quantity (>35% = risky) |",
        "| **HHI** | Herfindahl-Hirschman Index across market-sides (>0.15 = concentrated) |",
        "| **Promo** | Contexts passing all promotion gate criteria for this window |",
        "",
        "## Config Hashes",
        "",
    ])
    seen_configs: Dict[str, List[str]] = {}
    for r in rows:
        seen_configs.setdefault(r["config_hash"], []).append(r["run_id"])
    for cfg_hash, run_ids in seen_configs.items():
        md_lines.append(f"- `{cfg_hash}`: {', '.join(run_ids)}")

    md_lines.extend(["", f"*Generated: {ts}*", ""])
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"[rollup] {len(rows)} runs processed ({positive_runs} positive PnL)")
    print(f"[rollup] CSV:  {csv_path}")
    print(f"[rollup] JSON: {json_path}")
    print(f"[rollup] MD:   {md_path}")
    if rows:
        print(f"\n[rollup] Top run: {rows[0]['run_id']}  "
              f"pnl={rows[0]['total_pnl_usd']:+.2f}  "
              f"avg={rows[0]['avg_pnl_usd']:+.4f}  "
              f"robust={rows[0]['robust_score']:.4f}")


if __name__ == "__main__":
    main()
