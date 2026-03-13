#!/usr/bin/env python3
"""
Translate Polymarket wallet analysis into Kalshi Explore-sleeve hypotheses.

Usage:
  python scripts/polymarket_to_kalshi_hypotheses.py --analysis logs/analysis/polymarket/.../wallet_0x..._analysis.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _build_hypotheses(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    trades = analysis.get("trades", {})
    timing = analysis.get("timing", {})
    conc = analysis.get("concentration", {})
    style = str(analysis.get("inferred_style", "mixed"))
    avg_notional = float(trades.get("avg_notional_usd", 0.0) or 0.0)
    p90_notional = float(trades.get("p90_notional_usd", 0.0) or 0.0)
    median_gap = float(timing.get("median_intertrade_seconds", 0.0) or 0.0)
    hold_p50 = float(timing.get("hold_time_p50_minutes", 0.0) or 0.0)
    top_share = float(conc.get("top_market_share_notional", 0.0) or 0.0)

    h: List[Dict[str, Any]] = []

    # Sizing hypothesis
    risk_fraction = 0.005
    if p90_notional >= 1500:
        risk_fraction = 0.0075
    elif p90_notional <= 200:
        risk_fraction = 0.0035
    h.append(
        {
            "id": "sizing_proxy",
            "scope": "explore_sleeve",
            "change": {"sizing_risk_fraction": risk_fraction},
            "reason": f"wallet sizing profile avg=${avg_notional:.2f}, p90=${p90_notional:.2f}",
        }
    )

    # Time-to-expiry / cadence hypothesis
    if hold_p50 > 0 and hold_p50 <= 30:
        h.append(
            {
                "id": "short_hold_bias",
                "scope": "explore_sleeve",
                "change": {
                    "max_entry_minutes_to_expiry": 10,
                    "range_max_entry_minutes_to_expiry": 8,
                },
                "reason": f"wallet shows short hold profile (p50={hold_p50:.2f}m)",
            }
        )
    elif hold_p50 >= 240:
        h.append(
            {
                "id": "long_hold_bias",
                "scope": "explore_sleeve",
                "change": {
                    "max_entry_minutes_to_expiry": 20,
                    "range_max_entry_minutes_to_expiry": 15,
                },
                "reason": f"wallet shows long hold profile (p50={hold_p50:.2f}m)",
            }
        )

    # Concentration guard hypothesis
    if top_share >= 0.35:
        h.append(
            {
                "id": "concentration_guard",
                "scope": "explore_sleeve",
                "change": {
                    "family_side_cap_usd": 20000.0,
                    "market_side_cap_usd": 60000.0,
                },
                "reason": f"external wallet concentration high (top_share={top_share:.2%})",
            }
        )

    # High-frequency hint
    if median_gap > 0 and median_gap <= 30:
        h.append(
            {
                "id": "hf_reprice_focus",
                "scope": "explore_sleeve",
                "change": {
                    "scalp_reprice_window_s": 2.0,
                    "scalp_min_reprice_move_cents": 5,
                    "scalp_entry_cost_buffer_cents": 8,
                },
                "reason": f"external cadence suggests short opportunity windows (median_gap={median_gap:.2f}s)",
            }
        )

    # Optional mapping for style flags.
    if "mid-price-focus" in style:
        h.append(
            {
                "id": "mid_price_band",
                "scope": "explore_sleeve",
                "change": {"min_entry_cents": 25, "max_entry_cents": 75},
                "reason": "style indicates preference for mid-price contracts",
            }
        )
    if "tail-price-focus" in style:
        h.append(
            {
                "id": "tail_avoidance",
                "scope": "explore_sleeve",
                "change": {"min_entry_cents": 15, "max_entry_cents": 85},
                "reason": "style indicates tail-price usage; keep bounded for Kalshi fee structure",
            }
        )
    return h


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Kalshi Explore-sleeve hypotheses from Polymarket wallet analysis.")
    ap.add_argument("--analysis", required=True, help="Path to wallet analysis JSON")
    ap.add_argument("--out", default="logs/analysis/polymarket", help="Output root")
    args = ap.parse_args()

    analysis_path = Path(args.analysis)
    if not analysis_path.exists():
        raise SystemExit(f"analysis file not found: {analysis_path}")
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    hypotheses = _build_hypotheses(analysis)

    out = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "wallet": analysis.get("wallet"),
        "source_analysis": str(analysis_path),
        "apply_scope": "explore_sleeve_only",
        "hypotheses": hypotheses,
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    jpath = out_dir / "kalshi_hypotheses_from_polymarket.json"
    tpath = out_dir / "kalshi_hypotheses_from_polymarket.txt"
    jpath.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        f"wallet={out.get('wallet')}",
        f"source={analysis_path}",
        "scope=explore_sleeve_only",
        "",
        "hypotheses:",
    ]
    for h in hypotheses:
        lines.append(f"- {h['id']}: {h['reason']}")
        lines.append(f"  change={json.dumps(h['change'], sort_keys=True)}")
    tpath.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"[hypotheses] wrote {jpath}")
    print(f"[hypotheses] wrote {tpath}")


if __name__ == "__main__":
    main()

