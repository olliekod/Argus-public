#!/usr/bin/env python3
"""
Build a cross-wallet consensus profile from Polymarket wallet analysis JSON files.

Usage:
  python scripts/polymarket_wallet_consensus.py --analysis logs/analysis/polymarket/.../wallet_0x..._analysis.json --analysis ...
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _median(vals: List[float]) -> float:
    if not vals:
        return 0.0
    return float(statistics.median(vals))


def _mean(vals: List[float]) -> float:
    if not vals:
        return 0.0
    return float(statistics.mean(vals))


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def _build_argus_hypotheses(metrics: Dict[str, float], style_tags: Counter, price_bands: Dict[str, float]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    hold_p50 = metrics.get("hold_time_p50_minutes", 0.0)
    median_gap = metrics.get("median_intertrade_seconds", 0.0)
    p90_notional = metrics.get("p90_notional_usd", 0.0)
    top_share = metrics.get("top_market_share_notional", 0.0)
    mid_band = price_bands.get("40-60c", 0.0)
    high_tail = price_bands.get(">=80c", 0.0)

    if hold_p50 > 0 and hold_p50 <= 45:
        out.append(
            {
                "id": "short_horizon_entry_window",
                "change": {
                    "max_entry_minutes_to_expiry": 20,
                    "range_max_entry_minutes_to_expiry": 8,
                },
                "reason": f"consensus hold p50={hold_p50:.2f}m suggests short-horizon cadence",
            }
        )

    if median_gap > 0 and median_gap <= 60:
        out.append(
            {
                "id": "fast_reprice_execution",
                "change": {
                    "scalp_reprice_window_s": 2.0,
                    "scalp_min_reprice_move_cents": 4,
                    "scalp_entry_cost_buffer_cents": 7,
                },
                "reason": f"consensus intertrade median={median_gap:.2f}s implies tight repricing windows",
            }
        )

    if p90_notional >= 1000:
        out.append(
            {
                "id": "higher_conviction_sizing",
                "change": {"sizing_risk_fraction": 0.0065},
                "reason": f"consensus p90 notional=${p90_notional:.2f} supports moderate size increase",
            }
        )
    else:
        out.append(
            {
                "id": "conservative_sizing",
                "change": {"sizing_risk_fraction": 0.0050},
                "reason": f"consensus p90 notional=${p90_notional:.2f} supports baseline half-Kelly sizing",
            }
        )

    if top_share >= 0.35:
        out.append(
            {
                "id": "concentration_guardrails",
                "change": {
                    "family_side_cap_usd": 20000.0,
                    "market_side_cap_usd": 60000.0,
                },
                "reason": f"consensus top-market share={top_share:.2%} suggests concentration risk",
            }
        )

    if mid_band >= 0.35:
        out.append(
            {
                "id": "mid_band_focus",
                "change": {"min_entry_cents": 25, "max_entry_cents": 75},
                "reason": f"consensus mid-band activity={mid_band:.2%} aligns with 25-75c entry band",
            }
        )
    elif high_tail >= 0.30:
        out.append(
            {
                "id": "tail_risk_clamp",
                "change": {"yes_avoid_min_cents": 80, "yes_avoid_max_cents": 100},
                "reason": f"consensus high-tail activity={high_tail:.2%}; keep strict tail controls",
            }
        )

    if style_tags.get("high-frequency", 0) >= 2:
        out.append(
            {
                "id": "keep_arb_additive",
                "change": {"scalper_enabled": True, "arb_enabled": True, "arb_min_sum_ask_cents": 98},
                "reason": "multiple wallets exhibit high-frequency behavior; keep arb additive, not replacement",
            }
        )

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build cross-wallet consensus from wallet analysis JSON files.")
    ap.add_argument("--analysis", action="append", required=True, help="Path to wallet_0x..._analysis.json (repeatable)")
    ap.add_argument("--out", default="logs/analysis/polymarket", help="Output root")
    args = ap.parse_args()

    analysis_paths = [Path(p) for p in args.analysis]
    for p in analysis_paths:
        if not p.exists():
            raise SystemExit(f"analysis file not found: {p}")

    analyses: List[Dict[str, Any]] = [json.loads(p.read_text(encoding="utf-8")) for p in analysis_paths]
    wallets = [str(a.get("wallet", "")).lower() for a in analyses]

    avg_notional = [_safe_float(a.get("trades", {}).get("avg_notional_usd")) for a in analyses]
    p90_notional = [_safe_float(a.get("trades", {}).get("p90_notional_usd")) for a in analyses]
    buy_sell_ratio = [_safe_float(a.get("trades", {}).get("buy_sell_ratio")) for a in analyses if a.get("trades", {}).get("buy_sell_ratio") is not None]
    median_gap = [_safe_float(a.get("timing", {}).get("median_intertrade_seconds")) for a in analyses]
    hold_p50 = [_safe_float(a.get("timing", {}).get("hold_time_p50_minutes")) for a in analyses]
    hold_p90 = [_safe_float(a.get("timing", {}).get("hold_time_p90_minutes")) for a in analyses]
    top_share = [_safe_float(a.get("concentration", {}).get("top_market_share_notional")) for a in analyses]
    hhi = [_safe_float(a.get("concentration", {}).get("market_hhi_notional")) for a in analyses]

    style_counter: Counter = Counter()
    for a in analyses:
        style = str(a.get("inferred_style", "") or "")
        for tag in [t.strip() for t in style.split(",") if t.strip()]:
            style_counter[tag] += 1

    band_counter: Counter = Counter()
    band_total = 0.0
    for a in analyses:
        pb = a.get("price_bands", {}) or {}
        for k, v in pb.items():
            fv = _safe_float(v)
            band_counter[k] += fv
            band_total += fv
    band_share = {k: (v / band_total if band_total > 0 else 0.0) for k, v in sorted(band_counter.items())}

    consensus_metrics = {
        "avg_notional_usd": round(_median(avg_notional), 4),
        "p90_notional_usd": round(_median(p90_notional), 4),
        "buy_sell_ratio": round(_median(buy_sell_ratio), 6) if buy_sell_ratio else None,
        "median_intertrade_seconds": round(_median(median_gap), 4),
        "hold_time_p50_minutes": round(_median(hold_p50), 4),
        "hold_time_p90_minutes": round(_median(hold_p90), 4),
        "top_market_share_notional": round(_median(top_share), 6),
        "market_hhi_notional": round(_median(hhi), 8),
    }

    hypotheses = _build_argus_hypotheses(consensus_metrics, style_counter, band_share)

    out = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "wallet_count": len(analyses),
        "wallets": wallets,
        "source_analysis_files": [str(p) for p in analysis_paths],
        "consensus_metrics": consensus_metrics,
        "style_tag_counts": dict(style_counter),
        "price_band_share": {k: round(v, 6) for k, v in band_share.items()},
        "argus_consensus_hypotheses": hypotheses,
        "notes": [
            "Use these as test hypotheses, not direct copy-trading rules.",
            "Validate via 4h/8h A-B windows before promoting to 24h gating.",
        ],
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    jpath = out_dir / "wallet_consensus.json"
    tpath = out_dir / "wallet_consensus.txt"
    jpath.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        f"wallets={len(analyses)}",
        f"avg_notional_med=${consensus_metrics['avg_notional_usd']:.2f}",
        f"p90_notional_med=${consensus_metrics['p90_notional_usd']:.2f}",
        f"median_gap_s={consensus_metrics['median_intertrade_seconds']:.2f}",
        f"hold_p50_med_min={consensus_metrics['hold_time_p50_minutes']:.2f}",
        f"top_share_med={consensus_metrics['top_market_share_notional']:.2%}",
        "",
        "style_tags:",
    ]
    for k, v in style_counter.most_common():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("argus_consensus_hypotheses:")
    for h in hypotheses:
        lines.append(f"- {h['id']}: {h['reason']}")
        lines.append(f"  change={json.dumps(h['change'], sort_keys=True)}")
    tpath.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    print(f"[consensus] wrote {jpath}")
    print(f"[consensus] wrote {tpath}")


if __name__ == "__main__":
    main()

