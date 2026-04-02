# Created by Oliver Meihls

# Kalshi run-pack wrapper for post-window analytics.
#
# Generates one timestamped output directory containing:
# - context report (JSON + TXT)
# - promotion gate (JSON + TXT)
# - concentration summary (JSON + TXT)
# - cycle stability summary (JSON + TXT)
#
# Usage:
# python scripts/kalshi_run_pack.py
# python scripts/kalshi_run_pack.py --hours 8
# python scripts/kalshi_run_pack.py --windows 4,8,24

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import hashlib
import subprocess


@dataclass(frozen=True)
class WindowConfig:
    hours: float
    min_samples: int
    min_cycle_samples: int
    min_cycles: int
    max_concentration_share: float
    cycle_minutes: int


def _ctx_get(rec: Dict[str, Any], key: str, default: str = "na") -> str:
    ctx = rec.get("decision_context") or {}
    if not isinstance(ctx, dict):
        return default
    return str(ctx.get(key, default))


def _candidate_key(rec: Dict[str, Any]) -> str:
    family = str(rec.get("family") or "UNK")
    side = str(rec.get("side") or "?")
    return f"{family}|{side}|eb={_ctx_get(rec, 'eb')}|pb={_ctx_get(rec, 'pb')}"


def _load_records(path: Path) -> List[Dict[str, Any]]:
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
            rows.append(obj)
    return rows


def _max_timestamp(rows: Iterable[Dict[str, Any]]) -> Optional[float]:
    max_ts: Optional[float] = None
    for rec in rows:
        ts = rec.get("timestamp")
        if isinstance(ts, (int, float)):
            if max_ts is None or float(ts) > max_ts:
                max_ts = float(ts)
    return max_ts


def _window_settlements(rows: Iterable[Dict[str, Any]], since_ts: float) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rec in rows:
        if rec.get("type") != "settlement":
            continue
        ts = rec.get("timestamp")
        if not isinstance(ts, (int, float)):
            continue
        if float(ts) < since_ts:
            continue
        out.append(rec)
    return out


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


def _rank_section(
    rows: List[Tuple[str, int, float, float, float]],
    *,
    min_samples: int,
    top: int,
    reverse: bool,
) -> List[Dict[str, Any]]:
    filt = [r for r in rows if r[1] >= min_samples]
    filt.sort(key=lambda x: x[2], reverse=reverse)
    out: List[Dict[str, Any]] = []
    for key, n, pnl_sum, avg, wr in filt[:top]:
        out.append(
            {
                "key": key,
                "samples": n,
                "pnl_usd": round(pnl_sum, 6),
                "avg_pnl_usd": round(avg, 6),
                "win_rate_pct": round(wr, 4),
            }
        )
    return out


def _format_rank_lines(title: str, rows: List[Dict[str, Any]]) -> List[str]:
    lines = [title]
    if not rows:
        lines.append("  (no rows)")
        return lines
    for r in rows:
        lines.append(
            "  {key:45} n={samples:6d} pnl={pnl_usd:10.2f} avg={avg_pnl_usd:8.3f} wr={win_rate_pct:6.2f}%".format(
                **r
            )
        )
    return lines


def _build_context_report(
    settlements: List[Dict[str, Any]],
    *,
    hours: float,
    min_samples: int,
    top: int,
) -> Tuple[Dict[str, Any], str]:
    total = len(settlements)
    total_pnl = sum(float(r.get("pnl_usd") or 0.0) for r in settlements)
    total_wins = sum(1 for r in settlements if bool(r.get("won")))
    ctx_present = sum(1 for r in settlements if isinstance(r.get("decision_context"), dict) and r.get("decision_context"))
    wr = 100.0 * total_wins / total if total else 0.0
    coverage = 100.0 * ctx_present / total if total else 0.0

    by_family_side = _agg(settlements, lambda r: f"{r.get('family','UNK')}|{r.get('side','?')}")
    by_edge_bucket = _agg(settlements, lambda r: f"eb={_ctx_get(r, 'eb')}")
    by_price_bucket = _agg(settlements, lambda r: f"pb={_ctx_get(r, 'pb')}")
    by_tts_bucket = _agg(settlements, lambda r: f"tb={_ctx_get(r, 'tb')}")
    by_edge_price = _agg(settlements, lambda r: f"eb={_ctx_get(r, 'eb')}|pb={_ctx_get(r, 'pb')}")
    by_fam_edge_price = _agg(
        settlements,
        lambda r: f"{r.get('family','UNK')}|{r.get('side','?')}|eb={_ctx_get(r, 'eb')}|pb={_ctx_get(r, 'pb')}",
    )

    data = {
        "window_hours": hours,
        "settlements": total,
        "total_pnl_usd": round(total_pnl, 6),
        "win_rate_pct": round(wr, 4),
        "context_coverage_pct": round(coverage, 4),
        "sections": {
            "best_family_side": _rank_section(by_family_side, min_samples=min_samples, top=top, reverse=True),
            "worst_family_side": _rank_section(by_family_side, min_samples=min_samples, top=top, reverse=False),
            "best_edge_bucket": _rank_section(by_edge_bucket, min_samples=min_samples, top=top, reverse=True),
            "worst_edge_bucket": _rank_section(by_edge_bucket, min_samples=min_samples, top=top, reverse=False),
            "best_price_bucket": _rank_section(by_price_bucket, min_samples=min_samples, top=top, reverse=True),
            "worst_price_bucket": _rank_section(by_price_bucket, min_samples=min_samples, top=top, reverse=False),
            "best_tts_bucket": _rank_section(by_tts_bucket, min_samples=min_samples, top=top, reverse=True),
            "worst_tts_bucket": _rank_section(by_tts_bucket, min_samples=min_samples, top=top, reverse=False),
            "best_edge_price": _rank_section(by_edge_price, min_samples=min_samples, top=top, reverse=True),
            "worst_edge_price": _rank_section(by_edge_price, min_samples=min_samples, top=top, reverse=False),
            "best_family_side_edge_price": _rank_section(
                by_fam_edge_price, min_samples=min_samples, top=top, reverse=True
            ),
            "worst_family_side_edge_price": _rank_section(
                by_fam_edge_price, min_samples=min_samples, top=top, reverse=False
            ),
        },
    }

    lines = [
        f"window_hours={hours:.2f} settlements={total} pnl={total_pnl:.2f} wr={wr:.2f}%",
        f"context_coverage={ctx_present}/{total} ({coverage:.2f}%)",
        "",
    ]
    lines.extend(_format_rank_lines("Best Family|Side", data["sections"]["best_family_side"]))
    lines.append("")
    lines.extend(_format_rank_lines("Worst Family|Side", data["sections"]["worst_family_side"]))
    lines.append("")
    lines.extend(_format_rank_lines("Best Edge Bucket", data["sections"]["best_edge_bucket"]))
    lines.append("")
    lines.extend(_format_rank_lines("Worst Edge Bucket", data["sections"]["worst_edge_bucket"]))
    lines.append("")
    lines.extend(_format_rank_lines("Best Price Bucket", data["sections"]["best_price_bucket"]))
    lines.append("")
    lines.extend(_format_rank_lines("Worst Price Bucket", data["sections"]["worst_price_bucket"]))
    lines.append("")
    lines.extend(_format_rank_lines("Best TTS Bucket", data["sections"]["best_tts_bucket"]))
    lines.append("")
    lines.extend(_format_rank_lines("Worst TTS Bucket", data["sections"]["worst_tts_bucket"]))
    lines.append("")
    lines.extend(_format_rank_lines("Best Edge|Price", data["sections"]["best_edge_price"]))
    lines.append("")
    lines.extend(_format_rank_lines("Worst Edge|Price", data["sections"]["worst_edge_price"]))
    lines.append("")
    lines.extend(_format_rank_lines("Best Family|Side|Edge|Price", data["sections"]["best_family_side_edge_price"]))
    lines.append("")
    lines.extend(_format_rank_lines("Worst Family|Side|Edge|Price", data["sections"]["worst_family_side_edge_price"]))
    text = "\n".join(lines).rstrip() + "\n"
    return data, text


def _build_promotion_gate(
    settlements: List[Dict[str, Any]],
    cfg: WindowConfig,
    *,
    top: int,
) -> Tuple[Dict[str, Any], str]:
    cycle_s = max(60, int(cfg.cycle_minutes) * 60)
    agg: Dict[str, Dict[str, Any]] = {}
    for r in settlements:
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

    passing: List[Dict[str, Any]] = []
    failing: List[Dict[str, Any]] = []
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
            if int(cyc["n"]) < int(cfg.min_cycle_samples):
                continue
            if float(cyc["pnl"]) <= 0.0:
                break
            good_cycles += 1
            if good_cycles >= int(cfg.min_cycles):
                break

        reason = None
        if n < int(cfg.min_samples):
            reason = "min_samples"
        elif avg <= 0.0:
            reason = "non_positive_expectancy"
        elif top_share > float(cfg.max_concentration_share):
            reason = "concentration_too_high"
        elif good_cycles < int(cfg.min_cycles):
            reason = "insufficient_positive_cycles"

        row = {
            "candidate": key,
            "samples": n,
            "total_pnl_usd": round(pnl, 6),
            "avg_pnl_usd": round(avg, 6),
            "win_rate_pct": round(wr, 4),
            "positive_cycles": good_cycles,
            "top_market_side_share": round(top_share, 6),
        }
        if reason is None:
            passing.append(row)
        else:
            row["fail_reason"] = reason
            failing.append(row)

    passing.sort(key=lambda x: x["avg_pnl_usd"], reverse=True)
    failing.sort(key=lambda x: x["avg_pnl_usd"])

    total = len(settlements)
    total_pnl = sum(float(r.get("pnl_usd") or 0.0) for r in settlements)
    total_wr = 100.0 * sum(1 for r in settlements if bool(r.get("won"))) / max(1, total)

    data = {
        "window_hours": cfg.hours,
        "settlements": total,
        "total_pnl_usd": round(total_pnl, 6),
        "win_rate_pct": round(total_wr, 4),
        "gate": {
            "min_samples": cfg.min_samples,
            "min_cycle_samples": cfg.min_cycle_samples,
            "min_cycles": cfg.min_cycles,
            "max_concentration_share": cfg.max_concentration_share,
        },
        "promotable": passing[:top],
        "do_not_promote_worst": failing[:top],
        "counts": {"promotable_total": len(passing), "failing_total": len(failing)},
    }

    lines = [
        f"window_hours={cfg.hours:.2f} settlements={total} pnl={total_pnl:.2f} wr={total_wr:.2f}%",
        (
            "gate=min_samples:{min_samples} min_cycles:{min_cycles} min_cycle_samples:{min_cycle_samples} "
            "max_concentration_share:{max_concentration_share:.2f}"
        ).format(**data["gate"]),
        "",
        "PROMOTABLE CANDIDATES",
    ]
    if not data["promotable"]:
        lines.append("  (none)")
    else:
        for row in data["promotable"]:
            lines.append(
                "  {candidate:55} n={samples:6d} pnl={total_pnl_usd:10.2f} avg={avg_pnl_usd:8.3f} "
                "wr={win_rate_pct:6.2f}% cycles={positive_cycles} top_share={top_market_side_share:5.2f}".format(
                    **row
                )
            )
    lines.extend(["", "DO-NOT-PROMOTE (WORST)"])
    if not data["do_not_promote_worst"]:
        lines.append("  (none)")
    else:
        for row in data["do_not_promote_worst"]:
            lines.append(
                "  {candidate:55} n={samples:6d} pnl={total_pnl_usd:10.2f} avg={avg_pnl_usd:8.3f} "
                "wr={win_rate_pct:6.2f}% cycles={positive_cycles} top_share={top_market_side_share:5.2f} "
                "reason={fail_reason}".format(**row)
            )
    text = "\n".join(lines).rstrip() + "\n"
    return data, text


def _build_concentration_summary(
    settlements: List[Dict[str, Any]],
    *,
    top: int,
) -> Tuple[Dict[str, Any], str]:
    qty_by_market_side: Dict[str, float] = defaultdict(float)
    qty_by_family_side: Dict[str, float] = defaultdict(float)
    total_qty = 0.0
    for r in settlements:
        qty = float(r.get("quantity_contracts") or 0.0)
        if qty <= 0.0:
            continue
        mside = f"{r.get('market_ticker')}|{r.get('side')}"
        fside = f"{r.get('family','UNK')}|{r.get('side')}"
        qty_by_market_side[mside] += qty
        qty_by_family_side[fside] += qty
        total_qty += qty

    def _shares(src: Dict[str, float]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for key, qty in src.items():
            share = qty / total_qty if total_qty > 0 else 0.0
            out.append({"key": key, "qty_contracts": round(qty, 6), "share": round(share, 6)})
        out.sort(key=lambda x: x["share"], reverse=True)
        return out

    market_rows = _shares(qty_by_market_side)[:top]
    family_rows = _shares(qty_by_family_side)[:top]
    hhi = 0.0
    if total_qty > 0:
        for v in qty_by_market_side.values():
            p = v / total_qty
            hhi += p * p

    data = {
        "settlements": len(settlements),
        "total_qty_contracts": round(total_qty, 6),
        "market_side_hhi": round(hhi, 8),
        "top_market_side": market_rows,
        "top_family_side": family_rows,
    }

    lines = [
        f"settlements={data['settlements']} total_qty={data['total_qty_contracts']:.2f} market_side_hhi={data['market_side_hhi']:.6f}",
        "",
        "TOP MARKET|SIDE BY QTY SHARE",
    ]
    if not market_rows:
        lines.append("  (no rows)")
    else:
        for row in market_rows:
            lines.append(
                "  {key:55} qty={qty_contracts:10.2f} share={share:6.2%}".format(
                    key=row["key"], qty_contracts=row["qty_contracts"], share=row["share"]
                )
            )
    lines.extend(["", "TOP FAMILY|SIDE BY QTY SHARE"])
    if not family_rows:
        lines.append("  (no rows)")
    else:
        for row in family_rows:
            lines.append(
                "  {key:55} qty={qty_contracts:10.2f} share={share:6.2%}".format(
                    key=row["key"], qty_contracts=row["qty_contracts"], share=row["share"]
                )
            )
    text = "\n".join(lines).rstrip() + "\n"
    return data, text


def _build_cycle_stability_summary(
    settlements: List[Dict[str, Any]],
    *,
    cycle_minutes: int,
    top_contexts: int,
) -> Tuple[Dict[str, Any], str]:
    cycle_s = max(60, int(cycle_minutes) * 60)
    by_cycle: Dict[int, Dict[str, float]] = defaultdict(lambda: {"n": 0.0, "pnl": 0.0, "wins": 0.0})
    by_context_cycle: Dict[str, Dict[int, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: {"n": 0.0, "pnl": 0.0})
    )

    for r in settlements:
        ts = float(r.get("timestamp") or 0.0)
        cyc = int(ts // cycle_s)
        pnl = float(r.get("pnl_usd") or 0.0)
        won = 1.0 if bool(r.get("won")) else 0.0
        key = _candidate_key(r)

        by_cycle[cyc]["n"] += 1.0
        by_cycle[cyc]["pnl"] += pnl
        by_cycle[cyc]["wins"] += won

        by_context_cycle[key][cyc]["n"] += 1.0
        by_context_cycle[key][cyc]["pnl"] += pnl

    cycle_rows: List[Dict[str, Any]] = []
    for cyc, s in sorted(by_cycle.items(), key=lambda x: x[0]):
        n = int(s["n"])
        pnl = float(s["pnl"])
        wr = 100.0 * float(s["wins"]) / n if n else 0.0
        cycle_rows.append({"cycle": cyc, "samples": n, "pnl_usd": round(pnl, 6), "win_rate_pct": round(wr, 4)})

    context_rows: List[Dict[str, Any]] = []
    for key, cyc_map in by_context_cycle.items():
        cyc_stats = sorted(cyc_map.items(), key=lambda x: x[0])
        sample_cycles = 0
        positive_cycles = 0
        pnl_total = 0.0
        sample_total = 0
        for _, vals in cyc_stats:
            n = int(vals["n"])
            if n <= 0:
                continue
            sample_cycles += 1
            sample_total += n
            pnl = float(vals["pnl"])
            pnl_total += pnl
            if pnl > 0:
                positive_cycles += 1
        if sample_cycles == 0:
            continue
        stability = positive_cycles / sample_cycles
        avg = pnl_total / max(1, sample_total)
        context_rows.append(
            {
                "context": key,
                "cycles": sample_cycles,
                "positive_cycles": positive_cycles,
                "stability_ratio": round(stability, 6),
                "samples": sample_total,
                "total_pnl_usd": round(pnl_total, 6),
                "avg_pnl_usd": round(avg, 6),
            }
        )

    context_rows.sort(key=lambda x: (x["stability_ratio"], x["avg_pnl_usd"]), reverse=True)
    best_contexts = context_rows[:top_contexts]
    worst_contexts = sorted(context_rows, key=lambda x: (x["stability_ratio"], x["avg_pnl_usd"]))[:top_contexts]

    pnl_values = [r["pnl_usd"] for r in cycle_rows]
    mean_cycle_pnl = sum(pnl_values) / len(pnl_values) if pnl_values else 0.0
    std_cycle_pnl = 0.0
    if pnl_values:
        var = sum((v - mean_cycle_pnl) ** 2 for v in pnl_values) / len(pnl_values)
        std_cycle_pnl = math.sqrt(var)

    data = {
        "cycle_minutes": cycle_minutes,
        "cycle_count": len(cycle_rows),
        "cycle_mean_pnl_usd": round(mean_cycle_pnl, 6),
        "cycle_std_pnl_usd": round(std_cycle_pnl, 6),
        "cycles": cycle_rows,
        "best_context_stability": best_contexts,
        "worst_context_stability": worst_contexts,
    }

    lines = [
        (
            f"cycle_minutes={cycle_minutes} cycle_count={data['cycle_count']} "
            f"cycle_mean_pnl={data['cycle_mean_pnl_usd']:.4f} cycle_std_pnl={data['cycle_std_pnl_usd']:.4f}"
        ),
        "",
        "CYCLE SUMMARY",
    ]
    if not cycle_rows:
        lines.append("  (no rows)")
    else:
        for row in cycle_rows[-top_contexts:]:
            lines.append(
                "  cycle={cycle:10d} n={samples:6d} pnl={pnl_usd:10.2f} wr={win_rate_pct:6.2f}%".format(**row)
            )
    lines.extend(["", "BEST CONTEXT STABILITY"])
    if not best_contexts:
        lines.append("  (no rows)")
    else:
        for row in best_contexts:
            lines.append(
                "  {context:55} cycles={cycles:3d} pos={positive_cycles:3d} stab={stability_ratio:5.2%} "
                "n={samples:6d} avg={avg_pnl_usd:8.3f}".format(**row)
            )
    lines.extend(["", "WORST CONTEXT STABILITY"])
    if not worst_contexts:
        lines.append("  (no rows)")
    else:
        for row in worst_contexts:
            lines.append(
                "  {context:55} cycles={cycles:3d} pos={positive_cycles:3d} stab={stability_ratio:5.2%} "
                "n={samples:6d} avg={avg_pnl_usd:8.3f}".format(**row)
            )
    text = "\n".join(lines).rstrip() + "\n"
    return data, text


def _build_context_feature_coverage(
    settlements: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], str]:
    # Report coverage of new v2 decision_context fields.
    total = len(settlements)
    if total == 0:
        empty = {"total": 0, "coverage": {}}
        return empty, "No settlements.\n"

    fields = ["eb", "pb", "tb", "sdb", "nm", "sdp", "spb", "lb"]
    counts: Dict[str, int] = {f: 0 for f in fields}
    for r in settlements:
        ctx = r.get("decision_context") or {}
        if not isinstance(ctx, dict):
            continue
        for f in fields:
            if f in ctx:
                counts[f] += 1
    coverage = {f: round(100.0 * counts[f] / total, 2) for f in fields}
    # Strike distance bucket distribution
    sdb_dist: Dict[str, int] = defaultdict(int)
    nm_count = 0
    for r in settlements:
        ctx = r.get("decision_context") or {}
        if not isinstance(ctx, dict):
            continue
        sdb = ctx.get("sdb", "na")
        sdb_dist[sdb] += 1
        if ctx.get("nm"):
            nm_count += 1
    # Spread bucket distribution
    spb_dist: Dict[str, int] = defaultdict(int)
    for r in settlements:
        ctx = r.get("decision_context") or {}
        if isinstance(ctx, dict):
            spb_dist[ctx.get("spb", "na")] += 1
    # Liquidity bucket distribution
    lb_dist: Dict[str, int] = defaultdict(int)
    for r in settlements:
        ctx = r.get("decision_context") or {}
        if isinstance(ctx, dict):
            lb_dist[ctx.get("lb", "na")] += 1

    data = {
        "total": total,
        "coverage": coverage,
        "strike_distance_bucket_distribution": dict(sdb_dist),
        "near_money_count": nm_count,
        "near_money_pct": round(100.0 * nm_count / total, 2),
        "spread_bucket_distribution": dict(spb_dist),
        "liquidity_bucket_distribution": dict(lb_dist),
    }
    lines = [
        f"Context feature coverage (n={total})",
        *(f"  {f}: {coverage[f]:.1f}%" for f in fields),
        "",
        f"Near-money: {nm_count}/{total} ({data['near_money_pct']:.1f}%)",
        "",
        "Strike distance buckets:",
        *(f"  {k}: {v}" for k, v in sorted(sdb_dist.items())),
        "",
        "Spread buckets:",
        *(f"  {k}: {v}" for k, v in sorted(spb_dist.items())),
        "",
        "Liquidity buckets:",
        *(f"  {k}: {v}" for k, v in sorted(lb_dist.items())),
    ]
    return data, "\n".join(lines).rstrip() + "\n"


def _build_policy_snapshot() -> Tuple[Dict[str, Any], str]:
    # Load current policy file and produce a snapshot for run-pack.
    policy_path = Path("config/kalshi_context_policy.json")
    if not policy_path.exists():
        return {"loaded": False}, "No policy file found.\n"
    try:
        raw = json.loads(policy_path.read_text(encoding="utf-8"))
    except Exception:
        return {"loaded": False, "error": "parse_failed"}, "Policy file parse error.\n"
    keys_info = raw.get("keys", {})
    core = sum(1 for v in keys_info.values() if v.get("lane") == "core")
    explore = sum(1 for v in keys_info.values() if v.get("lane") == "explore")
    data = {
        "loaded": True,
        "version": raw.get("version"),
        "timestamp": raw.get("timestamp"),
        "total_keys": len(keys_info),
        "core_keys": core,
        "explore_keys": explore,
    }
    lines = [
        f"Policy: version={data['version']} keys={len(keys_info)} core={core} explore={explore}",
    ]
    return data, "\n".join(lines).rstrip() + "\n"


def _build_edge_retention_summary(
    settlements: List[Dict[str, Any]],
    *,
    min_samples: int = 20,
) -> Tuple[Dict[str, Any], str]:
    # Compute expected vs realized edge by context key.
    by_key: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for r in settlements:
        ctx = r.get("decision_context") or {}
        if not isinstance(ctx, dict):
            continue
        expected_edge = ctx.get("edge", 0.0)
        px = int(ctx.get("px", 0))
        qty = int(ctx.get("qty", 1))
        pnl = float(r.get("pnl_usd", 0.0))
        if px <= 0 or qty <= 0:
            continue
        cost = (px / 100.0) * qty
        realized = max(-1.0, min(2.0, pnl / cost))
        key = _candidate_key(r)
        by_key[key].append((expected_edge, realized))

    rows: List[Dict[str, Any]] = []
    for key, pairs in by_key.items():
        if len(pairs) < min_samples:
            continue
        exp_avg = sum(p[0] for p in pairs) / len(pairs)
        real_avg = sum(p[1] for p in pairs) / len(pairs)
        retention = real_avg / exp_avg if exp_avg > 0 else 0.0
        rows.append({
            "key": key,
            "samples": len(pairs),
            "expected_avg": round(exp_avg, 4),
            "realized_avg": round(real_avg, 4),
            "retention_ratio": round(retention, 4),
        })
    rows.sort(key=lambda x: x["retention_ratio"])

    data = {
        "total_keys": len(rows),
        "poor_retention_keys": sum(1 for r in rows if r["retention_ratio"] < 0.3),
        "best": rows[-5:] if rows else [],
        "worst": rows[:5] if rows else [],
    }
    lines = [
        f"Edge retention: {len(rows)} keys with >= {min_samples} samples",
        f"Poor retention (<30%): {data['poor_retention_keys']}",
        "",
        "WORST RETENTION",
    ]
    for r in data["worst"]:
        lines.append(
            f"  {r['key']:55} n={r['samples']:6d} exp={r['expected_avg']:.4f} real={r['realized_avg']:.4f} "
            f"ret={r['retention_ratio']:.2%}"
        )
    lines.extend(["", "BEST RETENTION"])
    for r in data["best"]:
        lines.append(
            f"  {r['key']:55} n={r['samples']:6d} exp={r['expected_avg']:.4f} real={r['realized_avg']:.4f} "
            f"ret={r['retention_ratio']:.2%}"
        )
    return data, "\n".join(lines).rstrip() + "\n"


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _file_hash(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    except Exception:
        return "unknown"


def _compute_max_drawdown(settlements: List[Dict[str, Any]]) -> float:
    # Peak-to-trough max drawdown on cumulative PnL sorted by timestamp.
    sorted_s = sorted(settlements, key=lambda r: float(r.get("timestamp") or 0.0))
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in sorted_s:
        cum += float(r.get("pnl_usd") or 0.0)
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd
    return round(max_dd, 6)


def _build_sleeve_metrics(
    settlements: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], str]:
    # Break down fills, PnL, and win rate by source (sleeve).
    by_source: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"fills": 0.0, "pnl": 0.0, "wins": 0.0}
    )
    total = len(settlements)
    total_pnl = 0.0
    for r in settlements:
        src = str(r.get("source") or "unknown")
        pnl = float(r.get("pnl_usd") or 0.0)
        won = 1.0 if bool(r.get("won")) else 0.0
        by_source[src]["fills"] += 1.0
        by_source[src]["pnl"] += pnl
        by_source[src]["wins"] += won
        total_pnl += pnl
    rows: List[Dict[str, Any]] = []
    for src, s in sorted(by_source.items()):
        n = int(s["fills"])
        pnl_s = float(s["pnl"])
        rows.append({
            "source": src,
            "fills": n,
            "fill_share_pct": round(100.0 * n / total, 2) if total else 0.0,
            "total_pnl_usd": round(pnl_s, 6),
            "pnl_share_pct": round(100.0 * pnl_s / total_pnl, 2) if total_pnl != 0 else 0.0,
            "avg_pnl_usd": round(pnl_s / n, 6) if n else 0.0,
            "win_rate_pct": round(100.0 * s["wins"] / n, 4) if n else 0.0,
        })
    data: Dict[str, Any] = {"total_fills": total, "by_source": rows}
    lines = [f"Sleeve metrics (total fills={total} total_pnl={total_pnl:.2f})", ""]
    for row in rows:
        lines.append(
            "  {source:30} fills={fills:7d} ({fill_share_pct:5.1f}%) pnl={total_pnl_usd:10.2f} "
            "({pnl_share_pct:+6.1f}%) avg={avg_pnl_usd:8.3f} wr={win_rate_pct:6.2f}%".format(**row)
        )
    return data, "\n".join(lines).rstrip() + "\n"


def _build_population_metrics(
    settlements: List[Dict[str, Any]],
    *,
    top: int = 20,
) -> Tuple[Dict[str, Any], str]:
    # Per-bot performance summary.
    by_bot: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"fills": 0.0, "pnl": 0.0, "wins": 0.0}
    )
    for r in settlements:
        bot = str(r.get("bot_id") or "unknown")
        by_bot[bot]["fills"] += 1.0
        by_bot[bot]["pnl"] += float(r.get("pnl_usd") or 0.0)
        by_bot[bot]["wins"] += 1.0 if bool(r.get("won")) else 0.0
    rows: List[Dict[str, Any]] = []
    for bot, s in by_bot.items():
        n = int(s["fills"])
        rows.append({
            "bot_id": bot,
            "fills": n,
            "total_pnl_usd": round(s["pnl"], 6),
            "avg_pnl_usd": round(s["pnl"] / n, 6) if n else 0.0,
            "win_rate_pct": round(100.0 * s["wins"] / n, 4) if n else 0.0,
        })
    rows.sort(key=lambda x: x["total_pnl_usd"], reverse=True)
    total_bots = len(rows)
    data: Dict[str, Any] = {
        "total_bots_with_fills": total_bots,
        "top_bots": rows[:top],
        "bottom_bots": rows[-top:] if total_bots > top else rows[:],
    }
    lines = [f"Population metrics: {total_bots} bots with fills", "", "TOP BOTS"]
    for r in data["top_bots"]:
        lines.append(
            "  {bot_id:30} fills={fills:6d} pnl={total_pnl_usd:10.2f} avg={avg_pnl_usd:8.3f} wr={win_rate_pct:6.2f}%".format(**r)
        )
    lines.extend(["", "BOTTOM BOTS"])
    for r in data["bottom_bots"]:
        lines.append(
            "  {bot_id:30} fills={fills:6d} pnl={total_pnl_usd:10.2f} avg={avg_pnl_usd:8.3f} wr={win_rate_pct:6.2f}%".format(**r)
        )
    return data, "\n".join(lines).rstrip() + "\n"


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_txt(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _parse_windows(value: str) -> List[float]:
    out: List[float] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        out.append(float(chunk))
    if not out:
        raise argparse.ArgumentTypeError("windows must contain at least one numeric value")
    return out


def _validate_runpack(out_dir: Path, manifest: Dict[str, Any]) -> None:
    # Validate all runpack artifacts exist and have required keys. Exits 1 on failure.
    REQUIRED_MANIFEST = {"run_id", "generated_at_utc", "git_hash", "config_hash", "windows_hours", "outputs"}
    REQUIRED_METRICS = {"run_id", "windows"}
    REQUIRED_CONTEXT = {"settlements", "total_pnl_usd", "sections"}
    REQUIRED_PROMOTION = {"settlements", "promotable", "do_not_promote_worst", "counts"}
    REQUIRED_CONCENTRATION = {"market_side_hhi", "top_market_side"}
    REQUIRED_SLEEVE = {"total_fills", "by_source"}
    REQUIRED_POPULATION = {"total_bots_with_fills"}

    failures: List[str] = []

    def _chk(path_str: str, required: set) -> None:
        p = Path(path_str)
        if not p.exists():
            failures.append(f"MISSING: {p}")
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(f"PARSE_ERROR: {p} — {exc}")
            return
        missing = required - set(data.keys())
        if missing:
            failures.append(f"MISSING_KEYS {missing}: {p}")

    _chk(str(out_dir / "manifest.json"), REQUIRED_MANIFEST)
    ms_path = manifest.get("metrics_summary", "")
    if ms_path:
        _chk(ms_path, REQUIRED_METRICS)
    for output in manifest.get("outputs", []):
        arts = output.get("artifacts", {})
        _chk(arts.get("context_report_json", ""), REQUIRED_CONTEXT)
        _chk(arts.get("promotion_gate_json", ""), REQUIRED_PROMOTION)
        _chk(arts.get("concentration_summary_json", ""), REQUIRED_CONCENTRATION)
        _chk(arts.get("sleeve_metrics_json", ""), REQUIRED_SLEEVE)
        _chk(arts.get("population_metrics_json", ""), REQUIRED_POPULATION)

    if failures:
        print("\n[run-pack] VALIDATION FAILED:")
        for f in failures:
            print(f"  {f}")
        raise SystemExit(1)
    print("[run-pack] VALIDATION PASSED — all artifacts present and valid")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Kalshi post-window run-pack artifacts")
    ap.add_argument("--log", default="logs/paper_trades.jsonl", help="Path to paper_trades.jsonl")
    ap.add_argument(
        "--windows",
        type=_parse_windows,
        default=[4.0, 8.0, 24.0],
        help="Comma-separated lookback windows in hours (default: 4,8,24)",
    )
    ap.add_argument("--hours", type=float, help="Single window override in hours")
    ap.add_argument("--min-samples", type=int, default=300, help="Minimum samples for context table slices")
    ap.add_argument("--promotion-min-samples", type=int, default=400, help="Promotion gate minimum samples")
    ap.add_argument("--promotion-min-cycle-samples", type=int, default=50, help="Promotion gate minimum per-cycle samples")
    ap.add_argument("--promotion-min-cycles", type=int, default=3, help="Promotion gate required positive cycles")
    ap.add_argument(
        "--promotion-max-concentration-share",
        type=float,
        default=0.35,
        help="Promotion gate maximum market-side concentration share",
    )
    ap.add_argument("--cycle-minutes", type=int, default=30, help="Cycle window used for stability and promotion")
    ap.add_argument("--top", type=int, default=15, help="Top rows per section")
    ap.add_argument("--out-root", default="logs/analysis", help="Run-pack output root directory")
    ap.add_argument(
        "--config",
        default="config/kalshi_family_adaptive.yaml",
        help="Config file to hash into manifest for reproducibility",
    )
    ap.add_argument("--validate", action="store_true", help="Validate all artifacts after writing")
    args = ap.parse_args()

    windows = [float(args.hours)] if args.hours is not None else [float(x) for x in args.windows]
    windows = sorted(set(windows))
    log_path = Path(args.log)
    if not log_path.exists():
        raise SystemExit(f"log not found: {log_path}")

    records = _load_records(log_path)
    max_ts = _max_timestamp(records)
    if max_ts is None:
        raise SystemExit("no timestamped records found")

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = hashlib.sha256(f"{run_ts}:{log_path}".encode()).hexdigest()[:8]
    git_hash = _git_hash()
    config_hash = _file_hash(Path(args.config))
    dir_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    out_dir = Path(args.out_root) / f"runpack_{dir_ts}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "generated_at_utc": run_ts,
        "git_hash": git_hash,
        "config_hash": config_hash,
        "log": str(log_path),
        "max_timestamp": max_ts,
        "windows_hours": windows,
        "outputs": [],
    }

    metrics_summary_windows: List[Dict[str, Any]] = []

    for hours in windows:
        since_ts = max_ts - (hours * 3600.0)
        settlements = _window_settlements(records, since_ts)
        label = f"{int(hours) if float(hours).is_integer() else hours}h"
        prefix = out_dir / f"window_{label}"

        cfg = WindowConfig(
            hours=hours,
            min_samples=args.promotion_min_samples,
            min_cycle_samples=args.promotion_min_cycle_samples,
            min_cycles=args.promotion_min_cycles,
            max_concentration_share=args.promotion_max_concentration_share,
            cycle_minutes=args.cycle_minutes,
        )

        context_json, context_txt = _build_context_report(
            settlements,
            hours=hours,
            min_samples=args.min_samples,
            top=args.top,
        )
        promotion_json, promotion_txt = _build_promotion_gate(settlements, cfg, top=args.top)
        concentration_json, concentration_txt = _build_concentration_summary(settlements, top=args.top)
        cycle_json, cycle_txt = _build_cycle_stability_summary(
            settlements,
            cycle_minutes=args.cycle_minutes,
            top_contexts=args.top,
        )
        feature_json, feature_txt = _build_context_feature_coverage(settlements)
        policy_json, policy_txt = _build_policy_snapshot()
        edge_json, edge_txt = _build_edge_retention_summary(settlements, min_samples=max(20, args.min_samples // 10))
        sleeve_json, sleeve_txt = _build_sleeve_metrics(settlements)
        pop_json, pop_txt = _build_population_metrics(settlements, top=args.top)

        artifacts = {
            "context_report_json": str(prefix) + "_context_report.json",
            "context_report_txt": str(prefix) + "_context_report.txt",
            "promotion_gate_json": str(prefix) + "_promotion_gate.json",
            "promotion_gate_txt": str(prefix) + "_promotion_gate.txt",
            "concentration_summary_json": str(prefix) + "_concentration_summary.json",
            "concentration_summary_txt": str(prefix) + "_concentration_summary.txt",
            "cycle_stability_summary_json": str(prefix) + "_cycle_stability_summary.json",
            "cycle_stability_summary_txt": str(prefix) + "_cycle_stability_summary.txt",
            "context_feature_coverage_json": str(prefix) + "_context_feature_coverage.json",
            "context_feature_coverage_txt": str(prefix) + "_context_feature_coverage.txt",
            "policy_snapshot_json": str(prefix) + "_policy_snapshot.json",
            "policy_snapshot_txt": str(prefix) + "_policy_snapshot.txt",
            "edge_retention_json": str(prefix) + "_edge_retention.json",
            "edge_retention_txt": str(prefix) + "_edge_retention.txt",
            "sleeve_metrics_json": str(prefix) + "_sleeve_metrics.json",
            "sleeve_metrics_txt": str(prefix) + "_sleeve_metrics.txt",
            "population_metrics_json": str(prefix) + "_population_metrics.json",
            "population_metrics_txt": str(prefix) + "_population_metrics.txt",
        }

        _write_json(Path(artifacts["context_report_json"]), context_json)
        _write_txt(Path(artifacts["context_report_txt"]), context_txt)
        _write_json(Path(artifacts["promotion_gate_json"]), promotion_json)
        _write_txt(Path(artifacts["promotion_gate_txt"]), promotion_txt)
        _write_json(Path(artifacts["concentration_summary_json"]), concentration_json)
        _write_txt(Path(artifacts["concentration_summary_txt"]), concentration_txt)
        _write_json(Path(artifacts["cycle_stability_summary_json"]), cycle_json)
        _write_txt(Path(artifacts["cycle_stability_summary_txt"]), cycle_txt)
        _write_json(Path(artifacts["context_feature_coverage_json"]), feature_json)
        _write_txt(Path(artifacts["context_feature_coverage_txt"]), feature_txt)
        _write_json(Path(artifacts["policy_snapshot_json"]), policy_json)
        _write_txt(Path(artifacts["policy_snapshot_txt"]), policy_txt)
        _write_json(Path(artifacts["edge_retention_json"]), edge_json)
        _write_txt(Path(artifacts["edge_retention_txt"]), edge_txt)
        _write_json(Path(artifacts["sleeve_metrics_json"]), sleeve_json)
        _write_txt(Path(artifacts["sleeve_metrics_txt"]), sleeve_txt)
        _write_json(Path(artifacts["population_metrics_json"]), pop_json)
        _write_txt(Path(artifacts["population_metrics_txt"]), pop_txt)

        # Collect window summary for metrics_summary.json
        top_conc = 0.0
        if concentration_json.get("top_market_side"):
            top_conc = float(concentration_json["top_market_side"][0].get("share", 0.0))
        total = len(settlements)
        total_pnl = float(context_json.get("total_pnl_usd", 0.0))
        metrics_summary_windows.append({
            "hours": hours,
            "settlements": total,
            "total_pnl_usd": total_pnl,
            "avg_pnl_usd": round(total_pnl / total, 6) if total else 0.0,
            "win_rate_pct": float(context_json.get("win_rate_pct", 0.0)),
            "max_drawdown_usd": _compute_max_drawdown(settlements),
            "top_concentration_share": round(top_conc, 6),
            "market_side_hhi": float(concentration_json.get("market_side_hhi", 0.0)),
            "context_coverage_pct": float(context_json.get("context_coverage_pct", 0.0)),
            "edge_retention_poor_keys": int(edge_json.get("poor_retention_keys", 0)),
            "promotable_contexts": int(promotion_json.get("counts", {}).get("promotable_total", 0)),
        })

        manifest["outputs"].append({
            "window_hours": hours,
            "settlements": len(settlements),
            "since_ts": since_ts,
            "artifacts": artifacts,
        })
        print(f"[run-pack] window={hours:g}h settlements={len(settlements)}")

    # Write top-level metrics_summary.json — primary input for rollup and readiness gate
    metrics_summary: Dict[str, Any] = {
        "run_id": run_id,
        "generated_at_utc": run_ts,
        "git_hash": git_hash,
        "config_hash": config_hash,
        "windows": metrics_summary_windows,
    }
    metrics_summary_path = out_dir / "metrics_summary.json"
    _write_json(metrics_summary_path, metrics_summary)
    manifest["metrics_summary"] = str(metrics_summary_path)

    manifest_path = out_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    print(f"[run-pack] wrote {out_dir}")
    print(f"[run-pack] run_id={run_id}  git={git_hash}  config_hash={config_hash}")

    if args.validate:
        _validate_runpack(out_dir, manifest)


if __name__ == "__main__":
    main()
