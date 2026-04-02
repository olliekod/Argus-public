# Created by Oliver Meihls

# Apply promotion gate: read paper trade settlements, classify context keys
# into core/explore/unknown lanes, and write a versioned policy JSON file.
#
# Usage:
# python scripts/kalshi_apply_promotion.py
# python scripts/kalshi_apply_promotion.py --log logs/paper_trades.jsonl --hours 8
# python scripts/kalshi_apply_promotion.py --output config/kalshi_context_policy.json

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


LANE_CORE = "core"
LANE_EXPLORE = "explore"
LANE_UNKNOWN = "unknown"

# Fixed fallback weights (continuous weights are computed per-key below)
WEIGHT_CORE = 1.3
WEIGHT_EXPLORE = 0.5
WEIGHT_UNKNOWN = 1.0

# Continuous weight bounds
_WEIGHT_CORE_MIN = 1.3
_WEIGHT_CORE_MAX = 1.5   # bonus for high-expectancy core keys
_WEIGHT_EXPLORE_MAX = 0.5
_WEIGHT_EXPLORE_MIN = 0.1  # near-block for deeply toxic keys
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIFETIME_MIN_FILLS = 10


def _continuous_weight(
    lane: str,
    avg_pnl: float,
    promote_threshold: float,
    demote_threshold: float,
) -> float:
    # Compute a continuous weight within the lane's range based on avg_pnl.
    #
    # Core:    [1.3, 1.5]  — extra boost for strong positive expectancy
    # Explore: [0.1, 0.5]  — deeper cuts for worse performers
    # Unknown: 1.0          — unchanged
    if lane == LANE_CORE:
        # At promote_threshold: 1.3. Each +1.0 of avg_pnl adds ~0.05 up to 1.5.
        bonus = min(0.2, max(0.0, (avg_pnl - promote_threshold) * 0.05))
        return round(_WEIGHT_CORE_MIN + bonus, 3)
    if lane == LANE_EXPLORE:
        # At demote_threshold: 0.5. Each -1.0 of avg_pnl cuts 0.04 down to 0.1.
        severity = max(0.0, abs(avg_pnl) - abs(demote_threshold))
        penalty = min(0.4, severity * 0.04)
        return round(max(_WEIGHT_EXPLORE_MIN, _WEIGHT_EXPLORE_MAX - penalty), 3)
    return WEIGHT_UNKNOWN


def _ctx_get(rec: Dict[str, Any], key: str, default: str = "na") -> str:
    ctx = rec.get("decision_context") or {}
    if not isinstance(ctx, dict):
        return default
    return str(ctx.get(key, default))


def _momentum_bucket(drift: float, scale: float = 1e-4) -> str:
    # Match momentum_bucket() in context_policy.py — must stay in sync.
    if drift > scale:
        return "up"
    if drift < -scale:
        return "dn"
    return "flat"


def _context_key(rec: Dict[str, Any]) -> str:
    # Build context key: family|side|edge_bucket|price_bucket|strike_distance_bucket|near_money|momentum.
    family = str(rec.get("family") or "UNK")
    side_code = _ctx_get(rec, "sd", "na")
    side = "yes" if side_code == "y" else ("no" if side_code == "n" else side_code)
    eb = _ctx_get(rec, "eb")
    pb = _ctx_get(rec, "pb")
    sdb = _ctx_get(rec, "sdb")
    # near_money stored as boolean; convert to "nm"/"far" string
    ctx = rec.get("decision_context") or {}
    nm_raw = ctx.get("nm", False) if isinstance(ctx, dict) else False
    nm = "nm" if nm_raw else "far"
    # mb stored since new records; derive from drift for older records
    ctx_mb = ctx.get("mb") if isinstance(ctx, dict) else None
    if ctx_mb in ("up", "dn", "flat"):
        mb = ctx_mb
    else:
        drift = float(ctx.get("drift") or 0.0) if isinstance(ctx, dict) else 0.0
        mb = _momentum_bucket(drift)
    return f"{family}|{side}|{eb}|{pb}|{sdb}|{nm}|{mb}"


def _load_settlements(
    log_path: Path,
    hours: float,
) -> Tuple[List[Dict[str, Any]], float]:
    # Load settlement records from JSONL within the lookback window.
    #
    # Returns (records, max_ts).
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
        return [], 0.0

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

    return rows, max_ts


def _classify(
    rows: List[Dict[str, Any]],
    min_samples: int,
    promote_threshold: float,
    demote_threshold: float,
) -> Dict[str, Dict[str, Any]]:
    # Group settlements by context key and classify each.
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

    result: Dict[str, Dict[str, Any]] = {}
    for key, item in agg.items():
        count = item["count"]
        total_pnl = item["total_pnl"]
        avg_pnl = total_pnl / count if count else 0.0
        win_rate = item["wins"] / count if count else 0.0

        if count >= min_samples and avg_pnl >= promote_threshold:
            lane = LANE_CORE
            weight = _continuous_weight(LANE_CORE, avg_pnl, promote_threshold, demote_threshold)
        elif count >= min_samples and avg_pnl <= demote_threshold:
            lane = LANE_EXPLORE
            weight = _continuous_weight(LANE_EXPLORE, avg_pnl, promote_threshold, demote_threshold)
        else:
            lane = LANE_UNKNOWN
            weight = WEIGHT_UNKNOWN

        result[key] = {
            "lane": lane,
            "weight": weight,
            "count": count,
            "total_pnl": round(total_pnl, 4),
            "avg_pnl": round(avg_pnl, 4),
            "win_rate": round(win_rate, 4),
        }

    return result


def _write_policy(
    output_path: Path,
    keys: Dict[str, Dict[str, Any]],
    total_settlements: int,
) -> Dict[str, Any]:
    # Write the policy JSON and return the document.
    core_count = sum(1 for v in keys.values() if v["lane"] == LANE_CORE)
    explore_count = sum(1 for v in keys.values() if v["lane"] == LANE_EXPLORE)
    unknown_count = sum(1 for v in keys.values() if v["lane"] == LANE_UNKNOWN)

    doc = {
        "version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "core_count": core_count,
            "explore_count": explore_count,
            "unknown_count": unknown_count,
            "total_settlements": total_settlements,
        },
        "keys": keys,
    }

    os.makedirs(output_path.parent, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    return doc


def _write_summary_txt(txt_path: Path, doc: Dict[str, Any]) -> None:
    # Write a human-readable .txt summary next to the JSON.
    lines: List[str] = []
    lines.append("Kalshi Context Policy Summary")
    lines.append("=" * 50)
    lines.append(f"Generated: {doc['timestamp']}")
    lines.append(f"Version:   {doc['version']}")
    lines.append("")

    s = doc["summary"]
    lines.append(f"Total settlements: {s['total_settlements']}")
    lines.append(f"Core keys:         {s['core_count']}")
    lines.append(f"Explore keys:      {s['explore_count']}")
    lines.append(f"Unknown keys:      {s['unknown_count']}")
    lines.append("")

    for lane_name in [LANE_CORE, LANE_EXPLORE, LANE_UNKNOWN]:
        lane_keys = {
            k: v for k, v in doc["keys"].items() if v["lane"] == lane_name
        }
        if not lane_keys:
            continue
        lines.append(f"--- {lane_name.upper()} (weight={lane_keys[next(iter(lane_keys))]['weight']}) ---")
        sorted_keys = sorted(
            lane_keys.items(), key=lambda x: x[1].get("avg_pnl", 0.0), reverse=True
        )
        for key, info in sorted_keys:
            wr_pct = info.get("win_rate", 0.0) * 100
            lines.append(
                f"  {key:60s} n={info.get('count', 0):5d}  avg_pnl={info.get('avg_pnl', 0.0):+8.4f}  "
                f"wr={wr_pct:5.1f}%  total_pnl={info.get('total_pnl', 0.0):+10.4f}"
            )
        lines.append("")

    os.makedirs(txt_path.parent, exist_ok=True)
    with txt_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
        fh.write("\n")


def _print_summary(doc: Dict[str, Any]) -> None:
    # Print a concise summary to stdout.
    s = doc["summary"]
    print(f"Kalshi Context Policy (v{doc['version']})")
    print(f"  timestamp:        {doc['timestamp']}")
    print(f"  total_settlements: {s['total_settlements']}")
    print(f"  core:             {s['core_count']} keys (weight {WEIGHT_CORE})")
    print(f"  explore:          {s['explore_count']} keys (weight {WEIGHT_EXPLORE})")
    print(f"  unknown:          {s['unknown_count']} keys (weight {WEIGHT_UNKNOWN})")
    print()

    for lane_name, label in [
        (LANE_CORE, "CORE"),
        (LANE_EXPLORE, "EXPLORE"),
    ]:
        lane_keys = {
            k: v for k, v in doc["keys"].items() if v["lane"] == lane_name
        }
        if not lane_keys:
            continue
        print(f"  {label}:")
        sorted_keys = sorted(
            lane_keys.items(), key=lambda x: x[1].get("avg_pnl", 0.0), reverse=True
        )
        for key, info in sorted_keys[:10]:
            wr_pct = info.get("win_rate", 0.0) * 100
            print(
                f"    {key:55s} n={info.get('count', 0):5d}  avg={info.get('avg_pnl', 0.0):+.4f}  "
                f"wr={wr_pct:5.1f}%"
            )
        if len(sorted_keys) > 10:
            print(f"    ... and {len(sorted_keys) - 10} more")
        print()


def _merge_policy_keys(
    current_keys: Dict[str, Dict[str, Any]],
    prior_keys: Dict[str, Dict[str, Any]],
    min_samples: int,
    promote_threshold: float,
    demote_threshold: float,
) -> Dict[str, Dict[str, Any]]:
    # Merge current and prior context-key stats using additive counts and PnL.
    merged: Dict[str, Dict[str, Any]] = {}

    for key in set(current_keys) | set(prior_keys):
        cur = current_keys.get(key)
        prv = prior_keys.get(key)

        if not isinstance(cur, dict):
            cur = None
        if not isinstance(prv, dict):
            prv = None

        if cur and prv:
            cur_count = int(cur.get("count") or 0)
            prv_count = int(prv.get("count") or 0)
            cnt = cur_count + prv_count
            tot = float(cur.get("total_pnl") or 0.0) + float(prv.get("total_pnl") or 0.0)
            wins = round(float(cur.get("win_rate") or 0.0) * cur_count) + round(
                float(prv.get("win_rate") or 0.0) * prv_count
            )
            avg = tot / cnt if cnt else 0.0
            wr = wins / cnt if cnt else 0.0
            if cnt >= min_samples and avg >= promote_threshold:
                lane = LANE_CORE
                weight = _continuous_weight(LANE_CORE, avg, promote_threshold, demote_threshold)
            elif cnt >= min_samples and avg <= demote_threshold:
                lane = LANE_EXPLORE
                weight = _continuous_weight(LANE_EXPLORE, avg, promote_threshold, demote_threshold)
            else:
                lane = LANE_UNKNOWN
                weight = WEIGHT_UNKNOWN
            merged[key] = {
                "lane": lane,
                "weight": weight,
                "count": cnt,
                "total_pnl": round(tot, 4),
                "avg_pnl": round(avg, 4),
                "win_rate": round(wr, 4),
            }
        elif cur:
            merged[key] = cur
        elif prv:
            merged[key] = prv

    return merged


def _load_bot_params_lookup(dwarf_names_file: str) -> Dict[str, Dict[str, Any]]:
    # Map bot_id to its static grid parameters.
    try:
        names = Path(dwarf_names_file).read_text(encoding="utf-8").split()
    except Exception:
        return {}

    try:
        import sys

        sys.path.insert(0, str(ROOT))
        from argus_kalshi.farm_grid import _default_grid_product

        grid = _default_grid_product()
    except Exception:
        return {}

    if not grid:
        return {}

    return {name: grid[idx % len(grid)] for idx, name in enumerate(names)}


def _rank_bot_entries(
    bot_stats: Dict[str, Dict[str, Any]],
    min_fills: int,
) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []
    for bot_id, stats in bot_stats.items():
        fills = int(stats.get("count", stats.get("fills", 0)) or 0)
        if fills < min_fills:
            continue
        wins = int(stats.get("wins", 0) or 0)
        total_pnl = float(stats.get("total_pnl") or 0.0)
        entry = {
            "bot_id": bot_id,
            "fills": fills,
            "total_pnl": round(total_pnl, 4),
            "avg_pnl": round(total_pnl / fills, 4) if fills else 0.0,
            "win_rate": round(wins / fills, 4) if fills else 0.0,
        }
        params = stats.get("params")
        if isinstance(params, dict):
            entry["params"] = params
        ranked.append(entry)

    ranked.sort(key=lambda x: x["total_pnl"], reverse=True)
    return ranked


def _compute_winner_zone(top_bots: List[Dict[str, Any]]) -> Dict[str, Any]:
    winner_params: Dict[str, List[float]] = defaultdict(list)
    for entry in top_bots:
        params = entry.get("params")
        if not isinstance(params, dict):
            continue
        for key, value in params.items():
            if isinstance(value, (int, float)):
                winner_params[key].append(float(value))

    winner_zone: Dict[str, Any] = {}
    for key, vals in winner_params.items():
        if vals:
            winner_zone[key] = {
                "min": min(vals),
                "max": max(vals),
                "mean": round(sum(vals) / len(vals), 4),
            }
    return winner_zone


def _bot_params_from_performance(doc: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    params_by_bot: Dict[str, Dict[str, Any]] = {}
    for section in ("top_bots", "bottom_bots"):
        for entry in doc.get(section, []):
            if not isinstance(entry, dict):
                continue
            bot_id = entry.get("bot_id")
            params = entry.get("params")
            if bot_id and isinstance(params, dict):
                params_by_bot[str(bot_id)] = params
    return params_by_bot


def _new_lifetime_doc() -> Dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_cycles": 0,
        "total_settlements": 0,
        "processed_run_files": [],
        "bots": {},
        "winner_zone": {},
        "top_bots": [],
    }


def _load_lifetime_doc(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return _new_lifetime_doc()
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _new_lifetime_doc()

    if not isinstance(doc, dict):
        return _new_lifetime_doc()

    doc.setdefault("processed_run_files", [])
    doc.setdefault("bots", {})
    doc.setdefault("winner_zone", {})
    doc.setdefault("top_bots", [])
    return doc


def _settlement_dedup_key(rec: Dict[str, Any]) -> Optional[Tuple[str, str, str]]:
    bot_id = str(rec.get("bot_id") or "").strip()
    market_ticker = str(rec.get("market_ticker") or "").strip()
    timestamp = rec.get("timestamp")
    if not bot_id or not market_ticker or timestamp is None:
        return None
    return bot_id, market_ticker, str(timestamp)


def _cycle_id_from_run_file(path: Path) -> str:
    return path.stem.removeprefix("run_")


def _winner_zone_delta(
    current_zone: Dict[str, Any],
    lifetime_zone: Dict[str, Any],
    mean_shift_threshold: float = 0.15,
) -> Tuple[bool, List[str]]:
    changed: List[str] = []
    for key in sorted(set(current_zone) | set(lifetime_zone)):
        cur = current_zone.get(key)
        life = lifetime_zone.get(key)
        if not isinstance(cur, dict) or not isinstance(life, dict):
            changed.append(key)
            continue
        cur_mean = float(cur.get("mean") or 0.0)
        life_mean = float(life.get("mean") or 0.0)
        span = abs(float(cur.get("max") or 0.0) - float(cur.get("min") or 0.0))
        baseline = span if span > 0 else max(abs(cur_mean), 1.0)
        if abs(life_mean - cur_mean) > baseline * mean_shift_threshold:
            changed.append(key)
    return bool(changed), changed


def _update_lifetime_performance(
    training_dir: Path,
    lifetime_path: Path,
    current_bot_performance: Dict[str, Any],
    dwarf_names_file: str,
    top_n: int,
    min_fills: int = DEFAULT_LIFETIME_MIN_FILLS,
) -> Dict[str, Any]:
    doc = _load_lifetime_doc(lifetime_path)
    bots = doc.setdefault("bots", {})
    processed_files: Set[str] = {
        str(name) for name in doc.get("processed_run_files", []) if name
    }
    run_files = sorted(training_dir.glob("run_*.jsonl"))
    params_lookup = _load_bot_params_lookup(dwarf_names_file)
    current_params = _bot_params_from_performance(current_bot_performance)
    processed_now: List[str] = []

    for run_file in run_files:
        if run_file.name in processed_files:
            continue
        cycle_id = _cycle_id_from_run_file(run_file)
        seen_in_file: Set[Tuple[str, str, str]] = set()
        cycle_bots: Set[str] = set()
        try:
            lines = run_file.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue

        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("type") != "settlement":
                continue
            dedup_key = _settlement_dedup_key(rec)
            if dedup_key is None or dedup_key in seen_in_file:
                continue
            seen_in_file.add(dedup_key)

            bot_id = dedup_key[0]
            pnl = float(rec.get("pnl_usd") or 0.0)
            won = 1 if bool(rec.get("won")) else 0
            bot = bots.setdefault(
                bot_id,
                {
                    "total_pnl": 0.0,
                    "fills": 0,
                    "wins": 0,
                    "avg_pnl": 0.0,
                    "win_rate": 0.0,
                    "cycles_active": 0,
                    "last_seen_cycle": None,
                    "params": None,
                },
            )
            bot["total_pnl"] = float(bot.get("total_pnl") or 0.0) + pnl
            bot["fills"] = int(bot.get("fills", 0) or 0) + 1
            bot["wins"] = int(bot.get("wins", 0) or 0) + won
            cycle_bots.add(bot_id)

        for bot_id in cycle_bots:
            bot = bots[bot_id]
            bot["cycles_active"] = int(bot.get("cycles_active", 0) or 0) + 1
            bot["last_seen_cycle"] = cycle_id

        processed_files.add(run_file.name)
        processed_now.append(run_file.name)

    for bot_id, bot in bots.items():
        fills = int(bot.get("fills", 0) or 0)
        wins = int(bot.get("wins", 0) or 0)
        total_pnl = float(bot.get("total_pnl") or 0.0)
        bot["total_pnl"] = round(total_pnl, 4)
        bot["fills"] = fills
        bot["wins"] = wins
        bot["avg_pnl"] = round(total_pnl / fills, 4) if fills else 0.0
        bot["win_rate"] = round(wins / fills, 4) if fills else 0.0
        params = current_params.get(bot_id) or bot.get("params") or params_lookup.get(bot_id)
        bot["params"] = params if isinstance(params, dict) else None

    ranked = _rank_bot_entries(
        {
            bot_id: {
                "count": bot.get("fills", 0),
                "total_pnl": bot.get("total_pnl", 0.0),
                "wins": bot.get("wins", 0),
                "params": bot.get("params"),
            }
            for bot_id, bot in bots.items()
        },
        min_fills=min_fills,
    )
    top_bots = ranked[:top_n]
    doc["generated_at"] = datetime.now(timezone.utc).isoformat()
    doc["total_cycles"] = len(processed_files)
    doc["total_settlements"] = sum(int(bot.get("fills", 0) or 0) for bot in bots.values())
    doc["processed_run_files"] = sorted(processed_files)
    doc["winner_zone"] = _compute_winner_zone(top_bots)
    doc["top_bots"] = top_bots
    doc["_summary"] = {
        "processed_now": processed_now,
        "unique_bots": len(bots),
        "ranked_bots": len(ranked),
    }

    os.makedirs(lifetime_path.parent, exist_ok=True)
    with lifetime_path.open("w", encoding="utf-8") as fh:
        json.dump({k: v for k, v in doc.items() if k != "_summary"}, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    return doc


def _print_lifetime_summary(
    lifetime_doc: Dict[str, Any],
    current_bot_performance: Dict[str, Any],
    lifetime_path: Path,
) -> None:
    summary = lifetime_doc.get("_summary", {})
    top_bots = lifetime_doc.get("top_bots", [])
    changed, changed_fields = _winner_zone_delta(
        current_bot_performance.get("winner_zone", {}),
        lifetime_doc.get("winner_zone", {}),
    )

    print(f"Lifetime bot performance written to: {lifetime_path}")
    print(f"  Total unique bots tracked: {summary.get('unique_bots', len(lifetime_doc.get('bots', {})))}")
    print(f"  Total cycles processed:    {lifetime_doc.get('total_cycles', 0)}")
    print(f"  Total settlements:         {lifetime_doc.get('total_settlements', 0)}")
    processed_now = summary.get("processed_now", [])
    if processed_now:
        print(f"  New run files ingested:    {len(processed_now)}")
    else:
        print("  New run files ingested:    0 (already up to date)")
    if top_bots:
        print("  Lifetime top 5:")
        for entry in top_bots[:5]:
            print(
                f"    {entry['bot_id']:16s} pnl={entry['total_pnl']:+.2f}  "
                f"fills={entry['fills']:4d}  win_rate={entry['win_rate'] * 100:5.1f}%"
            )
        print(f"  Winner zone differs from current cycle: {'yes' if changed else 'no'}")
        if changed_fields:
            print(f"    changed fields: {', '.join(changed_fields)}")
        top_bot = top_bots[0]
        print(
            f"  Top lifetime bot: {top_bot['bot_id']}  total_pnl={top_bot['total_pnl']:+.2f}  "
            f"fills={top_bot['fills']}  params={top_bot.get('params')}"
        )


def _extract_bot_performance(
    rows: List[Dict[str, Any]],
    dwarf_names_file: str,
    top_n: int = 50,
) -> Dict[str, Any]:
    # Build per-bot performance stats and map top/bottom bots to their grid parameters.
    #
    # Returns a dict ready to be written as JSON (genetic pressure output).
    # Aggregate per bot
    bot_stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "total_pnl": 0.0, "wins": 0}
    )
    for r in rows:
        bot_id = str(r.get("bot_id") or "unknown")
        if bot_id == "unknown":
            continue
        pnl = float(r.get("pnl_usd") or 0.0)
        won = 1 if bool(r.get("won")) else 0
        s = bot_stats[bot_id]
        s["count"] += 1
        s["total_pnl"] += pnl
        s["wins"] += won

    if not bot_stats:
        return {"error": "no bot_id data in settlements"}

    # Rank bots by total_pnl
    ranked = _rank_bot_entries(bot_stats, min_fills=5)

    if not ranked:
        return {"error": "no bots with enough fills"}

    # Load dwarf names to map bot_id → grid index
    params_lookup = _load_bot_params_lookup(dwarf_names_file)
    for entry in ranked:
        entry["params"] = params_lookup.get(entry["bot_id"])

    top_bots = ranked[:top_n]
    bottom_bots = ranked[-top_n:]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_bots_ranked": len(ranked),
        "top_n": top_n,
        "winner_zone": _compute_winner_zone(top_bots),
        "top_bots": top_bots,
        "bottom_bots": bottom_bots,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Consume promotion gate outputs and write a versioned context policy file."
    )
    ap.add_argument(
        "--log",
        default="logs/paper_trades.jsonl",
        help="Path to paper_trades.jsonl (default: logs/paper_trades.jsonl)",
    )
    ap.add_argument(
        "--hours",
        type=float,
        default=8.0,
        help="Lookback window in hours (default: 8.0)",
    )
    ap.add_argument(
        "--output",
        default="config/kalshi_context_policy.json",
        help="Output policy JSON path (default: config/kalshi_context_policy.json)",
    )
    ap.add_argument(
        "--min-samples",
        type=int,
        default=50,
        help="Minimum settlements per context key (default: 50)",
    )
    ap.add_argument(
        "--promote-threshold",
        type=float,
        default=0.50,
        help="Avg PnL threshold for core promotion (default: 0.50)",
    )
    ap.add_argument(
        "--demote-threshold",
        type=float,
        default=-0.50,
        help="Avg PnL threshold for explore demotion (default: -0.50)",
    )
    ap.add_argument(
        "--merge",
        action="store_true",
        help="Merge new settlements into existing policy (accumulate counts across cycles)",
    )
    ap.add_argument(
        "--bot-performance-output",
        default="config/kalshi_bot_performance.json",
        help="Path to write per-bot performance + winner zone (default: config/kalshi_bot_performance.json)",
    )
    ap.add_argument(
        "--lifetime-performance-output",
        default="config/kalshi_lifetime_performance.json",
        help="Path to write lifetime bot performance (default: config/kalshi_lifetime_performance.json)",
    )
    ap.add_argument(
        "--dwarf-names",
        default="argus_kalshi/dwarf_names.txt",
        help="Path to dwarf names file for grid parameter lookup (default: argus_kalshi/dwarf_names.txt)",
    )
    ap.add_argument(
        "--top-n-bots",
        type=int,
        default=50,
        help="Number of top/bottom bots to include in performance output (default: 50)",
    )
    ap.add_argument(
        "--lifetime-min-fills",
        type=int,
        default=DEFAULT_LIFETIME_MIN_FILLS,
        help="Minimum lifetime fills required for lifetime winner zone/top bot ranking (default: 10)",
    )
    args = ap.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        print("No settlements to process. Exiting.")
        raise SystemExit(0)

    output_path = Path(args.output)

    rows, max_ts = _load_settlements(log_path, args.hours)
    if not rows:
        print("No settlement records found in the specified window.")
        raise SystemExit(0)

    print(
        f"Loaded {len(rows)} settlements from last {args.hours:.1f}h "
        f"(min_samples={args.min_samples}, promote>={args.promote_threshold}, "
        f"demote<={args.demote_threshold})"
    )

    keys = _classify(rows, args.min_samples, args.promote_threshold, args.demote_threshold)

    # Merge with existing policy: add prior counts so signal accumulates across cycles.
    if args.merge and output_path.exists():
        try:
            prior = json.loads(output_path.read_text(encoding="utf-8"))
            prior_keys = prior.get("keys", {})
            prior_total = prior.get("summary", {}).get("total_settlements", 0)
            keys = _merge_policy_keys(
                current_keys=keys,
                prior_keys=prior_keys if isinstance(prior_keys, dict) else {},
                min_samples=args.min_samples,
                promote_threshold=args.promote_threshold,
                demote_threshold=args.demote_threshold,
            )
            print(f"Merged with prior policy ({prior_total} prior settlements + {len(rows)} new)")
        except Exception as e:
            print(f"Warning: could not merge prior policy ({e}), using current run only")

    total_settlements = sum(
        int(v.get("count") or 0)
        for v in keys.values()
        if isinstance(v, dict)
    )
    doc = _write_policy(output_path, keys, total_settlements)

    txt_path = output_path.with_suffix(".txt")
    _write_summary_txt(txt_path, doc)

    print(f"Policy written to: {output_path}")
    print(f"Summary written to: {txt_path}")
    print()
    _print_summary(doc)

    # Bot-level performance + winner zone for genetic pressure
    bp_path = Path(args.bot_performance_output)
    bp_data = _extract_bot_performance(rows, args.dwarf_names, args.top_n_bots)
    if "error" not in bp_data:
        os.makedirs(bp_path.parent, exist_ok=True)
        with bp_path.open("w", encoding="utf-8") as fh:
            json.dump(bp_data, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        wz = bp_data.get("winner_zone", {})
        print(f"Bot performance written to: {bp_path}")
        print(f"  Total bots ranked: {bp_data.get('total_bots_ranked', 0)}")
        if wz:
            print("  Winner zone (top bot parameter centroids):")
            for k, v in wz.items():
                print(f"    {k}: [{v['min']:.3g}, {v['max']:.3g}]  mean={v['mean']:.3g}")
        top = bp_data.get("top_bots", [])
        if top:
            print(f"  Top bot: {top[0]['bot_id']}  total_pnl={top[0]['total_pnl']:+.2f}  "
                  f"fills={top[0]['fills']}  params={top[0].get('params')}")
        lifetime_doc = _update_lifetime_performance(
            training_dir=ROOT / "logs" / "training_data",
            lifetime_path=Path(args.lifetime_performance_output),
            current_bot_performance=bp_data,
            dwarf_names_file=args.dwarf_names,
            top_n=args.top_n_bots,
            min_fills=args.lifetime_min_fills,
        )
        _print_lifetime_summary(
            lifetime_doc=lifetime_doc,
            current_bot_performance=bp_data,
            lifetime_path=Path(args.lifetime_performance_output),
        )
    else:
        print(f"  Bot performance: {bp_data['error']}")


if __name__ == "__main__":
    main()
