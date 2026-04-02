#!/usr/bin/env python3
# Created by Oliver Meihls

# Analyze a Polymarket wallet's behavior from ingested data.
#
# Usage:
# python scripts/polymarket_wallet_analyze.py --wallet 0xabc...
# python scripts/polymarket_wallet_analyze.py --wallet 0xabc --hours 168 --out logs/analysis/poly

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

DEFAULT_DB = "data/polymarket_wallets.db"


def _parse_ts(v: str) -> Optional[datetime]:
    if not v:
        return None
    try:
        if v.endswith("Z"):
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        return datetime.fromisoformat(v)
    except ValueError:
        return None


def _fmt_pct(x: float) -> str:
    return f"{x * 100.0:.2f}%"


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0:
        return min(values)
    if q >= 1:
        return max(values)
    arr = sorted(values)
    idx = int(round((len(arr) - 1) * q))
    return arr[idx]


def _infer_style(
    *,
    median_hold_minutes: float,
    p90_notional: float,
    price_band_center: float,
    buy_sell_ratio: float,
    median_gap_seconds: float,
) -> str:
    tags: List[str] = []
    if median_hold_minutes > 0 and median_hold_minutes <= 30:
        tags.append("short-hold")
    elif median_hold_minutes >= 240:
        tags.append("swing")
    if median_gap_seconds > 0 and median_gap_seconds <= 30:
        tags.append("high-frequency")
    if p90_notional >= 1000:
        tags.append("large-ticket")
    if 0.40 <= price_band_center <= 0.65:
        tags.append("mid-price-focus")
    elif price_band_center <= 0.20:
        tags.append("tail-price-focus")
    if buy_sell_ratio >= 2.0:
        tags.append("long-bias")
    elif buy_sell_ratio <= 0.5:
        tags.append("short-bias")
    if not tags:
        tags.append("mixed")
    return ",".join(tags)


def _load_trades(
    conn: sqlite3.Connection,
    wallet: str,
    since_iso: Optional[str],
) -> List[sqlite3.Row]:
    sql = """
        SELECT wallet, trade_uid, ts, market_slug, market_id, token_id, side, price, size,
               notional_usd, role, tx_hash, order_id, raw_json
        FROM poly_wallet_trades
        WHERE wallet = ?
    """
    params: List[Any] = [wallet]
    if since_iso:
        sql += " AND ts >= ?"
        params.append(since_iso)
    sql += " ORDER BY ts ASC"
    cur = conn.execute(sql, tuple(params))
    return cur.fetchall()


def _load_closed_positions(
    conn: sqlite3.Connection,
    wallet: str,
    since_iso: Optional[str],
) -> List[sqlite3.Row]:
    sql = """
        SELECT wallet, closed_uid, ts, market_slug, market_id, token_id, outcome, size,
               avg_entry, avg_exit, realized_pnl, raw_json
        FROM poly_wallet_closed_positions
        WHERE wallet = ?
    """
    params: List[Any] = [wallet]
    if since_iso:
        sql += " AND ts >= ?"
        params.append(since_iso)
    sql += " ORDER BY ts ASC"
    cur = conn.execute(sql, tuple(params))
    return cur.fetchall()


def _build_hold_stats(trades: Iterable[sqlite3.Row]) -> Tuple[List[float], int]:
    # FIFO hold-time approximation per token_id using opposite-side close.
    open_queues: Dict[str, deque] = defaultdict(deque)
    holds_min: List[float] = []
    closed_pairs = 0
    for row in trades:
        token = row["token_id"] or row["market_id"] or "na"
        side = (row["side"] or "").lower()
        ts = _parse_ts(row["ts"])
        if ts is None or side not in ("buy", "sell"):
            continue
        opposite = "sell" if side == "buy" else "buy"
        if open_queues[f"{token}|{opposite}"]:
            entry_ts = open_queues[f"{token}|{opposite}"].popleft()
            delta = (ts - entry_ts).total_seconds() / 60.0
            if delta >= 0:
                holds_min.append(delta)
                closed_pairs += 1
        else:
            open_queues[f"{token}|{side}"].append(ts)
    return holds_min, closed_pairs


def _top_items(counter: Dict[str, float], n: int) -> List[Tuple[str, float]]:
    return sorted(counter.items(), key=lambda x: x[1], reverse=True)[:n]


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze Polymarket wallet strategy from ingested data.")
    ap.add_argument("--wallet", required=True, help="Wallet address")
    ap.add_argument("--db", default=DEFAULT_DB, help=f"SQLite input path (default: {DEFAULT_DB})")
    ap.add_argument("--hours", type=float, default=0.0, help="Lookback window in hours (0=all)")
    ap.add_argument("--out", default="logs/analysis/polymarket", help="Output directory root")
    args = ap.parse_args()

    wallet = args.wallet.strip().lower()
    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"db not found: {db_path}")

    now = datetime.now(timezone.utc)
    since_iso = None
    if args.hours > 0:
        since = now.timestamp() - (args.hours * 3600.0)
        since_iso = datetime.fromtimestamp(since, tz=timezone.utc).isoformat()

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        trades = _load_trades(conn, wallet, since_iso)
        closed_positions = _load_closed_positions(conn, wallet, since_iso)
    finally:
        conn.close()

    mode = "trade_level"
    if not trades and not closed_positions:
        raise SystemExit(f"no trades/closed_positions for wallet={wallet} (hours={args.hours})")
    if not trades and closed_positions:
        mode = "closed_position_level"
        # Build pseudo-trades from closed positions so downstream metrics still run.
        pseudo: List[Dict[str, Any]] = []
        for cp in closed_positions:
            notional = float(cp["avg_entry"] or 0.0) * float(cp["size"] or 0.0)
            pseudo.append(
                {
                    "ts": cp["ts"],
                    "market_slug": cp["market_slug"],
                    "market_id": cp["market_id"],
                    "token_id": cp["token_id"],
                    "side": "buy",
                    "price": cp["avg_entry"],
                    "notional_usd": notional,
                }
            )
        # Convert dicts into row-like lookups.
        class _R(dict):
            def __getitem__(self, k: str) -> Any:
                return self.get(k)
        trades = [_R(p) for p in pseudo]

    # Core metrics
    n = len(trades)
    buy_n = sum(1 for r in trades if (r["side"] or "").lower() == "buy")
    sell_n = sum(1 for r in trades if (r["side"] or "").lower() == "sell")
    buy_sell_ratio = (buy_n / sell_n) if sell_n > 0 else float("inf")
    notionals = [float(r["notional_usd"] or 0.0) for r in trades]
    prices = [float(r["price"] or 0.0) for r in trades if float(r["price"] or 0.0) > 0.0]
    total_notional = sum(notionals)
    avg_notional = total_notional / n if n else 0.0
    p50_notional = _quantile(notionals, 0.50)
    p90_notional = _quantile(notionals, 0.90)
    p99_notional = _quantile(notionals, 0.99)
    price_center = (sum(prices) / len(prices)) if prices else 0.0

    # Timing metrics
    timestamps = [t for t in (_parse_ts(r["ts"]) for r in trades) if t is not None]
    intertrade_s: List[float] = []
    for i in range(1, len(timestamps)):
        intertrade_s.append(max(0.0, (timestamps[i] - timestamps[i - 1]).total_seconds()))
    median_gap_s = _quantile(intertrade_s, 0.50) if intertrade_s else 0.0

    # Hold-time approximation
    hold_mins, matched_round_trips = _build_hold_stats(trades)
    hold_p50 = _quantile(hold_mins, 0.50) if hold_mins else 0.0
    hold_p90 = _quantile(hold_mins, 0.90) if hold_mins else 0.0

    # Market concentration
    notional_by_market: Dict[str, float] = defaultdict(float)
    for r in trades:
        m = r["market_slug"] or r["market_id"] or "unknown_market"
        notional_by_market[m] += float(r["notional_usd"] or 0.0)
    top_markets = _top_items(notional_by_market, 20)
    top_share = (top_markets[0][1] / total_notional) if top_markets and total_notional > 0 else 0.0
    hhi = 0.0
    if total_notional > 0:
        for _, v in notional_by_market.items():
            p = v / total_notional
            hhi += p * p

    # Hour-of-day behavior
    hour_counts: Dict[int, int] = defaultdict(int)
    for ts in timestamps:
        hour_counts[ts.hour] += 1
    top_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:6]

    # Price bands
    price_bands = Counter()
    for p in prices:
        if p < 0.10:
            price_bands["<10c"] += 1
        elif p < 0.25:
            price_bands["10-25c"] += 1
        elif p < 0.40:
            price_bands["25-40c"] += 1
        elif p < 0.60:
            price_bands["40-60c"] += 1
        elif p < 0.80:
            price_bands["60-80c"] += 1
        else:
            price_bands[">=80c"] += 1

    inferred_style = _infer_style(
        median_hold_minutes=hold_p50,
        p90_notional=p90_notional,
        price_band_center=price_center,
        buy_sell_ratio=buy_sell_ratio if math.isfinite(buy_sell_ratio) else 10.0,
        median_gap_seconds=median_gap_s,
    )

    analysis: Dict[str, Any] = {
        "wallet": wallet,
        "generated_at_utc": now.isoformat(),
        "lookback_hours": args.hours,
        "analysis_mode": mode,
        "trades": {
            "count": n,
            "buy_count": buy_n,
            "sell_count": sell_n,
            "buy_sell_ratio": buy_sell_ratio if math.isfinite(buy_sell_ratio) else None,
            "total_notional_usd": round(total_notional, 4),
            "avg_notional_usd": round(avg_notional, 4),
            "p50_notional_usd": round(p50_notional, 4),
            "p90_notional_usd": round(p90_notional, 4),
            "p99_notional_usd": round(p99_notional, 4),
            "avg_price": round(price_center, 6),
        },
        "timing": {
            "median_intertrade_seconds": round(median_gap_s, 3),
            "top_hours_utc": [{"hour": h, "trades": c} for h, c in top_hours],
            "matched_round_trips": matched_round_trips,
            "hold_time_p50_minutes": round(hold_p50, 4),
            "hold_time_p90_minutes": round(hold_p90, 4),
        },
        "concentration": {
            "market_hhi_notional": round(hhi, 8),
            "top_market_share_notional": round(top_share, 6),
            "top_markets_by_notional": [{"market": m, "notional_usd": round(v, 4)} for m, v in top_markets],
        },
        "price_bands": dict(price_bands),
        "inferred_style": inferred_style,
    }

    lines = [
        f"wallet={wallet}",
        f"analysis_mode={mode}",
        f"lookback_hours={args.hours:g} trades={n} total_notional=${total_notional:.2f}",
        (
            f"buy/sell={buy_n}/{sell_n} ratio="
            f"{(buy_sell_ratio if math.isfinite(buy_sell_ratio) else float('nan')):.3f}"
        ),
        (
            f"sizing p50/p90/p99=${p50_notional:.2f}/${p90_notional:.2f}/${p99_notional:.2f} "
            f"avg=${avg_notional:.2f}"
        ),
        (
            f"timing median_gap={median_gap_s:.2f}s hold_p50/p90={hold_p50:.2f}/{hold_p90:.2f} min "
            f"matched_round_trips={matched_round_trips}"
        ),
        f"concentration top_share={_fmt_pct(top_share)} hhi={hhi:.6f}",
        f"inferred_style={inferred_style}",
        "",
        "top_hours_utc:",
    ]
    for h, c in top_hours:
        lines.append(f"  {h:02d}:00 -> {c} trades")
    lines.extend(["", "top_markets_by_notional:"])
    for m, v in top_markets[:10]:
        lines.append(f"  {m:60} ${v:,.2f}")
    lines.extend(["", "price_bands:"])
    for k, v in price_bands.most_common():
        lines.append(f"  {k:8} {v}")

    out_root = Path(args.out)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    out_dir = out_root / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    jpath = out_dir / f"wallet_{wallet}_analysis.json"
    tpath = out_dir / f"wallet_{wallet}_analysis.txt"
    jpath.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tpath.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"[analyze] wrote {jpath}")
    print(f"[analyze] wrote {tpath}")


if __name__ == "__main__":
    main()
