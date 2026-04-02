#!/usr/bin/env python3
# Created by Oliver Meihls

# Compare multiple Polymarket wallets on core behavior metrics.
#
# Usage:
# python scripts/polymarket_wallet_compare.py --wallet 0xaaa --wallet 0xbbb

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

DEFAULT_DB = "data/polymarket_wallets.db"


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    idx = int(round((len(arr) - 1) * q))
    return arr[max(0, min(len(arr) - 1, idx))]


def _wallet_stats(conn: sqlite3.Connection, wallet: str) -> Dict[str, Any]:
    rows = conn.execute(
        """
        SELECT ts, market_slug, market_id, side, price, notional_usd
        FROM poly_wallet_trades
        WHERE wallet = ?
        ORDER BY ts ASC
        """,
        (wallet,),
    ).fetchall()
    n = len(rows)
    if n == 0:
        return {"wallet": wallet, "trades": 0}
    notionals = [float(r["notional_usd"] or 0.0) for r in rows]
    buys = sum(1 for r in rows if (r["side"] or "").lower() == "buy")
    sells = sum(1 for r in rows if (r["side"] or "").lower() == "sell")
    market_notional = defaultdict(float)
    for r in rows:
        m = r["market_slug"] or r["market_id"] or "unknown"
        market_notional[m] += float(r["notional_usd"] or 0.0)
    total_notional = sum(notionals)
    top_share = (max(market_notional.values()) / total_notional) if total_notional > 0 and market_notional else 0.0
    hhi = 0.0
    if total_notional > 0:
        for v in market_notional.values():
            p = v / total_notional
            hhi += p * p
    return {
        "wallet": wallet,
        "trades": n,
        "total_notional_usd": round(total_notional, 4),
        "avg_notional_usd": round(total_notional / n, 4),
        "p90_notional_usd": round(_quantile(notionals, 0.90), 4),
        "buy_sell_ratio": round((buys / sells), 6) if sells > 0 else None,
        "market_hhi_notional": round(hhi, 8),
        "top_market_share_notional": round(top_share, 6),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare Polymarket wallet behavior metrics.")
    ap.add_argument("--wallet", action="append", required=True, help="Wallet address (repeatable).")
    ap.add_argument("--db", default=DEFAULT_DB, help=f"SQLite input path (default: {DEFAULT_DB})")
    ap.add_argument("--out", default="logs/analysis/polymarket", help="Output directory root")
    args = ap.parse_args()

    wallets = sorted({w.strip().lower() for w in args.wallet if w.strip()})
    conn = sqlite3.connect(str(args.db))
    conn.row_factory = sqlite3.Row
    try:
        rows = [_wallet_stats(conn, w) for w in wallets]
    finally:
        conn.close()

    rows = [r for r in rows if r.get("trades", 0) > 0]
    rows.sort(
        key=lambda r: (
            r["total_notional_usd"],
            -r["top_market_share_notional"],
        ),
        reverse=True,
    )

    out = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "wallets": rows,
    }
    lines = ["wallet comparison", ""]
    if not rows:
        lines.append("(no wallets with trade data)")
    else:
        for r in rows:
            lines.append(
                f"{r['wallet']} trades={r['trades']} total_notional=${r['total_notional_usd']:.2f} "
                f"avg=${r['avg_notional_usd']:.2f} p90=${r['p90_notional_usd']:.2f} "
                f"top_share={r['top_market_share_notional']:.2%} hhi={r['market_hhi_notional']:.6f}"
            )

    out_root = Path(args.out)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = out_root / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    jpath = out_dir / "wallet_compare.json"
    tpath = out_dir / "wallet_compare.txt"
    jpath.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tpath.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"[compare] wrote {jpath}")
    print(f"[compare] wrote {tpath}")


if __name__ == "__main__":
    main()

