"""Compare options snapshot quality between policy primary and secondary providers.

Uses data_sources.options_snapshots_primary (Tastytrade) and
options_snapshots_secondary (Public when enabled). Does not use Alpaca options."""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.core.database import Database


def _date_to_ms(s: str, end_of_day: bool = False) -> int:
    dt = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    ms = int(dt.timestamp() * 1000)
    if end_of_day:
        ms += (24 * 60 * 60 * 1000) - 1
    return ms


def _safe_float(v: Any) -> float | None:
    try:
        f = float(v)
        if f <= 0:
            return None
        return f
    except (TypeError, ValueError):
        return None


async def _load(db: Database, symbol: str, start_ms: int, end_ms: int) -> List[Dict[str, Any]]:
    rows = await db.get_option_chain_snapshots(symbol=symbol, start_ms=start_ms, end_ms=end_ms)
    result: List[Dict[str, Any]] = []
    for row in rows:
        provider = str(row.get("provider") or "")
        if provider not in {"tastytrade", "public"}:
            continue
        recv_ts_ms = int(row.get("recv_ts_ms") or row.get("timestamp_ms") or 0)
        expiration_ms = int(row.get("expiration_ms") or 0)
        atm_iv = _safe_float(row.get("atm_iv"))
        result.append(
            {
                "provider": provider,
                "recv_ts_ms": recv_ts_ms,
                "expiration_ms": expiration_ms,
                "bucket_ms": recv_ts_ms // 60_000,
                "atm_iv": atm_iv,
            }
        )
    return result


def _pair(rows: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    by_key: Dict[Tuple[int, int], Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: {"tastytrade": [], "public": []})
    for row in rows:
        key = (row["expiration_ms"], row["bucket_ms"])
        by_key[key][row["provider"]].append(row)

    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for providers in by_key.values():
        t_rows = sorted(providers["tastytrade"], key=lambda r: r["recv_ts_ms"])
        p_rows = sorted(providers["public"], key=lambda r: r["recv_ts_ms"])
        if not t_rows or not p_rows:
            continue
        n = min(len(t_rows), len(p_rows))
        for i in range(n):
            pairs.append((t_rows[i], p_rows[i]))
    return pairs


def _summarize(pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]]) -> Dict[str, Any]:
    recv_diffs = [p[1]["recv_ts_ms"] - p[0]["recv_ts_ms"] for p in pairs]
    iv_abs_diffs = [
        abs(p[0]["atm_iv"] - p[1]["atm_iv"])
        for p in pairs
        if p[0]["atm_iv"] is not None and p[1]["atm_iv"] is not None
    ]
    public_earlier = sum(1 for d in recv_diffs if d < 0)
    tasty_earlier = sum(1 for d in recv_diffs if d > 0)

    return {
        "pair_count": len(pairs),
        "recv_diff_mean_ms": statistics.mean(recv_diffs) if recv_diffs else None,
        "recv_diff_median_ms": statistics.median(recv_diffs) if recv_diffs else None,
        "public_earlier_count": public_earlier,
        "tastytrade_earlier_count": tasty_earlier,
        "atm_iv_pair_count": len(iv_abs_diffs),
        "atm_iv_mae": statistics.mean(iv_abs_diffs) if iv_abs_diffs else None,
        "atm_iv_median_abs_diff": statistics.median(iv_abs_diffs) if iv_abs_diffs else None,
    }


def _recommend(summary: Dict[str, Any]) -> str:
    recv_mean = summary.get("recv_diff_mean_ms")
    iv_mae = summary.get("atm_iv_mae")
    if summary.get("pair_count", 0) < 20:
        return "Insufficient overlap; keep current primary and re-run after more collection."
    if recv_mean is not None and recv_mean < -1500 and (iv_mae is None or iv_mae < 0.05):
        return "Recommend Public as primary (earlier receipt with acceptable IV agreement)."
    if recv_mean is not None and recv_mean > 1500:
        return "Recommend Tastytrade as primary (earlier receipt)."
    return "No strong winner; keep current primary and retain the other as secondary."


async def main_async(args: argparse.Namespace) -> int:
    db = Database(args.db)
    await db.connect()
    try:
        start_ms = _date_to_ms(args.start)
        end_ms = _date_to_ms(args.end, end_of_day=True)
        rows = await _load(db, args.symbol, start_ms, end_ms)
        pairs = _pair(rows)
        summary = _summarize(pairs)

        print(f"Symbol: {args.symbol}")
        print(f"Window: {args.start} -> {args.end}")
        print(f"Rows loaded (tastytrade/public): {len(rows)}")
        for k, v in summary.items():
            print(f"{k}: {v}")
        print(f"recommendation: {_recommend(summary)}")
        return 0
    finally:
        await db.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Tastytrade and Public options snapshots")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--db", default="data/argus.db")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async(parse_args())))
