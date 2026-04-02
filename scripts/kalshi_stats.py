# Created by Oliver Meihls

# Kalshi paper trading analytics — reads logs/paper_trades.jsonl.
#
# Each line in the JSONL is either a "paper_fill" (order placed) or a
# "settlement" (contract resolved win/loss with full P&L).
#
# Usage
# -----
# python scripts/kalshi_stats.py                  # all-time
# python scripts/kalshi_stats.py --days 1         # last 24 hours
# python scripts/kalshi_stats.py --days 7         # last 7 days
# python scripts/kalshi_stats.py --weeks 4        # last 4 weeks
# python scripts/kalshi_stats.py --since 2026-03-01
# python scripts/kalshi_stats.py --initial-bankroll 5000
# python scripts/kalshi_stats.py --log path/to/other.jsonl

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

DEFAULT_LOG = "logs/paper_trades.jsonl"
DEFAULT_BANKROLL = 5000.0

SEP_WIDE  = "═" * 62
SEP_THIN  = "─" * 62


# ── helpers ───────────────────────────────────────────────────────────────


def _ts_str(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%m-%d %H:%M:%S")


def _load(path: str, since_ts: Optional[float]) -> tuple[list, list]:
    p = Path(path)
    if not p.exists():
        print(f"[error] Log not found: {path}", file=sys.stderr)
        print("  Run argus_kalshi for a while — fills and settlements are written here.", file=sys.stderr)
        sys.exit(1)

    fills, settlements = [], []
    for raw in p.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError:
            continue
        ts = rec.get("timestamp", 0.0)
        if since_ts and ts < since_ts:
            continue
        t = rec.get("type", "")
        if t == "paper_fill":
            fills.append(rec)
        elif t == "settlement":
            settlements.append(rec)

    return fills, settlements


def _pct(n: int, d: int) -> str:
    return f"{100*n/d:.1f}%" if d else "  n/a"


def _avg(vals: list) -> float:
    return sum(vals) / len(vals) if vals else 0.0


# ── report ────────────────────────────────────────────────────────────────


def report(fills: list, settlements: list, initial_bankroll: float) -> None:
    total   = len(settlements)
    wins    = [s for s in settlements if s.get("won")]
    losses  = [s for s in settlements if not s.get("won")]
    total_pnl   = sum(s.get("pnl_usd", 0.0) for s in settlements)
    running_bal = initial_bankroll + total_pnl

    win_pnls  = [s["pnl_usd"] for s in wins]
    loss_pnls = [s["pnl_usd"] for s in losses]

    print()
    print(SEP_WIDE)
    print("  ARGUS KALSHI — PAPER TRADING ANALYTICS")
    print(SEP_WIDE)
    print(f"  Fills (orders placed) : {len(fills):,}")
    print(f"  Settlements           : {total:,}")
    print(f"  Win / Loss            : {len(wins)} W  /  {len(losses)} L  ({_pct(len(wins), total)} WR)")
    print(f"  Total PnL             : ${total_pnl:>+10.4f}")
    print(f"  Running Balance       : ${running_bal:>10.4f}  (started ${initial_bankroll:,.2f})")
    if wins:
        print(f"  Avg win               : ${_avg(win_pnls):>+10.4f}")
        print(f"  Best trade            : ${max(win_pnls):>+10.4f}")
    if losses:
        print(f"  Avg loss              : ${_avg(loss_pnls):>+10.4f}")
        print(f"  Worst trade           : ${min(loss_pnls):>+10.4f}")
    if wins and losses:
        ratio = abs(_avg(win_pnls) / _avg(loss_pnls)) if _avg(loss_pnls) != 0 else float("inf")
        print(f"  Win/Loss ratio        : {ratio:.2f}x")
        # Kelly fraction: f* = WR - (1-WR)/ratio
        wr = len(wins) / total
        kelly = wr - (1 - wr) / ratio if ratio else 0
        print(f"  Kelly fraction        : {kelly:.3f}  (theoretical optimal bet size)")

    # ── By asset ──────────────────────────────────────────────────────────
    if total:
        print()
        print(SEP_THIN)
        print(f"  {'BY ASSET':<20} {'Trades':>6}  {'WR':>6}  {'PnL':>10}  {'Avg PnL':>9}")
        print(SEP_THIN)
        by_asset: dict = defaultdict(list)
        for s in settlements:
            by_asset[s.get("asset", "?")].append(s)
        for asset in sorted(by_asset):
            ss = by_asset[asset]
            pnl = sum(x["pnl_usd"] for x in ss)
            w   = sum(1 for x in ss if x.get("won"))
            print(f"  {asset:<20} {len(ss):>6}  {_pct(w, len(ss)):>6}  ${pnl:>+9.4f}  ${_avg([x['pnl_usd'] for x in ss]):>+8.4f}")

    # ── By side ───────────────────────────────────────────────────────────
    if total:
        print()
        print(SEP_THIN)
        print(f"  {'BY SIDE':<20} {'Trades':>6}  {'WR':>6}  {'PnL':>10}  {'Avg PnL':>9}")
        print(SEP_THIN)
        by_side: dict = defaultdict(list)
        for s in settlements:
            by_side[s.get("side", "?").upper()].append(s)
        for side in sorted(by_side):
            ss = by_side[side]
            pnl = sum(x["pnl_usd"] for x in ss)
            w   = sum(1 for x in ss if x.get("won"))
            print(f"  {side:<20} {len(ss):>6}  {_pct(w, len(ss)):>6}  ${pnl:>+9.4f}  ${_avg([x['pnl_usd'] for x in ss]):>+8.4f}")

    # ── By source ─────────────────────────────────────────────────────────
    sources = {s.get("source") or "strategy" for s in settlements}
    if len(sources) > 1:
        print()
        print(SEP_THIN)
        print(f"  {'BY SOURCE':<20} {'Trades':>6}  {'WR':>6}  {'PnL':>10}  {'Avg PnL':>9}")
        print(SEP_THIN)
        by_src: dict = defaultdict(list)
        for s in settlements:
            by_src[s.get("source") or "strategy"].append(s)
        for src in sorted(by_src):
            ss = by_src[src]
            pnl = sum(x["pnl_usd"] for x in ss)
            w   = sum(1 for x in ss if x.get("won"))
            print(f"  {src:<20} {len(ss):>6}  {_pct(w, len(ss)):>6}  ${pnl:>+9.4f}  ${_avg([x['pnl_usd'] for x in ss]):>+8.4f}")

    # ── Running balance curve (sampled every 5 settlements) ───────────────
    if total >= 5:
        print()
        print(SEP_THIN)
        print("  RUNNING BALANCE")
        print(SEP_THIN)
        running = initial_bankroll
        step = max(1, total // 20)   # ~20 data points max
        for i, s in enumerate(settlements):
            running += s.get("pnl_usd", 0.0)
            if i % step == 0 or i == total - 1:
                bar_filled = int((running - initial_bankroll * 0.8) / (initial_bankroll * 0.4) * 20)
                bar_filled = max(0, min(20, bar_filled))
                bar = "█" * bar_filled + "░" * (20 - bar_filled)
                ts = _ts_str(s.get("timestamp", 0.0))
                print(f"  {ts}  [{bar}]  ${running:>8.2f}")

    # ── Recent 30 settlements ─────────────────────────────────────────────
    n_recent = min(30, total)
    if n_recent:
        print()
        print(SEP_THIN)
        print(f"  RECENT {n_recent} SETTLEMENTS")
        print(SEP_THIN)
        print(f"  {'Time (UTC)':<18}  {'Ticker':<32}  {'Side':<4}  {'Entry':>5}  {'Avg':>7}  {'Strike':>9}  {'PnL':>8}  Result")
        print(SEP_THIN)
        running = initial_bankroll + sum(s.get("pnl_usd", 0.0) for s in settlements[:-n_recent])
        for s in settlements[-n_recent:]:
            pnl = s.get("pnl_usd", 0.0)
            running += pnl
            ts      = _ts_str(s.get("timestamp", 0.0))
            ticker  = s.get("market_ticker", "")[-30:]
            side    = s.get("side", "?").upper()
            entry   = s.get("entry_price_cents", 0.0)
            avg     = s.get("final_avg", 0.0)
            strike  = s.get("strike", 0.0)
            result  = "WIN " if s.get("won") else "LOSS"
            pnl_s   = f"${pnl:>+7.4f}"
            print(f"  {ts:<18}  {ticker:<32}  {side:<4}  {entry:>5.1f}¢  {avg:>7.1f}  {strike:>9,.0f}  {pnl_s}  {result}  bal=${running:.2f}")

    print()
    print(SEP_WIDE)
    print()


# ── entry point ───────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Analyze Argus Kalshi paper trades from logs/paper_trades.jsonl",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--log", default=DEFAULT_LOG, help="Path to paper_trades.jsonl")
    ap.add_argument("--days",  type=int, help="Include only the last N days")
    ap.add_argument("--weeks", type=int, help="Include only the last N weeks")
    ap.add_argument("--since", help="Include only records on/after YYYY-MM-DD (UTC)")
    ap.add_argument(
        "--initial-bankroll", type=float, default=DEFAULT_BANKROLL,
        metavar="USD",
        help=f"Starting capital for running-balance calculation (default ${DEFAULT_BANKROLL:,.0f})",
    )
    args = ap.parse_args()

    since_ts: Optional[float] = None
    label = "all-time"
    if args.days:
        since_ts = (datetime.now(timezone.utc) - timedelta(days=args.days)).timestamp()
        label = f"last {args.days} day(s)"
    elif args.weeks:
        since_ts = (datetime.now(timezone.utc) - timedelta(weeks=args.weeks)).timestamp()
        label = f"last {args.weeks} week(s)"
    elif args.since:
        since_ts = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc).timestamp()
        label = f"since {args.since}"

    print(f"\nLoading {args.log}  [{label}]")
    fills, settlements = _load(args.log, since_ts)
    report(fills, settlements, args.initial_bankroll)


if __name__ == "__main__":
    main()
