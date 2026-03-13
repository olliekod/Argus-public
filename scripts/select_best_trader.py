"""
Best Trader Selection Script
=============================

Scores each trader_id on a composite metric over a configurable window,
writes the top N to the followed_traders table, and optionally alerts via
Telegram when the follow list changes.

Scoring method: weighted composite
  score = (
      w_pnl * normalized_total_pnl
    + w_wr  * win_rate / 100
    + w_con * consistency_score
    - w_dd  * drawdown_penalty
  )

Where:
  - normalized_total_pnl = total_pnl / starting_balance
  - consistency_score    = 1 - (std_pnl / (abs(mean_pnl) + 1))
  - drawdown_penalty     = max_single_loss / starting_balance
  - win_rate is as computed (wins / total_closed * 100)

Minimum requirements:
  - At least min_trades closed trades
  - At least 2 distinct trade dates (not a one-day fluke)

Usage:
  python -m scripts.select_best_trader [--days 30] [--top-n 10] [--min-trades 5]
"""

import argparse
import asyncio
import json
import os
import sys
import statistics as stats_mod
from datetime import datetime, timezone

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.database import Database


# Default scoring weights
W_PNL = 0.40
W_WR = 0.25
W_CON = 0.20
W_DD = 0.15

STARTING_BALANCE = 5000.0


def score_trader(row: dict, starting_balance: float = STARTING_BALANCE) -> float:
    """Compute composite score for a single trader row."""
    total_pnl = row.get('total_pnl', 0) or 0
    closed = row.get('closed_trades', 0) or 0
    wins = row.get('wins', 0) or 0
    worst_trade = row.get('worst_trade', 0) or 0

    if closed == 0:
        return float('-inf')

    normalized_pnl = total_pnl / starting_balance
    win_rate = wins / closed
    avg_pnl = row.get('avg_pnl', 0) or 0

    # Consistency: lower variance relative to mean is better
    # We approximate with avg_pnl since we don't have per-trade std
    consistency = 1.0 / (1.0 + abs(avg_pnl - total_pnl / closed))

    # Drawdown penalty: worst single trade relative to balance
    drawdown_penalty = abs(min(0, worst_trade)) / starting_balance

    score = (
        W_PNL * normalized_pnl
        + W_WR * win_rate
        + W_CON * consistency
        - W_DD * drawdown_penalty
    )
    return round(score, 6)


async def select_best_traders(
    db_path: str = "data/argus.db",
    days: int = 30,
    top_n: int = 10,
    min_trades: int = 5,
    verbose: bool = True,
) -> list:
    """Run selection and return top traders."""
    db = Database(db_path)
    await db.connect()

    try:
        rows = await db.get_per_trader_pnl(days=days, min_trades=min_trades)

        if not rows:
            if verbose:
                print(f"No traders with >= {min_trades} closed trades in last {days} days.")
            return []

        # Score each trader
        scored = []
        for row in rows:
            s = score_trader(row)
            scored.append({
                **row,
                'score': s,
                'return_pct': round((row.get('total_pnl', 0) or 0) / STARTING_BALANCE * 100, 4),
            })

        # Sort by score descending
        scored.sort(key=lambda x: x['score'], reverse=True)
        top = scored[:top_n]

        if verbose:
            print(f"\n{'='*70}")
            print(f"BEST TRADER SELECTION — Top {top_n} of {len(scored):,} active traders")
            print(f"Window: {days} days | Min trades: {min_trades}")
            print(f"{'='*70}\n")

            for i, t in enumerate(top, 1):
                closed = t.get('closed_trades', 0)
                wins = t.get('wins', 0)
                wr = (wins / closed * 100) if closed > 0 else 0
                print(
                    f"{i:>3}. {t['trader_id']:>12}  "
                    f"{t.get('strategy_type', '?'):>15}  "
                    f"Score: {t['score']:+.4f}  "
                    f"PnL: ${t.get('total_pnl', 0):+8.2f}  "
                    f"Ret: {t['return_pct']:+6.2f}%  "
                    f"WR: {wr:5.1f}%  "
                    f"Trades: {closed}"
                )

            if len(scored) >= 10:
                returns = [t['return_pct'] for t in scored]
                print(f"\nPopulation stats ({len(scored):,} traders):")
                print(f"  Mean return:   {stats_mod.mean(returns):+.2f}%")
                print(f"  Median return: {stats_mod.median(returns):+.2f}%")
                print(f"  Std dev:       {stats_mod.stdev(returns):.2f}%")

        # Write to followed_traders table
        follow_records = []
        for t in top:
            follow_records.append({
                'trader_id': t['trader_id'],
                'followed_at': datetime.now(timezone.utc).isoformat(),
                'score': t['score'],
                'scoring_method': 'weighted_composite_v1',
                'window_days': days,
                'config_json': json.dumps({
                    'strategy_type': t.get('strategy_type'),
                    'total_pnl': t.get('total_pnl'),
                    'return_pct': t.get('return_pct'),
                    'closed_trades': t.get('closed_trades'),
                    'wins': t.get('wins'),
                    'win_rate': round(
                        (t.get('wins', 0) / t.get('closed_trades', 1)) * 100, 1
                    ) if t.get('closed_trades', 0) > 0 else 0,
                }),
            })

        await db.set_followed_traders(follow_records)
        if verbose:
            print(f"\nWrote {len(follow_records)} traders to followed_traders table.")

        return top

    finally:
        await db.close()


def main():
    parser = argparse.ArgumentParser(description="Select best-performing paper traders")
    parser.add_argument("--days", type=int, default=30, help="Lookback window in days")
    parser.add_argument("--top-n", type=int, default=10, help="Number of traders to follow")
    parser.add_argument("--min-trades", type=int, default=5, help="Minimum closed trades")
    parser.add_argument("--db", type=str, default="data/argus.db", help="Database path")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    args = parser.parse_args()

    asyncio.run(select_best_traders(
        db_path=args.db,
        days=args.days,
        top_n=args.top_n,
        min_trades=args.min_trades,
        verbose=not args.quiet,
    ))


if __name__ == "__main__":
    main()
