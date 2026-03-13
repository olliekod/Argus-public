"""Verify terminal UI leaderboard numbers against paper_trades.jsonl.

The UI leaderboard is built from bus events (SettlementOutcome, FillEvent) in the
current session only. This script checks that the numbers are consistent with
the full paper_trades.jsonl (and can verify a subset like 'last N' for Balain).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

def main():
    path = Path(__file__).resolve().parent.parent / "logs" / "paper_trades.jsonl"
    if not path.exists():
        print(f"Not found: {path}", file=sys.stderr)
        sys.exit(1)

    fills = defaultdict(int)
    settlements = defaultdict(lambda: {"pnl": 0.0, "wins": 0, "losses": 0})
    balain_list = []  # all Balain settlements in order for subset check
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            bid = o.get("bot_id", "")
            if not bid:
                continue
            if o.get("type") == "paper_fill":
                fills[bid] += 1
            elif o.get("type") == "settlement":
                pnl = float(o.get("pnl_usd", 0))
                settlements[bid]["pnl"] += pnl
                if o.get("won"):
                    settlements[bid]["wins"] += 1
                else:
                    settlements[bid]["losses"] += 1
                if bid == "Balain":
                    balain_list.append({"pnl": pnl, "won": o.get("won")})

    # Full-file totals
    names = ["Balain", "Boriin", "Durrak", "Balorn", "Doridin", "Filidrin", "Dwalinor"]
    print("From paper_trades.jsonl (full file):")
    print(f"{'Bot':<12} {'PnL':>10} {'Wins':>5} {'Loss':>5} {'Trd':>4} {'Fills':>6} {'WR%':>6}")
    print("-" * 52)
    for name in names:
        s = settlements.get(name, {"pnl": 0, "wins": 0, "losses": 0})
        f = fills.get(name, 0)
        tot = s["wins"] + s["losses"]
        wr = (s["wins"] / tot * 100) if tot else 0
        print(f"{name:<12} {s['pnl']:>+10.2f} {s['wins']:>5} {s['losses']:>5} {tot:>4} {f:>6} {wr:>5.1f}%")

    # Terminal showed Balain: +308.80, 39 Trd, 64.1% WR. Check if last 39 match.
    if len(balain_list) >= 39:
        last39 = balain_list[-39:]
        sum39 = sum(x["pnl"] for x in last39)
        w39 = sum(1 for x in last39 if x["won"])
        print()
        print("Balain 'last 39' (matches UI if session had 39 outcomes):")
        print(f"  Sum PnL: {sum39:.2f}  Wins: {w39}  Losses: {39 - w39}  WR: {w39/39*100:.1f}%")
        if abs(sum39 - 308.80) < 0.02:
            print("  -> Matches terminal +308.80 and 64.1% WR.")
        else:
            print("  -> Terminal may show a different 39 (e.g. session subset).")


if __name__ == "__main__":
    main()
