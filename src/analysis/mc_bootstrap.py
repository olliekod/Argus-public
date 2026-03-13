"""Monte Carlo/bootstrap path stress on realized trades.

Phase 4C: resample realized trade PnLs from a single replay run to
estimate path sensitivity, ruin risk, and downside behavior.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Sequence


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0:
        return float(min(values))
    if q >= 1:
        return float(max(values))

    s = sorted(float(v) for v in values)
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return s[lo]
    frac = pos - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def _max_drawdown_pct(equity_path: Sequence[float], starting_cash: float) -> float:
    if not equity_path or starting_cash <= 0:
        return 0.0
    peak = float(equity_path[0])
    max_dd_pct = 0.0
    for equity in equity_path:
        if equity > peak:
            peak = equity
        dd_pct = (peak - equity) / peak if peak > 0 else 0.0
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct
    return max_dd_pct


def _sample_iid(trade_pnls: Sequence[float], rng: random.Random) -> List[float]:
    n = len(trade_pnls)
    return [float(trade_pnls[rng.randrange(n)]) for _ in range(n)]


def _sample_block_bootstrap(
    trade_pnls: Sequence[float],
    rng: random.Random,
    block_size: Optional[int] = None,
) -> List[float]:
    n = len(trade_pnls)
    if n == 0:
        return []

    # Fixed-block bootstrap with circular wrap; default preserves short clustering.
    b = int(block_size) if block_size and block_size > 0 else max(2, int(round(math.sqrt(n))))
    b = min(b, n)

    sampled: List[float] = []
    while len(sampled) < n:
        start = rng.randrange(n)
        for k in range(b):
            sampled.append(float(trade_pnls[(start + k) % n]))
            if len(sampled) >= n:
                break
    return sampled


def run_mc_paths(
    trade_pnls: List[float],
    starting_cash: float,
    n_paths: int,
    method: str = "bootstrap",
    block_size: Optional[int] = None,
    random_seed: Optional[int] = None,
    ruin_level: float = 0.2,
) -> Dict[str, Any]:
    """Run MC/bootstrap path simulation on realized trade outcomes."""
    method_norm = (method or "bootstrap").strip().lower()
    if method_norm not in {"bootstrap", "iid"}:
        raise ValueError(f"Unsupported method: {method}")

    n = len(trade_pnls)
    if n_paths <= 0:
        raise ValueError("n_paths must be > 0")

    rng = random.Random(random_seed)

    terminal_returns: List[float] = []
    max_drawdowns_pct: List[float] = []
    ruined_paths = 0
    positive_paths = 0

    ruin_threshold_equity = starting_cash * ruin_level

    for _ in range(n_paths):
        if n == 0:
            sampled = []
        elif method_norm == "iid":
            sampled = _sample_iid(trade_pnls, rng)
        else:
            sampled = _sample_block_bootstrap(trade_pnls, rng, block_size=block_size)

        equity = starting_cash
        path = [equity]
        ruined = equity <= ruin_threshold_equity
        for pnl in sampled:
            equity += pnl
            path.append(equity)
            if equity <= ruin_threshold_equity:
                ruined = True

        terminal_return = ((equity - starting_cash) / starting_cash) if starting_cash > 0 else 0.0
        terminal_returns.append(terminal_return)

        dd_pct = _max_drawdown_pct(path, starting_cash)
        max_drawdowns_pct.append(dd_pct)

        if ruined:
            ruined_paths += 1
        if terminal_return > 0:
            positive_paths += 1

    summary = {
        "method": method_norm,
        "n_paths": n_paths,
        "n_trades": n,
        "block_size": int(block_size) if (block_size and block_size > 0) else None,
        "ruin_level": ruin_level,
        "median_return": round(_percentile(terminal_returns, 0.5), 6),
        "p5_return": round(_percentile(terminal_returns, 0.05), 6),
        "p5_max_drawdown": round(_percentile(max_drawdowns_pct, 0.05), 6),
        "p95_max_drawdown": round(_percentile(max_drawdowns_pct, 0.95), 6),
        "ruin_probability": round(ruined_paths / n_paths, 6),
        "fraction_positive": round(positive_paths / n_paths, 6),
    }
    return summary


def evaluate_mc_kill(
    mc_summary: Dict[str, Any],
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Apply configurable MC kill rules to path summary metrics."""
    th = thresholds or {}
    checks = [
        ("mc_median_return", mc_summary.get("median_return"), th.get("mc_median_return_min"), "min"),
        ("mc_fraction_positive", mc_summary.get("fraction_positive"), th.get("mc_fraction_positive_min"), "min"),
        ("mc_ruin_prob", mc_summary.get("ruin_probability"), th.get("mc_ruin_prob_max"), "max"),
        ("mc_p5_drawdown", mc_summary.get("p95_max_drawdown"), th.get("mc_p5_drawdown_max"), "max"),
    ]

    reasons: List[Dict[str, Any]] = []
    for reason, value, threshold, mode in checks:
        if threshold is None or value is None:
            continue
        v = float(value)
        t = float(threshold)
        breached = (v < t) if mode == "min" else (v > t)
        if breached:
            reasons.append({"reason": reason, "value": round(v, 6), "threshold": t})

    return {
        "killed": len(reasons) > 0,
        "reasons": reasons,
    }
