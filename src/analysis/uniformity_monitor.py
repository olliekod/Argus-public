"""
Uniformity Monitor
==================

Detects when paper traders converge on identical decisions, distinguishing
bug-induced convergence from legitimate market crowding.

Key metrics:
- HHI (Herfindahl-Hirschman Index): Measures concentration of choices
  - HHI = sum(share_i^2) where share_i = count_i / total
  - HHI = 1.0 means all traders chose the same value (max concentration)
  - HHI = 1/N means perfectly uniform distribution across N categories
- Entropy: Information-theoretic measure of diversity
  - Higher entropy = more diverse choices
  - Zero entropy = all identical
- Modal percentage: What fraction chose the most popular value
"""

import logging
import math
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from ..core.logger import get_logger
    logger = get_logger('uniformity_monitor')
except (ImportError, ValueError):
    logger = logging.getLogger('uniformity_monitor')

# Alert thresholds
# HHI > 0.5 means one choice dominates > 70% of the time
HHI_ALERT_THRESHOLD = 0.5
# Modal % > 80% means bug-like convergence (real markets rarely have this)
MODAL_PCT_ALERT_THRESHOLD = 0.80
# Minimum sample size to evaluate (avoid false alerts with small samples)
MIN_SAMPLE_SIZE = 20


def compute_hhi(values: List[Any]) -> float:
    """Compute Herfindahl-Hirschman Index for a list of values."""
    if not values:
        return 0.0
    counts = Counter(values)
    total = len(values)
    return sum((c / total) ** 2 for c in counts.values())


def compute_entropy(values: List[Any]) -> float:
    """Compute Shannon entropy for a list of values."""
    if not values:
        return 0.0
    counts = Counter(values)
    total = len(values)
    entropy = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def analyze_variable(values: List[Any], variable_name: str) -> Dict[str, Any]:
    """Analyze a single variable's distribution across traders."""
    if len(values) < MIN_SAMPLE_SIZE:
        return {
            'variable_name': variable_name,
            'skipped': True,
            'reason': f'insufficient_data ({len(values)} < {MIN_SAMPLE_SIZE})',
        }

    counts = Counter(values)
    total = len(values)
    modal_value, modal_count = counts.most_common(1)[0]
    modal_pct = modal_count / total

    hhi = compute_hhi(values)
    entropy = compute_entropy(values)

    is_alert = False
    alert_reason = None

    if hhi > HHI_ALERT_THRESHOLD and modal_pct > MODAL_PCT_ALERT_THRESHOLD:
        is_alert = True
        alert_reason = (
            f"Bug-like convergence: HHI={hhi:.3f}, modal={modal_value} "
            f"at {modal_pct:.0%} ({modal_count}/{total})"
        )
    elif hhi > HHI_ALERT_THRESHOLD:
        is_alert = True
        alert_reason = f"High concentration: HHI={hhi:.3f} (threshold={HHI_ALERT_THRESHOLD})"

    return {
        'variable_name': variable_name,
        'unique_count': len(counts),
        'total_count': total,
        'modal_value': str(modal_value),
        'modal_pct': round(modal_pct, 4),
        'hhi': round(hhi, 6),
        'entropy': round(entropy, 4),
        'is_alert': is_alert,
        'alert_reason': alert_reason,
    }


async def run_uniformity_check(
    trades: List[Dict[str, Any]],
    db=None,
    strategy_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run uniformity check on recent trades.

    Args:
        trades: List of trade dicts with keys like 'strikes', 'expiry',
                'entry_credit', 'contracts', 'trader_id'
        db: Optional database for persisting results
        strategy_type: If set, only analyze trades of this strategy type

    Returns:
        List of analysis results, one per variable checked.
    """
    if strategy_type:
        trades = [t for t in trades if t.get('strategy_type') == strategy_type]

    if len(trades) < MIN_SAMPLE_SIZE:
        return []

    variables_to_check = {
        'strikes': [t.get('strikes') for t in trades],
        'expiry': [t.get('expiry') for t in trades],
        'entry_credit': [round(t.get('entry_credit', 0), 2) for t in trades],
        'contracts': [t.get('contracts') for t in trades],
    }

    results = []
    timestamp = datetime.now(timezone.utc).isoformat()

    for var_name, values in variables_to_check.items():
        # Filter out None/missing
        clean_values = [v for v in values if v is not None]
        if not clean_values:
            continue

        analysis = analyze_variable(clean_values, var_name)
        if analysis.get('skipped'):
            continue

        analysis['strategy_type'] = strategy_type
        analysis['timestamp'] = timestamp
        results.append(analysis)

        if analysis['is_alert']:
            logger.warning(
                f"UNIFORMITY ALERT [{strategy_type or 'all'}] "
                f"{var_name}: {analysis['alert_reason']}"
            )

    # Persist to DB if available
    if db and results:
        for r in results:
            try:
                await db.execute("""
                    INSERT INTO uniformity_snapshots
                    (timestamp, strategy_type, variable_name, unique_count,
                     total_count, modal_value, modal_pct, hhi, entropy,
                     is_alert, alert_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    r['timestamp'],
                    r.get('strategy_type'),
                    r['variable_name'],
                    r['unique_count'],
                    r['total_count'],
                    r['modal_value'],
                    r['modal_pct'],
                    r['hhi'],
                    r['entropy'],
                    1 if r['is_alert'] else 0,
                    r.get('alert_reason'),
                ))
                await db._connection.commit()
            except Exception as e:
                logger.error(f"Failed to persist uniformity snapshot: {e}")

    return results


def format_uniformity_report(results: List[Dict[str, Any]]) -> str:
    """Format uniformity results as a human-readable string."""
    if not results:
        return "No uniformity data available."

    alerts = [r for r in results if r.get('is_alert')]
    lines = []

    if alerts:
        lines.append(f"UNIFORMITY ALERTS ({len(alerts)}):")
        for a in alerts:
            lines.append(
                f"  {a['variable_name']}: HHI={a['hhi']:.3f}, "
                f"modal={a['modal_value']} ({a['modal_pct']:.0%}), "
                f"unique={a['unique_count']}/{a['total_count']}"
            )
            if a.get('alert_reason'):
                lines.append(f"    Reason: {a['alert_reason']}")
    else:
        lines.append("No uniformity alerts. Trader diversity looks healthy.")

    lines.append("")
    lines.append("Variable summary:")
    for r in results:
        if r.get('skipped'):
            continue
        status = "ALERT" if r.get('is_alert') else "ok"
        lines.append(
            f"  {r['variable_name']:>15}: "
            f"HHI={r['hhi']:.3f}  "
            f"entropy={r['entropy']:.2f}  "
            f"modal={r['modal_pct']:.0%}  "
            f"unique={r['unique_count']}  "
            f"[{status}]"
        )

    return "\n".join(lines)
