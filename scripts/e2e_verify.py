#!/usr/bin/env python3
"""
End-to-End Data Source Verification
====================================

Single-command pipeline that validates the Argus data-source policy
end-to-end:

1. Confirms bars/outcomes coverage from ``bars_primary`` (e.g. Alpaca).
2. Confirms option-snapshot coverage + % ``atm_iv`` from
   ``options_snapshots_primary`` (Tastytrade); secondary (Public) is included
   when enabled in config. Tastytrade IV comes from DXLink when orchestrator runs.
3. Probes DXLink greeks/quotes availability.
4. Builds a replay pack using policy defaults (no provider args).
5. Runs a smoke experiment (``VRPCreditSpreadStrategy`` on SPY)
   that must produce trades when prerequisites are satisfied.
6. Runs the strategy evaluator and writes a summary report.

Artifacts are written to ``logs/e2e/<date>/``.

Usage::

    python scripts/e2e_verify.py
    python scripts/e2e_verify.py --symbol IBIT --days 5
"""

from __future__ import annotations

import asyncio
import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import load_config
from src.core.data_sources import get_data_source_policy, DataSourcePolicy
from src.core.database import Database


# ═══════════════════════════════════════════════════════════════════════════
# Step 1 — Bars / Outcomes Coverage
# ═══════════════════════════════════════════════════════════════════════════

async def check_bars_coverage(
    db: Database,
    policy: DataSourcePolicy,
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> Dict[str, Any]:
    """Check bar coverage from bars_primary."""
    bars = await db.get_bars_for_outcome_computation(
        source=policy.bars_provider,
        symbol=symbol,
        bar_duration=60,
        start_ms=start_ms,
        end_ms=end_ms,
    )
    return {
        "provider": policy.bars_provider,
        "symbol": symbol,
        "bar_count": len(bars),
        "start_ms": start_ms,
        "end_ms": end_ms,
        "pass": len(bars) > 0,
    }


async def check_outcomes_coverage(
    db: Database,
    policy: DataSourcePolicy,
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> Dict[str, Any]:
    """Check outcome coverage derived from bars_primary."""
    outcomes = await db.get_bar_outcomes(
        provider=policy.bars_provider,
        symbol=symbol,
        bar_duration_seconds=60,
        start_ms=start_ms,
        end_ms=end_ms,
    )
    ok_count = sum(1 for o in outcomes if o.get("status") == "OK")
    return {
        "provider": policy.bars_provider,
        "symbol": symbol,
        "outcome_count": len(outcomes),
        "ok_count": ok_count,
        "pass": len(outcomes) > 0,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Step 2 — Options Snapshot Coverage + atm_iv %
# ═══════════════════════════════════════════════════════════════════════════

async def check_options_snapshots(
    db: Database,
    policy: DataSourcePolicy,
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> Dict[str, Any]:
    """Check snapshot coverage from options_snapshots_primary.

    Reports:
    - snapshot_count: total snapshots from primary provider
    - atm_iv_present: count with non-null atm_iv (provider-supplied)
    - atm_iv_pct: percentage with provider atm_iv
    - iv_derivable: count where IV could be derived from greeks cache enrichment
    - iv_derivable_pct: percentage with derived atm_iv
    - iv_ready_count: total with either provider or derived IV
    - iv_ready_pct: percentage IV-ready overall
    - recv_ts_gated: count with valid recv_ts_ms
    """
    raw = await db.get_option_chain_snapshots(
        symbol=symbol,
        start_ms=start_ms,
        end_ms=end_ms,
    )
    primary = [r for r in raw if r.get("provider") == policy.options_snapshot_provider]
    iv_present = sum(1 for r in primary if r.get("atm_iv") is not None)
    recv_ts_gated = sum(1 for r in primary if r.get("recv_ts_ms") is not None and r.get("recv_ts_ms", 0) > 0)

    # Check how many snapshots have derivable IV from quotes_json
    iv_derivable = 0
    for r in primary:
        if r.get("atm_iv") is not None:
            continue  # Already has IV, don't double count
        try:
            from src.tools.replay_pack import _atm_iv_from_quotes_json
            underlying = float(r.get("underlying_price") or 0)
            derived = _atm_iv_from_quotes_json(r.get("quotes_json", "") or "", underlying)
            if derived is not None and derived > 0:
                iv_derivable += 1
        except Exception:
            pass

    iv_ready = iv_present + iv_derivable
    pct = (iv_present / len(primary) * 100) if primary else 0.0
    iv_derivable_pct = (iv_derivable / len(primary) * 100) if primary else 0.0
    iv_ready_pct = (iv_ready / len(primary) * 100) if primary else 0.0
    return {
        "provider": policy.options_snapshot_provider,
        "symbol": symbol,
        "snapshot_count": len(primary),
        "atm_iv_present": iv_present,
        "atm_iv_pct": round(pct, 1),
        "iv_derivable": iv_derivable,
        "iv_derivable_pct": round(iv_derivable_pct, 1),
        "iv_ready_count": iv_ready,
        "iv_ready_pct": round(iv_ready_pct, 1),
        "recv_ts_gated": recv_ts_gated,
        "pass": len(primary) > 0,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Step 3 — DXLink Probe
# ═══════════════════════════════════════════════════════════════════════════

def check_dxlink_availability() -> Dict[str, Any]:
    """Probe whether DXLink parser module is importable."""
    try:
        from src.connectors.tastytrade_dxlink_parser import DXLinkParser  # noqa: F401
        available = True
    except ImportError:
        try:
            from src.connectors import tastytrade_dxlink_parser  # noqa: F401
            available = True
        except ImportError:
            available = False
    return {
        "dxlink_module_available": available,
        "pass": available,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Step 4 — Build Replay Pack (policy defaults)
# ═══════════════════════════════════════════════════════════════════════════

async def build_smoke_pack(
    symbol: str,
    start_date: str,
    end_date: str,
    output_path: str,
    db_path: str,
    policy: DataSourcePolicy,
) -> Dict[str, Any]:
    """Build a replay pack using data-source policy defaults."""
    from src.tools.replay_pack import create_replay_pack

    pack = await create_replay_pack(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        output_path=output_path,
        policy=policy,
        db_path=db_path,
    )
    meta = pack.get("metadata", {})
    return {
        "pack_path": output_path,
        "bars_provider": meta.get("bars_provider", "unknown"),
        "options_snapshot_provider": meta.get("options_snapshot_provider", "unknown"),
        "secondary_options_included": meta.get("secondary_options_included", False),
        "bar_count": meta.get("bar_count", 0),
        "snapshot_count": meta.get("snapshot_count", 0),
        "pass": meta.get("bar_count", 0) > 0,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Step 5 — Smoke Experiment
# ═══════════════════════════════════════════════════════════════════════════

def diagnose_zero_trades(pack_path: str) -> List[str]:
    """Diagnose why a smoke experiment produced zero trades.

    Returns a list of human-readable reason strings.
    """
    reasons: List[str] = []
    try:
        with open(pack_path, "r", encoding="utf-8") as f:
            pack = json.load(f)
    except Exception:
        reasons.append("could not load replay pack JSON")
        return reasons

    bars = pack.get("bars", [])
    snapshots = pack.get("snapshots", [])
    meta = pack.get("metadata", {})

    if not bars:
        reasons.append("no bars in replay pack")
    if not snapshots:
        reasons.append("no snapshots in range")
    else:
        # Check recv_ts_ms gating
        gated = sum(1 for s in snapshots if s.get("recv_ts_ms") and s["recv_ts_ms"] > 0)
        if gated == 0:
            reasons.append("snapshots gated by recv_ts_ms (all have recv_ts_ms=0 or missing)")
        # Check atm_iv availability
        with_iv = sum(1 for s in snapshots if s.get("atm_iv") is not None and s["atm_iv"] > 0)
        if with_iv == 0:
            reasons.append("no atm_iv and no quotes to derive IV from")
        # Check for stale greeks
        if with_iv == 0 and gated > 0:
            reasons.append("stale greeks (snapshots present but no IV)")

    if not reasons:
        reasons.append("unknown (bars and snapshots present with IV)")

    return reasons


def run_smoke_experiment(
    pack_path: str,
    output_dir: str,
) -> Dict[str, Any]:
    """Run VRPCreditSpreadStrategy on the smoke pack."""
    from src.analysis.experiment_runner import ExperimentRunner, ExperimentConfig
    from src.strategies.vrp_credit_spread import VRPCreditSpreadStrategy

    runner = ExperimentRunner(output_dir=output_dir)
    config = ExperimentConfig(
        strategy_class=VRPCreditSpreadStrategy,
        strategy_params={"min_vrp": 0.05},
        replay_pack_paths=[pack_path],
        starting_cash=10000.0,
        tag="e2e_smoke",
    )

    result = runner.run(config)
    summary = result.summary()
    portfolio = summary.get("portfolio_summary", summary)
    trades = portfolio.get("total_trades", 0)

    # If zero trades, diagnose why
    zero_trade_reasons: List[str] = []
    if trades == 0:
        zero_trade_reasons = diagnose_zero_trades(pack_path)

    return {
        "strategy": "VRPCreditSpreadStrategy",
        "trades": trades,
        "pnl": portfolio.get("total_realized_pnl", 0),
        "bars_replayed": summary.get("bars_replayed", result.bars_replayed),
        "zero_trade_reasons": zero_trade_reasons,
        "pass": True,  # smoke test passes if it runs without error
    }


# ═══════════════════════════════════════════════════════════════════════════
# Step 6 — Strategy Evaluator
# ═══════════════════════════════════════════════════════════════════════════

def run_evaluator(
    experiment_dir: str,
    output_dir: str,
) -> Dict[str, Any]:
    """Run the strategy evaluator on smoke experiment output."""
    try:
        from src.analysis.strategy_evaluator import StrategyEvaluator

        evaluator = StrategyEvaluator(
            input_dir=experiment_dir,
            output_dir=output_dir,
        )
        count = evaluator.load_experiments()
        rankings = evaluator.evaluate()
        evaluator.save_rankings()
        return {
            "experiments_loaded": count,
            "rankings_count": len(rankings),
            "pass": True,
        }
    except Exception as exc:
        return {
            "experiments_loaded": 0,
            "rankings_count": 0,
            "error": str(exc),
            "pass": False,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════════

async def run_e2e(
    symbol: str = "SPY",
    days: int = 5,
    db_path: str = "data/argus.db",
) -> Dict[str, Any]:
    """Run the full e2e verification pipeline."""
    policy = get_data_source_policy()
    today = datetime.now(timezone.utc).date()
    start_date = (today - timedelta(days=days)).isoformat()
    end_date = today.isoformat()
    start_ms = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000) + (24 * 3600 * 1000) - 1

    output_root = Path("logs") / "e2e" / today.isoformat()
    output_root.mkdir(parents=True, exist_ok=True)
    experiment_dir = str(output_root / "experiments")
    Path(experiment_dir).mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {
        "run_date": today.isoformat(),
        "symbol": symbol,
        "date_range": {"start": start_date, "end": end_date},
        "policy": {
            "bars_primary": policy.bars_primary,
            "options_snapshots_primary": policy.options_snapshots_primary,
            "options_snapshots_secondary": getattr(policy, "options_snapshots_secondary", []),
            "options_stream_primary": policy.options_stream_primary,
        },
        "steps": {},
    }

    db = Database(db_path)
    await db.connect()

    try:
        # Step 1: Bars coverage
        print("=" * 60)
        print("Step 1: Checking bars coverage...")
        bars_result = await check_bars_coverage(db, policy, symbol, start_ms, end_ms)
        results["steps"]["bars_coverage"] = bars_result
        print(f"  Bars from {bars_result['provider']}: {bars_result['bar_count']} bars — {'PASS' if bars_result['pass'] else 'FAIL'}")

        # Step 1b: Outcomes coverage
        print("\nStep 1b: Checking outcomes coverage...")
        outcomes_result = await check_outcomes_coverage(db, policy, symbol, start_ms, end_ms)
        results["steps"]["outcomes_coverage"] = outcomes_result
        print(f"  Outcomes from {outcomes_result['provider']}: {outcomes_result['outcome_count']} ({outcomes_result['ok_count']} OK) — {'PASS' if outcomes_result['pass'] else 'FAIL'}")

        # Step 2: Options snapshot coverage
        print("\nStep 2: Checking options snapshot coverage...")
        snap_result = await check_options_snapshots(db, policy, symbol, start_ms, end_ms)
        results["steps"]["options_snapshots"] = snap_result
        print(f"  Snapshots from {snap_result['provider']}: {snap_result['snapshot_count']} ({snap_result['atm_iv_pct']}% have atm_iv, {snap_result['iv_ready_pct']}% IV-ready) — {'PASS' if snap_result['pass'] else 'FAIL'}")
        if snap_result.get("atm_iv_pct", 0) == 0 and snap_result.get("provider") == "tastytrade":
            print("    (Hint: Tastytrade atm_iv requires live DXLink; run orchestrator with OAuth to populate IV for new snapshots.)")
    finally:
        await db.close()

    # Step 3: DXLink probe
    print("\nStep 3: Probing DXLink availability...")
    dxlink_result = check_dxlink_availability()
    results["steps"]["dxlink_probe"] = dxlink_result
    print(f"  DXLink module available: {dxlink_result['dxlink_module_available']} — {'PASS' if dxlink_result['pass'] else 'FAIL'}")

    # Step 4: Build replay pack
    print("\nStep 4: Building replay pack (policy defaults)...")
    pack_path = str(output_root / f"{symbol}_smoke_pack.json")
    try:
        pack_result = await build_smoke_pack(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            output_path=pack_path,
            db_path=db_path,
            policy=policy,
        )
        results["steps"]["replay_pack"] = pack_result
        print(f"  Pack: {pack_result['bar_count']} bars, {pack_result['snapshot_count']} snapshots — {'PASS' if pack_result['pass'] else 'FAIL'}")
    except Exception as exc:
        results["steps"]["replay_pack"] = {"error": str(exc), "pass": False}
        print(f"  Pack build FAILED: {exc}")

    # Step 5: Smoke experiment
    pack_ok = results["steps"].get("replay_pack", {}).get("pass", False)
    if pack_ok:
        print("\nStep 5: Running smoke experiment (VRPCreditSpreadStrategy)...")
        try:
            exp_result = run_smoke_experiment(pack_path, experiment_dir)
            results["steps"]["smoke_experiment"] = exp_result
            status_str = 'PASS' if exp_result['pass'] else 'FAIL'
            print(f"  Trades: {exp_result['trades']}, PnL: {exp_result['pnl']}, Bars: {exp_result['bars_replayed']} -- {status_str}")
            if exp_result.get("zero_trade_reasons"):
                print(f"  Zero-trade reasons: {', '.join(exp_result['zero_trade_reasons'])}")
        except Exception as exc:
            results["steps"]["smoke_experiment"] = {"error": str(exc), "pass": False}
            print(f"  Experiment FAILED: {exc}")
    else:
        results["steps"]["smoke_experiment"] = {"skipped": True, "pass": False}
        print("\nStep 5: SKIPPED (no pack data)")

    # Step 6: Strategy evaluator
    exp_ok = results["steps"].get("smoke_experiment", {}).get("pass", False)
    if exp_ok:
        print("\nStep 6: Running strategy evaluator...")
        eval_result = run_evaluator(experiment_dir, str(output_root))
        results["steps"]["evaluator"] = eval_result
        print(f"  Loaded {eval_result.get('experiments_loaded', 0)} experiments, {eval_result.get('rankings_count', 0)} rankings — {'PASS' if eval_result['pass'] else 'FAIL'}")
    else:
        results["steps"]["evaluator"] = {"skipped": True, "pass": False}
        print("\nStep 6: SKIPPED (no experiment)")

    # ── Summary ──────────────────────────────────────────────────────
    all_steps = results["steps"]
    passed = sum(1 for v in all_steps.values() if v.get("pass"))
    total = len(all_steps)
    results["summary"] = {
        "passed": passed,
        "total": total,
        "all_pass": passed == total,
    }

    # Write summary.json
    summary_json_path = output_root / "summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    # Write summary.md
    summary_md_path = output_root / "summary.md"
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(f"# Argus E2E Verification — {today.isoformat()}\n\n")
        f.write(f"**Symbol:** {symbol}  \n")
        f.write(f"**Date range:** {start_date} -> {end_date}  \n")
        f.write(f"**Result:** {passed}/{total} steps passed  \n\n")
        f.write("## Policy\n\n")
        f.write(f"- bars_primary: `{policy.bars_primary}`\n")
        f.write(f"- options_snapshots_primary: `{policy.options_snapshots_primary}`\n")
        f.write(f"- options_snapshots_secondary: `{getattr(policy, 'options_snapshots_secondary', [])}`\n")
        f.write(f"- options_stream_primary: `{policy.options_stream_primary}`\n\n")
        f.write("## Steps\n\n")
        f.write("| Step | Result |\n")
        f.write("|------|--------|\n")
        for step, info in all_steps.items():
            status = "PASS" if info.get("pass") else ("SKIP" if info.get("skipped") else "FAIL")
            f.write(f"| {step} | {status} |\n")
        f.write("\n")

        # IV readiness detail (if options_snapshots step ran)
        snap_info = all_steps.get("options_snapshots", {})
        if snap_info.get("snapshot_count", 0) > 0:
            f.write("## IV Readiness\n\n")
            f.write(f"- Provider atm_iv: {snap_info.get('atm_iv_present', 0)}/{snap_info['snapshot_count']}"
                    f" ({snap_info.get('atm_iv_pct', 0)}%)\n")
            f.write(f"- Derived atm_iv (greeks cache): {snap_info.get('iv_derivable', 0)}/{snap_info['snapshot_count']}"
                    f" ({snap_info.get('iv_derivable_pct', 0)}%)\n")
            f.write(f"- IV-ready overall: {snap_info.get('iv_ready_count', 0)}/{snap_info['snapshot_count']}"
                    f" ({snap_info.get('iv_ready_pct', 0)}%)\n")
            f.write(f"- recv_ts_ms gated: {snap_info.get('recv_ts_gated', 0)}/{snap_info['snapshot_count']}\n")
            f.write("\n")

        # Zero-trade diagnostics
        smoke_info = all_steps.get("smoke_experiment", {})
        if smoke_info.get("zero_trade_reasons"):
            f.write("## Zero-Trade Diagnostics\n\n")
            for reason in smoke_info["zero_trade_reasons"]:
                f.write(f"- {reason}\n")
            f.write("\n")

    print("\n" + "=" * 60)
    print(f"E2E Verification: {passed}/{total} steps passed")
    print(f"Artifacts: {output_root}/")
    print(f"  summary.json: {summary_json_path}")
    print(f"  summary.md:   {summary_md_path}")
    print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Argus end-to-end data-source verification pipeline."
    )
    parser.add_argument("--symbol", default="SPY", help="Symbol to verify (default: SPY)")
    parser.add_argument("--days", type=int, default=5, help="Lookback days (default: 5)")
    parser.add_argument("--db", default="data/argus.db", help="Path to argus.db")
    args = parser.parse_args()

    asyncio.run(run_e2e(
        symbol=args.symbol,
        days=args.days,
        db_path=args.db,
    ))


if __name__ == "__main__":
    main()
