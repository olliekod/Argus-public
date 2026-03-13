#!/usr/bin/env python3
"""
Argus Comprehensive Verification
================================

Single entry point to verify the full data and research pipeline using the
**data-source policy** (config/config.yaml data_sources). No hard-coded
Alpaca options: bars come from bars_primary (e.g. Alpaca); options snapshots
from options_snapshots_primary (Tastytrade) and options_snapshots_secondary (Public when enabled).

Modes:
  --quick   Core imports + DB existence (no data checks).
  --data    + Bars coverage, outcomes coverage, options snapshots (primary + secondary).
  --replay  + Replay pack build + VRP smoke experiment.
  --full    + Research loop dry-run.

Use --validate to exit with code 1 if any run step fails (for CI or scripts).

Usage:
  python scripts/verify_argus.py --quick
  python scripts/verify_argus.py --data --symbol SPY --days 5
  python scripts/verify_argus.py --full --validate
  python scripts/verify_argus.py --replay --symbol SPY --start 2026-02-11 --end 2026-02-11
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Project root
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# ─── Output helpers ─────────────────────────────────────────────────────

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"


def _ok(msg: str) -> None:
    print(f"  {GREEN}[OK]{RESET} {msg}")


def _fail(msg: str, detail: Optional[str] = None) -> None:
    print(f"  {RED}[FAIL]{RESET} {msg}")
    if detail:
        print(f"        {detail}")


def _warn(msg: str) -> None:
    print(f"  {YELLOW}[WARN]{RESET} {msg}")


def _info(msg: str) -> None:
    print(f"  {CYAN}[INFO]{RESET} {msg}")


# ─── Step 1: Core imports & config ─────────────────────────────────────

def step_core_imports() -> Dict[str, Any]:
    """Verify core modules and load config + data-source policy."""
    result: Dict[str, Any] = {"passed": True, "details": {}}
    try:
        from src.core.config import load_config, load_secrets
        from src.core.data_sources import get_data_source_policy, DataSourcePolicy
        from src.core.database import Database
        cfg = load_config()
        policy = get_data_source_policy(cfg)
        result["policy"] = {
            "bars_primary": policy.bars_primary,
            "options_snapshots_primary": policy.options_snapshots_primary,
            "options_snapshots_secondary": policy.options_snapshots_secondary,
        }
        result["details"]["config"] = "loaded"
        result["details"]["policy"] = "loaded"
        return result
    except Exception as e:
        result["passed"] = False
        result["error"] = str(e)
        return result


# ─── Step 2: Database ───────────────────────────────────────────────────

def step_database(db_path: str) -> Dict[str, Any]:
    """Check DB exists and has expected tables."""
    result: Dict[str, Any] = {"passed": False, "path": db_path}
    path = Path(db_path)
    if not path.exists():
        result["error"] = f"Database not found: {path}"
        return result
    result["exists"] = True
    result["size_kb"] = round(path.stat().st_size / 1024, 1)
    try:
        import sqlite3
        conn = sqlite3.connect(str(path))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [r[0] for r in cursor.fetchall()]
        conn.close()
        result["tables"] = tables
        expected = {"market_bars", "bar_outcomes", "option_chain_snapshots"}
        missing = expected - set(tables)
        if missing:
            result["warn"] = f"Missing tables: {missing}"
        result["passed"] = True
        return result
    except Exception as e:
        result["error"] = str(e)
        return result


# ─── Step 3: Bars coverage ──────────────────────────────────────────────

async def step_bars_coverage(
    db_path: str,
    policy: Any,
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> Dict[str, Any]:
    """Bars from bars_primary."""
    from src.core.database import Database
    result: Dict[str, Any] = {"passed": False, "provider": policy.bars_primary}
    db = Database(db_path)
    await db.connect()
    try:
        bars = await db.get_bars_for_outcome_computation(
            source=policy.bars_provider,
            symbol=symbol,
            bar_duration=60,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        result["bar_count"] = len(bars)
        result["passed"] = len(bars) > 0
        return result
    finally:
        await db.close()


# ─── Step 4: Outcomes coverage ──────────────────────────────────────────

async def step_outcomes_coverage(
    db_path: str,
    policy: Any,
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> Dict[str, Any]:
    """Outcomes from outcomes_from (bars_primary)."""
    from src.core.database import Database
    result: Dict[str, Any] = {"passed": False, "provider": policy.bars_provider}
    db = Database(db_path)
    await db.connect()
    try:
        outcomes = await db.get_bar_outcomes(
            provider=policy.bars_provider,
            symbol=symbol,
            bar_duration_seconds=60,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        ok_count = sum(1 for o in outcomes if o.get("status") == "OK")
        result["outcome_count"] = len(outcomes)
        result["ok_count"] = ok_count
        result["passed"] = len(outcomes) > 0
        return result
    finally:
        await db.close()


# ─── Step 5: Options snapshots (primary + secondary from policy) ────────

async def step_options_snapshots(
    db_path: str,
    policy: Any,
    config: Dict[str, Any],
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> Dict[str, Any]:
    """Options snapshot coverage: primary and (when enabled) secondary."""
    from src.core.database import Database
    providers = policy.snapshot_providers(include_secondary=True)
    # If public_options.enabled is false, do not require public
    public_enabled = (config.get("public_options") or {}).get("enabled", False)
    if not public_enabled and "public" in providers:
        providers = [p for p in providers if p != "public"]
    result: Dict[str, Any] = {"passed": False, "providers": {}}
    db = Database(db_path)
    await db.connect()
    try:
        raw = await db.get_option_chain_snapshots(
            symbol=symbol,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        for prov in providers:
            rows = [r for r in raw if (r.get("provider") or "").lower() == prov]
            total = len(rows)
            with_iv = sum(1 for r in rows if r.get("atm_iv") is not None and float(r.get("atm_iv") or 0) > 0)
            pct = (100.0 * with_iv / total) if total else 0.0
            result["providers"][prov] = {
                "count": total,
                "with_atm_iv": with_iv,
                "atm_iv_pct": round(pct, 1),
            }
        primary = policy.options_snapshots_primary
        primary_data = result["providers"].get(primary, {})
        result["primary_count"] = primary_data.get("count", 0)
        result["primary_iv_pct"] = primary_data.get("atm_iv_pct", 0)
        result["passed"] = result["primary_count"] > 0
        return result
    finally:
        await db.close()


# ─── Step 6: Replay pack build ─────────────────────────────────────────

async def step_replay_pack(
    db_path: str,
    policy: Any,
    symbol: str,
    start_date: str,
    end_date: str,
    pack_path: str,
) -> Dict[str, Any]:
    """Build replay pack using policy (bars_primary + options_snapshots_primary)."""
    from src.tools.replay_pack import create_replay_pack
    result: Dict[str, Any] = {"passed": False}
    try:
        pack = await create_replay_pack(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            output_path=pack_path,
            db_path=db_path,
            policy=policy,
        )
        bars = len(pack.get("bars", []))
        outcomes = len(pack.get("outcomes", []))
        snapshots = len(pack.get("snapshots", []))
        result["bar_count"] = bars
        result["outcome_count"] = outcomes
        result["snapshot_count"] = snapshots
        result["passed"] = bars > 0
        result["pack_path"] = pack_path
        return result
    except Exception as e:
        result["error"] = str(e)
        return result


# ─── Step 7: VRP smoke experiment ──────────────────────────────────────

def step_vrp_smoke(pack_path: str, output_dir: str) -> Dict[str, Any]:
    """Run VRPCreditSpreadStrategy once on the pack."""
    from src.analysis.experiment_runner import ExperimentConfig, ExperimentRunner
    from src.strategies.vrp_credit_spread import VRPCreditSpreadStrategy
    result: Dict[str, Any] = {"passed": True}
    try:
        runner = ExperimentRunner(output_dir=output_dir)
        res = runner.run(
            ExperimentConfig(
                strategy_class=VRPCreditSpreadStrategy,
                strategy_params={"min_vrp": 0.05},
                replay_pack_paths=[pack_path],
                starting_cash=10_000.0,
            )
        )
        trades = int(res.portfolio_summary.get("total_trades", 0))
        result["trades"] = trades
        result["bars_replayed"] = getattr(res, "bars_replayed", None)
        result["zero_trades"] = trades == 0
        # Smoke "passes" if the run completed; zero trades is informational
        return result
    except Exception as e:
        result["passed"] = False
        result["error"] = str(e)
        return result


# ─── Step 8: Research loop dry-run ─────────────────────────────────────

def step_research_loop_dry_run(config_path: str) -> Dict[str, Any]:
    """Validate research loop config (dry-run)."""
    result: Dict[str, Any] = {"passed": False}
    try:
        from src.analysis.research_loop_config import load_research_loop_config
        load_research_loop_config(config_path)
        result["passed"] = True
        result["config_path"] = config_path
        return result
    except Exception as e:
        result["error"] = str(e)
        return result


# ─── Orchestrator ───────────────────────────────────────────────────────

async def run_verification(
    mode: str,
    db_path: str = "data/argus.db",
    symbol: str = "SPY",
    days: Optional[int] = 5,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    research_config: str = "config/research_loop.yaml",
) -> Dict[str, Any]:
    """Run verification steps according to mode. Returns results dict."""
    today = datetime.now(timezone.utc).date()
    if start_date and end_date:
        start_d = start_date
        end_d = end_date
    else:
        lookback = days or 5
        end_d = today.isoformat()
        start_d = (today - timedelta(days=lookback)).isoformat()
    start_ms = int(datetime.strptime(start_d, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(datetime.strptime(end_d, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000) + (24 * 3600 * 1000) - 1

    results: Dict[str, Any] = {
        "mode": mode,
        "symbol": symbol,
        "start_date": start_d,
        "end_date": end_d,
        "db_path": db_path,
        "steps": {},
        "all_passed": True,
    }

    # Step 1: Core
    print("\n[1] Core imports & data-source policy")
    print("-" * 50)
    r1 = step_core_imports()
    results["steps"]["core"] = r1
    if not r1.get("passed"):
        _fail("Core imports/config", r1.get("error"))
        results["all_passed"] = False
        return results
    _ok("Config and data-source policy loaded")
    policy = None
    try:
        from src.core.config import load_config
        from src.core.data_sources import get_data_source_policy
        policy = get_data_source_policy(load_config())
    except Exception:
        pass
    assert policy is not None

    # Step 2: Database
    print("\n[2] Database")
    print("-" * 50)
    r2 = step_database(db_path)
    results["steps"]["database"] = r2
    if not r2.get("passed"):
        _fail("Database", r2.get("error"))
        results["all_passed"] = False
        return results
    _ok(f"DB exists: {r2.get('path')} ({r2.get('size_kb')} KB)")
    if r2.get("warn"):
        _warn(r2["warn"])

    if mode == "quick":
        return results

    from src.core.config import load_config as _load_config
    try:
        config = _load_config()
    except Exception:
        config = {}

    # Step 3: Bars
    print("\n[3] Bars coverage (bars_primary)")
    print("-" * 50)
    r3 = await step_bars_coverage(db_path, policy, symbol, start_ms, end_ms)
    results["steps"]["bars"] = r3
    if r3.get("passed"):
        _ok(f"Bars from {r3['provider']}: {r3.get('bar_count', 0)} bars")
    else:
        _fail(f"No bars from {r3['provider']} for {symbol} in range")
        results["all_passed"] = False

    # Step 4: Outcomes
    print("\n[4] Outcomes coverage")
    print("-" * 50)
    r4 = await step_outcomes_coverage(db_path, policy, symbol, start_ms, end_ms)
    results["steps"]["outcomes"] = r4
    if r4.get("passed"):
        _ok(f"Outcomes: {r4.get('outcome_count', 0)} ({r4.get('ok_count', 0)} OK)")
    else:
        _fail(f"No outcomes for {symbol}; run: python -m src.outcomes backfill --provider {policy.bars_provider} --symbol {symbol} --bar 60 --start {start_d} --end {end_d}")
        results["all_passed"] = False

    # Step 5: Options snapshots
    print("\n[5] Options snapshots (primary + secondary from policy)")
    print("-" * 50)
    r5 = await step_options_snapshots(db_path, policy, config, symbol, start_ms, end_ms)
    results["steps"]["options_snapshots"] = r5
    if r5.get("passed"):
        for prov, data in r5.get("providers", {}).items():
            _ok(f"{prov}: {data.get('count', 0)} snapshots, {data.get('atm_iv_pct', 0)}% with atm_iv")
    else:
        _fail(f"No option snapshots from primary ({policy.options_snapshots_primary}) for {symbol}")
        results["all_passed"] = False

    if mode == "data":
        return results

    # Step 6: Replay pack
    pack_path = str(Path("data") / "packs" / f"{symbol}_{start_d}_{end_d}_verify.json")
    Path(pack_path).parent.mkdir(parents=True, exist_ok=True)
    print("\n[6] Replay pack build")
    print("-" * 50)
    r6 = await step_replay_pack(db_path, policy, symbol, start_d, end_d, pack_path)
    results["steps"]["replay_pack"] = r6
    if r6.get("passed"):
        _ok(f"Pack: {r6.get('bar_count')} bars, {r6.get('outcome_count')} outcomes, {r6.get('snapshot_count')} snapshots")
    else:
        _fail("Replay pack build", r6.get("error"))
        results["all_passed"] = False

    # Step 7: VRP smoke
    print("\n[7] VRP smoke experiment")
    print("-" * 50)
    if r6.get("passed") and Path(pack_path).exists():
        out_dir = str(Path("logs") / "verify_argus" / "experiments")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        r7 = step_vrp_smoke(pack_path, out_dir)
        results["steps"]["vrp_smoke"] = r7
        if r7.get("error"):
            _fail("VRP smoke", r7["error"])
            results["all_passed"] = False
        else:
            _ok(f"Trades: {r7.get('trades', 0)}, bars replayed: {r7.get('bars_replayed', 'N/A')}")
            if r7.get("zero_trades"):
                _warn("Zero trades (check bars/outcomes/snapshots/IV and strategy gating)")
    else:
        results["steps"]["vrp_smoke"] = {"skipped": True, "passed": False}

    if mode == "replay":
        return results

    # Step 8: Research loop dry-run
    print("\n[8] Research loop dry-run")
    print("-" * 50)
    r8 = step_research_loop_dry_run(research_config)
    results["steps"]["research_loop_dry_run"] = r8
    if r8.get("passed"):
        _ok(f"Config valid: {r8.get('config_path')}")
    else:
        _fail("Research loop config", r8.get("error"))
        results["all_passed"] = False

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Argus comprehensive verification (policy-driven: bars_primary, options primary + secondary).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Core imports + DB only",
    )
    parser.add_argument(
        "--data",
        action="store_true",
        help="+ Bars, outcomes, options snapshots",
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help="+ Replay pack + VRP smoke",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="+ Research loop dry-run",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Exit 1 if any step failed",
    )
    parser.add_argument("--db", default="data/argus.db", help="Database path")
    parser.add_argument("--symbol", default="SPY", help="Symbol for data/replay steps")
    parser.add_argument("--days", type=int, default=5, help="Lookback days when no --start/--end")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--research-config", default="config/research_loop.yaml", help="Research loop YAML")
    args = parser.parse_args()

    if not any([args.quick, args.data, args.replay, args.full]):
        parser.error("Specify one of --quick, --data, --replay, --full")

    if args.full:
        mode = "full"
    elif args.replay:
        mode = "replay"
    elif args.data:
        mode = "data"
    else:
        mode = "quick"

    print("=" * 60)
    print("ARGUS VERIFICATION")
    print("=" * 60)
    print(f"Mode: {mode}  |  DB: {args.db}  |  Symbol: {args.symbol}")

    results = asyncio.run(run_verification(
        mode=mode,
        db_path=args.db,
        symbol=args.symbol,
        days=args.days,
        start_date=args.start,
        end_date=args.end,
        research_config=args.research_config,
    ))

    print("\n" + "=" * 60)
    if results.get("all_passed"):
        print(f"{GREEN}Verification passed.{RESET}")
    else:
        print(f"{RED}Verification had failures.{RESET}")
    print("=" * 60)

    if args.validate and not results.get("all_passed"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
