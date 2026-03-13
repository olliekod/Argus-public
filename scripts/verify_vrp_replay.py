"""Repeatable E2E replay verifier for strategies.

Builds a replay pack using the data-source policy (bars_primary, options_snapshots_primary),
runs the selected strategy, and prints core counts plus diagnostics.

Strategies:
  vrp              - Sell put spreads when VRP > 0, skip VOL_SPIKE (default)
  high_vol         - Sell premium when VOL_SPIKE or VOL_HIGH
  overnight_session - Long momentum at session transitions
  router           - Regime-conditional: picks vrp, high_vol, or overnight based on conditions

When --provider is omitted, uses bars_primary from config (e.g. alpaca).
Dates: use YYYY-MM-DD (e.g. 2026-02-13). If --start/--end omitted, defaults to today
in US Eastern time (so evening ET still uses the current calendar day).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path

# Add project root to import path for script execution
sys.path.append(str(Path(__file__).parent.parent))

from src.core.data_sources import get_data_source_policy
from src.analysis.experiment_runner import ExperimentConfig, ExperimentRunner
from src.strategies.vrp_credit_spread import VRPCreditSpreadStrategy
from src.strategies.high_vol_credit import HighVolCreditStrategy
from src.strategies.overnight_session import OvernightSessionStrategy
from src.strategies.regime_conditional import RegimeConditionalStrategy
from src.tools.replay_pack import create_replay_pack

_STRATEGY_REGISTRY = {
    "vrp": (VRPCreditSpreadStrategy, {"min_vrp": 0.05}),
    "high_vol": (HighVolCreditStrategy, {"min_iv": 0.12, "allowed_vol_regimes": ["VOL_SPIKE", "VOL_HIGH"]}),
    "overnight_session": (OvernightSessionStrategy, {"fwd_return_threshold": 0.005, "horizon_seconds": 14400}),
    "router": (RegimeConditionalStrategy, {}),
}


def _get_strategy_class_and_params(name: str) -> tuple:
    entry = _STRATEGY_REGISTRY.get(name)
    if not entry:
        valid = ", ".join(_STRATEGY_REGISTRY)
        raise ValueError(f"Unknown --strategy {name!r}. Valid: {valid}")
    cls, params = entry
    return cls, params


def _parse_date(s: str) -> str:
    """Parse date string to YYYY-MM-DD. Accepts YYYY-MM-DD, MM-DD-YYYY, DD-MM-YYYY."""
    s = s.strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s
    m = re.match(r"^(\d{1,2})[-/](\d{1,2})[-/](\d{4})$", s)
    if m:
        a, b, year = m.groups()
        ai, bi = int(a), int(b)
        if ai > 12:
            return f"{year}-{bi:02d}-{ai:02d}"  # DD-MM-YYYY
        if bi > 12:
            return f"{year}-{ai:02d}-{bi:02d}"  # MM-DD-YYYY
        return f"{year}-{ai:02d}-{bi:02d}"  # assume MM-DD-YYYY
    raise ValueError(f"Invalid date {s!r}. Use YYYY-MM-DD (e.g. 2026-02-13).")


def _iv_diagnostic(snapshots: list) -> dict:
    """Count how many snapshots have atm_iv or derivable IV from quotes; include recv_ts range."""
    with_atm_iv = 0
    with_quotes = 0
    with_quotes_usable = 0
    sample_ivs: list[float] = []
    recv_ts_all: list[int] = []
    recv_ts_with_iv: list[int] = []
    for s in snapshots:
        recv = s.get("recv_ts_ms") or s.get("timestamp_ms") or 0
        if recv:
            recv_ts_all.append(int(recv))
        atm_iv = s.get("atm_iv")
        if atm_iv is not None:
            try:
                v = float(atm_iv)
                if v > 0:
                    with_atm_iv += 1
                    if recv:
                        recv_ts_with_iv.append(int(recv))
                    if len(sample_ivs) < 5:
                        sample_ivs.append(v)
            except (TypeError, ValueError):
                pass
        qj = s.get("quotes_json")
        if qj and isinstance(qj, str) and qj.strip():
            with_quotes += 1
            try:
                data = json.loads(qj)
                puts = data.get("puts") or []
                underlying = s.get("underlying_price") or 0
                if puts and underlying and underlying > 0:
                    for p in puts:
                        if p.get("strike") is not None and p.get("bid") is not None and p.get("ask") is not None:
                            with_quotes_usable += 1
                            break
            except Exception:
                pass
        elif qj and isinstance(qj, dict):
            with_quotes += 1
            puts = qj.get("puts") or []
            if puts:
                with_quotes_usable += 1
    return {
        "with_atm_iv": with_atm_iv,
        "with_quotes_json": with_quotes,
        "with_quotes_usable": with_quotes_usable,
        "sample_ivs": sample_ivs,
        "recv_ts_min": min(recv_ts_all) if recv_ts_all else None,
        "recv_ts_max": max(recv_ts_all) if recv_ts_all else None,
        "recv_ts_with_iv_min": min(recv_ts_with_iv) if recv_ts_with_iv else None,
        "recv_ts_with_iv_max": max(recv_ts_with_iv) if recv_ts_with_iv else None,
    }


def _default_pack_path(symbol: str, start: str, end: str, provider: str) -> Path:
    safe_provider = (provider or "default").replace("/", "_")
    return Path("data/packs") / f"{symbol}_{start}_{end}_{safe_provider}.json"


def _reasons_for_zero_trades(pack: dict, expected_provider: str) -> list[str]:
    reasons: list[str] = []
    bars = pack.get("bars", [])
    outcomes = pack.get("outcomes", [])
    snapshots = pack.get("snapshots", [])

    if not bars:
        reasons.append("missing bars for the selected provider/date range")
    if not outcomes:
        reasons.append("missing outcomes (RV unavailable); run outcomes backfill for the same provider as bars")
    if not snapshots:
        reasons.append("missing option snapshots (IV unavailable)")

    iv_ready = 0
    for s in snapshots:
        atm_iv = s.get("atm_iv")
        try:
            if atm_iv is not None and float(atm_iv) > 0:
                iv_ready += 1
        except (TypeError, ValueError):
            continue
    if snapshots and iv_ready == 0:
        reasons.append("snapshots exist but none have atm_iv>0 (indicative feed or provider mismatch likely)")

    meta_provider = str(pack.get("metadata", {}).get("provider", ""))
    if meta_provider and expected_provider and meta_provider != expected_provider:
        reasons.append(f"provider mismatch: pack provider={meta_provider}, expected={expected_provider}")

    if not reasons:
        reasons.append("VRP threshold/regime gating too strict for this window (inspect strategy diagnostics)")

    return reasons


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build replay pack + run VRP replay smoke check (policy-driven when --provider omitted).",
        epilog="Dates: YYYY-MM-DD (e.g. 2026-02-13). Omit --start/--end to use today.",
    )
    parser.add_argument("--symbol", required=True, help="Symbol (e.g. SPY)")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD (default: today)")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--provider", default=None, help="Bars/outcomes provider (default: from data_sources.bars_primary)")
    parser.add_argument("--pack_out", default=None, help="Optional output JSON path for replay pack")
    parser.add_argument("--db", default="data/argus.db", help="Path to argus.db")
    parser.add_argument("--options-snapshot-fallback", action="store_true",
        help="Fill primary option snapshots with secondary when gap >= 3m")
    parser.add_argument("--include-secondary-options", action="store_true",
        help="Include both primary and secondary option snapshot providers in pack")
    parser.add_argument("--options-snapshot-gap-minutes", type=int, default=3,
        help="Gap threshold in minutes for --options-snapshot-fallback (default: 3)")
    parser.add_argument("--strategy", default="router",
        choices=list(_STRATEGY_REGISTRY),
        help="Strategy to run: router (auto by regime), vrp, high_vol, overnight_session (default: router)")
    args = parser.parse_args()

    # Use US Eastern date for "today" so pack date matches trading calendar (e.g. evening ET = still "today")
    today = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    if args.start is None and args.end is None:
        start_date = end_date = today
    elif args.start is not None and args.end is not None:
        try:
            start_date = _parse_date(args.start)
            end_date = _parse_date(args.end)
        except ValueError as e:
            print(f"Error: {e}")
            return 2
    else:
        print("Error: provide both --start and --end, or omit both to use today.")
        return 2

    policy = get_data_source_policy()
    provider = args.provider or policy.bars_provider

    pack_path = Path(args.pack_out) if args.pack_out else _default_pack_path(args.symbol, start_date, end_date, provider)
    pack = asyncio.run(
        create_replay_pack(
            symbol=args.symbol,
            start_date=start_date,
            end_date=end_date,
            output_path=str(pack_path),
            provider=provider,
            db_path=args.db,
            options_snapshot_fallback=args.options_snapshot_fallback,
            include_secondary_options=args.include_secondary_options,
            options_snapshot_gap_minutes=args.options_snapshot_gap_minutes,
        )
    )

    snapshots = pack.get("snapshots", [])
    iv_diag = _iv_diagnostic(snapshots)

    def _ts_iso(ms: int | None) -> str:
        if ms is None:
            return "N/A"
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    print("IV diagnostic (pack snapshots):")
    print(f"  with atm_iv > 0:     {iv_diag['with_atm_iv']} / {len(snapshots)}")
    print(f"  with quotes_json:    {iv_diag['with_quotes_json']} / {len(snapshots)}")
    print(f"  quotes usable (IV): {iv_diag['with_quotes_usable']} / {len(snapshots)}")
    if iv_diag["sample_ivs"]:
        print(f"  sample atm_iv:      {iv_diag['sample_ivs']}")
    else:
        print("  sample atm_iv:      (none — no provider atm_iv in pack; VRP needs IV from DB or derivable from quotes)")
    if iv_diag.get("recv_ts_min") is not None:
        print(f"  recv_ts range:      {_ts_iso(iv_diag['recv_ts_min'])} .. {_ts_iso(iv_diag['recv_ts_max'])}")
    if iv_diag.get("recv_ts_with_iv_min") is not None:
        print(f"  recv_ts (w/ IV):    {_ts_iso(iv_diag['recv_ts_with_iv_min'])} .. {_ts_iso(iv_diag['recv_ts_with_iv_max'])}")
        bars_meta = pack.get("metadata", {})
        start_date = bars_meta.get("start_date") or start_date
        end_date_meta = bars_meta.get("end_date") or end_date
        print(f"  pack bar range:     {start_date} .. {end_date_meta} (bars use timestamp_ms; replay only shows snapshots where recv_ts_ms <= bar time)")
    if snapshots and (iv_diag["with_atm_iv"] or iv_diag["with_quotes_usable"]) and iv_diag["with_atm_iv"] < len(snapshots):
        print("  Note: VRP uses the *most recent* snapshot (by recv_ts_ms) first. If the latest snapshots lack atm_iv, it may still report 'no IV' even when older snapshots have IV.")
    print()

    bars_count = len(pack.get("bars", []))
    outcomes_count = len(pack.get("outcomes", []))
    snapshots_count = len(snapshots)

    strategy_cls, strategy_params = _get_strategy_class_and_params(args.strategy)
    print(f"Strategy: {args.strategy} ({strategy_cls.__name__})")
    runner = ExperimentRunner(output_dir="logs/experiments")
    result = runner.run(
        ExperimentConfig(
            strategy_class=strategy_cls,
            strategy_params=strategy_params,
            replay_pack_paths=[str(pack_path)],
            starting_cash=10_000.0,
        )
    )
    trade_count = int(result.portfolio_summary.get("total_trades", 0))
    execution = result.execution_summary or {}
    fills_count = int(execution.get("fills", execution.get("fills_count", 0)))
    rejects_count = int(execution.get("rejects", execution.get("rejects_count", 0)))

    # Print router breakdown if using regime-conditional
    if args.strategy == "router" and hasattr(result, "strategy_state"):
        state = result.strategy_state or {}
        if "routed_counts" in state:
            print("Router breakdown:", state["routed_counts"])

    print(f"bars_count={bars_count}")
    print(f"outcomes_count={outcomes_count}")
    print(f"snapshots_count={snapshots_count}")
    print(f"fills_count={fills_count} (open intents filled)")
    print(f"trade_count={trade_count} (closed positions)")

    if fills_count == 0 and trade_count == 0:
        print("WARNING: no fills and no closed trades. Likely reasons:")
        for reason in _reasons_for_zero_trades(pack, provider):
            print(f"- {reason}")
        return 1
    if fills_count > 0 and trade_count == 0:
        print("Note: fills > 0 but trade_count=0 (VRP/HighVol open-only; no CLOSE intents).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
