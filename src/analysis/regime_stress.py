"""Regime subset stress testing for replay experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Type

from src.analysis.execution_model import ExecutionConfig, ExecutionModel
from src.analysis.replay_harness import (
    MarketDataSnapshot,
    ReplayConfig,
    ReplayHarness,
    ReplayStrategy,
)
from src.core.outcome_engine import BarData


@dataclass
class RegimeSubsetResult:
    regime_key: str
    bars: int
    pnl: float
    sharpe: float
    trades: int


def _pack_snapshots_to_objects(snapshot_dicts: List[Dict[str, Any]]) -> List[MarketDataSnapshot]:
    out: List[MarketDataSnapshot] = []
    for s in snapshot_dicts or []:
        recv_ts = s.get("recv_ts_ms")
        if recv_ts is None:
            recv_ts = s.get("timestamp_ms", 0)
        qj = s.get("quotes_json")
        if qj is not None and not isinstance(qj, str):
            qj = json.dumps(qj)
        out.append(
            MarketDataSnapshot(
                symbol=s.get("symbol", "SPY"),
                recv_ts_ms=recv_ts,
                underlying_price=float(s.get("underlying_price", 0.0)),
                atm_iv=s.get("atm_iv") if s.get("atm_iv") is not None else None,
                source=s.get("provider", ""),
                quotes_json=qj,
            )
        )
    return out


def map_bars_to_regime_keys(
    bars: List[BarData],
    regimes: List[Dict[str, Any]],
    include_session: bool = True,
) -> Dict[int, Set[str]]:
    """Map each bar timestamp to active regime keys using latest-known regime state."""
    bars_sorted = sorted(bars, key=lambda b: b.timestamp_ms)
    regimes_sorted = sorted(regimes, key=lambda r: int(r.get("timestamp_ms", 0)))

    active_by_scope: Dict[str, Dict[str, Any]] = {}
    regime_idx = 0
    ts_to_keys: Dict[int, Set[str]] = {}

    for bar in bars_sorted:
        while regime_idx < len(regimes_sorted) and int(regimes_sorted[regime_idx].get("timestamp_ms", 0)) <= bar.timestamp_ms:
            regime = regimes_sorted[regime_idx]
            scope = str(regime.get("scope", "MARKET"))
            active_by_scope[scope] = regime
            regime_idx += 1

        keys: Set[str] = set()
        for scope, reg in active_by_scope.items():
            vol_regime = reg.get("vol_regime")
            trend_regime = reg.get("trend_regime")
            if vol_regime:
                keys.add(f"regime:{scope}:{vol_regime}")
            if trend_regime:
                keys.add(f"regime:{scope}_trend:{trend_regime}")
            if include_session and reg.get("session_regime"):
                keys.add(f"session:{reg.get('session_regime')}")

        ts_to_keys[bar.timestamp_ms] = keys

    return ts_to_keys


def _filter_dicts_by_time(records: List[Dict[str, Any]], start_ms: int, end_ms: int, ts_key: str = "timestamp_ms") -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rec in records:
        ts = rec.get(ts_key)
        if ts is None:
            continue
        ts_i = int(ts)
        if start_ms <= ts_i <= end_ms:
            out.append(rec)
    return out


def run_regime_subset_stress(
    *,
    bars: List[BarData],
    outcomes: List[Dict[str, Any]],
    regimes: List[Dict[str, Any]],
    snapshots: List[Dict[str, Any]],
    strategy_class: Type[ReplayStrategy],
    strategy_params: Dict[str, Any],
    starting_cash: float,
    execution_config: Optional[ExecutionConfig] = None,
    selected_regime_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run one replay per regime subset and return stress metrics."""
    if not bars:
        return {"subsets": [], "fraction_profitable": 0.0, "stress_score": 0.0}

    ts_to_keys = map_bars_to_regime_keys(bars, regimes)
    all_keys: Set[str] = set()
    for keys in ts_to_keys.values():
        all_keys.update(keys)

    regime_keys = selected_regime_keys or sorted(all_keys)
    subset_results: List[RegimeSubsetResult] = []

    for regime_key in regime_keys:
        subset_bars = [b for b in bars if regime_key in ts_to_keys.get(b.timestamp_ms, set())]
        if not subset_bars:
            continue

        timestamps = [b.timestamp_ms for b in subset_bars]
        start_ms = min(timestamps)
        end_ms = max(timestamps)

        subset_outcomes = _filter_dicts_by_time(outcomes, start_ms, end_ms, ts_key="timestamp_ms")
        subset_regimes = _filter_dicts_by_time(regimes, start_ms, end_ms, ts_key="timestamp_ms")
        subset_snapshots_dict = _filter_dicts_by_time(snapshots, start_ms, end_ms, ts_key="recv_ts_ms")
        subset_snapshots = _pack_snapshots_to_objects(subset_snapshots_dict)

        harness = ReplayHarness(
            bars=subset_bars,
            outcomes=subset_outcomes,
            strategy=strategy_class(strategy_params),
            execution_model=ExecutionModel(execution_config),
            regimes=subset_regimes,
            snapshots=subset_snapshots,
            config=ReplayConfig(starting_cash=starting_cash),
        )
        result = harness.run()
        ps = result.portfolio_summary
        subset_results.append(
            RegimeSubsetResult(
                regime_key=regime_key,
                bars=len(subset_bars),
                pnl=float(ps.get("total_realized_pnl", 0.0)),
                sharpe=float(ps.get("sharpe_annualized_proxy", 0.0)),
                trades=int(ps.get("total_trades", 0)),
            )
        )

    profitable = [r for r in subset_results if r.pnl > 0]
    fraction_profitable = len(profitable) / len(subset_results) if subset_results else 0.0

    return {
        "subsets": [r.__dict__ for r in subset_results],
        "fraction_profitable": round(fraction_profitable, 4),
        "stress_score": round(fraction_profitable, 4),
    }
