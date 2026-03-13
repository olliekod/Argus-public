"""
Phase 2 E2E Verification Tests
================================

Tests for Overnight Session Strategy — Phase 2 (Data Enhancement).

Covers:
- Task A: Replay pack contains global_risk_flow in regime metrics_json
- Task B: Strategy respects gate_on_risk_flow gating
- Task C: Full replay harness integration with risk flow + overnight strategy
- Deterministic injection behavior across multiple pack builds
- Edge cases: missing AV data, partial components, boundary values

These tests use mocked DB layers to avoid requiring a live database, but
exercise the full code path through replay_pack injection, strategy gating,
and replay harness integration.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass

import pytest

# Ensure repo root on sys.path so "python tests/test_phase2_e2e.py" and pytest both work
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.core.global_risk_flow import (
    ASIA_SYMBOLS,
    EUROPE_SYMBOLS,
    FX_RISK_SYMBOL,
    compute_global_risk_flow,
)
from src.core.outcome_engine import BarData, OutcomeResult
from src.strategies.overnight_session import OvernightSessionStrategy
from src.analysis.replay_harness import (
    ReplayHarness,
    ReplayConfig,
    ReplayResult,
    TradeIntent,
)
from src.analysis.execution_model import ExecutionModel, ExecutionConfig


# ═══════════════════════════════════════════════════════════════════════════
# Shared test data builders
# ═══════════════════════════════════════════════════════════════════════════

_DAY_MS = 86_400_000


def _make_av_bars(symbol: str, closes: List[float], start_day: int = 1):
    """Create daily bars for Alpha Vantage symbols."""
    return [
        {"timestamp_ms": (start_day + i) * _DAY_MS, "close": c, "symbol": symbol}
        for i, c in enumerate(closes)
    ]


def _build_full_av_bars(
    asia_ret: float = 0.01,
    europe_ret: float = 0.02,
    fx_ret: float = 0.005,
) -> Dict[str, List[Dict[str, Any]]]:
    """Build a complete set of AV bars for all risk-flow symbols.

    Uses synthetic close prices to produce the specified returns.
    """
    bars: Dict[str, List[Dict[str, Any]]] = {}

    base = 100.0
    for sym in ASIA_SYMBOLS:
        bars[sym] = _make_av_bars(sym, [base, base * (1 + asia_ret)])
    for sym in EUROPE_SYMBOLS:
        bars[sym] = _make_av_bars(sym, [base, base * (1 + europe_ret)])
    bars[FX_RISK_SYMBOL] = _make_av_bars(FX_RISK_SYMBOL, [base, base * (1 + fx_ret)])

    return bars


def _make_bar(ts_ms: int, close: float = 100.0, symbol: str = "SPY") -> BarData:
    bar = BarData(
        timestamp_ms=ts_ms,
        open=close - 0.10,
        high=close + 0.50,
        low=close - 0.50,
        close=close,
        volume=1000.0,
    )
    object.__setattr__(bar, "symbol", symbol)
    return bar


def _make_outcome_dict(
    ts_ms: int,
    horizon_seconds: int = 14400,
    fwd_return: float = 0.01,
    symbol: str = "SPY",
) -> Dict[str, Any]:
    """Create outcome dict as stored in replay packs."""
    return {
        "timestamp_ms": ts_ms,
        "symbol": symbol,
        "provider": "test",
        "bar_duration_seconds": 60,
        "horizon_seconds": horizon_seconds,
        "outcome_version": "TEST_V1",
        "close_now": 100.0,
        "close_at_horizon": 100.0 * (1 + fwd_return),
        "fwd_return": fwd_return,
        "max_runup": abs(fwd_return),
        "max_drawdown": 0.0,
        "realized_vol": 0.1,
        "max_high_in_window": None,
        "min_low_in_window": None,
        "max_runup_ts_ms": None,
        "max_drawdown_ts_ms": None,
        "time_to_max_runup_ms": None,
        "time_to_max_drawdown_ms": None,
        "status": "OK",
        "close_ref_ms": ts_ms,
        "window_start_ms": ts_ms,
        "window_end_ms": ts_ms + horizon_seconds * 1000,
        "bars_expected": horizon_seconds // 60,
        "bars_found": horizon_seconds // 60,
        "gap_count": 0,
        "computed_at_ms": ts_ms,
    }


def _make_outcome(
    ts_ms: int,
    horizon_seconds: int = 14400,
    fwd_return: float = 0.01,
    symbol: str = "SPY",
) -> OutcomeResult:
    return OutcomeResult(
        provider="test",
        symbol=symbol,
        bar_duration_seconds=60,
        timestamp_ms=ts_ms,
        horizon_seconds=horizon_seconds,
        outcome_version="TEST_V1",
        close_now=100.0,
        close_at_horizon=100.0 * (1 + fwd_return),
        fwd_return=fwd_return,
        max_runup=abs(fwd_return),
        max_drawdown=0.0,
        realized_vol=0.1,
        max_high_in_window=None,
        min_low_in_window=None,
        max_runup_ts_ms=None,
        max_drawdown_ts_ms=None,
        time_to_max_runup_ms=None,
        time_to_max_drawdown_ms=None,
        status="OK",
        close_ref_ms=ts_ms,
        window_start_ms=ts_ms,
        window_end_ms=ts_ms + horizon_seconds * 1000,
        bars_expected=horizon_seconds // 60,
        bars_found=horizon_seconds // 60,
        gap_count=0,
        computed_at_ms=ts_ms,
    )


def _make_regimes(
    risk_flow: Optional[float] = None,
) -> Dict[str, Dict[str, Any]]:
    metrics = {}
    if risk_flow is not None:
        metrics["global_risk_flow"] = risk_flow
    return {
        "EQUITIES": {
            "vol_regime": "VOL_NORMAL",
            "trend_regime": "TREND_UP",
            "metrics_json": json.dumps(metrics) if metrics else "",
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# Task A — Replay Pack Contains Risk Flow
# ═══════════════════════════════════════════════════════════════════════════


class TestReplayPackRiskFlowInjection:
    """E2E: Replay pack building injects global_risk_flow into regimes."""

    @pytest.mark.asyncio
    async def test_pack_injects_risk_flow_into_regimes(self, tmp_path):
        """Pack building should inject global_risk_flow into regime metrics_json."""
        from src.tools.replay_pack import create_replay_pack

        output_file = tmp_path / "pack.json"

        db = AsyncMock()
        # Return bars (not important for this test — focus is on regimes)
        db.get_bars_for_outcome_computation.return_value = []
        db.get_bar_outcomes.return_value = []
        db.get_bar_inventory.return_value = []
        db.get_outcome_inventory.return_value = []
        db.get_option_chain_snapshots.return_value = []

        # Regimes with no global_risk_flow yet
        # get_regimes is called twice: once for symbol scope, once for market scope
        # Use side_effect to return different regimes for each call
        db.get_regimes.side_effect = [
            # Symbol regimes (scope=SPY)
            [
                {
                    "scope": "SPY",
                    "timestamp": "2023-11-15T10:00:00Z",
                    "timestamp_ms": 1700042400000,
                    "metrics_json": '{"rsi": 55}',
                },
            ],
            # Market regimes (scope=EQUITIES)
            [
                {
                    "scope": "EQUITIES",
                    "timestamp": "2023-11-15T10:00:00Z",
                    "timestamp_ms": 1700042400000,
                    "metrics_json": '{"vol_ema": 0.15}',
                },
            ],
        ]

        # Alpha Vantage bars for risk-flow computation
        av_bars = _build_full_av_bars(asia_ret=0.01, europe_ret=0.02, fx_ret=0.005)
        db.get_bars_daily_for_risk_flow.return_value = av_bars

        with patch("src.tools.replay_pack.Database", return_value=db):
            with patch("src.tools.replay_pack.get_data_source_policy") as mock_policy:
                mock_policy.return_value.bars_provider = "alpaca"
                mock_policy.return_value.options_snapshot_provider = "tastytrade"

                pack = await create_replay_pack(
                    "SPY", "2023-11-15", "2023-11-15", str(output_file)
                )

        # Verify regimes have global_risk_flow injected
        assert len(pack["regimes"]) == 2
        for regime in pack["regimes"]:
            metrics = json.loads(regime["metrics_json"])
            assert "global_risk_flow" in metrics, (
                f"global_risk_flow missing from regime metrics: {metrics}"
            )
            # Value should be a number
            assert isinstance(metrics["global_risk_flow"], float)

        # Verify the injected value matches expected computation
        expected_flow = compute_global_risk_flow(
            av_bars, pack["regimes"][0].get("timestamp_ms", 1700042400000)
        )
        actual_flow = json.loads(pack["regimes"][0]["metrics_json"])["global_risk_flow"]
        assert abs(actual_flow - round(expected_flow, 8)) < 1e-10

    @pytest.mark.asyncio
    async def test_pack_preserves_existing_metrics(self, tmp_path):
        """Injection should preserve existing metrics in metrics_json."""
        from src.tools.replay_pack import create_replay_pack

        output_file = tmp_path / "pack_preserve.json"

        db = AsyncMock()
        db.get_bars_for_outcome_computation.return_value = []
        db.get_bar_outcomes.return_value = []
        db.get_bar_inventory.return_value = []
        db.get_outcome_inventory.return_value = []
        db.get_option_chain_snapshots.return_value = []

        db.get_regimes.return_value = [
            {
                "scope": "EQUITIES",
                "timestamp": "2023-11-15T10:00:00Z",
                "timestamp_ms": 1700042400000,
                "metrics_json": '{"rsi": 55, "vol_ema": 0.15}',
            },
        ]

        av_bars = _build_full_av_bars()
        db.get_bars_daily_for_risk_flow.return_value = av_bars

        with patch("src.tools.replay_pack.Database", return_value=db):
            with patch("src.tools.replay_pack.get_data_source_policy") as mock_policy:
                mock_policy.return_value.bars_provider = "alpaca"
                mock_policy.return_value.options_snapshot_provider = "tastytrade"

                pack = await create_replay_pack(
                    "SPY", "2023-11-15", "2023-11-15", str(output_file)
                )

        metrics = json.loads(pack["regimes"][0]["metrics_json"])
        # Original metrics preserved
        assert metrics["rsi"] == 55
        assert metrics["vol_ema"] == 0.15
        # Risk flow added
        assert "global_risk_flow" in metrics

    @pytest.mark.asyncio
    async def test_pack_injection_deterministic(self, tmp_path):
        """Two pack builds with identical data produce identical risk_flow values."""
        from src.tools.replay_pack import create_replay_pack

        av_bars = _build_full_av_bars(asia_ret=0.015, europe_ret=-0.005, fx_ret=0.003)

        packs = []
        for i in range(2):
            output_file = tmp_path / f"pack_{i}.json"

            db = AsyncMock()
            db.get_bars_for_outcome_computation.return_value = []
            db.get_bar_outcomes.return_value = []
            db.get_bar_inventory.return_value = []
            db.get_outcome_inventory.return_value = []
            db.get_option_chain_snapshots.return_value = []
            db.get_regimes.return_value = [
                {
                    "scope": "EQUITIES",
                    "timestamp": "2023-11-15T10:00:00Z",
                    "timestamp_ms": 1700042400000,
                    "metrics_json": '{"other": 1}',
                },
            ]
            db.get_bars_daily_for_risk_flow.return_value = av_bars

            with patch("src.tools.replay_pack.Database", return_value=db):
                with patch("src.tools.replay_pack.get_data_source_policy") as mock_policy:
                    mock_policy.return_value.bars_provider = "alpaca"
                    mock_policy.return_value.options_snapshot_provider = "tastytrade"

                    pack = await create_replay_pack(
                        "SPY", "2023-11-15", "2023-11-15", str(output_file)
                    )
                    packs.append(pack)

        # Exact string equality of metrics_json (deterministic serialization)
        m1 = packs[0]["regimes"][0]["metrics_json"]
        m2 = packs[1]["regimes"][0]["metrics_json"]
        assert m1 == m2, f"Non-deterministic injection: {m1} != {m2}"

    @pytest.mark.asyncio
    async def test_pack_no_av_data_no_injection(self, tmp_path):
        """When no Alpha Vantage bars exist, regimes should not have risk flow."""
        from src.tools.replay_pack import create_replay_pack

        output_file = tmp_path / "pack_no_av.json"

        db = AsyncMock()
        db.get_bars_for_outcome_computation.return_value = []
        db.get_bar_outcomes.return_value = []
        db.get_bar_inventory.return_value = []
        db.get_outcome_inventory.return_value = []
        db.get_option_chain_snapshots.return_value = []
        db.get_regimes.return_value = [
            {
                "scope": "EQUITIES",
                "timestamp": "2023-11-15T10:00:00Z",
                "timestamp_ms": 1700042400000,
                "metrics_json": '{"vol_ema": 0.15}',
            },
        ]
        # Empty AV bars
        db.get_bars_daily_for_risk_flow.return_value = {}

        with patch("src.tools.replay_pack.Database", return_value=db):
            with patch("src.tools.replay_pack.get_data_source_policy") as mock_policy:
                mock_policy.return_value.bars_provider = "alpaca"
                mock_policy.return_value.options_snapshot_provider = "tastytrade"

                pack = await create_replay_pack(
                    "SPY", "2023-11-15", "2023-11-15", str(output_file)
                )

        metrics = json.loads(pack["regimes"][0]["metrics_json"])
        assert "global_risk_flow" not in metrics
        # Original metric still present
        assert metrics["vol_ema"] == 0.15

    @pytest.mark.asyncio
    async def test_pack_sort_keys_for_determinism(self, tmp_path):
        """Verify metrics_json uses sort_keys for deterministic serialization."""
        from src.tools.replay_pack import create_replay_pack

        output_file = tmp_path / "pack_sort.json"

        db = AsyncMock()
        db.get_bars_for_outcome_computation.return_value = []
        db.get_bar_outcomes.return_value = []
        db.get_bar_inventory.return_value = []
        db.get_outcome_inventory.return_value = []
        db.get_option_chain_snapshots.return_value = []
        db.get_regimes.return_value = [
            {
                "scope": "EQUITIES",
                "timestamp": "2023-11-15T10:00:00Z",
                "timestamp_ms": 1700042400000,
                "metrics_json": '{"z_key": 99, "a_key": 1}',
            },
        ]
        av_bars = _build_full_av_bars()
        db.get_bars_daily_for_risk_flow.return_value = av_bars

        with patch("src.tools.replay_pack.Database", return_value=db):
            with patch("src.tools.replay_pack.get_data_source_policy") as mock_policy:
                mock_policy.return_value.bars_provider = "alpaca"
                mock_policy.return_value.options_snapshot_provider = "tastytrade"

                pack = await create_replay_pack(
                    "SPY", "2023-11-15", "2023-11-15", str(output_file)
                )

        raw = pack["regimes"][0]["metrics_json"]
        parsed = json.loads(raw)
        # Verify sort_keys: a_key < global_risk_flow < z_key
        keys = list(parsed.keys())
        assert keys == sorted(keys), f"Keys not sorted: {keys}"

    @pytest.mark.asyncio
    async def test_pack_partial_av_data_still_injects(self, tmp_path):
        """With only Asia data (no Europe, no FX), risk flow is still computed."""
        from src.tools.replay_pack import create_replay_pack

        output_file = tmp_path / "pack_partial.json"

        db = AsyncMock()
        db.get_bars_for_outcome_computation.return_value = []
        db.get_bar_outcomes.return_value = []
        db.get_bar_inventory.return_value = []
        db.get_outcome_inventory.return_value = []
        db.get_option_chain_snapshots.return_value = []
        db.get_regimes.return_value = [
            {
                "scope": "EQUITIES",
                "timestamp": "2023-11-15T10:00:00Z",
                "timestamp_ms": 1700042400000,
                "metrics_json": "{}",
            },
        ]
        # Only Asia bars (no Europe, no FX)
        partial_bars: Dict[str, List[Dict[str, Any]]] = {}
        for sym in ASIA_SYMBOLS:
            partial_bars[sym] = _make_av_bars(sym, [100.0, 102.0])
        db.get_bars_daily_for_risk_flow.return_value = partial_bars

        with patch("src.tools.replay_pack.Database", return_value=db):
            with patch("src.tools.replay_pack.get_data_source_policy") as mock_policy:
                mock_policy.return_value.bars_provider = "alpaca"
                mock_policy.return_value.options_snapshot_provider = "tastytrade"

                pack = await create_replay_pack(
                    "SPY", "2023-11-15", "2023-11-15", str(output_file)
                )

        metrics = json.loads(pack["regimes"][0]["metrics_json"])
        assert "global_risk_flow" in metrics
        # With only Asia at +2%, full weight on Asia: should be ~0.02
        assert abs(metrics["global_risk_flow"] - 0.02) < 0.001


# ═══════════════════════════════════════════════════════════════════════════
# Task B — Strategy Respects Gate
# ═══════════════════════════════════════════════════════════════════════════


class TestStrategyGateOnRiskFlow:
    """E2E: OvernightSessionStrategy gates entries based on risk flow."""

    def test_gate_suppresses_all_entries_when_risk_flow_below_threshold(self):
        """With gate_on_risk_flow=True and min above regime values, no entries."""
        strategy = OvernightSessionStrategy(params={
            "fwd_return_threshold": 0.001,
            "horizon_seconds": 14400,
            "gate_on_risk_flow": True,
            "min_global_risk_flow": 0.05,  # very high threshold
        })

        all_intents = []
        # Simulate 10 bars across CLOSED → PRE transition
        # Risk flow is -0.02 (below threshold of 0.05)
        regimes = _make_regimes(risk_flow=-0.02)

        # Bar 1: CLOSED (to set prev_session)
        t0 = 100_000
        bar0 = _make_bar(t0, symbol="SPY")
        strategy.on_bar(bar0, t0, "CLOSED", {})
        all_intents.extend(strategy.generate_intents(t0))

        # Bars 2-10: PRE bars with good outcomes — should all be gated
        for i in range(1, 10):
            ts = t0 + i * 60_000
            bar = _make_bar(ts, symbol="SPY")
            outcomes = {ts - 60_000: _make_outcome(ts - 60_000, 14400, 0.02)}
            strategy.on_bar(
                bar, ts, "PRE", outcomes, visible_regimes=regimes,
            )
            intents = strategy.generate_intents(ts)
            all_intents.extend(intents)

        # Verify: no OPEN intents (all gated by risk flow)
        open_intents = [i for i in all_intents if i.intent_type == "OPEN"]
        assert len(open_intents) == 0, (
            f"Expected 0 entries, got {len(open_intents)} — gate not working"
        )

        state = strategy.finalize()
        assert state["entries_emitted"] == 0

    def test_gate_allows_entries_when_risk_flow_above_threshold(self):
        """With gate_on_risk_flow=True and risk flow above threshold, entries occur."""
        strategy = OvernightSessionStrategy(params={
            "fwd_return_threshold": 0.001,
            "horizon_seconds": 14400,
            "gate_on_risk_flow": True,
            "min_global_risk_flow": -0.01,
        })

        regimes = _make_regimes(risk_flow=0.05)  # well above -0.01

        # CLOSED → PRE transition
        t0 = 100_000
        strategy.on_bar(_make_bar(t0, symbol="SPY"), t0, "CLOSED", {})
        strategy.generate_intents(t0)

        # PRE bar with good outcome
        t1 = t0 + 60_000
        outcomes = {t0: _make_outcome(t0, 14400, 0.02)}
        strategy.on_bar(
            _make_bar(t1, symbol="SPY"), t1, "PRE", outcomes,
            visible_regimes=regimes,
        )
        intents = strategy.generate_intents(t1)

        open_intents = [i for i in intents if i.intent_type == "OPEN"]
        assert len(open_intents) == 1

    def test_gate_exact_boundary_value(self):
        """Risk flow exactly at threshold should NOT be gated (>= semantics)."""
        strategy = OvernightSessionStrategy(params={
            "fwd_return_threshold": 0.001,
            "horizon_seconds": 14400,
            "gate_on_risk_flow": True,
            "min_global_risk_flow": -0.005,
        })

        # Risk flow exactly at threshold
        regimes = _make_regimes(risk_flow=-0.005)

        t0 = 100_000
        strategy.on_bar(_make_bar(t0, symbol="SPY"), t0, "CLOSED", {})
        strategy.generate_intents(t0)

        t1 = t0 + 60_000
        outcomes = {t0: _make_outcome(t0, 14400, 0.02)}
        strategy.on_bar(
            _make_bar(t1, symbol="SPY"), t1, "PRE", outcomes,
            visible_regimes=regimes,
        )
        intents = strategy.generate_intents(t1)

        # At boundary: risk_flow (-0.005) is NOT < min (-0.005), so entry should proceed
        open_intents = [i for i in intents if i.intent_type == "OPEN"]
        assert len(open_intents) == 1

    def test_gate_still_emits_close_intents(self):
        """Even when gated, existing positions should still be closed."""
        horizon_s = 3600
        strategy = OvernightSessionStrategy(params={
            "fwd_return_threshold": 0.001,
            "horizon_seconds": horizon_s,
            "gate_on_risk_flow": True,
            "min_global_risk_flow": -0.005,
        })

        # Phase 1: Enter with risk flow OK
        good_regimes = _make_regimes(risk_flow=0.05)

        t0 = 100_000
        strategy.on_bar(_make_bar(t0, symbol="SPY"), t0, "CLOSED", {})
        strategy.generate_intents(t0)

        t1 = t0 + 60_000
        outcomes = {t0: _make_outcome(t0, horizon_s, 0.01)}
        strategy.on_bar(
            _make_bar(t1, symbol="SPY"), t1, "PRE", outcomes,
            visible_regimes=good_regimes,
        )
        open_intents = strategy.generate_intents(t1)
        assert len([i for i in open_intents if i.intent_type == "OPEN"]) == 1

        # Phase 2: Risk flow drops below threshold — no new entries
        bad_regimes = _make_regimes(risk_flow=-0.1)

        # Advance past horizon to trigger close
        t2 = t1 + (horizon_s * 1000) + 1
        strategy.on_bar(
            _make_bar(t2, symbol="SPY"), t2, "RTH", {},
            visible_regimes=bad_regimes,
        )
        close_intents = strategy.generate_intents(t2)

        # Close intents should still fire (risk flow only gates entries)
        close_only = [i for i in close_intents if i.intent_type == "CLOSE"]
        assert len(close_only) == 1, "Close intent should still fire even when gated"

    def test_gate_with_multiple_risk_flow_transitions(self):
        """Strategy should respect risk flow changes across bars."""
        strategy = OvernightSessionStrategy(params={
            "fwd_return_threshold": 0.001,
            "horizon_seconds": 14400,
            "gate_on_risk_flow": True,
            "min_global_risk_flow": 0.0,  # require positive risk flow
        })

        entry_count = 0

        # Bar 1: CLOSED
        t = 100_000
        strategy.on_bar(_make_bar(t, symbol="SPY"), t, "CLOSED", {})
        strategy.generate_intents(t)

        # Bar 2: PRE with negative risk flow — gated
        t += 60_000
        outcomes = {t - 60_000: _make_outcome(t - 60_000, 14400, 0.02)}
        strategy.on_bar(
            _make_bar(t, symbol="SPY"), t, "PRE", outcomes,
            visible_regimes=_make_regimes(risk_flow=-0.01),
        )
        intents = strategy.generate_intents(t)
        entry_count += len([i for i in intents if i.intent_type == "OPEN"])

        # Bar 3: Still PRE, risk flow goes positive — should enter
        t += 60_000
        outcomes = {t - 60_000: _make_outcome(t - 60_000, 14400, 0.02)}
        strategy.on_bar(
            _make_bar(t, symbol="SPY"), t, "PRE", outcomes,
            visible_regimes=_make_regimes(risk_flow=0.02),
        )
        intents = strategy.generate_intents(t)
        entry_count += len([i for i in intents if i.intent_type == "OPEN"])

        # Bar 2 should have been gated, Bar 3 should have entered
        assert entry_count == 1


# ═══════════════════════════════════════════════════════════════════════════
# Task C — Full Replay Harness Integration
# ═══════════════════════════════════════════════════════════════════════════


class TestReplayHarnessIntegration:
    """E2E: Run OvernightSessionStrategy through ReplayHarness with regimes."""

    def test_harness_passes_regimes_to_strategy(self):
        """ReplayHarness feeds visible_regimes (with risk flow) to strategy."""
        # Build minimal replay data: CLOSED → PRE transition with outcomes
        # 2024-01-15 03:00 UTC = CLOSED, 04:30 UTC = PRE
        t_closed = 1705287600000  # 03:00 UTC
        t_pre = 1705293000000     # 04:30 UTC (PRE starts 09:00 UTC → 04:00 ET)

        # Use simpler timestamps for deterministic testing
        t0 = 100_000  # CLOSED
        t1 = t0 + 60_000  # still CLOSED
        # t2 and beyond are in PRE (we'll use the regime to drive this)

        bars = [
            _make_bar(t0, close=100.0, symbol="SPY"),
            _make_bar(t0 + 60_000, close=100.1, symbol="SPY"),
            _make_bar(t0 + 120_000, close=100.2, symbol="SPY"),
        ]

        # Outcomes with positive forward returns, visible after window_end
        # Make window_end early so they're visible during replay
        outcomes = [
            _make_outcome_dict(t0, 14400, 0.02, "SPY"),
        ]
        # Set window_end_ms to be before the bars we want the outcome visible for
        outcomes[0]["window_end_ms"] = t0 + 60_000  # visible at bar 2

        # Regimes with global_risk_flow
        regimes = [
            {
                "scope": "EQUITIES",
                "timestamp_ms": t0,
                "vol_regime": "VOL_NORMAL",
                "trend_regime": "TREND_UP",
                "metrics_json": json.dumps({"global_risk_flow": 0.015}),
            },
        ]

        strategy = OvernightSessionStrategy(params={
            "fwd_return_threshold": 0.001,
            "horizon_seconds": 14400,
            "gate_on_risk_flow": True,
            "min_global_risk_flow": -0.005,
        })

        exec_model = ExecutionModel()
        config = ReplayConfig(
            starting_cash=10_000.0,
            bar_duration_seconds=60,
            market="EQUITIES",
        )

        harness = ReplayHarness(
            bars=bars,
            outcomes=outcomes,
            strategy=strategy,
            execution_model=exec_model,
            config=config,
            regimes=regimes,
        )

        result = harness.run()

        # Verify the harness ran successfully
        assert result.bars_replayed == 3
        assert result.strategy_id == "OVERNIGHT_SESSION_V1"
        # Regimes should have been loaded
        assert result.regimes_loaded == 1

    def test_harness_gated_strategy_no_entries(self):
        """ReplayHarness with risk flow below threshold → no entries."""
        t0 = 100_000
        bars = [
            _make_bar(t0, close=100.0, symbol="SPY"),
            _make_bar(t0 + 60_000, close=100.1, symbol="SPY"),
            _make_bar(t0 + 120_000, close=100.2, symbol="SPY"),
        ]

        outcomes = [_make_outcome_dict(t0, 14400, 0.02, "SPY")]
        outcomes[0]["window_end_ms"] = t0 + 60_000

        # Risk flow very negative — should gate
        regimes = [
            {
                "scope": "EQUITIES",
                "timestamp_ms": t0,
                "vol_regime": "VOL_NORMAL",
                "trend_regime": "TREND_UP",
                "metrics_json": json.dumps({"global_risk_flow": -0.5}),
            },
        ]

        strategy = OvernightSessionStrategy(params={
            "fwd_return_threshold": 0.001,
            "horizon_seconds": 14400,
            "gate_on_risk_flow": True,
            "min_global_risk_flow": 0.0,  # require positive
        })

        exec_model = ExecutionModel()
        config = ReplayConfig(
            starting_cash=10_000.0,
            bar_duration_seconds=60,
            market="EQUITIES",
        )

        harness = ReplayHarness(
            bars=bars,
            outcomes=outcomes,
            strategy=strategy,
            execution_model=exec_model,
            config=config,
            regimes=regimes,
        )

        result = harness.run()
        state = result.strategy_state

        assert state["entries_emitted"] == 0, (
            f"Expected 0 entries (gated), got {state['entries_emitted']}"
        )

    def test_harness_deterministic_replay_with_regimes(self):
        """Two runs of same data produce identical results (with regimes)."""
        t0 = 100_000
        bars = [
            _make_bar(t0, close=100.0, symbol="SPY"),
            _make_bar(t0 + 60_000, close=100.1, symbol="SPY"),
        ]

        outcomes = [_make_outcome_dict(t0, 14400, 0.015, "SPY")]
        outcomes[0]["window_end_ms"] = t0 + 60_000

        regimes = [
            {
                "scope": "EQUITIES",
                "timestamp_ms": t0,
                "vol_regime": "VOL_NORMAL",
                "trend_regime": "TREND_UP",
                "metrics_json": json.dumps({"global_risk_flow": 0.01}),
            },
        ]

        params = {
            "fwd_return_threshold": 0.001,
            "horizon_seconds": 14400,
            "gate_on_risk_flow": True,
            "min_global_risk_flow": -0.005,
        }

        results = []
        for _ in range(2):
            strategy = OvernightSessionStrategy(params=params)
            exec_model = ExecutionModel()
            config = ReplayConfig(
                starting_cash=10_000.0,
                bar_duration_seconds=60,
                market="EQUITIES",
            )

            harness = ReplayHarness(
                bars=bars,
                outcomes=outcomes,
                strategy=strategy,
                execution_model=exec_model,
                config=config,
                regimes=regimes,
            )
            results.append(harness.run())

        assert results[0].strategy_state == results[1].strategy_state
        assert results[0].portfolio_summary["total_trades"] == results[1].portfolio_summary["total_trades"]


# ═══════════════════════════════════════════════════════════════════════════
# Global Risk Flow Computation Verification
# ═══════════════════════════════════════════════════════════════════════════


class TestGlobalRiskFlowIntegrity:
    """Verify risk flow computation matches expected formula end-to-end."""

    def test_full_computation_matches_formula(self):
        """GlobalRiskFlow = 0.4*Asia + 0.4*Europe + 0.2*FX."""
        asia_ret = 0.01
        europe_ret = 0.02
        fx_ret = 0.005

        bars = _build_full_av_bars(asia_ret, europe_ret, fx_ret)
        flow = compute_global_risk_flow(bars, 10 * _DAY_MS)

        expected = 0.4 * asia_ret + 0.4 * europe_ret + 0.2 * fx_ret
        assert flow is not None
        assert abs(flow - expected) < 1e-10

    def test_risk_flow_sign_semantics(self):
        """Positive = risk-on, negative = risk-off."""
        # All positive: risk-on
        bars_on = _build_full_av_bars(0.02, 0.03, 0.01)
        flow_on = compute_global_risk_flow(bars_on, 10 * _DAY_MS)
        assert flow_on is not None and flow_on > 0

        # All negative: risk-off
        bars_off = _build_full_av_bars(-0.02, -0.03, -0.01)
        flow_off = compute_global_risk_flow(bars_off, 10 * _DAY_MS)
        assert flow_off is not None and flow_off < 0

    def test_risk_flow_with_only_fx(self):
        """FX-only data should compute with full weight on FX."""
        bars = {}
        bars[FX_RISK_SYMBOL] = _make_av_bars(FX_RISK_SYMBOL, [100.0, 103.0])

        flow = compute_global_risk_flow(bars, 10 * _DAY_MS)
        assert flow is not None
        # Only FX: weight redistributes to 1.0 * 0.03
        assert abs(flow - 0.03) < 1e-10

    def test_risk_flow_lookahead_prevention(self):
        """Bars at exactly sim_time are excluded (strict less-than)."""
        bars = _build_full_av_bars(0.01, 0.02, 0.005)
        # sim_time exactly at the second bar's timestamp (day 2 = 2 * _DAY_MS)
        flow = compute_global_risk_flow(bars, 2 * _DAY_MS)
        # Only first bar visible (need 2 for return) → None
        assert flow is None


# ═══════════════════════════════════════════════════════════════════════════
# Research Loop Config Validation
# ═══════════════════════════════════════════════════════════════════════════


class TestResearchLoopConfig:
    """Verify research loop config includes overnight strategy with gating."""

    def test_overnight_strategy_in_research_loop_config(self):
        """OvernightSessionStrategy is registered in research_loop.yaml."""
        import yaml
        config_path = "config/research_loop.yaml"

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        strategies = config.get("strategies", [])
        overnight_entries = [
            s for s in strategies
            if s.get("strategy_class") == "OvernightSessionStrategy"
        ]
        assert len(overnight_entries) >= 1, (
            "OvernightSessionStrategy not found in research_loop.yaml"
        )

    def test_overnight_sweep_includes_risk_flow_gate(self):
        """overnight_sweep.yaml includes gate_on_risk_flow parameter."""
        import yaml
        sweep_path = "config/overnight_sweep.yaml"

        with open(sweep_path, "r") as f:
            sweep = yaml.safe_load(f)

        assert "gate_on_risk_flow" in sweep, (
            "gate_on_risk_flow missing from overnight_sweep.yaml"
        )
        values = sweep["gate_on_risk_flow"].get("values", [])
        assert True in values or "true" in values, (
            "gate_on_risk_flow sweep should include true"
        )
        assert False in values or "false" in values, (
            "gate_on_risk_flow sweep should include false"
        )

    def test_strategy_loadable_by_research_loop(self):
        """OvernightSessionStrategy can be loaded by the research loop loader."""
        import importlib
        modules = [
            "src.strategies.vrp_credit_spread",
            "src.strategies.dow_regime_timing",
            "src.strategies.overnight_session",
        ]
        found = False
        for mod_name in modules:
            try:
                mod = importlib.import_module(mod_name)
                if hasattr(mod, "OvernightSessionStrategy"):
                    found = True
                    break
            except ImportError:
                continue
        assert found, "OvernightSessionStrategy not loadable from strategy modules"


# ═══════════════════════════════════════════════════════════════════════════
# Strategy _extract_global_risk_flow Edge Cases
# ═══════════════════════════════════════════════════════════════════════════


class TestExtractGlobalRiskFlow:
    """Verify the strategy's extraction of risk flow from visible_regimes."""

    def test_extract_from_valid_regimes(self):
        """Normal extraction path."""
        regimes = _make_regimes(risk_flow=0.0123)
        val = OvernightSessionStrategy._extract_global_risk_flow(regimes)
        assert val == 0.0123

    def test_extract_none_when_no_regimes(self):
        val = OvernightSessionStrategy._extract_global_risk_flow(None)
        assert val is None

    def test_extract_none_when_empty_regimes(self):
        val = OvernightSessionStrategy._extract_global_risk_flow({})
        assert val is None

    def test_extract_none_when_no_equities(self):
        regimes = {"CRYPTO": {"metrics_json": '{"global_risk_flow": 0.01}'}}
        val = OvernightSessionStrategy._extract_global_risk_flow(regimes)
        assert val is None

    def test_extract_none_when_empty_metrics_json(self):
        regimes = {"EQUITIES": {"metrics_json": ""}}
        val = OvernightSessionStrategy._extract_global_risk_flow(regimes)
        assert val is None

    def test_extract_none_when_invalid_json(self):
        regimes = {"EQUITIES": {"metrics_json": "not json"}}
        val = OvernightSessionStrategy._extract_global_risk_flow(regimes)
        assert val is None

    def test_extract_none_when_key_missing(self):
        regimes = {"EQUITIES": {"metrics_json": '{"other": 1}'}}
        val = OvernightSessionStrategy._extract_global_risk_flow(regimes)
        assert val is None

    def test_extract_zero_risk_flow(self):
        """Zero is a valid risk flow value."""
        regimes = _make_regimes(risk_flow=0.0)
        val = OvernightSessionStrategy._extract_global_risk_flow(regimes)
        assert val == 0.0

    def test_extract_negative_risk_flow(self):
        regimes = _make_regimes(risk_flow=-0.05)
        val = OvernightSessionStrategy._extract_global_risk_flow(regimes)
        assert val == -0.05


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
