"""
Tests for OvernightSessionStrategy.

Covers:
- Constructor / defaults
- Entry window detection (equities RTH tail, PRE, crypto transitions)
- Forward-return signal gating
- Explicit close intents (horizon-based)
- Risk-flow gating
- Determinism (same inputs → same outputs)
"""

from __future__ import annotations

import json
import pytest
from typing import Any, Dict, List, Optional

from src.core.outcome_engine import BarData, OutcomeResult
from src.strategies.overnight_session import OvernightSessionStrategy


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _make_bar(
    ts_ms: int,
    close: float = 100.0,
    symbol: str = "SPY",
) -> BarData:
    """Create a minimal BarData with a symbol attribute."""
    bar = BarData(
        timestamp_ms=ts_ms,
        open=close - 0.10,
        high=close + 0.50,
        low=close - 0.50,
        close=close,
        volume=1000.0,
    )
    # ReplayHarness uses object.__setattr__ for symbol
    object.__setattr__(bar, "symbol", symbol)
    return bar


def _make_outcome(
    ts_ms: int,
    horizon_seconds: int = 14400,
    fwd_return: float = 0.01,
    symbol: str = "SPY",
) -> OutcomeResult:
    """Create a minimal OutcomeResult."""
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
    news_sentiment: Optional[Any] = None,
) -> Dict[str, Dict[str, Any]]:
    """Build a visible_regimes dict with optional external metrics."""
    metrics = {}
    if risk_flow is not None:
        metrics["global_risk_flow"] = risk_flow
    if news_sentiment is not None:
        metrics["news_sentiment"] = news_sentiment

    return {
        "EQUITIES": {
            "vol_regime": "VOL_NORMAL",
            "trend_regime": "TREND_UP",
            "metrics_json": json.dumps(metrics) if metrics else "",
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestDefaults:
    """Verify constructor defaults and parameter override."""

    def test_default_params(self):
        s = OvernightSessionStrategy()
        assert s._cfg["fwd_return_threshold"] == 0.005
        assert s._cfg["horizon_seconds"] == 14400
        assert s._cfg["gate_on_risk_flow"] is False
        assert s._cfg["entry_window_minutes"] == 30
        assert s._cfg["gate_on_news_sentiment"] is False

    def test_custom_params(self):
        s = OvernightSessionStrategy(params={
            "fwd_return_threshold": 0.010,
            "horizon_seconds": 28800,
        })
        assert s._cfg["fwd_return_threshold"] == 0.010
        assert s._cfg["horizon_seconds"] == 28800
        # Other defaults remain
        assert s._cfg["entry_window_minutes"] == 30
        assert s._cfg["gate_on_news_sentiment"] is False

    def test_strategy_id(self):
        s = OvernightSessionStrategy()
        assert s.strategy_id == "OVERNIGHT_SESSION_V1"


class TestEntryWindow:
    """Verify entry window detection across sessions."""

    def test_rth_tail_equities_entry(self):
        """Strategy should signal during end-of-RTH in equities."""
        s = OvernightSessionStrategy(params={
            "fwd_return_threshold": 0.001,
            "horizon_seconds": 14400,
        })

        # 2024-01-15 15:45 EST = 20:45 UTC → last 15 min of RTH
        # RTH close is 16:00 ET → this is within 30 min window
        ts_ms = 1705351500000  # Mon Jan 15 2024 20:45:00 UTC

        bar = _make_bar(ts_ms, symbol="SPY")
        outcomes = {ts_ms - 60000: _make_outcome(ts_ms - 60000, 14400, 0.008)}

        s.on_bar(bar, ts_ms, "RTH", outcomes)
        intents = s.generate_intents(ts_ms)

        # Should have generated an entry
        assert len(intents) >= 1
        assert intents[-1].intent_type == "OPEN"
        assert intents[-1].tag == "OVERNIGHT_ENTRY"

    def test_pre_market_equities_entry(self):
        """Strategy should signal during PRE session in equities."""
        s = OvernightSessionStrategy(params={
            "fwd_return_threshold": 0.001,
            "horizon_seconds": 14400,
        })

        # First: feed a CLOSED bar to set prev_session
        closed_ts = 1705309200000  # some CLOSED time
        bar_closed = _make_bar(closed_ts, symbol="SPY")
        s.on_bar(bar_closed, closed_ts, "CLOSED", {})
        s.generate_intents(closed_ts)

        # Now: PRE session bar with positive outcome
        pre_ts = 1705312800000  # PRE session time
        bar_pre = _make_bar(pre_ts, symbol="SPY")
        outcomes = {pre_ts - 60000: _make_outcome(pre_ts - 60000, 14400, 0.008)}

        s.on_bar(bar_pre, pre_ts, "PRE", outcomes)
        intents = s.generate_intents(pre_ts)

        assert len(intents) >= 1
        open_intents = [i for i in intents if i.intent_type == "OPEN"]
        assert len(open_intents) == 1
        assert open_intents[0].tag == "OVERNIGHT_ENTRY"

    def test_no_entry_during_rth_middle(self):
        """No entry during middle of RTH (not in last N minutes)."""
        s = OvernightSessionStrategy(params={
            "fwd_return_threshold": 0.001,
            "entry_window_minutes": 15,
            "horizon_seconds": 14400,
        })

        # 2024-01-15 12:00 EST = 17:00 UTC → early RTH, not in last 15min
        ts_ms = 1705338000000
        bar = _make_bar(ts_ms, symbol="SPY")
        outcomes = {ts_ms - 60000: _make_outcome(ts_ms - 60000, 14400, 0.02)}

        s.on_bar(bar, ts_ms, "RTH", outcomes)
        intents = s.generate_intents(ts_ms)

        assert len(intents) == 0

    def test_crypto_session_transition(self):
        """Crypto entry at ASIA → EU transition."""
        s = OvernightSessionStrategy(params={
            "fwd_return_threshold": 0.001,
            "horizon_seconds": 3600,
        })

        # First: ASIA bar
        asia_ts = 1705280400000
        bar_asia = _make_bar(asia_ts, symbol="BTCUSD")
        s.on_bar(bar_asia, asia_ts, "ASIA", {})
        s.generate_intents(asia_ts)

        # Then: EU bar with positive outcome
        eu_ts = 1705284000000
        bar_eu = _make_bar(eu_ts, symbol="BTCUSD")
        outcomes = {eu_ts - 60000: _make_outcome(eu_ts - 60000, 3600, 0.005)}

        s.on_bar(bar_eu, eu_ts, "EU", outcomes)
        intents = s.generate_intents(eu_ts)

        open_intents = [i for i in intents if i.intent_type == "OPEN"]
        assert len(open_intents) == 1
        assert open_intents[0].symbol == "BTCUSD"

    def test_no_crypto_entry_within_same_session(self):
        """No entry when staying in the same crypto session."""
        s = OvernightSessionStrategy(params={
            "fwd_return_threshold": 0.001,
            "horizon_seconds": 3600,
        })

        # Two consecutive ASIA bars
        t1 = 1705280400000
        t2 = t1 + 60000

        s.on_bar(_make_bar(t1, symbol="BTCUSD"), t1, "ASIA", {})
        s.generate_intents(t1)

        outcomes = {t1: _make_outcome(t1, 3600, 0.01)}
        s.on_bar(_make_bar(t2, symbol="BTCUSD"), t2, "ASIA", outcomes)
        intents = s.generate_intents(t2)

        assert len(intents) == 0  # No transition → no entry


class TestSignalGating:
    """Verify forward-return threshold gating."""

    def test_below_threshold_no_entry(self):
        """No entry when fwd_return < threshold."""
        s = OvernightSessionStrategy(params={
            "fwd_return_threshold": 0.010,
            "horizon_seconds": 14400,
        })

        # Force entry window via PRE after CLOSED
        s.on_bar(_make_bar(1000, symbol="SPY"), 1000, "CLOSED", {})
        s.generate_intents(1000)

        ts = 2000
        outcomes = {ts - 60000: _make_outcome(ts - 60000, 14400, 0.005)}  # below 0.010
        s.on_bar(_make_bar(ts, symbol="SPY"), ts, "PRE", outcomes)
        intents = s.generate_intents(ts)

        assert len(intents) == 0

    def test_no_matching_horizon_no_entry(self):
        """No entry when outcomes exist but at wrong horizon."""
        s = OvernightSessionStrategy(params={
            "fwd_return_threshold": 0.001,
            "horizon_seconds": 28800,  # strategy wants 8h
        })

        s.on_bar(_make_bar(1000, symbol="SPY"), 1000, "CLOSED", {})
        s.generate_intents(1000)

        ts = 2000
        # Outcome has horizon 3600 (1h), not 28800 (8h)
        outcomes = {ts - 60000: _make_outcome(ts - 60000, 3600, 0.02)}
        s.on_bar(_make_bar(ts, symbol="SPY"), ts, "PRE", outcomes)
        intents = s.generate_intents(ts)

        assert len(intents) == 0


class TestExplicitClose:
    """Verify explicit close intent after horizon elapsed."""

    def test_close_after_horizon(self):
        """CLOSE intent emitted when hold time >= horizon_seconds."""
        horizon_s = 3600  # 1h for faster test
        s = OvernightSessionStrategy(params={
            "fwd_return_threshold": 0.001,
            "horizon_seconds": horizon_s,
        })

        # Set up: CLOSED → PRE transition to enter
        t0 = 100_000
        s.on_bar(_make_bar(t0, symbol="SPY"), t0, "CLOSED", {})
        s.generate_intents(t0)

        # Enter at t1
        t1 = t0 + 60_000
        outcomes = {t0: _make_outcome(t0, horizon_s, 0.01)}
        s.on_bar(_make_bar(t1, symbol="SPY"), t1, "PRE", outcomes)
        open_intents = s.generate_intents(t1)

        assert len(open_intents) == 1
        assert open_intents[0].intent_type == "OPEN"
        assert s._entries_emitted == 1

        # Advance past horizon
        t2 = t1 + (horizon_s * 1000) + 1  # just past horizon
        s.on_bar(_make_bar(t2, symbol="SPY"), t2, "RTH", {})
        close_intents = s.generate_intents(t2)

        close_only = [i for i in close_intents if i.intent_type == "CLOSE"]
        assert len(close_only) == 1
        assert close_only[0].tag == "OVERNIGHT_EXIT"
        assert close_only[0].side == "SELL"
        assert s._closes_emitted == 1

    def test_no_close_before_horizon(self):
        """No CLOSE intent before horizon elapsed."""
        horizon_s = 14400
        s = OvernightSessionStrategy(params={
            "fwd_return_threshold": 0.001,
            "horizon_seconds": horizon_s,
        })

        t0 = 100_000
        s.on_bar(_make_bar(t0, symbol="SPY"), t0, "CLOSED", {})
        s.generate_intents(t0)

        t1 = t0 + 60_000
        outcomes = {t0: _make_outcome(t0, horizon_s, 0.01)}
        s.on_bar(_make_bar(t1, symbol="SPY"), t1, "PRE", outcomes)
        s.generate_intents(t1)

        # Advance but still within horizon
        t2 = t1 + (horizon_s * 500)  # half the horizon
        s.on_bar(_make_bar(t2, symbol="SPY"), t2, "RTH", {})
        intents = s.generate_intents(t2)

        close_only = [i for i in intents if i.intent_type == "CLOSE"]
        assert len(close_only) == 0


class TestRiskFlowGating:
    """Verify global risk flow gating logic."""

    def test_gated_by_low_risk_flow(self):
        """No entry when risk_flow < min_global_risk_flow."""
        s = OvernightSessionStrategy(params={
            "fwd_return_threshold": 0.001,
            "horizon_seconds": 14400,
            "gate_on_risk_flow": True,
            "min_global_risk_flow": -0.005,
        })

        s.on_bar(_make_bar(1000, symbol="SPY"), 1000, "CLOSED", {})
        s.generate_intents(1000)

        ts = 2000
        outcomes = {ts - 60000: _make_outcome(ts - 60000, 14400, 0.02)}
        regimes = _make_regimes(risk_flow=-0.1)  # well below threshold

        s.on_bar(_make_bar(ts, symbol="SPY"), ts, "PRE", outcomes, visible_regimes=regimes)
        intents = s.generate_intents(ts)

        assert len(intents) == 0

    def test_not_gated_when_risk_flow_ok(self):
        """Entry proceeds when risk_flow >= min_global_risk_flow."""
        s = OvernightSessionStrategy(params={
            "fwd_return_threshold": 0.001,
            "horizon_seconds": 14400,
            "gate_on_risk_flow": True,
            "min_global_risk_flow": -0.005,
        })

        s.on_bar(_make_bar(1000, symbol="SPY"), 1000, "CLOSED", {})
        s.generate_intents(1000)

        ts = 2000
        outcomes = {ts - 60000: _make_outcome(ts - 60000, 14400, 0.02)}
        regimes = _make_regimes(risk_flow=0.01)  # above threshold

        s.on_bar(_make_bar(ts, symbol="SPY"), ts, "PRE", outcomes, visible_regimes=regimes)
        intents = s.generate_intents(ts)

        open_intents = [i for i in intents if i.intent_type == "OPEN"]
        assert len(open_intents) == 1

    def test_gating_disabled_by_default(self):
        """With gate_on_risk_flow=False, risk flow is ignored."""
        s = OvernightSessionStrategy(params={
            "fwd_return_threshold": 0.001,
            "horizon_seconds": 14400,
            "gate_on_risk_flow": False,
        })

        s.on_bar(_make_bar(1000, symbol="SPY"), 1000, "CLOSED", {})
        s.generate_intents(1000)

        ts = 2000
        outcomes = {ts - 60000: _make_outcome(ts - 60000, 14400, 0.02)}
        regimes = _make_regimes(risk_flow=-999)  # extremely negative

        s.on_bar(_make_bar(ts, symbol="SPY"), ts, "PRE", outcomes, visible_regimes=regimes)
        intents = s.generate_intents(ts)

        open_intents = [i for i in intents if i.intent_type == "OPEN"]
        assert len(open_intents) == 1

    def test_missing_risk_flow_no_block(self):
        """When gate_on_risk_flow=True but metric is missing, don't block."""
        s = OvernightSessionStrategy(params={
            "fwd_return_threshold": 0.001,
            "horizon_seconds": 14400,
            "gate_on_risk_flow": True,
            "min_global_risk_flow": -0.005,
        })

        s.on_bar(_make_bar(1000, symbol="SPY"), 1000, "CLOSED", {})
        s.generate_intents(1000)

        ts = 2000
        outcomes = {ts - 60000: _make_outcome(ts - 60000, 14400, 0.02)}
        regimes = _make_regimes(risk_flow=None)  # no risk flow metric

        s.on_bar(_make_bar(ts, symbol="SPY"), ts, "PRE", outcomes, visible_regimes=regimes)
        intents = s.generate_intents(ts)

        # Should still enter because risk_flow is None (not below threshold)
        open_intents = [i for i in intents if i.intent_type == "OPEN"]
        assert len(open_intents) == 1


class TestDeterminism:
    """Replay determinism: same inputs → same outputs."""

    def test_deterministic_replay(self):
        """Two identical replays produce identical intents."""
        params = {
            "fwd_return_threshold": 0.003,
            "horizon_seconds": 3600,
        }

        def _run_replay():
            s = OvernightSessionStrategy(params=params)
            all_intents = []

            # Build a sequence: CLOSED → PRE (entry) → RTH → RTH (close)
            bars_and_sessions = [
                (100_000, "CLOSED", {}),
                (160_000, "PRE", {100_000: _make_outcome(100_000, 3600, 0.01)}),
                (220_000, "RTH", {}),
                (3_760_000, "RTH", {}),  # past horizon
            ]

            for ts, session, outcomes in bars_and_sessions:
                bar = _make_bar(ts, symbol="SPY")
                s.on_bar(bar, ts, session, outcomes)
                intents = s.generate_intents(ts)
                all_intents.extend(intents)

            state = s.finalize()
            return all_intents, state

        intents_a, state_a = _run_replay()
        intents_b, state_b = _run_replay()

        assert len(intents_a) == len(intents_b)
        for a, b in zip(intents_a, intents_b):
            assert a.symbol == b.symbol
            assert a.side == b.side
            assert a.intent_type == b.intent_type
            assert a.tag == b.tag

        assert state_a == state_b


class TestFinalize:
    """Verify finalize returns correct summary."""

    def test_finalize_empty(self):
        s = OvernightSessionStrategy()
        state = s.finalize()
        assert state["bars_seen"] == 0
        assert state["entries_emitted"] == 0
        assert state["closes_emitted"] == 0
        assert state["open_at_end"] == 0
        assert "params" in state

    def test_finalize_after_entry(self):
        s = OvernightSessionStrategy(params={
            "fwd_return_threshold": 0.001,
            "horizon_seconds": 3600,
        })

        s.on_bar(_make_bar(1000, symbol="SPY"), 1000, "CLOSED", {})
        s.generate_intents(1000)

        ts = 2000
        outcomes = {1000: _make_outcome(1000, 3600, 0.01)}
        s.on_bar(_make_bar(ts, symbol="SPY"), ts, "PRE", outcomes)
        s.generate_intents(ts)

        state = s.finalize()
        assert state["bars_seen"] == 2
        assert state["entries_emitted"] == 1
        assert state["open_at_end"] == 1  # not yet closed


class TestLongOnlyV1:
    """V1 is long-only — all entries are BUY side."""

    def test_all_entries_are_buy(self):
        s = OvernightSessionStrategy(params={
            "fwd_return_threshold": 0.001,
            "horizon_seconds": 3600,
        })

        s.on_bar(_make_bar(1000, symbol="SPY"), 1000, "CLOSED", {})
        s.generate_intents(1000)

        ts = 2000
        outcomes = {1000: _make_outcome(1000, 3600, 0.01)}
        s.on_bar(_make_bar(ts, symbol="SPY"), ts, "PRE", outcomes)
        intents = s.generate_intents(ts)

        for intent in intents:
            if intent.intent_type == "OPEN":
                assert intent.side == "BUY"


class TestNewsSentimentGating:
    """Verify news sentiment gating logic."""

    def test_gated_by_stub_news_sentiment(self):
        """No LONG entries with gate enabled and threshold above stub sentiment."""
        s = OvernightSessionStrategy(params={
            "fwd_return_threshold": 0.001,
            "horizon_seconds": 14400,
            "gate_on_news_sentiment": True,
            "min_news_sentiment": 0.5,
        })

        s.on_bar(_make_bar(1000, symbol="SPY"), 1000, "CLOSED", {})
        s.generate_intents(1000)

        ts = 2000
        outcomes = {ts - 60000: _make_outcome(ts - 60000, 14400, 0.02)}
        regimes = _make_regimes(news_sentiment={"score": 0.0, "label": "stub", "n_headlines": 0})

        s.on_bar(_make_bar(ts, symbol="SPY"), ts, "PRE", outcomes, visible_regimes=regimes)
        intents = s.generate_intents(ts)

        long_entries = [i for i in intents if i.intent_type == "OPEN" and i.side == "BUY"]
        assert len(long_entries) == 0

    def test_news_sentiment_allows_entry_when_above_threshold(self):
        s = OvernightSessionStrategy(params={
            "fwd_return_threshold": 0.001,
            "horizon_seconds": 14400,
            "gate_on_news_sentiment": True,
            "min_news_sentiment": 0.5,
        })

        s.on_bar(_make_bar(1000, symbol="SPY"), 1000, "CLOSED", {})
        s.generate_intents(1000)

        ts = 2000
        outcomes = {ts - 60000: _make_outcome(ts - 60000, 14400, 0.02)}
        regimes = _make_regimes(news_sentiment={"score": 0.75, "label": "bullish", "n_headlines": 10})

        s.on_bar(_make_bar(ts, symbol="SPY"), ts, "PRE", outcomes, visible_regimes=regimes)
        intents = s.generate_intents(ts)

        long_entries = [i for i in intents if i.intent_type == "OPEN" and i.side == "BUY"]
        assert len(long_entries) == 1
