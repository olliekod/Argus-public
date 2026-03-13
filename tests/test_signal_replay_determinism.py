"""
Signal Replay Determinism Tests
===============================

Ensure signal emissions are deterministic across runs.
"""

from datetime import datetime, timezone

from src.core.events import BarEvent
from src.core.regimes import MarketRegimeEvent, SymbolRegimeEvent
from src.core.signals import signal_to_dict
from src.strategies.dow_regime_timing import DowRegimeTimingGateStrategy


class DummyBus:
    def __init__(self) -> None:
        self.published = []

    def subscribe(self, topic, handler) -> None:
        return None

    def publish(self, topic, event) -> None:
        self.published.append((topic, event))


def _make_bar(symbol: str, ts: float) -> BarEvent:
    return BarEvent(
        symbol=symbol,
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=1.0,
        timestamp=ts,
        source="yahoo",
        bar_duration=60,
    )


def _make_symbol_regime(symbol: str, ts_ms: int) -> SymbolRegimeEvent:
    return SymbolRegimeEvent(
        symbol=symbol,
        timeframe=60,
        timestamp_ms=ts_ms,
        vol_regime="VOL_NORMAL",
        trend_regime="RANGE",
        liquidity_regime="LIQ_NORMAL",
        atr=1.0,
        atr_pct=0.01,
        vol_z=0.1,
        ema_fast=100.0,
        ema_slow=100.0,
        ema_slope=0.0,
        rsi=50.0,
        spread_pct=0.001,
        volume_pctile=50.0,
        confidence=1.0,
        is_warm=True,
        data_quality_flags=0,
        config_hash="test",
    )


def _make_market_regime(ts_ms: int, session: str) -> MarketRegimeEvent:
    return MarketRegimeEvent(
        market="EQUITIES",
        timeframe=60,
        timestamp_ms=ts_ms,
        session_regime=session,
        risk_regime="UNKNOWN",
        confidence=1.0,
        data_quality_flags=0,
        config_hash="test",
    )


def _default_config():
    return {
        "gates": {
            "SELL_PUT_SPREAD": {
                "symbols": ["IBIT"],
                "market": "EQUITIES",
                "enable_market_scope": False,
                "allow_pre_market": False,
                "allow_post_market": False,
                "avoid_last_n_minutes_rth": 15,
            },
        },
        "dow_weights": {"Mon": 1.0, "Tue": 1.0, "Wed": 1.0, "Thu": 1.0, "Fri": 1.0},
        "gate_score_base": 1.0,
        "gate_score_threshold": 0.5,
        "gate_score_penalties": {},
        "vol_spike_avoid": True,
        "dq_avoid_flags": ["GAP_WINDOW", "REPAIRED_INPUT"],
    }


def _run_sequence():
    bus = DummyBus()
    strat = DowRegimeTimingGateStrategy(bus, config=_default_config())

    dt1 = datetime(2024, 1, 2, 15, 0, tzinfo=timezone.utc)
    dt2 = datetime(2024, 1, 2, 22, 0, tzinfo=timezone.utc)
    ts1_ms = int(dt1.timestamp() * 1000)
    ts2_ms = int(dt2.timestamp() * 1000)

    strat._on_symbol_regime(_make_symbol_regime("IBIT", ts1_ms))
    strat._on_market_regime(_make_market_regime(ts1_ms, session="RTH"))
    strat._on_bar(_make_bar("IBIT", dt1.timestamp()))

    strat._on_market_regime(_make_market_regime(ts2_ms, session="CLOSED"))
    strat._on_bar(_make_bar("IBIT", dt2.timestamp()))

    return [signal_to_dict(event) for _, event in bus.published]


def test_signal_replay_determinism():
    signals_run_1 = _run_sequence()
    signals_run_2 = _run_sequence()

    assert signals_run_1 == signals_run_2
