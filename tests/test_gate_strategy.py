"""
Gate Strategy Tests
===================

Tests for the Day-of-Week + Regime timing gate strategy.
"""

from datetime import datetime, timezone

from src.core.events import BarEvent
from src.core.regimes import (
    MarketRegimeEvent,
    SymbolRegimeEvent,
    DQ_GAP_WINDOW,
)
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


def _make_symbol_regime(symbol: str, ts_ms: int, is_warm: bool = True, dq: int = 0) -> SymbolRegimeEvent:
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
        is_warm=is_warm,
        data_quality_flags=dq,
        config_hash="test",
    )


def _make_market_regime(ts_ms: int, session: str = "RTH") -> MarketRegimeEvent:
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


def test_gate_determinism_same_inputs():
    bus1 = DummyBus()
    bus2 = DummyBus()
    config = _default_config()

    strat1 = DowRegimeTimingGateStrategy(bus1, config=config)
    strat2 = DowRegimeTimingGateStrategy(bus2, config=config)

    dt = datetime(2024, 1, 2, 15, 0, tzinfo=timezone.utc)
    ts_ms = int(dt.timestamp() * 1000)
    bar = _make_bar("IBIT", dt.timestamp())
    symbol_regime = _make_symbol_regime("IBIT", ts_ms)
    market_regime = _make_market_regime(ts_ms, session="RTH")

    strat1._on_symbol_regime(symbol_regime)
    strat1._on_market_regime(market_regime)
    strat1._on_bar(bar)

    strat2._on_symbol_regime(symbol_regime)
    strat2._on_market_regime(market_regime)
    strat2._on_bar(bar)

    signal1 = bus1.published[0][1]
    signal2 = bus2.published[0][1]
    assert signal_to_dict(signal1) == signal_to_dict(signal2)


def test_gate_mixed_regime_arrival_order():
    config = _default_config()
    dt = datetime(2024, 1, 2, 15, 0, tzinfo=timezone.utc)
    ts_ms = int(dt.timestamp() * 1000)
    bar = _make_bar("IBIT", dt.timestamp())
    symbol_regime = _make_symbol_regime("IBIT", ts_ms)
    market_regime = _make_market_regime(ts_ms, session="RTH")

    bus1 = DummyBus()
    strat1 = DowRegimeTimingGateStrategy(bus1, config=config)
    strat1._on_symbol_regime(symbol_regime)
    strat1._on_market_regime(market_regime)
    strat1._on_bar(bar)
    signal1 = bus1.published[0][1]

    bus2 = DummyBus()
    strat2 = DowRegimeTimingGateStrategy(bus2, config=config)
    strat2._on_market_regime(market_regime)
    strat2._on_symbol_regime(symbol_regime)
    strat2._on_bar(bar)
    signal2 = bus2.published[0][1]

    assert signal_to_dict(signal1) == signal_to_dict(signal2)


def test_gate_no_duplicate_emits_for_same_state():
    config = _default_config()
    dt = datetime(2024, 1, 2, 15, 0, tzinfo=timezone.utc)
    ts_ms = int(dt.timestamp() * 1000)
    bar = _make_bar("IBIT", dt.timestamp())
    symbol_regime = _make_symbol_regime("IBIT", ts_ms)
    market_regime = _make_market_regime(ts_ms, session="RTH")

    bus = DummyBus()
    strat = DowRegimeTimingGateStrategy(bus, config=config)
    strat._on_symbol_regime(symbol_regime)
    strat._on_market_regime(market_regime)

    strat._on_bar(bar)
    strat._on_bar(bar)

    assert len(bus.published) == 1


def test_gate_session_last_n_minutes():
    config = _default_config()
    dt = datetime(2024, 1, 2, 20, 55, tzinfo=timezone.utc)
    ts_ms = int(dt.timestamp() * 1000)
    bar = _make_bar("IBIT", dt.timestamp())
    symbol_regime = _make_symbol_regime("IBIT", ts_ms)
    market_regime = _make_market_regime(ts_ms, session="RTH")

    bus = DummyBus()
    strat = DowRegimeTimingGateStrategy(bus, config=config)
    strat._on_symbol_regime(symbol_regime)
    strat._on_market_regime(market_regime)
    strat._on_bar(bar)

    signal = bus.published[0][1]
    assert "RTH_LAST_N" in signal.explain
    assert signal.features_snapshot["gate_allow"] == 0.0


def test_gate_dow_weight_applied():
    config = _default_config()
    config["dow_weights"]["Mon"] = 0.7
    dt = datetime(2024, 1, 1, 15, 0, tzinfo=timezone.utc)  # Monday
    ts_ms = int(dt.timestamp() * 1000)
    bar = _make_bar("IBIT", dt.timestamp())
    symbol_regime = _make_symbol_regime("IBIT", ts_ms)
    market_regime = _make_market_regime(ts_ms, session="RTH")

    bus = DummyBus()
    strat = DowRegimeTimingGateStrategy(bus, config=config)
    strat._on_symbol_regime(symbol_regime)
    strat._on_market_regime(market_regime)
    strat._on_bar(bar)

    signal = bus.published[0][1]
    assert signal.features_snapshot["gate_score"] == 0.7
    assert signal.features_snapshot["gate_allow"] == 1.0


def test_gate_data_quality_avoid():
    config = _default_config()
    dt = datetime(2024, 1, 2, 15, 0, tzinfo=timezone.utc)
    ts_ms = int(dt.timestamp() * 1000)
    bar = _make_bar("IBIT", dt.timestamp())
    symbol_regime = _make_symbol_regime("IBIT", ts_ms, dq=DQ_GAP_WINDOW)
    market_regime = _make_market_regime(ts_ms, session="RTH")

    bus = DummyBus()
    strat = DowRegimeTimingGateStrategy(bus, config=config)
    strat._on_symbol_regime(symbol_regime)
    strat._on_market_regime(market_regime)
    strat._on_bar(bar)

    signal = bus.published[0][1]
    assert "DQ_GAP_WINDOW" in signal.explain
    assert signal.features_snapshot["gate_allow"] == 0.0


def test_gate_missing_symbol_regime_avoids():
    config = _default_config()
    dt = datetime(2024, 1, 2, 15, 0, tzinfo=timezone.utc)
    ts_ms = int(dt.timestamp() * 1000)
    bar = _make_bar("IBIT", dt.timestamp())
    market_regime = _make_market_regime(ts_ms, session="RTH")

    bus = DummyBus()
    strat = DowRegimeTimingGateStrategy(bus, config=config)
    strat._on_market_regime(market_regime)
    strat._on_bar(bar)

    signal = bus.published[0][1]
    assert "SYMBOL_MISSING" in signal.explain
    assert signal.features_snapshot["gate_allow"] == 0.0
