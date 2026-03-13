from __future__ import annotations

from unittest.mock import MagicMock

from src.core.market_regime_detector import MarketRegimeDetector
from src.core.regimes import SymbolRegimeEvent
from src.core.events import TOPIC_REGIMES_MARKET


def _symbol(symbol: str, trend: str, vol: str, ts_ms: int = 1_700_000_000_000) -> SymbolRegimeEvent:
    return SymbolRegimeEvent(
        symbol=symbol,
        timeframe=60,
        timestamp_ms=ts_ms,
        vol_regime=vol,
        trend_regime=trend,
        liquidity_regime="LIQ_NORMAL",
        atr=1.0,
        atr_pct=0.01,
        vol_z=0.0,
        ema_fast=100.0,
        ema_slow=99.0,
        ema_slope=0.1,
        rsi=55.0,
        spread_pct=0.01,
        volume_pctile=50.0,
        confidence=1.0,
        is_warm=True,
        data_quality_flags=0,
        config_hash="x",
    )


def test_market_regime_detector_disabled_by_default():
    bus = MagicMock()
    bus.subscribe = MagicMock()
    published = []
    bus.publish.side_effect = lambda topic, event: published.append((topic, event))

    d = MarketRegimeDetector(bus)
    d._on_symbol_regime(_symbol("SPY", "TREND_UP", "VOL_NORMAL"))
    assert not published


def test_market_regime_detector_emits_when_basket_ready():
    bus = MagicMock()
    bus.subscribe = MagicMock()
    published = []
    bus.publish.side_effect = lambda topic, event: published.append((topic, event))

    d = MarketRegimeDetector(bus, risk_basket_symbols=["SPY", "TLT"])
    d._on_symbol_regime(_symbol("SPY", "TREND_UP", "VOL_NORMAL"))
    d._on_symbol_regime(_symbol("TLT", "RANGE", "VOL_NORMAL"))

    assert published
    topic, event = published[-1]
    assert topic == TOPIC_REGIMES_MARKET
    assert event.risk_regime in {"RISK_ON", "RISK_OFF", "NEUTRAL", "UNKNOWN"}
    assert event.market == "GLOBAL"
