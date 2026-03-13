import pytest
import math
from typing import List, Dict, Any
from src.core.bus import EventBus
from src.core.events import BarEvent, QuoteEvent, TOPIC_REGIMES_SYMBOL, TOPIC_REGIMES_MARKET
from src.core.regime_detector import RegimeDetector
from src.core.regimes import SymbolRegimeEvent, MarketRegimeEvent

class SyncBus(EventBus):
    def __init__(self):
        super().__init__()
        self.events = []

    def publish(self, topic: str, event: Any) -> None:
        # Collect for verification
        self.events.append((topic, event))
        # Sync dispatch
        handlers = self._subscribers.get(topic, [])
        for handler in handlers:
            handler(event)

def create_bar(symbol: str, ts: float, close: float, high: float = None, low: float = None) -> BarEvent:
    return BarEvent(
        symbol=symbol,
        timestamp=ts,
        open=close,
        high=high or close,
        low=low or close,
        close=close,
        volume=1000,
        bar_duration=60,
        source="test"
    )

def test_regime_hysteresis():
    bus = SyncBus()
    thresholds = {
        "warmup_bars": 5,
        "vol_hysteresis_enabled": True,
        "vol_hysteresis_band": 0.5,
        "vol_high_z": 1.0,
        "trend_hysteresis_enabled": True,
        "trend_hysteresis_slope_band": 0.2,
        "trend_slope_threshold": 0.5,
        "trend_strength_threshold": 0.1,
    }
    detector = RegimeDetector(bus, thresholds=thresholds)
    
    symbol = "TEST"
    # 1. Warm up
    # Need > 14 bars for RSI/EMA warmth
    for i in range(40): 
        bus.publish("market.bars", create_bar(symbol, 1000 + i*60, 100 + (0.1 if i % 2 == 0 else -0.1)))
    
    bus.events.clear()
    
    # 2. Induce high volatility (z-score should spike)
    # We need to feed it returns that increase vol_z
    for i in range(10, 20):
        # alternate big moves to spike vol
        close = 100 + (5.0 if i % 2 == 0 else -5.0)
        bus.publish("market.bars", create_bar(symbol, 1000 + i*60, close))

    # Check that we hit VOL_HIGH or VOL_SPIKE
    last_event = [e for t, e in bus.events if t == TOPIC_REGIMES_SYMBOL][-1]
    assert last_event.vol_regime in ["VOL_HIGH", "VOL_SPIKE"]
    
    prev_regime = last_event.vol_regime
    prev_z = last_event.vol_z
    
    # 3. Bring z-score just below threshold but within hysteresis band
    # If high_z is 1.0 and band is 0.5, exit is at 0.5.
    # We want a z-score of 0.8.
    
    # This is tricky without a full math model, so we'll just verify the logic branches in the code:
    # We'll mock the _classify_vol_regime return for a unit-ish test if needed, 
    # but here we'll try to push the data.
    
    for i in range(20, 25):
        bus.publish("market.bars", create_bar(symbol, 1000 + i*60, 100))
    
    events = [e for t, e in bus.events if t == TOPIC_REGIMES_SYMBOL]
    # With hysteresis, vol can step down (e.g. SPIKE -> HIGH) as z-score drops; it should not
    # immediately flip to VOL_NORMAL. So we allow prev_vol_regime to stay elevated (HIGH or SPIKE).
    current_prev = detector._symbol_states[f"{symbol}:60"].prev_vol_regime
    assert current_prev in ["VOL_HIGH", "VOL_SPIKE"], (
        f"Hysteresis should keep regime elevated; got {current_prev}"
    )

def test_gap_warmth_decay():
    bus = SyncBus()
    thresholds = {
        "warmup_bars": 10,
        "gap_confidence_decay_threshold_ms": 3600 * 1000, # 1 hour
        "gap_warmth_decay_bars": 5,
        "gap_confidence_decay_multiplier": 0.5
    }
    detector = RegimeDetector(bus, thresholds=thresholds)
    
    symbol = "GAP_TEST"
    # 1. Warm up fully
    for i in range(15):
        bus.publish("market.bars", create_bar(symbol, 1000 + i*60, 100))
    
    state = detector._symbol_states[f"{symbol}:60"]
    assert state.bars_processed == 15
    
    # 2. Induce a 2-hour gap
    bus.publish("market.bars", create_bar(symbol, 1000 + 15*60 + 7200, 101))
    
    # 3. Check decay
    # bars_processed should have dropped by 5
    assert state.bars_processed == 11 # 15 - 5 + 1 (for the new bar)
    
    last_event = [e for t, e in bus.events if t == TOPIC_REGIMES_SYMBOL][-1]
    assert last_event.confidence < 1.0
    assert last_event.data_quality_flags & 2 # DQ_GAP_WINDOW

def test_risk_regime_aggregation():
    bus = SyncBus()
    thresholds = {
        "warmup_bars": 1,
        "risk_basket": ["SPY", "TLT"]
    }
    detector = RegimeDetector(bus, thresholds=thresholds)
    
    # 1. Provide SPY data (Bullish)
    # Need > 14 bars for RSI warmth. Provide 40 for stability.
    for i in range(40):
        # trending up: price 400 to 440
        bus.publish("market.bars", create_bar("SPY", 1000 + i*60, 400 + i))
    
    # Check Market Regime
    market_events = [e for t, e in bus.events if t == TOPIC_REGIMES_MARKET]
    assert market_events[-1].risk_regime == "RISK_ON"
    
    # 2. Induce SPY crash (Bearish)
    for i in range(40, 60):
        # sharp drop
        bus.publish("market.bars", create_bar("SPY", 1000 + i*60, 440 - (i-40)*10))
    
    market_events = [e for t, e in bus.events if t == TOPIC_REGIMES_MARKET]
    assert market_events[-1].risk_regime == "RISK_OFF"

if __name__ == "__main__":
    print("Running test_risk_regime_aggregation manually...")
    try:
        test_risk_regime_aggregation()
        print("test_risk_regime_aggregation PASSED")
    except Exception as e:
        print(f"test_risk_regime_aggregation FAILED: {e}")
        import traceback
        traceback.print_exc()
