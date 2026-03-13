import pytest
import asyncio
import json
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import replace, dataclass

from src.core.bus import EventBus
from src.core.regime_detector import RegimeDetector
from src.core.global_risk_flow_updater import GlobalRiskFlowUpdater
from src.core.news_sentiment_updater import NewsSentimentUpdater
from src.core.events import (
    TOPIC_EXTERNAL_METRICS,
    TOPIC_REGIMES_MARKET,
    TOPIC_MARKET_BARS,
    ExternalMetricEvent,
    BarEvent
)
from src.core.regimes import MarketRegimeEvent
from src.tools.replay_pack import create_replay_pack

@pytest.mark.asyncio
async def test_global_risk_flow_updater_publishes_from_db():
    bus = MagicMock(spec=EventBus)
    db = AsyncMock()
    
    # Mock DB to return some bars
    db.get_bars_daily_for_risk_flow.return_value = {
        "EWJ": [{"timestamp_ms": 1000, "close": 100.0}, {"timestamp_ms": 2000, "close": 101.0}],
        "FX:USDJPY": [{"timestamp_ms": 1000, "close": 150.0}, {"timestamp_ms": 2000, "close": 151.0}]
    }
    
    config = {
        "exchanges": {
            "alphavantage": {
                "enabled": True,
                "daily_symbols": ["EWJ"],
                "fx_pairs": ["USD/JPY"]
            }
        }
    }
    
    updater = GlobalRiskFlowUpdater(bus, db, config)
    # Ensure sim_time_ms is after the bars
    with patch("time.time", return_value=3.0):
        val = await updater.update()
    
    assert val is not None
    # Check that publish was called on the bus
    bus.publish.assert_called()
    args, _ = bus.publish.call_args
    assert args[0] == TOPIC_EXTERNAL_METRICS
    assert isinstance(args[1], ExternalMetricEvent)
    assert args[1].key == "global_risk_flow"
    assert args[1].value == round(val, 8)

@pytest.mark.asyncio
async def test_regime_detector_subscription_and_versioning():
    bus = EventBus()
    detector = RegimeDetector(bus)
    
    # 1. Valid v1 event
    event1 = ExternalMetricEvent(
        key="global_risk_flow",
        value=0.0123,
        timestamp_ms=123456789,
        v=1
    )
    bus.publish(TOPIC_EXTERNAL_METRICS, event1)
    
    # 2. Unknown version event (should be ignored)
    @dataclass
    class LegacyEvent:
        key: str
        value: Any
        timestamp_ms: int
        v: int
    
    event2 = LegacyEvent(key="global_risk_flow", value=0.9999, timestamp_ms=123456799, v=2)
    bus.publish(TOPIC_EXTERNAL_METRICS, event2) # type: ignore
    
    bus.start()
    await asyncio.sleep(0.1)
    
    with detector._lock:
        # Should have event1 value, NOT event2
        assert detector._risk_metrics["global_risk_flow"] == 0.0123
    
    bus.stop()

@pytest.mark.asyncio
async def test_alphavantage_disabled_noop():
    bus = MagicMock(spec=EventBus)
    config = {
        "exchanges": {
            "alphavantage": {
                "enabled": False
            }
        }
    }
    updater = GlobalRiskFlowUpdater(bus, None, config)
    val = await updater.update()
    assert val is None
    bus.publish.assert_not_called()

@pytest.mark.asyncio
async def test_replay_pack_injection_determinism(tmp_path):
    output_file1 = tmp_path / "pack1.json"
    output_file2 = tmp_path / "pack2.json"
    
    # Mock Database
    db = AsyncMock()
    regimes = [
        {
            "scope": "EQUITIES",
            "timestamp": "2023-11-15T00:00:00Z",
            "metrics_json": '{"other": 1}'
        }
    ]
    db.get_regimes.return_value = regimes
    db.get_bars_daily_for_risk_flow.return_value = {
        "EWJ": [{"timestamp_ms": 1000, "close": 100.0}, {"timestamp_ms": 2000, "close": 101.0}],
        "FX:USDJPY": [{"timestamp_ms": 1000, "close": 150.0}, {"timestamp_ms": 2000, "close": 151.0}]
    }
    # Mocking other DB calls to avoid errors
    db.get_bars_for_outcome_computation.return_value = []
    db.get_bar_outcomes.return_value = []
    db.get_bar_inventory.return_value = []
    db.get_outcome_inventory.return_value = []
    db.get_option_chain_snapshots.return_value = []
    
    from src.tools.replay_pack import create_replay_pack
    from unittest.mock import patch
    
    with patch("src.tools.replay_pack.Database", return_value=db):
        with patch("src.tools.replay_pack.get_data_source_policy") as mock_policy:
            mock_policy.return_value.bars_provider = "alpaca"
            mock_policy.return_value.options_snapshot_provider = "alpaca"
            
            await create_replay_pack("IBIT", "2023-11-15", "2023-11-15", str(output_file1))
            # Reset mock for second call
            db.get_regimes.return_value = [
                {
                    "scope": "EQUITIES",
                    "timestamp": "2023-11-15T00:00:00Z",
                    "metrics_json": '{"other": 1}'
                }
            ]
            await create_replay_pack("IBIT", "2023-11-15", "2023-11-15", str(output_file2))
            
    with open(output_file1) as f:
        p1 = json.load(f)
    with open(output_file2) as f:
        p2 = json.load(f)
        
    # Data check
    assert p1["regimes"][0]["metrics_json"] == p2["regimes"][0]["metrics_json"]
    # String check (ensuring sort_keys works)
    m1 = p1["regimes"][0]["metrics_json"]
    m2 = p2["regimes"][0]["metrics_json"]
    assert m1 == m2
    assert "global_risk_flow" in m1
    assert "news_sentiment" in m1

@pytest.mark.asyncio
async def test_bus_to_regime_integration():
    """Prove signal exists end-to-end: Bus -> RegimeDetector -> Emitted Event."""
    bus = EventBus()
    detector = RegimeDetector(bus)
    
    regime_events = []
    def on_market_regime(event):
        regime_events.append(event)
    
    bus.subscribe(TOPIC_REGIMES_MARKET, on_market_regime)
    bus.start()
    
    # 1. Publish external metric
    bus.publish(TOPIC_EXTERNAL_METRICS, ExternalMetricEvent(
        key="global_risk_flow",
        value=0.5555,
        timestamp_ms=int(time.time() * 1000)
    ))
    
    # Wait for the metric to be registered in the detector to avoid race conditions
    # (since the bus uses a separate worker thread per topic)
    for _ in range(20):
        with detector._lock:
            if detector._risk_metrics.get("global_risk_flow") == 0.5555:
                break
        await asyncio.sleep(0.05)
    
    # 2. Emit a bar to trigger regime emission
    bus.publish(TOPIC_MARKET_BARS, BarEvent(
        symbol="SPY", open=400, high=401, low=399, close=400.5,
        volume=1000, timestamp=time.time(), source="yahoo"
    ))
    
    # Let workers process
    await asyncio.sleep(0.2)
    
    assert len(regime_events) > 0
    last_event = regime_events[-1]
    metrics = json.loads(last_event.metrics_json)
    assert metrics["global_risk_flow"] == 0.5555
    
    bus.stop()


@pytest.mark.asyncio
async def test_news_sentiment_updater_and_regime_integration():
    """News sentiment publishes ExternalMetricEvent and appears in metrics_json."""
    bus = EventBus()
    detector = RegimeDetector(bus)

    regime_events = []

    def on_market_regime(event):
        regime_events.append(event)

    bus.subscribe(TOPIC_REGIMES_MARKET, on_market_regime)
    bus.start()

    updater = NewsSentimentUpdater(bus=bus, config={
        "news_sentiment": {
            "enabled": True,
            "interval_seconds": 3600,
            "feeds": [],
            "max_headlines": 50,
        }
    })

    payload = await updater.update()
    assert payload == {"score": 0.0, "label": "stub", "n_headlines": 0}

    # Wait for the metric to be registered in the detector (bus uses worker threads)
    for _ in range(30):
        with detector._lock:
            if detector._risk_metrics.get("news_sentiment") == payload:
                break
        await asyncio.sleep(0.05)

    # Emit a bar to force a market regime event with merged metrics_json
    bus.publish(TOPIC_MARKET_BARS, BarEvent(
        symbol="SPY", open=400, high=401, low=399, close=400.5,
        volume=1000, timestamp=time.time(), source="yahoo"
    ))

    await asyncio.sleep(0.2)

    assert len(regime_events) > 0
    metrics = json.loads(regime_events[-1].metrics_json)
    assert metrics["news_sentiment"] == payload

    await updater.close()
    bus.stop()
