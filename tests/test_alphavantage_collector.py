import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo
from src.connectors.alphavantage_collector import AlphaVantageCollector
from src.core.bus import EventBus

# Config with 10 ETFs + 4 FX (matches config.yaml budget)
_AV_CONFIG_14 = {
    "exchanges": {
        "alphavantage": {
            "enabled": True,
            "daily_symbols": ["EWJ", "FXI", "EWT", "EWY", "INDA", "EWG", "EWU", "FEZ", "EWL", "EEM"],
            "fx_pairs": ["EUR/USD", "USD/JPY", "GBP/USD", "AUD/USD"],
        }
    }
}


@pytest.mark.asyncio
async def test_collector_batch_logic():
    av_client = AsyncMock()
    db = AsyncMock()
    bus = MagicMock(spec=EventBus)
    config = _AV_CONFIG_14
    
    collector = AlphaVantageCollector(av_client, db, bus, config)
    
    # Mock AV to return a single bar
    mock_bar = {
        "timestamp_ms": 1700000000000,
        "open": 100.0, "high": 110.0, "low": 90.0, "close": 105.0, "volume": 1000.0
    }
    av_client.fetch_daily_bars.return_value = [mock_bar]
    av_client.fetch_fx_daily.return_value = [mock_bar]
    db.upsert_bars_backfill.return_value = 1
    
    # Run batch (must set _running=True or it breaks early)
    # Patch sleep to avoid 15s delay between symbols
    collector._running = True
    with patch("asyncio.sleep", AsyncMock()):
        await collector._run_batch()
    
    # Should have polled all 14 symbols (10 ETFs + 4 FX)
    assert av_client.fetch_daily_bars.call_count == 10
    assert av_client.fetch_fx_daily.call_count == 4
    assert db.upsert_bars_backfill.call_count == 14
    
    # Verify last run date was set to today (ET)
    assert collector._last_run_date == datetime.now(ZoneInfo("America/New_York")).date()

def test_collector_scheduling_logic():
    av_client = MagicMock()
    db = MagicMock()
    bus = MagicMock()
    config = _AV_CONFIG_14
    collector = AlphaVantageCollector(av_client, db, bus, config)
    
    target_time = dt_time(9, 0)
    
    # Monday 8:00 AM ET -> Should NOT run
    monday_8am = datetime(2026, 2, 16, 8, 0, tzinfo=ZoneInfo("America/New_York"))
    assert collector._should_run_now(monday_8am, target_time) is False
    
    # Monday 9:01 AM ET -> Should run
    monday_9am = datetime(2026, 2, 16, 9, 1, tzinfo=ZoneInfo("America/New_York"))
    assert collector._should_run_now(monday_9am, target_time) is True
    
    # Sunday 10:00 AM ET -> Should NOT run (weekend)
    sunday_10am = datetime(2026, 2, 15, 10, 0, tzinfo=ZoneInfo("America/New_York"))
    assert collector._should_run_now(sunday_10am, target_time) is False
    
    # Already ran today -> Should NOT run again
    collector._last_run_date = monday_9am.date()
    assert collector._should_run_now(monday_9am, target_time) is False

@pytest.mark.asyncio
async def test_collector_fx_mapping():
    av_client = AsyncMock()
    db = AsyncMock()
    bus = MagicMock()
    config = _AV_CONFIG_14
    collector = AlphaVantageCollector(av_client, db, bus, config)
    
    # Mock bar
    mock_bar = {
        "timestamp_ms": 1700000000000,
        "open": 1.1, "high": 1.2, "low": 1.0, "close": 1.15
    }
    av_client.fetch_fx_daily.return_value = [mock_bar]
    db.upsert_bars_backfill.return_value = 1
    
    await collector._collect_symbol("FX:USDJPY")
    
    av_client.fetch_fx_daily.assert_called_with("USD", "JPY")
    db.upsert_bars_backfill.assert_called()
    rows = db.upsert_bars_backfill.call_args[0][0]
    assert rows[0][1] == "FX:USDJPY"
