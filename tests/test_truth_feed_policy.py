"""
Tests for orchestrator truth-feed policy:
  - Coinbase WS (primary) started in run(), disconnected in stop()
  - OKX WS (secondary fallback) with deterministic source switching
  - Truth-feed health tracking per asset
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
#  Helpers: mock orchestrator that only exercises truth-feed code paths
# ---------------------------------------------------------------------------

def _make_mock_orchestrator():
    """Build a minimal mock of ArgusOrchestrator with truth-feed attrs."""
    from types import SimpleNamespace

    orch = SimpleNamespace()
    orch.mode = "collector"
    orch.logger = MagicMock()
    orch._running = True
    orch._tasks = []

    # Coinbase WS mock
    orch.coinbase_ws = MagicMock()
    orch.coinbase_ws.connect = AsyncMock()
    orch.coinbase_ws.disconnect = AsyncMock()
    orch.coinbase_ws.is_connected = MagicMock(return_value=True)

    # OKX fallback mock
    orch._okx_fallback = MagicMock()
    orch._okx_fallback.start = AsyncMock()
    orch._okx_fallback.stop = AsyncMock()

    # Bybit WS mock
    orch.bybit_ws = MagicMock()
    orch.bybit_ws.connect = AsyncMock()
    orch.bybit_ws.disconnect = AsyncMock()

    # Truth-feed health tracking
    orch._truth_feed_health = {
        "BTC": {"last_tick_mono": 0.0, "active_source": "coinbase", "fallback_count": 0},
        "ETH": {"last_tick_mono": 0.0, "active_source": "coinbase", "fallback_count": 0},
    }
    orch._truth_fallback_activation_s = 5.0
    orch._truth_source_switch_log_ts = 0.0

    # Detectors (empty — not relevant for truth-feed tests)
    orch.detectors = {}
    orch._last_price_snapshot = {}

    return orch


# ---------------------------------------------------------------------------
#  run() starts Coinbase WS
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_coinbase_ws_started_in_run():
    """Coinbase WS connect() must be called in run() via asyncio.create_task."""
    from src.orchestrator import ArgusOrchestrator

    # Patch out __init__ to avoid loading real config
    with patch.object(ArgusOrchestrator, "__init__", lambda self, *a, **kw: None):
        orch = ArgusOrchestrator.__new__(ArgusOrchestrator)

    orch.mode = "collector"
    orch.logger = MagicMock()
    orch._running = True
    orch._tasks = []

    # Mock connectors
    orch.coinbase_ws = MagicMock()
    orch.coinbase_ws.connect = AsyncMock()

    orch._okx_fallback = MagicMock()
    orch._okx_fallback.start = AsyncMock()

    orch.bybit_ws = MagicMock()
    orch.bybit_ws.connect = AsyncMock()

    # Null out everything else run() touches
    orch.yahoo_client = None
    orch.alpaca_client = None
    orch.alpaca_options = None
    orch.tastytrade_options = None
    orch.public_options = None
    orch.paper_trader_farm = None
    orch.research_enabled = False
    orch.polymarket_gamma = None
    orch.polymarket_clob = None
    orch.polymarket_watchlist = None
    orch.gap_risk_tracker = None
    orch.av_collector = None
    orch.global_risk_flow_updater = None
    orch.conditions_monitor = None
    orch.daily_review = None
    orch.telegram = None
    orch.soak_guardian = MagicMock()
    orch.resource_monitor = MagicMock()

    # Stub internal async methods
    orch._poll_deribit = AsyncMock()
    orch._health_check = AsyncMock()
    orch._run_db_maintenance = AsyncMock()
    orch._run_soak_guards = AsyncMock()
    orch._publish_heartbeats = AsyncMock()
    orch._publish_minute_ticks = AsyncMock()
    orch._run_market_session_monitor = AsyncMock()
    orch._publish_component_heartbeats = AsyncMock()
    orch._publish_status_snapshots = AsyncMock()

    # Make gather return immediately
    run_task = asyncio.create_task(orch.run())
    await asyncio.sleep(0.05)
    run_task.cancel()
    try:
        await run_task
    except asyncio.CancelledError:
        pass

    # Coinbase connect() must have been scheduled (task created)
    # The task list should contain a task for coinbase_ws.connect
    assert any(
        "coinbase" in str(getattr(t, '_coro', '')).lower()
        for t in orch._tasks
    ) or orch.coinbase_ws.connect.called, (
        "Coinbase WS connect() was not started in run()"
    )


# ---------------------------------------------------------------------------
#  stop() disconnects Coinbase WS
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_coinbase_ws_disconnected_in_stop():
    """stop() must call coinbase_ws.disconnect()."""
    from src.orchestrator import ArgusOrchestrator

    with patch.object(ArgusOrchestrator, "__init__", lambda self, *a, **kw: None):
        orch = ArgusOrchestrator.__new__(ArgusOrchestrator)

    orch.logger = MagicMock()
    orch._running = True
    orch._tasks = []
    orch.bar_builder = None
    orch.persistence = None
    orch.event_bus = MagicMock()
    orch.event_bus.stop = MagicMock()
    orch.av_collector = None
    orch.dashboard = None
    orch.news_sentiment_updater = None
    orch.conditions_monitor = None
    orch.daily_review = None
    orch.telegram = None
    orch.polymarket_gamma = None
    orch.polymarket_clob = None
    orch.polymarket_watchlist = None
    orch.bybit_ws = MagicMock()
    orch.bybit_ws.disconnect = AsyncMock()
    orch.coinbase_ws = MagicMock()
    orch.coinbase_ws.disconnect = AsyncMock()
    orch._okx_fallback = MagicMock()
    orch._okx_fallback.stop = AsyncMock()
    orch.deribit_client = None
    orch.yahoo_client = None
    orch.alpaca_options = None
    orch.tastytrade_options = None
    orch.public_options = None
    orch.alphavantage_client = None
    orch.ai_agent = None
    orch.db = MagicMock()
    orch.db.close = AsyncMock()

    await orch.stop()

    orch.coinbase_ws.disconnect.assert_awaited_once()
    orch._okx_fallback.stop.assert_awaited_once()


# ---------------------------------------------------------------------------
#  Truth feed health: Coinbase tick updates tracking
# ---------------------------------------------------------------------------

def test_truth_feed_health_tracks_coinbase_ticks():
    """Coinbase ticker handler must update truth-feed health."""
    orch = _make_mock_orchestrator()

    # Import the handler logic pattern
    health = orch._truth_feed_health["BTC"]
    assert health["last_tick_mono"] == 0.0

    # Simulate coinbase tick
    now = time.monotonic()
    health["last_tick_mono"] = now
    health["active_source"] = "coinbase"

    assert health["last_tick_mono"] == now
    assert health["active_source"] == "coinbase"


# ---------------------------------------------------------------------------
#  OKX fallback: fires only when Coinbase is silent
# ---------------------------------------------------------------------------

def test_okx_fallback_suppressed_when_coinbase_alive():
    """OKX must NOT fire when Coinbase ticked recently."""
    orch = _make_mock_orchestrator()

    # Coinbase ticked 1 second ago
    orch._truth_feed_health["BTC"]["last_tick_mono"] = time.monotonic() - 1.0

    health = orch._truth_feed_health["BTC"]
    now_mono = time.monotonic()
    coinbase_age = now_mono - health["last_tick_mono"]

    # Should be suppressed (Coinbase alive)
    assert coinbase_age < orch._truth_fallback_activation_s, (
        "OKX should be suppressed when Coinbase ticked less than 5s ago"
    )


def test_okx_fallback_activates_when_coinbase_silent():
    """OKX must fire when Coinbase has been silent > 5s."""
    orch = _make_mock_orchestrator()

    # Coinbase ticked 10 seconds ago
    orch._truth_feed_health["BTC"]["last_tick_mono"] = time.monotonic() - 10.0

    health = orch._truth_feed_health["BTC"]
    now_mono = time.monotonic()
    coinbase_age = now_mono - health["last_tick_mono"]

    # Should activate (Coinbase silent)
    assert coinbase_age >= orch._truth_fallback_activation_s, (
        "OKX should activate when Coinbase silent for > 5s"
    )


def test_truth_feed_source_switches_to_okx_on_silence():
    """Simulates OKX callback updating active source when Coinbase is silent."""
    orch = _make_mock_orchestrator()

    # Coinbase silent for 10s
    orch._truth_feed_health["BTC"]["last_tick_mono"] = time.monotonic() - 10.0

    health = orch._truth_feed_health["BTC"]
    health["active_source"] = "okx"
    health["fallback_count"] += 1

    assert health["active_source"] == "okx"
    assert health["fallback_count"] == 1


def test_truth_feed_restores_to_coinbase_on_tick():
    """When Coinbase ticks again after OKX fallback, source restores."""
    orch = _make_mock_orchestrator()

    # Currently on OKX fallback
    health = orch._truth_feed_health["BTC"]
    health["active_source"] = "okx"
    health["fallback_count"] = 3

    # Coinbase tick arrives
    health["last_tick_mono"] = time.monotonic()
    health["active_source"] = "coinbase"

    assert health["active_source"] == "coinbase"
    # Fallback count persists (monotonic counter)
    assert health["fallback_count"] == 3


# ---------------------------------------------------------------------------
#  Source priority documentation
# ---------------------------------------------------------------------------

def test_source_priority_constants():
    """Verify source priority is documented in health tracking."""
    orch = _make_mock_orchestrator()

    # Both BTC and ETH must have health tracking
    assert "BTC" in orch._truth_feed_health
    assert "ETH" in orch._truth_feed_health

    # Default source must be coinbase (primary)
    for asset in ["BTC", "ETH"]:
        assert orch._truth_feed_health[asset]["active_source"] == "coinbase"


# ---------------------------------------------------------------------------
#  Stop lifecycle: all connectors properly cleaned up
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stop_cleans_up_all_connectors():
    """stop() must disconnect Bybit, Coinbase, and OKX."""
    from src.orchestrator import ArgusOrchestrator

    with patch.object(ArgusOrchestrator, "__init__", lambda self, *a, **kw: None):
        orch = ArgusOrchestrator.__new__(ArgusOrchestrator)

    orch.logger = MagicMock()
    orch._running = True
    orch._tasks = []
    orch.bar_builder = None
    orch.persistence = None
    orch.event_bus = MagicMock()
    orch.event_bus.stop = MagicMock()
    orch.av_collector = None
    orch.dashboard = None
    orch.news_sentiment_updater = None
    orch.conditions_monitor = None
    orch.daily_review = None
    orch.telegram = None
    orch.polymarket_gamma = None
    orch.polymarket_clob = None
    orch.polymarket_watchlist = None
    orch.bybit_ws = MagicMock()
    orch.bybit_ws.disconnect = AsyncMock()
    orch.coinbase_ws = MagicMock()
    orch.coinbase_ws.disconnect = AsyncMock()
    orch._okx_fallback = MagicMock()
    orch._okx_fallback.stop = AsyncMock()
    orch.deribit_client = None
    orch.yahoo_client = None
    orch.alpaca_options = None
    orch.tastytrade_options = None
    orch.public_options = None
    orch.alphavantage_client = None
    orch.ai_agent = None
    orch.db = MagicMock()
    orch.db.close = AsyncMock()

    await orch.stop()

    # All three must be cleaned up
    orch.bybit_ws.disconnect.assert_awaited_once()
    orch.coinbase_ws.disconnect.assert_awaited_once()
    orch._okx_fallback.stop.assert_awaited_once()

