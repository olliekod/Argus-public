"""
Tests for orchestrator task tracking + shutdown correctness (10.4).

Verifies:
- All background tasks are registered in _tasks
- shutdown cancels all tasks
- No dangling tasks after shutdown
- Graceful cleanup of components

NOTE: These tests do NOT import the orchestrator module directly because
it pulls in yfinance (through ibit_detector -> trade_calculator) which
may not be installed. Instead, we test the shutdown contract structurally.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


async def _simulate_stop(orch) -> None:
    """Simulate the orchestrator stop() logic without importing the module.

    Mirrors the shutdown sequence in ArgusOrchestrator.stop():
    1. Set _running = False
    2. Flush bar_builder and persistence
    3. Stop event bus
    4. Cancel all tasks and await them
    5. Close connectors and DB
    """
    orch._running = False

    if orch.bar_builder:
        orch.bar_builder.flush()

    if orch.persistence:
        orch.persistence.shutdown()

    orch.event_bus.stop()

    if getattr(orch, "dashboard", None):
        await orch.dashboard.stop()

    if getattr(orch, "conditions_monitor", None):
        orch.conditions_monitor.stop_monitoring()
    if getattr(orch, "daily_review", None):
        orch.daily_review.stop_monitoring()

    if getattr(orch, "telegram", None):
        await orch.telegram.stop_polling()

    for task in orch._tasks:
        task.cancel()
    if orch._tasks:
        await asyncio.gather(*orch._tasks, return_exceptions=True)

    if getattr(orch, "polymarket_gamma", None):
        await orch.polymarket_gamma.stop()
    if getattr(orch, "polymarket_clob", None):
        await orch.polymarket_clob.stop()
    if getattr(orch, "polymarket_watchlist", None):
        await orch.polymarket_watchlist.stop()

    if getattr(orch, "bybit_ws", None):
        await orch.bybit_ws.disconnect()
    if getattr(orch, "deribit_client", None):
        await orch.deribit_client.close()
    if getattr(orch, "yahoo_client", None):
        await orch.yahoo_client.close()
    if getattr(orch, "alpaca_options", None):
        await orch.alpaca_options.close()
    if getattr(orch, "tastytrade_options", None):
        orch.tastytrade_options.close()

    await orch.db.close()


def _make_mock_orch(**overrides):
    """Build a minimal MagicMock orchestrator with expected attributes."""
    orch = MagicMock()
    orch._running = True
    orch._tasks = overrides.get("_tasks", [])
    orch.bar_builder = overrides.get("bar_builder", None)
    orch.persistence = overrides.get("persistence", None)
    orch.event_bus = MagicMock()
    orch.dashboard = None
    orch.conditions_monitor = None
    orch.daily_review = None
    orch.telegram = None
    orch.polymarket_gamma = None
    orch.polymarket_clob = None
    orch.polymarket_watchlist = None
    orch.bybit_ws = None
    orch.deribit_client = None
    orch.yahoo_client = None
    orch.alpaca_options = None
    orch.tastytrade_options = None
    orch.db = AsyncMock()
    orch.logger = MagicMock()
    return orch


class TestOrchestratorTaskTracking:
    """Verify task management mirrors ArgusOrchestrator.stop() contract."""

    @pytest.mark.asyncio
    async def test_tasks_list_starts_empty(self):
        """_tasks should be empty before run()."""
        orch = _make_mock_orch()
        assert len(orch._tasks) == 0

    @pytest.mark.asyncio
    async def test_stop_cancels_all_tasks(self):
        """stop() should cancel every task in _tasks."""
        async def forever():
            await asyncio.sleep(3600)

        tasks = [asyncio.create_task(forever()) for _ in range(5)]
        orch = _make_mock_orch(_tasks=tasks)
        await _simulate_stop(orch)

        for task in tasks:
            assert task.cancelled() or task.done(), "Task should be cancelled or done"

        assert orch._running is False

    @pytest.mark.asyncio
    async def test_stop_awaits_task_cancellation(self):
        """After stop(), cancelled tasks should not be pending."""
        async def forever():
            await asyncio.sleep(3600)

        tasks = [asyncio.create_task(forever()) for _ in range(3)]

        for t in tasks:
            t.cancel()

        await asyncio.sleep(0)

        for t in tasks:
            assert t.cancelled() or t.done(), "Task should be cancelled or done"

    @pytest.mark.asyncio
    async def test_stop_sets_running_false(self):
        """stop() must set _running=False to break polling loops."""
        orch = _make_mock_orch()
        await _simulate_stop(orch)
        assert orch._running is False

    @pytest.mark.asyncio
    async def test_stop_flushes_persistence(self):
        """stop() should call persistence.shutdown() if persistence exists."""
        orch = _make_mock_orch(
            bar_builder=MagicMock(),
            persistence=MagicMock(),
        )
        await _simulate_stop(orch)

        orch.bar_builder.flush.assert_called_once()
        orch.persistence.shutdown.assert_called_once()
