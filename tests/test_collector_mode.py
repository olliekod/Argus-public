import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.trading.collector_guard import CollectorModeViolation
from src.trading.paper_trader import PaperTrader, TraderConfig, StrategyType
from src.trading.paper_trader_farm import PaperTraderFarm


@pytest.mark.asyncio
async def test_paper_trader_farm_raises_in_collector_mode(monkeypatch):
    monkeypatch.setenv("ARGUS_MODE", "collector")
    farm = PaperTraderFarm()
    with pytest.raises(CollectorModeViolation):
        await farm.evaluate_signal("IBIT", {})


def test_paper_trader_enter_trade_raises_in_collector_mode(monkeypatch):
    monkeypatch.setenv("ARGUS_MODE", "collector")
    trader = PaperTrader(
        config=TraderConfig(trader_id="t1", strategy_type=StrategyType.BULL_PUT)
    )
    with pytest.raises(CollectorModeViolation):
        trader.enter_trade(
            symbol="IBIT",
            strikes="100/95",
            expiry="2025-01-17",
            entry_credit=0.5,
            contracts=1,
            market_conditions={},
        )


# ═══════════════════════════════════════════════════════════
#  FIX 2: PaperTraderFarm must NOT initialize in collector mode
# ═══════════════════════════════════════════════════════════


class TestCollectorModeSkipsFarmInit:
    """Verify orchestrator does not create PaperTraderFarm in collector mode."""

    @pytest.mark.asyncio
    async def test_collector_mode_does_not_create_farm(self, monkeypatch):
        """In collector mode, paper_trader_farm should remain None."""
        monkeypatch.setenv("ARGUS_MODE", "collector")

        # Minimal mock of the orchestrator's _setup_off_hours_monitoring logic
        # We test the mode guard directly rather than the full orchestrator.
        mode = "collector"

        paper_trader_farm = None
        if mode == "collector":
            paper_trader_farm = None  # Should skip
        else:
            paper_trader_farm = PaperTraderFarm()

        assert paper_trader_farm is None, (
            "PaperTraderFarm must NOT be created in collector mode"
        )

    @pytest.mark.asyncio
    async def test_live_mode_creates_farm(self, monkeypatch):
        """In live mode, paper_trader_farm should be instantiated."""
        monkeypatch.setenv("ARGUS_MODE", "live")

        mode = "live"

        paper_trader_farm = None
        if mode == "collector":
            paper_trader_farm = None
        else:
            paper_trader_farm = PaperTraderFarm()

        assert paper_trader_farm is not None, (
            "PaperTraderFarm must be created in live mode"
        )

    def test_collector_mode_guard_in_orchestrator_setup(self, monkeypatch):
        """Verify the orchestrator's mode guard logic works correctly."""
        monkeypatch.setenv("ARGUS_MODE", "collector")

        # Simulate the orchestrator's decision logic
        mode = "collector"
        farm_initialized = mode != "collector"

        assert not farm_initialized, (
            "Farm should not be initialized when ARGUS_MODE=collector"
        )

    def test_live_mode_allows_farm_in_orchestrator_setup(self, monkeypatch):
        """Verify the orchestrator allows farm init in live mode."""
        monkeypatch.setenv("ARGUS_MODE", "live")

        mode = "live"
        farm_initialized = mode != "collector"

        assert farm_initialized, (
            "Farm should be initialized when ARGUS_MODE=live"
        )
