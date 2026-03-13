"""
Test: AllocationEngine + RiskEngine Integration
==================================================

Integration test verifying that AllocationEngine + RiskEngine produces
clamped allocations, clamp reasons, and attribution artifacts.

Also includes a "no lookahead" test ensuring that loaders respect
as_of_ts_ms and do not read future data.
"""

import json
import os
import tempfile
import pytest

from src.analysis.allocation_engine import AllocationEngine, AllocationConfig, Allocation
from src.analysis.sizing import Forecast
from src.analysis.drawdown_containment import DrawdownConfig
from src.analysis.correlation_exposure import CorrelationConfig
from src.analysis.greek_limits import GreekLimitsConfig
from src.analysis.portfolio_state import PortfolioState, build_portfolio_state_from_context
from src.analysis.risk_engine import RiskEngine, RiskEngineConfig, ClampReason
from src.analysis.risk_attribution import RiskAttribution, persist_risk_attribution
from src.analysis.tail_risk_scenario import TailRiskConfig


def _make_forecasts():
    """Create test forecasts."""
    return [
        Forecast(
            strategy_id="VRP_v1",
            instrument="IBIT",
            mu=0.15,
            sigma=0.30,
            edge_score=0.8,
            confidence=0.7,
        ),
        Forecast(
            strategy_id="Overnight_v1",
            instrument="SPY",
            mu=0.10,
            sigma=0.15,
            edge_score=0.6,
            confidence=0.8,
        ),
        Forecast(
            strategy_id="VRP_v2",
            instrument="IBIT",
            mu=0.12,
            sigma=0.25,
            edge_score=0.7,
            confidence=0.6,
        ),
    ]


def _make_portfolio_state(**overrides):
    defaults = dict(
        as_of_ts_ms=1700000000000,
        equity_usd=10_000.0,
    )
    defaults.update(overrides)
    return build_portfolio_state_from_context(**defaults)


def _make_risk_config(**overrides):
    defaults = dict(
        enabled=True,
        aggregate_exposure_cap=1.0,
        drawdown=DrawdownConfig(threshold_pct=0.10),
        correlation=CorrelationConfig(),
        greek_limits=GreekLimitsConfig(),
        tail_risk=TailRiskConfig(enabled_for_options=False),
        enforce_idempotence_check=False,
        enforce_monotone_check=True,
    )
    defaults.update(overrides)
    return RiskEngineConfig(**defaults)


class TestAllocationWithRiskEngine:
    """Integration: AllocationEngine + RiskEngine."""

    def test_basic_integration(self):
        """allocate_with_risk_engine returns valid tuple."""
        engine = AllocationEngine(
            config=AllocationConfig(kelly_fraction=0.25, per_play_cap=0.07),
            equity=10_000.0,
        )
        forecasts = _make_forecasts()
        state = _make_portfolio_state()
        risk_config = _make_risk_config()

        allocations, reasons, attribution = engine.allocate_with_risk_engine(
            forecasts=forecasts,
            portfolio_state=state,
            risk_config=risk_config,
        )

        assert isinstance(allocations, list)
        assert all(isinstance(a, Allocation) for a in allocations)
        assert isinstance(reasons, list)
        assert isinstance(attribution, RiskAttribution)

    def test_produces_clamp_reasons_with_drawdown(self):
        """Drawdown in portfolio → produces clamp reasons.

        The drawdown throttle works as a tighter aggregate cap
        (effective_cap = aggregate_cap * throttle).  For the throttle to
        produce clamp reasons, total allocation exposure must exceed the
        effective cap.  We use a tight aggregate cap so allocations exceed
        effective_cap = 0.10 * 0.75 = 0.075.
        """
        engine = AllocationEngine(
            config=AllocationConfig(kelly_fraction=0.25, per_play_cap=0.07),
            equity=10_000.0,
        )
        forecasts = _make_forecasts()
        state = _make_portfolio_state(
            equity_usd=8_500.0,
            peak_equity_usd=10_000.0,
        )
        risk_config = _make_risk_config(
            aggregate_exposure_cap=0.10,
            drawdown=DrawdownConfig(
                threshold_pct=0.10,
                throttle_mode="linear",
                k=5.0,
            ),
        )

        allocations, reasons, attribution = engine.allocate_with_risk_engine(
            forecasts=forecasts,
            portfolio_state=state,
            risk_config=risk_config,
        )

        # Should have drawdown throttle reasons
        dd_reasons = [r for r in reasons if r.constraint_id == "drawdown_throttle"]
        assert len(dd_reasons) > 0, "Expected drawdown clamp reasons"

        # Attribution should reflect the throttle
        assert attribution.portfolio_summary.drawdown_throttle_factor < 1.0

    def test_produces_attribution_artifact(self):
        """Risk attribution artifact has expected structure."""
        engine = AllocationEngine(
            config=AllocationConfig(kelly_fraction=0.25, per_play_cap=0.07),
            equity=10_000.0,
        )
        forecasts = _make_forecasts()
        state = _make_portfolio_state()
        risk_config = _make_risk_config()

        _, _, attribution = engine.allocate_with_risk_engine(
            forecasts=forecasts,
            portfolio_state=state,
            risk_config=risk_config,
        )

        attr_dict = attribution.to_dict()
        assert "per_strategy" in attr_dict
        assert "portfolio_summary" in attr_dict
        assert "clamp_summary" in attr_dict
        assert "config_hash" in attr_dict

    def test_attribution_persistence(self):
        """Risk attribution can be persisted to JSON."""
        engine = AllocationEngine(
            config=AllocationConfig(kelly_fraction=0.25),
            equity=10_000.0,
        )
        forecasts = _make_forecasts()
        state = _make_portfolio_state()
        risk_config = _make_risk_config()

        _, _, attribution = engine.allocate_with_risk_engine(
            forecasts=forecasts,
            portfolio_state=state,
            risk_config=risk_config,
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            persist_risk_attribution(attribution, path)
            with open(path) as f:
                data = json.load(f)
            assert "per_strategy" in data
            assert "portfolio_summary" in data
        finally:
            os.unlink(path)

    def test_aggregate_cap_with_risk_engine(self):
        """Aggregate cap enforced by risk engine."""
        engine = AllocationEngine(
            config=AllocationConfig(kelly_fraction=0.50, per_play_cap=0.20),
            equity=10_000.0,
        )
        forecasts = _make_forecasts()
        state = _make_portfolio_state()
        risk_config = _make_risk_config(aggregate_exposure_cap=0.10)

        allocations, reasons, _ = engine.allocate_with_risk_engine(
            forecasts=forecasts,
            portfolio_state=state,
            risk_config=risk_config,
        )

        total_exposure = sum(abs(a.weight) for a in allocations)
        assert total_exposure <= 0.10 + 1e-8, (
            f"Aggregate cap violated: {total_exposure}"
        )

    def test_config_hash_deterministic(self):
        """Same config → same hash."""
        config1 = _make_risk_config()
        config2 = _make_risk_config()
        assert config1.config_hash() == config2.config_hash()

    def test_config_hash_changes_with_params(self):
        """Different config → different hash."""
        config1 = _make_risk_config(aggregate_exposure_cap=0.5)
        config2 = _make_risk_config(aggregate_exposure_cap=1.0)
        assert config1.config_hash() != config2.config_hash()

    def test_clamp_reason_structure(self):
        """ClampReason objects have required fields."""
        engine = AllocationEngine(
            config=AllocationConfig(kelly_fraction=0.25),
            equity=10_000.0,
        )
        forecasts = _make_forecasts()
        state = _make_portfolio_state(
            equity_usd=8_500.0,
            peak_equity_usd=10_000.0,
        )
        risk_config = _make_risk_config(
            drawdown=DrawdownConfig(threshold_pct=0.10, throttle_mode="linear", k=5.0),
        )

        _, reasons, _ = engine.allocate_with_risk_engine(
            forecasts=forecasts,
            portfolio_state=state,
            risk_config=risk_config,
        )

        for r in reasons:
            assert isinstance(r, ClampReason)
            assert r.constraint_id
            assert r.allocation_id
            assert isinstance(r.before, dict)
            assert isinstance(r.after, dict)
            assert r.reason
            assert r.severity in ("info", "warn", "kill")

            # Verify serializable
            d = r.to_dict()
            assert isinstance(d, dict)


class TestNoLookahead:
    """Ensure data loaders respect as_of_ts_ms and don't read future data."""

    def test_strategy_returns_no_future_data(self):
        """get_strategy_returns_for_correlation respects as_of_ts_ms."""
        from datetime import datetime, timezone
        from src.analysis.correlation_exposure import get_strategy_returns_for_correlation

        as_of_ts = int(datetime(2024, 1, 15, tzinfo=timezone.utc).timestamp() * 1000)

        series = {
            "A": {
                "2024-01-01": 0.01,
                "2024-01-10": 0.02,
                "2024-01-15": 0.03,   # at boundary — should be included
                "2024-01-20": 0.04,   # future — must NOT be included
                "2024-02-01": 0.05,   # future — must NOT be included
            },
            "B": {
                "2024-01-01": -0.01,
                "2024-01-10": -0.02,
                "2024-01-15": -0.03,
                "2024-01-20": -0.04,
                "2024-02-01": -0.05,
            },
        }

        result = get_strategy_returns_for_correlation(
            series, ["A", "B"], as_of_ts, rolling_days=60,
        )

        # Should have exactly 3 dates (Jan 1, 10, 15)
        assert len(result["A"]) == 3
        assert len(result["B"]) == 3

        # No future returns should be present
        assert 0.04 not in result["A"]
        assert 0.05 not in result["A"]

    def test_portfolio_state_as_of_ts(self):
        """PortfolioState records as_of_ts_ms correctly."""
        ts = 1700000000000
        state = build_portfolio_state_from_context(
            as_of_ts_ms=ts,
            equity_usd=10_000.0,
        )
        assert state.as_of_ts_ms == ts

    def test_risk_engine_uses_state_timestamp(self):
        """RiskEngine clamp reasons use the state's timestamp."""
        engine = RiskEngine()
        allocs = [
            Allocation(
                strategy_id="test",
                instrument="SPY",
                weight=0.50,
                dollar_risk=5000.0,
            ),
        ]
        state = PortfolioState(
            as_of_ts_ms=1700000000000,
            equity_usd=10_000.0,
            current_drawdown_pct=0.15,
        )
        config = RiskEngineConfig(
            enabled=True,
            drawdown=DrawdownConfig(threshold_pct=0.10, throttle_mode="linear", k=5.0),
            tail_risk=TailRiskConfig(enabled_for_options=False),
            enforce_idempotence_check=False,
            enforce_monotone_check=False,
        )

        _, reasons, _ = engine.clamp(allocs, state, config)

        for r in reasons:
            assert r.ts_ms == 1700000000000, (
                f"ClampReason should use state timestamp, got {r.ts_ms}"
            )
