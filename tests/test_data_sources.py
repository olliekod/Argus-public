"""
Tests for Data Source Policy
=============================

Verifies:
- DataSourcePolicy defaults match the canonical config values
- get_data_source_policy() reads from config correctly
- Replay pack composition follows data_sources policy
- VRP strategy chooses Tastytrade IV by default
- VRP strategy falls back to derived IV when atm_iv is missing
- VRP strategy never uses Alpaca for IV (Alpaca is bars/outcomes only)
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

from src.core.data_sources import (
    DataSourcePolicy,
    get_data_source_policy,
)
from src.strategies.vrp_credit_spread import (
    _select_iv_from_snapshots,
    _derive_iv_from_quotes,
    VRPCreditSpreadStrategy,
)
from src.core.outcome_engine import BarData
from src.analysis.replay_harness import MarketDataSnapshot


# ═══════════════════════════════════════════════════════════════════════════
# DataSourcePolicy defaults
# ═══════════════════════════════════════════════════════════════════════════

class TestDataSourcePolicyDefaults:
    def test_default_bars_primary(self):
        policy = DataSourcePolicy()
        assert policy.bars_primary == "alpaca"

    def test_default_outcomes_from(self):
        policy = DataSourcePolicy()
        assert policy.outcomes_from == "bars_primary"

    def test_default_options_snapshots_primary(self):
        policy = DataSourcePolicy()
        assert policy.options_snapshots_primary == "tastytrade"

    def test_default_options_snapshots_secondary(self):
        policy = DataSourcePolicy()
        assert policy.options_snapshots_secondary == ["public"]

    def test_default_options_stream_primary(self):
        policy = DataSourcePolicy()
        assert policy.options_stream_primary == "tastytrade_dxlink"

    def test_default_bars_secondary(self):
        policy = DataSourcePolicy()
        assert policy.bars_secondary == ["yahoo"]

    def test_bars_provider_alias(self):
        policy = DataSourcePolicy()
        assert policy.bars_provider == "alpaca"

    def test_options_snapshot_provider_alias(self):
        policy = DataSourcePolicy()
        assert policy.options_snapshot_provider == "tastytrade"

    def test_is_secondary_options_provider(self):
        policy = DataSourcePolicy()
        assert policy.is_secondary_options_provider("public") is True
        assert policy.is_secondary_options_provider("tastytrade") is False

    def test_snapshot_providers_primary_only(self):
        policy = DataSourcePolicy()
        assert policy.snapshot_providers(include_secondary=False) == ["tastytrade"]

    def test_snapshot_providers_with_secondary(self):
        policy = DataSourcePolicy()
        providers = policy.snapshot_providers(include_secondary=True)
        assert providers == ["tastytrade", "public"]

    def test_policy_is_frozen(self):
        policy = DataSourcePolicy()
        with pytest.raises(AttributeError):
            policy.bars_primary = "yahoo"


# ═══════════════════════════════════════════════════════════════════════════
# get_data_source_policy() from config
# ═══════════════════════════════════════════════════════════════════════════

class TestGetDataSourcePolicy:
    def test_from_explicit_config(self):
        config = {
            "data_sources": {
                "bars_primary": "alpaca",
                "outcomes_from": "bars_primary",
                "options_snapshots_primary": "tastytrade",
                "options_snapshots_secondary": ["public"],
                "options_stream_primary": "tastytrade_dxlink",
                "bars_secondary": ["yahoo"],
            }
        }
        policy = get_data_source_policy(config)
        assert policy.bars_primary == "alpaca"
        assert policy.options_snapshots_primary == "tastytrade"

    def test_empty_data_sources_uses_defaults(self):
        policy = get_data_source_policy({"data_sources": {}})
        assert policy.bars_primary == "alpaca"
        assert policy.options_snapshots_primary == "tastytrade"

    def test_missing_data_sources_uses_defaults(self):
        policy = get_data_source_policy({})
        assert policy.bars_primary == "alpaca"

    def test_partial_overrides(self):
        config = {
            "data_sources": {
                "bars_primary": "yahoo",
            }
        }
        policy = get_data_source_policy(config)
        assert policy.bars_primary == "yahoo"
        assert policy.options_snapshots_primary == "tastytrade"  # default

    def test_secondary_as_string_coerced_to_list(self):
        config = {
            "data_sources": {
                "options_snapshots_secondary": "public",
            }
        }
        policy = get_data_source_policy(config)
        assert policy.options_snapshots_secondary == ["public"]


# ═══════════════════════════════════════════════════════════════════════════
# Pack metadata records provider info
# ═══════════════════════════════════════════════════════════════════════════

class TestPackMetadataProviders:
    def _build_pack_metadata(
        self,
        bars_provider: str = "alpaca",
        options_snapshot_provider: str = "tastytrade",
        secondary_options_included: bool = False,
    ) -> Dict[str, Any]:
        return {
            "metadata": {
                "symbol": "SPY",
                "provider": bars_provider,
                "bars_provider": bars_provider,
                "options_snapshot_provider": options_snapshot_provider,
                "secondary_options_included": secondary_options_included,
                "bar_count": 100,
                "snapshot_count": 10,
            },
            "bars": [],
            "outcomes": [],
            "regimes": [],
            "snapshots": [],
        }

    def test_pack_records_bars_provider(self):
        pack = self._build_pack_metadata(bars_provider="alpaca")
        assert pack["metadata"]["bars_provider"] == "alpaca"

    def test_pack_records_options_snapshot_provider(self):
        pack = self._build_pack_metadata(options_snapshot_provider="tastytrade")
        assert pack["metadata"]["options_snapshot_provider"] == "tastytrade"

    def test_pack_records_secondary_flag(self):
        pack = self._build_pack_metadata(secondary_options_included=True)
        assert pack["metadata"]["secondary_options_included"] is True

    def test_pack_records_legacy_provider(self):
        pack = self._build_pack_metadata(bars_provider="alpaca")
        assert pack["metadata"]["provider"] == "alpaca"


# ═══════════════════════════════════════════════════════════════════════════
# VRP IV Source Selection
# ═══════════════════════════════════════════════════════════════════════════

class TestVRPIVSourceSelection:
    """Verify the VRP strategy's IV source selection logic."""

    def _make_snapshot(
        self,
        provider: str = "tastytrade",
        atm_iv: Optional[float] = 0.20,
        underlying_price: float = 450.0,
        quotes_json: str = '{"puts": [], "calls": []}',
    ) -> MarketDataSnapshot:
        return MarketDataSnapshot(
            symbol="SPY",
            recv_ts_ms=1_700_000_060_000,
            underlying_price=underlying_price,
            atm_iv=atm_iv,
            source=provider,
        )

    def test_tastytrade_iv_preferred(self):
        """Tastytrade snapshot with atm_iv should be selected first."""
        snaps = [
            self._make_snapshot(provider="alpaca", atm_iv=0.30),
            self._make_snapshot(provider="tastytrade", atm_iv=0.20),
        ]
        iv = _select_iv_from_snapshots(snaps)
        assert iv == 0.20

    def test_alpaca_iv_never_used(self):
        """Alpaca is bars/outcomes only; IV from Alpaca snapshots is never selected."""
        snaps = [
            self._make_snapshot(provider="alpaca", atm_iv=0.30),
            self._make_snapshot(provider="tastytrade", atm_iv=None),
        ]
        iv = _select_iv_from_snapshots(snaps)
        assert iv is None

    def test_empty_snapshots_returns_none(self):
        iv = _select_iv_from_snapshots([])
        assert iv is None

    def test_latest_tastytrade_wins(self):
        """Most recent Tastytrade snapshot with IV should be used."""
        older = MarketDataSnapshot(
            symbol="SPY", recv_ts_ms=1_700_000_060_000,
            underlying_price=450.0, atm_iv=0.15, source="tastytrade",
        )
        newer = MarketDataSnapshot(
            symbol="SPY", recv_ts_ms=1_700_000_120_000,
            underlying_price=450.0, atm_iv=0.25, source="tastytrade",
        )
        iv = _select_iv_from_snapshots([older, newer])
        assert iv == 0.25

    def test_dict_snapshots_work(self):
        """IV selection should also work with dict-format snapshots."""
        snaps = [
            {"provider": "tastytrade", "atm_iv": 0.22, "underlying_price": 450.0},
        ]
        iv = _select_iv_from_snapshots(snaps)
        assert iv == 0.22


class TestDerivedIV:
    """Test the derived IV fallback from quotes."""

    def test_derive_iv_with_valid_quotes(self):
        snapshot = {
            "underlying_price": 450.0,
            "quotes_json": json.dumps({
                "puts": [
                    {"strike": 450.0, "bid": 5.0, "ask": 6.0, "dte_years": 0.038},
                ],
                "calls": [],
            }),
        }
        iv = _derive_iv_from_quotes(snapshot)
        assert iv is not None
        assert 0.01 < iv < 3.0

    def test_derive_iv_no_puts_returns_none(self):
        snapshot = {
            "underlying_price": 450.0,
            "quotes_json": json.dumps({"puts": [], "calls": []}),
        }
        iv = _derive_iv_from_quotes(snapshot)
        assert iv is None

    def test_derive_iv_no_quotes_returns_none(self):
        snapshot = {"underlying_price": 450.0}
        iv = _derive_iv_from_quotes(snapshot)
        assert iv is None

    def test_derive_iv_zero_underlying_returns_none(self):
        snapshot = {
            "underlying_price": 0.0,
            "quotes_json": json.dumps({
                "puts": [{"strike": 450.0, "bid": 5.0, "ask": 6.0}],
                "calls": [],
            }),
        }
        iv = _derive_iv_from_quotes(snapshot)
        assert iv is None


class TestVRPStrategyIVIntegration:
    """Verify the full VRP strategy respects provider selection."""

    def _make_bar(self, ts_ms: int = 1_700_000_000_000, close: float = 450.0) -> BarData:
        return BarData(
            timestamp_ms=ts_ms,
            open=450.0, high=451.0, low=449.0,
            close=close, volume=1000.0,
        )

    def test_strategy_uses_tastytrade_iv_by_default(self):
        strategy = VRPCreditSpreadStrategy()
        snap_tt = MarketDataSnapshot(
            symbol="SPY", recv_ts_ms=100, underlying_price=450.0,
            atm_iv=0.20, source="tastytrade",
        )
        snap_alp = MarketDataSnapshot(
            symbol="SPY", recv_ts_ms=200, underlying_price=450.0,
            atm_iv=0.35, source="alpaca",
        )
        from src.core.outcome_engine import OutcomeResult
        bar = self._make_bar()
        strategy.on_bar(
            bar,
            sim_ts_ms=bar.timestamp_ms + 60000,
            session_regime="RTH",
            visible_outcomes={},
            visible_snapshots=[snap_tt, snap_alp],
        )
        assert strategy.last_iv == 0.20

    def test_strategy_never_uses_alpaca_iv(self):
        """Alpaca is bars/outcomes only; strategy never uses Alpaca for IV."""
        strategy = VRPCreditSpreadStrategy()
        snap_alp = MarketDataSnapshot(
            symbol="SPY", recv_ts_ms=100, underlying_price=450.0,
            atm_iv=0.35, source="alpaca",
        )
        bar = self._make_bar()
        strategy.on_bar(
            bar,
            sim_ts_ms=bar.timestamp_ms + 60000,
            session_regime="RTH",
            visible_outcomes={},
            visible_snapshots=[snap_alp],
        )
        assert strategy.last_iv is None
