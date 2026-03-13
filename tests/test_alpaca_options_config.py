"""
Tests for Alpaca Options Connector Improvements
=================================================

Tests for configurable feed and improved logging in AlpacaOptionsConnector.
"""

import pytest
from src.connectors.alpaca_options import (
    AlpacaOptionsConfig,
    AlpacaOptionsConnector,
    _now_ms,
    _date_to_ms,
    _compute_contract_id,
)


class TestAlpacaOptionsConfig:
    """Tests for AlpacaOptionsConfig."""

    def test_default_feed(self):
        """Default feed should be 'indicative'."""
        config = AlpacaOptionsConfig()
        assert config.feed == "indicative"

    def test_custom_feed(self):
        """Feed should be configurable."""
        config = AlpacaOptionsConfig(feed="opra")
        assert config.feed == "opra"

    def test_default_values(self):
        config = AlpacaOptionsConfig()
        assert config.base_url == "https://data.alpaca.markets"
        assert config.timeout_seconds == 30.0
        assert config.rate_limit_per_min == 200
        assert config.cache_ttl_seconds == 60


class TestAlpacaOptionsConnector:
    """Tests for AlpacaOptionsConnector initialization."""

    def test_provider_name(self):
        conn = AlpacaOptionsConnector()
        assert conn.PROVIDER == "alpaca"

    def test_health_status_initial(self):
        conn = AlpacaOptionsConnector()
        health = conn.get_health_status()
        assert health["provider"] == "alpaca"
        assert health["request_count"] == 0
        assert health["error_count"] == 0
        assert health["health"] == "ok"

    def test_config_feed_propagation(self):
        """Feed value should be stored on config."""
        config = AlpacaOptionsConfig(feed="opra")
        conn = AlpacaOptionsConnector(config=config)
        assert conn._config.feed == "opra"

    def test_sequence_id_monotonic(self):
        conn = AlpacaOptionsConnector()
        ids = [conn._next_sequence_id() for _ in range(5)]
        assert ids == [1, 2, 3, 4, 5]


class TestOCCSymbolParsing:
    """Tests for OCC symbol parsing."""

    def test_parse_put(self):
        conn = AlpacaOptionsConnector()
        strike, opt_type = conn._parse_occ_symbol("IBIT250221P00045000")
        assert strike == 45.0
        assert opt_type == "PUT"

    def test_parse_call(self):
        conn = AlpacaOptionsConnector()
        strike, opt_type = conn._parse_occ_symbol("SPY250321C00590000")
        assert strike == 590.0
        assert opt_type == "CALL"

    def test_parse_invalid(self):
        conn = AlpacaOptionsConnector()
        with pytest.raises(ValueError):
            conn._parse_occ_symbol("INVALID")


class TestHelpers:
    def test_now_ms(self):
        ms = _now_ms()
        assert isinstance(ms, int)
        assert ms > 0

    def test_date_to_ms(self):
        ms = _date_to_ms("2025-03-21")
        assert ms == 1742515200000

    def test_contract_id_deterministic(self):
        id1 = _compute_contract_id("IBIT250221P00045000")
        id2 = _compute_contract_id("IBIT250221P00045000")
        assert id1 == id2
        assert len(id1) == 16
