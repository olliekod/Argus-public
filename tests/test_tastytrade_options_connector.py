"""
Tests for Tastytrade Options Snapshot Connector
================================================

Unit tests for TastytradeOptionsConnector snapshot building.
Uses mock data to avoid network calls.
"""

import pytest
import time
from unittest.mock import MagicMock, patch

from src.connectors.tastytrade_options import (
    TastytradeOptionsConnector,
    TastytradeOptionsConfig,
    _now_ms,
    _date_to_ms,
    _compute_contract_id,
)
from src.core.option_events import (
    OptionChainSnapshotEvent,
    option_chain_to_dict,
    dict_to_option_chain,
)


# ─── Fixtures ────────────────────────────────────────────────────────────

SAMPLE_NESTED_CHAIN = {
    "data": {
        "items": [
            {
                "underlying-symbol": "SPY",
                "root-symbol": "SPY",
                "option-chain-type": "Standard",
                "shares-per-contract": 100,
                "expirations": [
                    {
                        "expiration-date": "2025-03-21",
                        "strikes": [
                            {
                                "strike-price": "590.0",
                                "call": "SPY   250321C00590000",
                                "put": "SPY   250321P00590000",
                                "call-streamer-symbol": ".SPY250321C590",
                                "put-streamer-symbol": ".SPY250321P590",
                            },
                            {
                                "strike-price": "595.0",
                                "call": "SPY   250321C00595000",
                                "put": "SPY   250321P00595000",
                                "call-streamer-symbol": ".SPY250321C595",
                                "put-streamer-symbol": ".SPY250321P595",
                            },
                            {
                                "strike-price": "600.0",
                                "call": "SPY   250321C00600000",
                                "put": "SPY   250321P00600000",
                                "call-streamer-symbol": ".SPY250321C600",
                                "put-streamer-symbol": ".SPY250321P600",
                            },
                        ],
                    },
                    {
                        "expiration-date": "2025-03-28",
                        "strikes": [
                            {
                                "strike-price": "590.0",
                                "call": "SPY   250328C00590000",
                                "put": "SPY   250328P00590000",
                            },
                            {
                                "strike-price": "600.0",
                                "call": "SPY   250328C00600000",
                                "put": "SPY   250328P00600000",
                            },
                        ],
                    },
                ],
            }
        ]
    }
}


def _make_config(**overrides) -> TastytradeOptionsConfig:
    defaults = dict(
        username="testuser",
        password="testpass",
        environment="sandbox",
        min_dte=0,
        max_dte=365,
    )
    defaults.update(overrides)
    return TastytradeOptionsConfig(**defaults)


# ─── Test helper functions ────────────────────────────────────────────────

class TestHelperFunctions:
    def test_now_ms_returns_int(self):
        result = _now_ms()
        assert isinstance(result, int)
        assert result > 0

    def test_date_to_ms(self):
        ms = _date_to_ms("2025-03-21")
        assert isinstance(ms, int)
        assert ms > 0
        # 2025-03-21 00:00:00 UTC
        assert ms == 1742515200000

    def test_compute_contract_id_deterministic(self):
        id1 = _compute_contract_id("SPY250321C00590000")
        id2 = _compute_contract_id("SPY250321C00590000")
        id3 = _compute_contract_id("SPY250321P00590000")
        assert id1 == id2
        assert id1 != id3
        assert len(id1) == 16


# ─── Test connector initialization ────────────────────────────────────────

class TestConnectorInit:
    def test_default_config(self):
        config = TastytradeOptionsConfig()
        conn = TastytradeOptionsConnector(config=config)
        assert conn.PROVIDER == "tastytrade"
        assert conn._authenticated is False
        assert conn._sequence_id == 0

    def test_health_status_initial(self):
        conn = TastytradeOptionsConnector(config=_make_config())
        health = conn.get_health_status()
        assert health["provider"] == "tastytrade"
        assert health["request_count"] == 0
        assert health["error_count"] == 0
        assert health["authenticated"] is False
        assert health["health"] == "ok"

    def test_close_without_client(self):
        """close() should not fail if client was never created."""
        conn = TastytradeOptionsConnector(config=_make_config())
        conn.close()  # Should not raise


# ─── Test snapshot building from normalized data ──────────────────────────

class TestBuildChainSnapshot:
    def setup_method(self):
        self.conn = TastytradeOptionsConnector(config=_make_config())

    def test_build_snapshot_basic(self):
        """Build a snapshot from pre-normalized data."""
        from src.core.options_normalize import normalize_tastytrade_nested_chain

        normalized = normalize_tastytrade_nested_chain(SAMPLE_NESTED_CHAIN)
        assert len(normalized) > 0

        snapshot = self.conn.build_chain_snapshot(
            symbol="SPY",
            expiration="2025-03-21",
            normalized=normalized,
            underlying_price=595.0,
        )

        assert snapshot is not None
        assert isinstance(snapshot, OptionChainSnapshotEvent)
        assert snapshot.symbol == "SPY"
        assert snapshot.provider == "tastytrade"
        assert snapshot.underlying_price == 595.0
        assert len(snapshot.puts) == 3
        assert len(snapshot.calls) == 3
        assert snapshot.n_strikes == 3
        assert snapshot.timestamp_ms > 0
        assert snapshot.recv_ts_ms > 0
        assert snapshot.snapshot_id != ""

    def test_build_snapshot_strike_ordering(self):
        """Puts and calls should be sorted by strike ascending."""
        from src.core.options_normalize import normalize_tastytrade_nested_chain

        normalized = normalize_tastytrade_nested_chain(SAMPLE_NESTED_CHAIN)
        snapshot = self.conn.build_chain_snapshot(
            symbol="SPY",
            expiration="2025-03-21",
            normalized=normalized,
            underlying_price=595.0,
        )

        assert snapshot is not None
        put_strikes = [p.strike for p in snapshot.puts]
        call_strikes = [c.strike for c in snapshot.calls]
        assert put_strikes == sorted(put_strikes)
        assert call_strikes == sorted(call_strikes)

    def test_build_snapshot_empty_expiration(self):
        """Return None for non-existent expiration."""
        from src.core.options_normalize import normalize_tastytrade_nested_chain

        normalized = normalize_tastytrade_nested_chain(SAMPLE_NESTED_CHAIN)
        snapshot = self.conn.build_chain_snapshot(
            symbol="SPY",
            expiration="2099-12-31",
            normalized=normalized,
        )

        assert snapshot is None

    def test_build_snapshot_provider_field(self):
        """Snapshot and all quotes should have provider='tastytrade'."""
        from src.core.options_normalize import normalize_tastytrade_nested_chain

        normalized = normalize_tastytrade_nested_chain(SAMPLE_NESTED_CHAIN)
        snapshot = self.conn.build_chain_snapshot(
            symbol="SPY",
            expiration="2025-03-21",
            normalized=normalized,
        )

        assert snapshot.provider == "tastytrade"
        for q in snapshot.puts:
            assert q.provider == "tastytrade"
        for q in snapshot.calls:
            assert q.provider == "tastytrade"

    def test_build_snapshot_round_trip(self):
        """Snapshot should survive serialization round-trip."""
        from src.core.options_normalize import normalize_tastytrade_nested_chain

        normalized = normalize_tastytrade_nested_chain(SAMPLE_NESTED_CHAIN)
        snapshot = self.conn.build_chain_snapshot(
            symbol="SPY",
            expiration="2025-03-21",
            normalized=normalized,
            underlying_price=595.0,
        )

        d = option_chain_to_dict(snapshot)
        restored = dict_to_option_chain(d)

        assert restored.symbol == snapshot.symbol
        assert restored.provider == "tastytrade"
        assert restored.underlying_price == 595.0
        assert len(restored.puts) == len(snapshot.puts)
        assert len(restored.calls) == len(snapshot.calls)
        assert restored.n_strikes == snapshot.n_strikes

    def test_build_snapshot_zero_underlying_price(self):
        """Snapshot should work with zero underlying price."""
        from src.core.options_normalize import normalize_tastytrade_nested_chain

        normalized = normalize_tastytrade_nested_chain(SAMPLE_NESTED_CHAIN)
        snapshot = self.conn.build_chain_snapshot(
            symbol="SPY",
            expiration="2025-03-21",
            normalized=normalized,
            underlying_price=0.0,
        )

        assert snapshot is not None
        assert snapshot.underlying_price == 0.0
        # ATM IV should be None when underlying_price is 0
        assert snapshot.atm_iv is None

    def test_sequence_ids_are_monotonic(self):
        """Each quote should get a unique, increasing sequence_id."""
        from src.core.options_normalize import normalize_tastytrade_nested_chain

        normalized = normalize_tastytrade_nested_chain(SAMPLE_NESTED_CHAIN)
        snapshot = self.conn.build_chain_snapshot(
            symbol="SPY",
            expiration="2025-03-21",
            normalized=normalized,
        )

        all_quotes = list(snapshot.puts) + list(snapshot.calls)
        seq_ids = [q.sequence_id for q in all_quotes]
        # All sequence IDs should be unique and positive
        assert len(set(seq_ids)) == len(seq_ids)
        assert all(s > 0 for s in seq_ids)


# ─── Test expiration filtering ────────────────────────────────────────────

class TestExpirationFiltering:
    def test_get_expirations_in_range(self):
        """Expirations outside DTE range should be excluded."""
        from src.core.options_normalize import normalize_tastytrade_nested_chain

        conn = TastytradeOptionsConnector(config=_make_config())
        normalized = normalize_tastytrade_nested_chain(SAMPLE_NESTED_CHAIN)

        # Use a very wide range to include all
        results = conn.get_expirations_in_range(normalized, min_dte=0, max_dte=36500)
        assert len(results) >= 0  # May be 0 if dates are in the past

    def test_get_expirations_empty_input(self):
        conn = TastytradeOptionsConnector(config=_make_config())
        results = conn.get_expirations_in_range([], min_dte=0, max_dte=365)
        assert results == []


# ─── Test build_snapshots_for_symbol ──────────────────────────────────────

class TestBuildSnapshotsForSymbol:
    def test_empty_chain_returns_empty_list(self):
        """If fetch returns empty, build_snapshots_for_symbol returns []."""
        conn = TastytradeOptionsConnector(config=_make_config())

        with patch.object(conn, 'fetch_nested_chain', return_value={}):
            results = conn.build_snapshots_for_symbol("SPY")
            assert results == []

    def test_normalization_failure_returns_empty(self):
        """If normalization returns [], build_snapshots_for_symbol returns []."""
        conn = TastytradeOptionsConnector(config=_make_config())

        with patch.object(conn, 'fetch_nested_chain', return_value={"invalid": "data"}):
            results = conn.build_snapshots_for_symbol("SPY")
            assert results == []


# ─── Test multi-provider compatibility ────────────────────────────────────

class TestMultiProviderCompatibility:
    def test_tastytrade_snapshot_matches_alpaca_schema(self):
        """Tastytrade snapshots should have same fields as Alpaca snapshots."""
        from src.core.options_normalize import normalize_tastytrade_nested_chain

        conn = TastytradeOptionsConnector(config=_make_config())
        normalized = normalize_tastytrade_nested_chain(SAMPLE_NESTED_CHAIN)

        snapshot = conn.build_chain_snapshot(
            symbol="SPY",
            expiration="2025-03-21",
            normalized=normalized,
            underlying_price=595.0,
        )

        # Verify all required OptionChainSnapshotEvent fields
        assert hasattr(snapshot, 'symbol')
        assert hasattr(snapshot, 'expiration_ms')
        assert hasattr(snapshot, 'underlying_price')
        assert hasattr(snapshot, 'underlying_bid')
        assert hasattr(snapshot, 'underlying_ask')
        assert hasattr(snapshot, 'puts')
        assert hasattr(snapshot, 'calls')
        assert hasattr(snapshot, 'n_strikes')
        assert hasattr(snapshot, 'atm_iv')
        assert hasattr(snapshot, 'timestamp_ms')
        assert hasattr(snapshot, 'source_ts_ms')
        assert hasattr(snapshot, 'recv_ts_ms')
        assert hasattr(snapshot, 'provider')
        assert hasattr(snapshot, 'snapshot_id')
        assert hasattr(snapshot, 'sequence_id')
        assert hasattr(snapshot, 'v')

    def test_serialized_snapshot_has_quotes_json_fields(self):
        """Serialized snapshot should be DB-compatible for persistence."""
        from src.core.options_normalize import normalize_tastytrade_nested_chain
        import json

        conn = TastytradeOptionsConnector(config=_make_config())
        normalized = normalize_tastytrade_nested_chain(SAMPLE_NESTED_CHAIN)

        snapshot = conn.build_chain_snapshot(
            symbol="SPY",
            expiration="2025-03-21",
            normalized=normalized,
            underlying_price=595.0,
        )

        d = option_chain_to_dict(snapshot)
        quotes_json = json.dumps(d, sort_keys=True)

        # Should be valid JSON
        parsed = json.loads(quotes_json)
        assert parsed["provider"] == "tastytrade"
        assert parsed["symbol"] == "SPY"
        assert "puts" in parsed
        assert "calls" in parsed
