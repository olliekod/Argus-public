"""
Tests for Option Events Module
==============================

Unit tests for Phase 3B option event schemas and serialization.
"""

import pytest
from src.core.option_events import (
    OptionContractEvent,
    OptionQuoteEvent,
    OptionChainSnapshotEvent,
    option_contract_to_dict,
    option_quote_to_dict,
    option_chain_to_dict,
    dict_to_option_contract,
    dict_to_option_quote,
    dict_to_option_chain,
    compute_snapshot_id,
    compute_chain_hash,
)


class TestOptionContractEvent:
    """Tests for OptionContractEvent."""
    
    def test_create_contract(self):
        """Test creating an option contract event."""
        contract = OptionContractEvent(
            symbol="IBIT",
            contract_id="abc123",
            option_symbol="IBIT250221P00045000",
            strike=45.0,
            expiration_ms=1740124800000,
            option_type="PUT",
            provider="alpaca",
        )
        
        assert contract.symbol == "IBIT"
        assert contract.strike == 45.0
        assert contract.option_type == "PUT"
        assert contract.multiplier == 100
        assert contract.style == "american"
    
    def test_contract_round_trip(self):
        """Test serialization round-trip."""
        contract = OptionContractEvent(
            symbol="BITO",
            contract_id="def456",
            option_symbol="BITO250228P00020000",
            strike=20.0,
            expiration_ms=1740729600000,
            option_type="PUT",
            provider="tradier",
            timestamp_ms=1700000000000,
        )
        
        d = option_contract_to_dict(contract)
        restored = dict_to_option_contract(d)
        
        assert restored.symbol == contract.symbol
        assert restored.contract_id == contract.contract_id
        assert restored.strike == contract.strike
        assert restored.expiration_ms == contract.expiration_ms
        assert restored.option_type == contract.option_type
        assert restored.provider == contract.provider


class TestOptionQuoteEvent:
    """Tests for OptionQuoteEvent."""
    
    def test_create_quote(self):
        """Test creating an option quote event."""
        quote = OptionQuoteEvent(
            contract_id="abc123",
            symbol="IBIT",
            strike=45.0,
            expiration_ms=1740124800000,
            option_type="PUT",
            bid=1.50,
            ask=1.60,
            last=1.55,
            mid=1.55,
            volume=1000,
            open_interest=5000,
            iv=0.45,
            delta=-0.18,
            provider="alpaca",
        )
        
        assert quote.bid == 1.50
        assert quote.ask == 1.60
        assert quote.delta == -0.18
        assert quote.iv == 0.45
    
    def test_quote_with_optional_greeks(self):
        """Test quote with missing Greeks."""
        quote = OptionQuoteEvent(
            contract_id="abc123",
            symbol="IBIT",
            strike=45.0,
            expiration_ms=1740124800000,
            option_type="PUT",
            bid=1.50,
            ask=1.60,
        )
        
        assert quote.iv is None
        assert quote.delta is None
        assert quote.gamma is None
    
    def test_quote_round_trip(self):
        """Test serialization round-trip."""
        quote = OptionQuoteEvent(
            contract_id="abc123",
            symbol="IBIT",
            strike=45.0,
            expiration_ms=1740124800000,
            option_type="PUT",
            bid=1.50,
            ask=1.60,
            iv=0.45,
            delta=-0.18,
            timestamp_ms=1700000000000,
            provider="alpaca",
        )
        
        d = option_quote_to_dict(quote)
        restored = dict_to_option_quote(d)
        
        assert restored.contract_id == quote.contract_id
        assert restored.bid == quote.bid
        assert restored.ask == quote.ask
        assert restored.iv == quote.iv
        assert restored.delta == quote.delta


class TestOptionChainSnapshotEvent:
    """Tests for OptionChainSnapshotEvent."""
    
    def test_create_snapshot(self):
        """Test creating a chain snapshot."""
        put1 = OptionQuoteEvent(
            contract_id="p1", symbol="IBIT", strike=44.0,
            expiration_ms=1740124800000, option_type="PUT", bid=1.0, ask=1.1,
        )
        put2 = OptionQuoteEvent(
            contract_id="p2", symbol="IBIT", strike=45.0,
            expiration_ms=1740124800000, option_type="PUT", bid=1.5, ask=1.6,
        )
        
        snapshot = OptionChainSnapshotEvent(
            symbol="IBIT",
            expiration_ms=1740124800000,
            underlying_price=50.0,
            puts=(put1, put2),
            calls=(),
            n_strikes=2,
            timestamp_ms=1700000000000,
            provider="alpaca",
            snapshot_id="IBIT_1740124800000_1700000000000",
        )
        
        assert snapshot.symbol == "IBIT"
        assert len(snapshot.puts) == 2
        assert snapshot.puts[0].strike == 44.0  # Sorted ascending
    
    def test_snapshot_is_immutable(self):
        """Test snapshot immutability."""
        snapshot = OptionChainSnapshotEvent(
            symbol="IBIT",
            expiration_ms=1740124800000,
            underlying_price=50.0,
            puts=(),
            calls=(),
            timestamp_ms=1700000000000,
        )
        
        with pytest.raises(AttributeError):
            snapshot.symbol = "BITO"
    
    def test_snapshot_round_trip(self):
        """Test serialization round-trip."""
        put = OptionQuoteEvent(
            contract_id="p1", symbol="IBIT", strike=45.0,
            expiration_ms=1740124800000, option_type="PUT", bid=1.5, ask=1.6,
            iv=0.45, delta=-0.18, timestamp_ms=1700000000000, provider="alpaca",
        )
        
        snapshot = OptionChainSnapshotEvent(
            symbol="IBIT",
            expiration_ms=1740124800000,
            underlying_price=50.0,
            underlying_bid=49.95,
            underlying_ask=50.05,
            puts=(put,),
            calls=(),
            n_strikes=1,
            atm_iv=0.45,
            timestamp_ms=1700000000000,
            provider="alpaca",
            snapshot_id="IBIT_1740124800000_1700000000000",
        )
        
        d = option_chain_to_dict(snapshot)
        restored = dict_to_option_chain(d)
        
        assert restored.symbol == snapshot.symbol
        assert restored.underlying_price == snapshot.underlying_price
        assert len(restored.puts) == 1
        assert restored.puts[0].strike == 45.0
        assert restored.puts[0].delta == -0.18


class TestHelperFunctions:
    """Tests for helper functions."""
    
    def test_compute_snapshot_id(self):
        """Test deterministic snapshot ID."""
        id1 = compute_snapshot_id("IBIT", 1740124800000, 1700000000000)
        id2 = compute_snapshot_id("IBIT", 1740124800000, 1700000000000)
        id3 = compute_snapshot_id("BITO", 1740124800000, 1700000000000)
        
        assert id1 == id2  # Same inputs → same output
        assert id1 != id3  # Different symbol → different output
    
    def test_compute_chain_hash(self):
        """Test deterministic chain hash."""
        snapshot1 = OptionChainSnapshotEvent(
            symbol="IBIT",
            expiration_ms=1740124800000,
            underlying_price=50.0,
            puts=(),
            calls=(),
            timestamp_ms=1700000000000,
        )
        snapshot2 = OptionChainSnapshotEvent(
            symbol="IBIT",
            expiration_ms=1740124800000,
            underlying_price=50.0,
            puts=(),
            calls=(),
            timestamp_ms=1700000000000,
        )
        
        hash1 = compute_chain_hash(snapshot1)
        hash2 = compute_chain_hash(snapshot2)
        
        assert hash1 == hash2  # Same data → same hash
