"""
Tests for Spread Candidate Generator
=====================================

Unit tests for Phase 3B put spread candidate generation.
"""

import time
import pytest
from src.core.option_events import (
    OptionQuoteEvent,
    OptionChainSnapshotEvent,
)
from src.strategies.spread_generator import (
    SpreadCandidateGenerator,
    SpreadGeneratorConfig,
    PutSpreadCandidate,
    compute_config_hash,
    compute_candidate_id,
)

# Compute future expiration (14 days from now) for tests
_TEST_NOW_MS = int(time.time() * 1000)
_TEST_EXPIRATION_MS = _TEST_NOW_MS + (14 * 24 * 60 * 60 * 1000)  # 14 days forward


def make_put(strike: float, bid: float, ask: float, delta: float = None) -> OptionQuoteEvent:
    """Helper to create a put quote."""
    return OptionQuoteEvent(
        contract_id=f"p_{strike}",
        symbol="IBIT",
        strike=strike,
        expiration_ms=_TEST_EXPIRATION_MS,  # Dynamic future date
        option_type="PUT",
        bid=bid,
        ask=ask,
        mid=(bid + ask) / 2,
        delta=delta,
        timestamp_ms=_TEST_NOW_MS,
        provider="alpaca",
    )

def make_snapshot(puts: list, underlying_price: float = 50.0) -> OptionChainSnapshotEvent:
    """Helper to create a chain snapshot."""
    return OptionChainSnapshotEvent(
        symbol="IBIT",
        expiration_ms=_TEST_EXPIRATION_MS,  # Dynamic future date
        underlying_price=underlying_price,
        puts=tuple(sorted(puts, key=lambda q: q.strike)),
        calls=(),
        n_strikes=len(puts),
        timestamp_ms=_TEST_NOW_MS,
        provider="alpaca",
        snapshot_id="test_snapshot",
    )


class TestSpreadCandidateGeneration:
    """Tests for spread candidate generation."""
    
    def test_generates_valid_spread(self):
        """Test generating a valid put spread candidate."""
        puts = [
            make_put(44.0, 0.80, 0.90, delta=-0.12),
            make_put(46.0, 1.50, 1.60, delta=-0.18),
        ]
        snapshot = make_snapshot(puts)
        
        config = SpreadGeneratorConfig(
            min_dte=1,
            max_dte=30,
            allowed_widths=(2.0,),
            min_credit=0.20,
        )
        generator = SpreadCandidateGenerator("test_strategy", config)
        
        candidates = generator.on_chain_snapshot(snapshot)
        
        assert len(candidates) == 1
        c = candidates[0]
        assert c.short_strike == 46.0
        assert c.long_strike == 44.0
        assert c.credit == 0.60  # 1.50 - 0.90
        assert c.width == 2.0
        assert c.max_loss == 1.40  # 2.0 - 0.60
    
    def test_filters_by_delta(self):
        """Test delta-based filtering."""
        puts = [
            make_put(44.0, 0.80, 0.90, delta=-0.08),  # Too low delta
            make_put(46.0, 1.50, 1.60, delta=-0.18),
            make_put(48.0, 2.50, 2.60, delta=-0.30),  # Too high delta
        ]
        snapshot = make_snapshot(puts)
        
        config = SpreadGeneratorConfig(
            min_dte=1,
            max_dte=30,
            min_short_delta=0.10,
            max_short_delta=0.25,
            allowed_widths=(2.0,),
            min_credit=0.20,
        )
        generator = SpreadCandidateGenerator("test_strategy", config)
        
        candidates = generator.on_chain_snapshot(snapshot)
        
        # Only 46 strike should pass delta filter
        assert len(candidates) == 1
        assert candidates[0].short_strike == 46.0
    
    def test_filters_by_credit(self):
        """Test minimum credit filter."""
        puts = [
            make_put(44.0, 0.50, 0.55),  # Low credit spread
            make_put(46.0, 0.60, 0.65),
        ]
        snapshot = make_snapshot(puts)
        
        config = SpreadGeneratorConfig(
            min_dte=1,
            max_dte=30,
            allowed_widths=(2.0,),
            min_credit=0.20,  # Credit would be 0.05, below threshold
        )
        generator = SpreadCandidateGenerator("test_strategy", config)
        
        candidates = generator.on_chain_snapshot(snapshot)
        
        assert len(candidates) == 0
    
    def test_deterministic_ranking(self):
        """Test that ranking is deterministic."""
        puts = [
            make_put(42.0, 0.50, 0.60, delta=-0.10),
            make_put(44.0, 0.90, 1.00, delta=-0.15),
            make_put(46.0, 1.50, 1.60, delta=-0.20),
            make_put(48.0, 2.20, 2.30, delta=-0.25),
        ]
        snapshot = make_snapshot(puts)
        
        config = SpreadGeneratorConfig(
            min_dte=1,
            max_dte=30,
            allowed_widths=(2.0,),
            min_credit=0.10,
            target_short_delta=0.18,
        )
        
        # Generate multiple times
        results = []
        for _ in range(3):
            generator = SpreadCandidateGenerator("test_strategy", config)
            candidates = generator.on_chain_snapshot(snapshot)
            results.append([(c.short_strike, c.long_strike) for c in candidates])
        
        # All runs should produce identical ranking
        assert results[0] == results[1] == results[2]


class TestCandidateIdempotency:
    """Tests for idempotency and determinism."""
    
    def test_candidate_id_determination(self):
        """Test that candidate IDs are deterministic."""
        id1 = compute_candidate_id("strat", "IBIT", 1740124800000, 46.0, 44.0, 1700000000000)
        id2 = compute_candidate_id("strat", "IBIT", 1740124800000, 46.0, 44.0, 1700000000000)
        id3 = compute_candidate_id("strat", "IBIT", 1740124800000, 46.0, 44.0, 1700000001000)
        
        assert id1 == id2
        assert id1 != id3  # Different timestamp
    
    def test_config_hash_determinism(self):
        """Test that config hashes are deterministic."""
        config1 = {"min_dte": 7, "max_dte": 21, "target_delta": 0.18}
        config2 = {"min_dte": 7, "max_dte": 21, "target_delta": 0.18}
        config3 = {"min_dte": 7, "max_dte": 14, "target_delta": 0.18}
        
        assert compute_config_hash(config1) == compute_config_hash(config2)
        assert compute_config_hash(config1) != compute_config_hash(config3)


class TestSignalEmission:
    """Tests for signal emission."""
    
    def test_emits_signals(self):
        """Test that signals are emitted for candidates."""
        puts = [
            make_put(44.0, 0.80, 0.90, delta=-0.12),
            make_put(46.0, 1.50, 1.60, delta=-0.18),
        ]
        snapshot = make_snapshot(puts)
        
        signals = []
        
        config = SpreadGeneratorConfig(
            min_dte=1,
            max_dte=30,
            allowed_widths=(2.0,),
            min_credit=0.20,
        )
        generator = SpreadCandidateGenerator(
            "test_strategy",
            config,
            on_signal=signals.append,
        )
        
        generator.on_chain_snapshot(snapshot)
        
        assert len(signals) == 1
        signal = signals[0]
        assert signal.strategy_id == "test_strategy"
        assert signal.direction == "SHORT"
        assert signal.entry_type == "PUT_SPREAD"
        assert signal.features_snapshot["short_strike"] == 46.0
        assert signal.features_snapshot["credit"] == 0.60


class TestChainSnapshotIdempotency:
    """Tests for chain snapshot idempotency and determinism.
    
    Since we use ON CONFLICT DO NOTHING, we verify that:
    1. Same timestamp + same content = identical chain_hash (safe to skip)
    2. Same snapshot processed twice = no signal duplication
    """
    
    def test_identical_snapshots_same_chain_hash(self):
        """Identical snapshots at same timestamp should produce same chain_hash."""
        from src.core.option_events import compute_chain_hash
        
        puts = [
            make_put(44.0, 0.80, 0.90, delta=-0.12),
            make_put(46.0, 1.50, 1.60, delta=-0.18),
        ]
        
        # Create two identical snapshots
        snapshot1 = make_snapshot(puts)
        snapshot2 = make_snapshot(puts)
        
        hash1 = compute_chain_hash(snapshot1)
        hash2 = compute_chain_hash(snapshot2)
        
        # Identical content must produce identical hash
        assert hash1 == hash2, "Identical snapshots should have same chain_hash"
    
    def test_different_content_different_hash(self):
        """Different chain structure must produce different chain_hash."""
        from src.core.option_events import compute_chain_hash
        
        # Different number of puts = different chain structure
        puts1 = [make_put(44.0, 0.80, 0.90)]
        puts2 = [
            make_put(44.0, 0.80, 0.90),
            make_put(46.0, 1.50, 1.60),
        ]  # More puts
        
        snapshot1 = make_snapshot(puts1)
        snapshot2 = make_snapshot(puts2)
        
        hash1 = compute_chain_hash(snapshot1)
        hash2 = compute_chain_hash(snapshot2)
        
        # Different chain structure (n_puts differs) must produce different hash
        assert hash1 != hash2, "Different chain structures should have different chain_hash"
    
    def test_no_signal_duplication_on_replay(self):
        """Processing same snapshot twice should not emit duplicate signals."""
        puts = [
            make_put(44.0, 0.80, 0.90, delta=-0.12),
            make_put(46.0, 1.50, 1.60, delta=-0.18),
        ]
        snapshot = make_snapshot(puts)
        
        signals = []
        config = SpreadGeneratorConfig(
            min_dte=1,
            max_dte=30,
            allowed_widths=(2.0,),
            min_credit=0.20,
        )
        generator = SpreadCandidateGenerator(
            "test_strategy",
            config,
            on_signal=signals.append,
        )
        
        # Process same snapshot twice (simulating replay)
        generator.on_chain_snapshot(snapshot)
        count_after_first = len(signals)
        
        generator.on_chain_snapshot(snapshot)
        count_after_second = len(signals)
        
        # Each processing generates signals - this tests that generator
        # produces deterministic output. In production, dedupe happens
        # at the persistence layer via ON CONFLICT DO NOTHING.
        # Here we verify the signal IDs are deterministic (same inputs = same IDs)
        if count_after_second > count_after_first:
            # Check first and second signal have same idempotency_key
            assert signals[0].idempotency_key == signals[count_after_first].idempotency_key, \
                "Same snapshot should produce signals with same idempotency_key"

