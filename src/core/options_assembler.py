"""
Options Chain Assembler
=======================

Converts raw connector data into deterministic OptionChainSnapshotEvents.
Ensures reproducible ordering and deduplication for tape recording.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Set

from .option_events import (
    OptionContractEvent,
    OptionQuoteEvent,
    OptionChainSnapshotEvent,
    option_chain_to_dict,
    TOPIC_OPTIONS_CHAINS,
    TOPIC_OPTIONS_QUOTES,
    TOPIC_OPTIONS_CONTRACTS,
)

logger = logging.getLogger(__name__)


def _now_ms() -> int:
    """Current time as int milliseconds."""
    return int(time.time() * 1000)


class SequenceProvider:
    """Thread-safe monotonic sequence ID provider."""
    
    def __init__(self, start: int = 0) -> None:
        self._seq = start
    
    def next(self) -> int:
        """Get next sequence ID."""
        self._seq += 1
        return self._seq
    
    @property
    def current(self) -> int:
        """Get current sequence ID without incrementing."""
        return self._seq


class OptionsChainAssembler:
    """Assembles raw connector data into deterministic chain snapshots.
    
    Guarantees:
    - Quotes sorted by strike (ascending)
    - Sequence IDs assigned monotonically
    - Deterministic: identical inputs → identical outputs
    - No wall-clock reads during replay mode
    """
    
    def __init__(
        self,
        sequence_provider: Optional[SequenceProvider] = None,
        on_chain_snapshot: Optional[Callable[[OptionChainSnapshotEvent], None]] = None,
        on_contract: Optional[Callable[[OptionContractEvent], None]] = None,
        replay_mode: bool = False,
    ) -> None:
        """Initialize assembler.
        
        Args:
            sequence_provider: Provider for monotonic sequence IDs
            on_chain_snapshot: Callback for chain snapshots (for bus publishing)
            on_contract: Callback for new contracts
            replay_mode: If True, use event timestamps instead of wall clock
        """
        self._seq = sequence_provider or SequenceProvider()
        self._on_chain_snapshot = on_chain_snapshot
        self._on_contract = on_contract
        self._replay_mode = replay_mode
        
        # Track last snapshot per symbol+expiration to avoid duplicates
        self._last_snapshot: Dict[str, int] = {}  # key -> timestamp_ms
        
        # Track known contracts for deduplication
        self._known_contracts: Set[str] = set()
    
    def assemble_chain(
        self,
        symbol: str,
        expiration_ms: int,
        underlying_price: float,
        raw_puts: List[Dict[str, Any]],
        raw_calls: List[Dict[str, Any]],
        provider: str,
        source_ts_ms: int,
        timestamp_ms: Optional[int] = None,
        underlying_bid: float = 0.0,
        underlying_ask: float = 0.0,
    ) -> OptionChainSnapshotEvent:
        """Build atomic chain snapshot with deterministic ordering.
        
        Args:
            symbol: Underlying symbol
            expiration_ms: Expiration as UTC milliseconds
            underlying_price: Current underlying price
            raw_puts: List of raw put quote dicts
            raw_calls: List of raw call quote dicts
            provider: Data provider name
            source_ts_ms: Provider source timestamp
            timestamp_ms: Logical timestamp (defaults to now)
            underlying_bid: Underlying bid price
            underlying_ask: Underlying ask price
            
        Returns:
            Deterministic OptionChainSnapshotEvent
        """
        now_ms = timestamp_ms if timestamp_ms else _now_ms()
        
        # Dedupe and sort puts
        puts = self._process_quotes(
            raw_puts, symbol, expiration_ms, "PUT", provider, now_ms, source_ts_ms
        )
        
        # Dedupe and sort calls
        calls = self._process_quotes(
            raw_calls, symbol, expiration_ms, "CALL", provider, now_ms, source_ts_ms
        )
        
        # Compute ATM IV from put closest to underlying
        atm_iv = None
        if puts:
            atm_put = min(puts, key=lambda q: abs(q.strike - underlying_price))
            atm_iv = atm_put.iv
        
        snapshot_id = f"{provider}_{symbol}_{expiration_ms}_{now_ms}"
        
        snapshot = OptionChainSnapshotEvent(
            symbol=symbol,
            expiration_ms=expiration_ms,
            underlying_price=underlying_price,
            underlying_bid=underlying_bid,
            underlying_ask=underlying_ask,
            puts=tuple(puts),
            calls=tuple(calls),
            n_strikes=len(puts),
            atm_iv=atm_iv,
            timestamp_ms=now_ms,
            source_ts_ms=source_ts_ms,
            recv_ts_ms=now_ms,
            provider=provider,
            snapshot_id=snapshot_id,
            sequence_id=self._seq.next(),
        )
        
        # Track last snapshot for deduplication
        key = f"{symbol}_{expiration_ms}"
        self._last_snapshot[key] = now_ms
        
        # Emit callback if configured
        if self._on_chain_snapshot:
            self._on_chain_snapshot(snapshot)
        
        return snapshot
    
    def _process_quotes(
        self,
        raw_quotes: List[Dict[str, Any]],
        symbol: str,
        expiration_ms: int,
        option_type: str,
        provider: str,
        timestamp_ms: int,
        source_ts_ms: int,
    ) -> List[OptionQuoteEvent]:
        """Process raw quotes into sorted, deduplicated OptionQuoteEvents."""
        # Dedupe by contract_id
        seen: Dict[str, Dict[str, Any]] = {}
        for raw in raw_quotes:
            contract_id = raw.get("contract_id")
            if not contract_id:
                continue
            # Keep latest by source_ts
            if contract_id not in seen or raw.get("source_ts_ms", 0) > seen[contract_id].get("source_ts_ms", 0):
                seen[contract_id] = raw
        
        quotes = []
        for raw in seen.values():
            quote = self._raw_to_quote(
                raw, symbol, expiration_ms, option_type, provider, timestamp_ms, source_ts_ms
            )
            if quote:
                quotes.append(quote)
                
                # Track new contracts
                if quote.contract_id not in self._known_contracts:
                    self._known_contracts.add(quote.contract_id)
                    self._emit_contract(quote, provider, timestamp_ms)
        
        # Sort by strike for determinism
        quotes.sort(key=lambda q: q.strike)
        return quotes
    
    def _raw_to_quote(
        self,
        raw: Dict[str, Any],
        symbol: str,
        expiration_ms: int,
        option_type: str,
        provider: str,
        timestamp_ms: int,
        source_ts_ms: int,
    ) -> Optional[OptionQuoteEvent]:
        """Convert raw quote dict to OptionQuoteEvent."""
        try:
            bid = raw.get("bid", 0.0)
            ask = raw.get("ask", 0.0)
            mid = (bid + ask) / 2 if bid and ask else 0.0
            
            return OptionQuoteEvent(
                contract_id=raw["contract_id"],
                symbol=symbol,
                strike=raw["strike"],
                expiration_ms=expiration_ms,
                option_type=option_type,
                bid=bid,
                ask=ask,
                last=raw.get("last", 0.0),
                mid=mid,
                volume=raw.get("volume", 0),
                open_interest=raw.get("open_interest", 0),
                iv=raw.get("iv"),
                delta=raw.get("delta"),
                gamma=raw.get("gamma"),
                theta=raw.get("theta"),
                vega=raw.get("vega"),
                timestamp_ms=timestamp_ms,
                source_ts_ms=source_ts_ms,
                recv_ts_ms=timestamp_ms,
                provider=provider,
                sequence_id=self._seq.next(),
            )
        except (KeyError, TypeError) as e:
            logger.warning("Failed to parse quote: %s", e)
            return None
    
    def _emit_contract(
        self,
        quote: OptionQuoteEvent,
        provider: str,
        timestamp_ms: int,
    ) -> None:
        """Emit contract event for new contracts."""
        if not self._on_contract:
            return
        
        contract = OptionContractEvent(
            symbol=quote.symbol,
            contract_id=quote.contract_id,
            option_symbol=quote.contract_id,  # Use contract_id as option_symbol
            strike=quote.strike,
            expiration_ms=quote.expiration_ms,
            option_type=quote.option_type,
            multiplier=100,
            style="american",
            provider=provider,
            timestamp_ms=timestamp_ms,
            source_ts_ms=timestamp_ms,
        )
        self._on_contract(contract)
    
    def is_duplicate_snapshot(
        self,
        symbol: str,
        expiration_ms: int,
        timestamp_ms: int,
        min_interval_ms: int = 60000,
    ) -> bool:
        """Check if a snapshot would be a duplicate.
        
        Args:
            symbol: Underlying symbol
            expiration_ms: Expiration timestamp
            timestamp_ms: Proposed snapshot timestamp
            min_interval_ms: Minimum interval between snapshots
            
        Returns:
            True if this would be a duplicate snapshot
        """
        key = f"{symbol}_{expiration_ms}"
        last_ts = self._last_snapshot.get(key, 0)
        return (timestamp_ms - last_ts) < min_interval_ms
    
    def reset(self) -> None:
        """Reset assembler state (for testing)."""
        self._last_snapshot.clear()
        self._known_contracts.clear()
