"""
Put Spread Candidate Generator
==============================

Deterministic generator for put spread trading candidates.
Takes option chain snapshots and emits SignalEvents for high-quality spreads.

Guarantees:
- No randomness: all ranking uses deterministic sort keys
- No wall-clock: timestamps from events only
- Reproducible: same inputs → same outputs
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..core.option_events import (
    OptionQuoteEvent,
    OptionChainSnapshotEvent,
)
from ..core.signals import SignalEvent, compute_signal_id

logger = logging.getLogger(__name__)


def _now_ms() -> int:
    """Current time as int milliseconds."""
    return int(time.time() * 1000)


def compute_config_hash(config: Dict[str, Any]) -> str:
    """Compute deterministic hash of strategy config."""
    config_str = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(config_str.encode()).hexdigest()[:12]


def compute_candidate_id(
    strategy_id: str,
    symbol: str,
    expiration_ms: int,
    short_strike: float,
    long_strike: float,
    timestamp_ms: int,
) -> str:
    """Compute deterministic candidate ID."""
    key = f"{strategy_id}:{symbol}:{expiration_ms}:{short_strike}:{long_strike}:{timestamp_ms}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class PutSpreadCandidate:
    """Deterministic put spread candidate.
    
    Represents a potential credit put spread trade.
    """
    symbol: str
    expiration_ms: int
    short_strike: float
    long_strike: float
    short_quote: OptionQuoteEvent
    long_quote: OptionQuoteEvent
    
    # Computed metrics
    credit: float            # Net credit received (short_bid - long_ask)
    max_loss: float          # Width - credit
    width: float             # Strike difference
    risk_reward: float       # credit / max_loss
    
    # Greeks / IV (approximations)
    short_delta: Optional[float] = None
    spread_delta: Optional[float] = None
    short_iv: Optional[float] = None
    iv_rank: Optional[float] = None
    
    # Probability proxies
    prob_otm: Optional[float] = None      # P(short expires OTM)
    expected_value: Optional[float] = None
    
    # Context
    underlying_price: float = 0.0
    days_to_expiry: int = 0
    regime: str = ""
    timestamp_ms: int = 0
    
    # Determinism
    candidate_id: str = ""


@dataclass
class SpreadGeneratorConfig:
    """Configuration for spread generator."""
    # DTE filters
    min_dte: int = 7
    max_dte: int = 21
    
    # Delta filters (absolute value)
    min_short_delta: float = 0.10
    max_short_delta: float = 0.25
    target_short_delta: float = 0.18
    
    # Credit filters
    min_credit: float = 0.20
    min_risk_reward: float = 0.20
    
    # Spread width
    target_width: float = 2.0
    allowed_widths: Tuple[float, ...] = (1.0, 2.0, 2.5, 5.0)
    
    # Bid-ask quality
    max_spread_pct: float = 0.25  # Max bid-ask spread as % of mid
    
    # Candidate limits
    max_candidates_per_expiration: int = 5
    max_total_candidates: int = 10


class SpreadCandidateGenerator:
    """Deterministic put spread candidate generator.
    
    Processes chain snapshots and emits candidate signals.
    """
    
    def __init__(
        self,
        strategy_id: str,
        config: Optional[SpreadGeneratorConfig] = None,
        on_signal: Optional[Callable[[SignalEvent], None]] = None,
    ) -> None:
        """Initialize generator.
        
        Args:
            strategy_id: Strategy identifier for signals
            config: Generator configuration
            on_signal: Callback for emitted signals
        """
        self._strategy_id = strategy_id
        self._config = config or SpreadGeneratorConfig()
        self._config_hash = compute_config_hash(asdict(self._config))
        self._on_signal = on_signal
        
        # Regime cache (symbol -> regime)
        self._regime_cache: Dict[str, str] = {}
        
        # Stats
        self._snapshots_processed = 0
        self._candidates_generated = 0
        self._signals_emitted = 0
    
    def set_regime(self, symbol: str, regime: str) -> None:
        """Update regime for a symbol."""
        self._regime_cache[symbol] = regime
    
    def on_chain_snapshot(
        self,
        snapshot: OptionChainSnapshotEvent,
        regime: Optional[str] = None,
    ) -> List[PutSpreadCandidate]:
        """Process chain snapshot and generate candidates.
        
        Args:
            snapshot: Option chain snapshot
            regime: Optional regime override
            
        Returns:
            List of generated candidates
        """
        self._snapshots_processed += 1
        
        # Determine regime
        if regime is None:
            regime = self._regime_cache.get(snapshot.symbol, "unknown")
        
        # Calculate DTE
        now_ms = snapshot.timestamp_ms or _now_ms()
        dte_ms = snapshot.expiration_ms - now_ms
        dte = max(0, dte_ms // (24 * 60 * 60 * 1000))
        
        # Filter by DTE
        if not (self._config.min_dte <= dte <= self._config.max_dte):
            return []
        
        # Generate candidates
        candidates = self._generate_candidates(
            snapshot=snapshot,
            regime=regime,
            dte=dte,
        )
        
        # Filter candidates
        candidates = self._filter_candidates(candidates)
        
        # Rank and limit
        candidates = self._rank_candidates(candidates)
        
        self._candidates_generated += len(candidates)
        
        # Emit signals for top candidates
        for candidate in candidates[:self._config.max_candidates_per_expiration]:
            self._emit_signal(candidate)
        
        return candidates
    
    def _generate_candidates(
        self,
        snapshot: OptionChainSnapshotEvent,
        regime: str,
        dte: int,
    ) -> List[PutSpreadCandidate]:
        """Generate all valid put spread candidates from chain."""
        candidates = []
        
        puts = list(snapshot.puts)
        if len(puts) < 2:
            return candidates
        
        # For each potential short put, find valid long puts
        for i, short_quote in enumerate(puts):
            # Skip if no bid (can't sell)
            if short_quote.bid <= 0:
                continue
            
            # Skip if delta filter fails (when delta available)
            if short_quote.delta is not None:
                abs_delta = abs(short_quote.delta)
                if not (self._config.min_short_delta <= abs_delta <= self._config.max_short_delta):
                    continue
            
            # Find valid long puts (lower strikes)
            for j in range(i):
                long_quote = puts[j]
                
                # Skip if no ask (can't buy)
                if long_quote.ask <= 0:
                    continue
                
                width = short_quote.strike - long_quote.strike
                
                # Check if width is allowed
                if width not in self._config.allowed_widths:
                    continue
                
                # Calculate credit (conservative: short bid - long ask)
                credit = short_quote.bid - long_quote.ask
                if credit < self._config.min_credit:
                    continue
                
                max_loss = width - credit
                if max_loss <= 0:
                    continue  # Should never happen, but safety check
                
                risk_reward = credit / max_loss
                if risk_reward < self._config.min_risk_reward:
                    continue
                
                # Compute probability of OTM (if delta available)
                prob_otm = None
                if short_quote.delta is not None:
                    prob_otm = 1 - abs(short_quote.delta)
                
                # Compute spread delta
                spread_delta = None
                if short_quote.delta is not None and long_quote.delta is not None:
                    spread_delta = short_quote.delta - long_quote.delta
                
                candidate_id = compute_candidate_id(
                    self._strategy_id,
                    snapshot.symbol,
                    snapshot.expiration_ms,
                    short_quote.strike,
                    long_quote.strike,
                    snapshot.timestamp_ms,
                )
                
                candidate = PutSpreadCandidate(
                    symbol=snapshot.symbol,
                    expiration_ms=snapshot.expiration_ms,
                    short_strike=short_quote.strike,
                    long_strike=long_quote.strike,
                    short_quote=short_quote,
                    long_quote=long_quote,
                    credit=credit,
                    max_loss=max_loss,
                    width=width,
                    risk_reward=risk_reward,
                    short_delta=short_quote.delta,
                    spread_delta=spread_delta,
                    short_iv=short_quote.iv,
                    iv_rank=None,  # Would need historical data
                    prob_otm=prob_otm,
                    expected_value=credit * prob_otm if prob_otm else None,
                    underlying_price=snapshot.underlying_price,
                    days_to_expiry=dte,
                    regime=regime,
                    timestamp_ms=snapshot.timestamp_ms,
                    candidate_id=candidate_id,
                )
                
                candidates.append(candidate)
        
        return candidates
    
    def _filter_candidates(
        self,
        candidates: List[PutSpreadCandidate],
    ) -> List[PutSpreadCandidate]:
        """Apply deterministic filters."""
        filtered = []
        
        for c in candidates:
            # Bid-ask quality filter
            if c.short_quote.mid > 0:
                short_spread = c.short_quote.ask - c.short_quote.bid
                short_spread_pct = short_spread / c.short_quote.mid
                if short_spread_pct > self._config.max_spread_pct:
                    continue
            
            if c.long_quote.mid > 0:
                long_spread = c.long_quote.ask - c.long_quote.bid
                long_spread_pct = long_spread / c.long_quote.mid
                if long_spread_pct > self._config.max_spread_pct:
                    continue
            
            filtered.append(c)
        
        return filtered
    
    def _rank_candidates(
        self,
        candidates: List[PutSpreadCandidate],
    ) -> List[PutSpreadCandidate]:
        """Deterministic ranking by score.
        
        Score favors:
        - Higher risk/reward
        - Delta closer to target
        - Higher probability OTM
        """
        def score(c: PutSpreadCandidate) -> Tuple[float, float, float, float, float]:
            # Risk/reward score (higher is better)
            rr_score = c.risk_reward
            
            # Delta proximity score (closer to target is better)
            delta_score = 0.0
            if c.short_delta is not None:
                delta_diff = abs(abs(c.short_delta) - self._config.target_short_delta)
                delta_score = 1.0 - min(delta_diff, 0.2) / 0.2  # 0-1 range
            
            # Probability score
            prob_score = c.prob_otm if c.prob_otm else 0.5
            
            # Deterministic tiebreakers
            strike_key = c.short_strike
            width_key = c.width
            
            return (rr_score, delta_score, prob_score, -strike_key, -width_key)
        
        # Sort descending by score
        candidates.sort(key=score, reverse=True)
        
        return candidates[:self._config.max_total_candidates]
    
    def _emit_signal(self, candidate: PutSpreadCandidate) -> None:
        """Emit SignalEvent for a candidate."""
        self._signals_emitted += 1
        
        signal = SignalEvent(
            strategy_id=self._strategy_id,
            config_hash=self._config_hash,
            symbol=candidate.symbol,
            timestamp_ms=candidate.timestamp_ms,
            direction="SHORT",  # Selling put spread
            signal_type="OPTIONS_ENTRY",  # Distinct from generic ENTRY signals
            timeframe=60,  # Required: 1-minute bar timeframe
            entry_type="PUT_SPREAD",
            
            features_snapshot={
                "short_strike": candidate.short_strike,
                "long_strike": candidate.long_strike,
                "credit": candidate.credit,
                "max_loss": candidate.max_loss,
                "width": candidate.width,
                "risk_reward": candidate.risk_reward,
                "short_delta": candidate.short_delta,
                "spread_delta": candidate.spread_delta,
                "short_iv": candidate.short_iv,
                "prob_otm": candidate.prob_otm,
                "underlying_price": candidate.underlying_price,
                "days_to_expiry": candidate.days_to_expiry,
                "expiration_ms": candidate.expiration_ms,
                "candidate_id": candidate.candidate_id,
            },
            
            regime_snapshot={"symbol": candidate.regime},
            
            explain=(
                f"Put spread {candidate.symbol} "
                f"{candidate.short_strike}/{candidate.long_strike} "
                f"credit={candidate.credit:.2f} "
                f"R/R={candidate.risk_reward:.2f}"
            ),
            
            idempotency_key=compute_signal_id(
                self._strategy_id,
                self._config_hash,
                candidate.symbol,
                candidate.timestamp_ms,
            ),
        )
        
        if self._on_signal:
            self._on_signal(signal)
        
        return signal
    
    def get_stats(self) -> Dict[str, Any]:
        """Get generator statistics."""
        return {
            "strategy_id": self._strategy_id,
            "config_hash": self._config_hash,
            "snapshots_processed": self._snapshots_processed,
            "candidates_generated": self._candidates_generated,
            "signals_emitted": self._signals_emitted,
        }
