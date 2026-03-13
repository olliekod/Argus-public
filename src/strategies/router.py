"""
Signal Router and Ranker
========================

Collects raw signals from strategy modules, scores them,
and emits ranked signals for downstream consumption.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import deque
from typing import Any, Deque, Dict, List, Optional

from ..core.events import TOPIC_SIGNALS_RAW, TOPIC_SIGNALS_RANKED
from ..core.signals import (
    SignalEvent,
    RankedSignalEvent,
    compute_config_hash,
)

logger = logging.getLogger("argus.strategies.router")


# ═══════════════════════════════════════════════════════════════════════════
# Default Ranker Configuration
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_RANKER_CONFIG = {
    # Data quality penalties (subtracted from score)
    "penalty_repaired_input": 10,  # DQ_REPAIRED_INPUT
    "penalty_gap_window": 20,       # DQ_GAP_WINDOW
    "penalty_stale_input": 15,      # DQ_STALE_INPUT
    
    # Regime bonuses (added to score)
    "bonus_trend_aligned": 15,      # direction matches trend regime
    "bonus_vol_appropriate": 10,    # vol regime matches strategy preference
    "bonus_session_optimal": 5,     # RTH or preferred session
    
    # Base score
    "base_score": 50,
    
    # Suppression thresholds
    "min_score_threshold": 20,      # suppress if below this
    "max_signals_per_bucket": 5,    # top N per timestamp bucket
}


# ═══════════════════════════════════════════════════════════════════════════
# Signal Router
# ═══════════════════════════════════════════════════════════════════════════

class SignalRouter:
    """
    Collects, scores, and ranks signals from all active strategies.
    
    Signals are bucketed by timestamp (to the second) and ranked
    within each bucket. Top signals are emitted to signals.ranked.
    """
    
    def __init__(
        self,
        bus,
        config: Optional[Dict[str, Any]] = None,
        bucket_window_ms: int = 1000,  # 1 second buckets
    ) -> None:
        self._bus = bus
        self._config = config or DEFAULT_RANKER_CONFIG
        self._config_hash = compute_config_hash(self._config)
        self._bucket_window_ms = bucket_window_ms
        
        # Current bucket
        self._current_bucket_ts: Optional[int] = None
        self._pending_signals: List[SignalEvent] = []
        
        # Telemetry
        self._signals_received = 0
        self._signals_ranked = 0
        self._signals_suppressed = 0
        self._buckets_processed = 0
        
        self._lock = threading.Lock()
        
        # Subscribe
        self._bus.subscribe(TOPIC_SIGNALS_RAW, self._on_raw_signal)
        
        logger.info(
            "SignalRouter initialized — config_hash=%s, bucket_window=%dms",
            self._config_hash, self._bucket_window_ms
        )
    
    def _on_raw_signal(self, signal: SignalEvent) -> None:
        """Handle incoming raw signal."""
        with self._lock:
            self._signals_received += 1
            
            # Determine bucket timestamp
            bucket_ts = (signal.timestamp_ms // self._bucket_window_ms) * self._bucket_window_ms
            
            # If new bucket, flush previous
            if self._current_bucket_ts is not None and bucket_ts != self._current_bucket_ts:
                self._flush_bucket()
            
            self._current_bucket_ts = bucket_ts
            self._pending_signals.append(signal)
    
    def _flush_bucket(self) -> None:
        """Score and rank pending signals, emit top N."""
        if not self._pending_signals:
            return
        
        self._buckets_processed += 1
        
        # Score all signals
        scored: List[tuple] = []  # (score, breakdown, signal)
        for signal in self._pending_signals:
            score, breakdown = self._score_signal(signal)
            scored.append((score, breakdown, signal))
        
        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Emit top N
        max_signals = self._config.get("max_signals_per_bucket", 5)
        min_score = self._config.get("min_score_threshold", 20)
        
        for rank, (score, breakdown, signal) in enumerate(scored, start=1):
            suppressed = score < min_score or rank > max_signals
            
            if suppressed:
                self._signals_suppressed += 1
                reason = f"score={score:.1f}<{min_score}" if score < min_score else f"rank={rank}>{max_signals}"
            else:
                self._signals_ranked += 1
                reason = ""
            
            ranked_event = RankedSignalEvent(
                signal=signal,
                rank=rank,
                final_score=score,
                score_breakdown=breakdown,
                suppressed=suppressed,
                suppression_reason=reason,
            )
            
            self._bus.publish(TOPIC_SIGNALS_RANKED, ranked_event)
        
        # Clear bucket
        self._pending_signals = []
    
    def _score_signal(self, signal: SignalEvent) -> tuple:
        """
        Compute deterministic score for a signal.
        
        Returns (total_score, breakdown_dict).
        """
        cfg = self._config
        breakdown: Dict[str, float] = {}
        
        # Start with base + strategy confidence
        base = cfg.get("base_score", 50)
        conf_contribution = signal.confidence * 30  # max 30 points from confidence
        breakdown["base"] = base
        breakdown["confidence"] = conf_contribution
        
        score = base + conf_contribution
        
        # Data quality penalties
        dq = signal.data_quality_flags
        if dq & 1:  # DQ_REPAIRED_INPUT
            penalty = cfg.get("penalty_repaired_input", 10)
            score -= penalty
            breakdown["penalty_repaired"] = -penalty
        if dq & 2:  # DQ_GAP_WINDOW
            penalty = cfg.get("penalty_gap_window", 20)
            score -= penalty
            breakdown["penalty_gap"] = -penalty
        if dq & 4:  # DQ_STALE_INPUT
            penalty = cfg.get("penalty_stale_input", 15)
            score -= penalty
            breakdown["penalty_stale"] = -penalty
        
        # Regime alignment bonuses
        regime = signal.regime_snapshot
        
        # Trend alignment
        trend = regime.get("trend", "")
        if signal.direction == "LONG" and trend == "TREND_UP":
            bonus = cfg.get("bonus_trend_aligned", 15)
            score += bonus
            breakdown["bonus_trend"] = bonus
        elif signal.direction == "SHORT" and trend == "TREND_DOWN":
            bonus = cfg.get("bonus_trend_aligned", 15)
            score += bonus
            breakdown["bonus_trend"] = bonus
        
        # Volatility appropriateness
        vol = regime.get("vol", "")
        if vol in ("VOL_NORMAL", "VOL_HIGH"):
            bonus = cfg.get("bonus_vol_appropriate", 10)
            score += bonus
            breakdown["bonus_vol"] = bonus
        
        # Session bonus
        session = regime.get("session", "")
        if session in ("RTH", "US"):
            bonus = cfg.get("bonus_session_optimal", 5)
            score += bonus
            breakdown["bonus_session"] = bonus
        
        # Clamp to 0-100
        score = max(0, min(100, score))
        
        return score, breakdown
    
    def flush(self) -> None:
        """Force flush current bucket (e.g., on shutdown)."""
        with self._lock:
            self._flush_bucket()
    
    def get_status(self) -> Dict[str, Any]:
        """Return router telemetry."""
        with self._lock:
            pending = len(self._pending_signals)
        return {
            "config_hash": self._config_hash,
            "signals_received": self._signals_received,
            "signals_ranked": self._signals_ranked,
            "signals_suppressed": self._signals_suppressed,
            "buckets_processed": self._buckets_processed,
            "pending_signals": pending,
        }
