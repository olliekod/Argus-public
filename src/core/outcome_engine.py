"""
Argus Outcome Engine
====================

Compute bar-level forward outcomes from persisted bars.
Produces deterministic, idempotent ground truth for backtesting.

Phase 4A.1: Bar Outcomes Foundation
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .sessions import get_session_regime

logger = logging.getLogger(__name__)


# Status values for outcomes
STATUS_OK = "OK"
STATUS_INCOMPLETE = "INCOMPLETE"
STATUS_GAP = "GAP"


@dataclass(frozen=True)
class BarData:
    """Lightweight bar representation for outcome computation."""
    timestamp_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class OutcomeResult:
    """Result of outcome computation for a single bar+horizon."""
    provider: str
    symbol: str
    bar_duration_seconds: int
    timestamp_ms: int
    horizon_seconds: int
    outcome_version: str
    
    # Core metrics
    close_now: float
    close_at_horizon: Optional[float]
    fwd_return: Optional[float]
    max_runup: Optional[float]
    max_drawdown: Optional[float]
    realized_vol: Optional[float]
    
    # Path helpers
    max_high_in_window: Optional[float]
    min_low_in_window: Optional[float]
    max_runup_ts_ms: Optional[int]
    max_drawdown_ts_ms: Optional[int]
    time_to_max_runup_ms: Optional[int]
    time_to_max_drawdown_ms: Optional[int]
    
    # Coverage
    status: str
    close_ref_ms: int
    window_start_ms: int
    window_end_ms: int
    bars_expected: int
    bars_found: int
    gap_count: int
    computed_at_ms: Optional[int]

    # Session awareness (not persisted to DB — computed on-the-fly)
    session_regime: str = ""
    
    def to_tuple(self) -> tuple:
        """Convert to tuple for batch DB insert."""
        return (
            self.provider, self.symbol, self.bar_duration_seconds, self.timestamp_ms,
            self.horizon_seconds, self.outcome_version,
            self.close_now, self.close_at_horizon, self.fwd_return,
            self.max_runup, self.max_drawdown, self.realized_vol,
            self.max_high_in_window, self.min_low_in_window,
            self.max_runup_ts_ms, self.max_drawdown_ts_ms,
            self.time_to_max_runup_ms, self.time_to_max_drawdown_ms,
            self.status, self.close_ref_ms, self.window_start_ms, self.window_end_ms,
            self.bars_expected, self.bars_found, self.gap_count, self.computed_at_ms,
        )


def _quantize(value: Optional[float], decimals: int) -> Optional[float]:
    """Quantize a float to fixed decimal precision for determinism."""
    if value is None or math.isnan(value) or math.isinf(value):
        return None
    return round(value, decimals)


def _timestamp_to_ms(ts_str: str) -> int:
    """Convert ISO timestamp string to milliseconds (always UTC).

    Naive timestamps (no timezone info) are treated as UTC to ensure
    determinism regardless of the machine's local timezone.
    """
    from datetime import timezone as _tz

    try:
        if "T" in ts_str:
            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        else:
            dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        # Force UTC for naive datetimes to guarantee determinism
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_tz.utc)
        return int(dt.timestamp() * 1000)
    except Exception:
        # Fallback: try to parse as float seconds
        try:
            return int(float(ts_str) * 1000)
        except Exception:
            return 0


class OutcomeEngine:
    """Compute bar-level forward outcomes from persisted bars (DB source of truth).
    
    Produces deterministic, idempotent outcome records for backtesting.
    
    Key properties:
    - Determinism: same bars → same outcomes (no wall-clock dependency in metrics)
    - Idempotency: reruns don't duplicate or drift values
    - Explicit status: OK, INCOMPLETE, or GAP
    - Quantized metrics for exact equality in tests
    """
    
    def __init__(
        self,
        db: Any,  # Database instance
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the outcome engine.
        
        Args:
            db: Database instance for reading bars and writing outcomes.
            config: Outcome configuration dict with keys:
                - outcome_version: Version string for outcomes
                - gap_tolerance_bars: Max missing bars for OK status
                - quantize_decimals: Decimal precision for metrics
                - horizons_seconds_by_bar: Dict mapping bar_duration to horizon list
        """
        self.db = db
        config = config or {}
        
        self.outcome_version = config.get("outcome_version", "BAR_OUTCOMES_V1")
        self.gap_tolerance_bars = config.get("gap_tolerance_bars", 1)
        self.quantize_decimals = config.get("quantize_decimals", 10)
        self.horizons_by_bar = config.get("horizons_seconds_by_bar", {
            60: [300, 900, 3600, 14400, 28800, 86400],
            300: [900, 3600, 14400, 28800, 86400],
            900: [3600, 14400, 86400],
            3600: [14400, 86400],
            14400: [86400],
            86400: [604800],
        })
        
        # Convert string keys to int if needed (from YAML)
        self.horizons_by_bar = {
            int(k): v for k, v in self.horizons_by_bar.items()
        }
    
    async def compute_outcomes_for_range(
        self,
        provider: str,
        symbol: str,
        bar_duration_seconds: int,
        start_ms: int,
        end_ms: int,
        batch_size: int = 1000,
    ) -> Tuple[int, int, int]:
        """Compute outcomes for bars in [start_ms, end_ms].
        
        Args:
            provider: Data provider (maps to source in market_bars)
            symbol: Symbol to process
            bar_duration_seconds: Bar duration in seconds
            start_ms: Start of range (inclusive)
            end_ms: End of range (inclusive)
            batch_size: Batch size for DB writes
            
        Returns:
            Tuple of (bars_processed, outcomes_computed, outcomes_upserted)
        """
        horizons = self.horizons_by_bar.get(bar_duration_seconds, [])
        if not horizons:
            logger.warning(
                "No horizons configured for bar_duration=%d, skipping",
                bar_duration_seconds
            )
            return 0, 0, 0
        
        # Compute max lookahead needed
        max_horizon_seconds = max(horizons)
        max_lookahead_ms = max_horizon_seconds * 1000
        
        # Fetch bars with lookahead
        bars_raw = await self.db.get_bars_for_outcome_computation(
            source=provider,
            symbol=symbol,
            bar_duration=bar_duration_seconds,
            start_ms=start_ms,
            end_ms=end_ms + max_lookahead_ms,
        )
        
        if not bars_raw:
            logger.info("No bars found for %s/%s", provider, symbol)
            return 0, 0, 0
        
        # Convert to BarData objects
        bars = self._convert_bars(bars_raw, bar_duration_seconds)
        
        # Build timestamp index for efficient lookups
        bars_by_ts = {b.timestamp_ms: b for b in bars}
        
        # Compute outcomes
        outcomes: List[OutcomeResult] = []
        bars_processed = 0
        
        for bar in bars:
            # Only process bars in the target range
            if bar.timestamp_ms < start_ms or bar.timestamp_ms > end_ms:
                continue
            
            bars_processed += 1
            
            for horizon in horizons:
                result = self._compute_single_outcome(
                    bar=bar,
                    bars_by_ts=bars_by_ts,
                    all_bars=bars,
                    provider=provider,
                    symbol=symbol,
                    bar_duration_seconds=bar_duration_seconds,
                    horizon_seconds=horizon,
                )
                outcomes.append(result)
        
        # Batch upsert with deliberate yield between batches.
        # After each commit we sleep 100 ms so the OS-level SQLite write lock
        # is left open long enough for the live-feed process to acquire it and
        # insert its single bar (~2 ms).  Without this pause the backfiller
        # re-acquires the lock almost instantly, starving the live feed even
        # when SQLite is configured in WAL mode with a 30-second busy_timeout.
        outcomes_upserted = 0
        for i in range(0, len(outcomes), batch_size):
            batch = outcomes[i:i + batch_size]
            tuples = [o.to_tuple() for o in batch]
            count = await self.db.upsert_bar_outcomes_batch(tuples)
            outcomes_upserted += count
            # Yield the write lock so the live feed can insert its bar.
            await asyncio.sleep(0.1)
        
        logger.info(
            "Computed outcomes: provider=%s symbol=%s bars=%d outcomes=%d upserted=%d",
            provider, symbol, bars_processed, len(outcomes), outcomes_upserted
        )
        
        return bars_processed, len(outcomes), outcomes_upserted
    
    def _convert_bars(
        self,
        bars_raw: List[Dict[str, Any]],
        bar_duration_seconds: int,
    ) -> List[BarData]:
        """Convert raw DB rows to BarData objects."""
        result = []
        for row in bars_raw:
            ts = row.get("timestamp")
            if isinstance(ts, str):
                ts_ms = _timestamp_to_ms(ts)
            elif isinstance(ts, (int, float)):
                # If already numeric, assume it's epoch seconds or ms
                if ts > 1e12:  # Already ms
                    ts_ms = int(ts)
                else:
                    ts_ms = int(ts * 1000)
            else:
                continue
            
            result.append(BarData(
                timestamp_ms=ts_ms,
                open=float(row.get("open", 0)),
                high=float(row.get("high", 0)),
                low=float(row.get("low", 0)),
                close=float(row.get("close", 0)),
                volume=float(row.get("volume", 0) or 0),
            ))
        
        # Sort by timestamp
        result.sort(key=lambda b: b.timestamp_ms)
        return result
    
    def _compute_single_outcome(
        self,
        bar: BarData,
        bars_by_ts: Dict[int, BarData],
        all_bars: List[BarData],
        provider: str,
        symbol: str,
        bar_duration_seconds: int,
        horizon_seconds: int,
    ) -> OutcomeResult:
        """Compute outcome for a single bar and horizon."""
        bar_duration_ms = bar_duration_seconds * 1000
        horizon_ms = horizon_seconds * 1000
        
        # Reference point is bar close
        close_ref_ms = bar.timestamp_ms + bar_duration_ms
        window_start_ms = close_ref_ms
        window_end_ms = window_start_ms + horizon_ms
        
        # Expected bars in window
        bars_expected = horizon_seconds // bar_duration_seconds
        
        # Find future bars strictly within the window
        # A bar is in window if its close time <= window_end_ms
        future_bars: List[BarData] = []
        for b in all_bars:
            bar_close_ms = b.timestamp_ms + bar_duration_ms
            if b.timestamp_ms > bar.timestamp_ms and bar_close_ms <= window_end_ms:
                future_bars.append(b)
        
        bars_found = len(future_bars)
        gap_count = max(0, bars_expected - bars_found)
        
        # Determine status
        if bars_found == 0:
            status = STATUS_INCOMPLETE
        elif gap_count > self.gap_tolerance_bars:
            status = STATUS_GAP
        else:
            status = STATUS_OK
        
        # Compute metrics (only for OK status, or partial for INCOMPLETE)
        close_now = bar.close
        close_at_horizon: Optional[float] = None
        fwd_return: Optional[float] = None
        max_runup: Optional[float] = None
        max_drawdown: Optional[float] = None
        realized_vol: Optional[float] = None
        max_high_in_window: Optional[float] = None
        min_low_in_window: Optional[float] = None
        max_runup_ts_ms: Optional[int] = None
        max_drawdown_ts_ms: Optional[int] = None
        time_to_max_runup_ms: Optional[int] = None
        time_to_max_drawdown_ms: Optional[int] = None
        
        if future_bars:
            # Close at horizon is the close of the last bar in window
            close_at_horizon = future_bars[-1].close
            
            if close_now > 0:
                fwd_return = (close_at_horizon / close_now) - 1
            
            # Only compute path metrics if status is OK
            if status == STATUS_OK:
                # Find max high and min low
                max_high = future_bars[0].high
                max_high_bar = future_bars[0]
                min_low = future_bars[0].low
                min_low_bar = future_bars[0]
                
                for b in future_bars:
                    if b.high > max_high:
                        max_high = b.high
                        max_high_bar = b
                    if b.low < min_low:
                        min_low = b.low
                        min_low_bar = b
                
                max_high_in_window = max_high
                min_low_in_window = min_low
                max_runup_ts_ms = max_high_bar.timestamp_ms + bar_duration_ms
                max_drawdown_ts_ms = min_low_bar.timestamp_ms + bar_duration_ms
                time_to_max_runup_ms = max_runup_ts_ms - window_start_ms
                time_to_max_drawdown_ms = max_drawdown_ts_ms - window_start_ms
                
                if close_now > 0:
                    max_runup = (max_high / close_now) - 1
                    max_drawdown = (min_low / close_now) - 1
                
                # Realized volatility (annualized stddev of log returns)
                realized_vol = self._compute_realized_vol(
                    [bar] + future_bars,
                    bar_duration_seconds,
                )
        
        # Quantize all float metrics
        q = self.quantize_decimals
        # Deterministic computed_at_ms: derived from window_end so reruns
        # on the same bars produce the exact same output (no wall-clock).
        computed_at_ms = window_end_ms
        
        # Session regime at bar open (deterministic from timestamp)
        session = get_session_regime("EQUITIES", bar.timestamp_ms)

        return OutcomeResult(
            provider=provider,
            symbol=symbol,
            bar_duration_seconds=bar_duration_seconds,
            timestamp_ms=bar.timestamp_ms,
            horizon_seconds=horizon_seconds,
            outcome_version=self.outcome_version,
            close_now=_quantize(close_now, q),
            close_at_horizon=_quantize(close_at_horizon, q),
            fwd_return=_quantize(fwd_return, q),
            max_runup=_quantize(max_runup, q),
            max_drawdown=_quantize(max_drawdown, q),
            realized_vol=_quantize(realized_vol, q),
            max_high_in_window=_quantize(max_high_in_window, q),
            min_low_in_window=_quantize(min_low_in_window, q),
            max_runup_ts_ms=max_runup_ts_ms,
            max_drawdown_ts_ms=max_drawdown_ts_ms,
            time_to_max_runup_ms=time_to_max_runup_ms,
            time_to_max_drawdown_ms=time_to_max_drawdown_ms,
            status=status,
            close_ref_ms=close_ref_ms,
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
            bars_expected=bars_expected,
            bars_found=bars_found,
            gap_count=gap_count,
            computed_at_ms=computed_at_ms,
            session_regime=session,
        )
    
    def _compute_realized_vol(
        self, bars: List[BarData], bar_duration_seconds: int
    ) -> Optional[float]:
        """Compute annualized realized volatility as stddev of log returns.

        Returns volatility in same units as IV (annualized decimal, e.g. 0.15 = 15%).
        """
        if len(bars) < 2:
            return None
        
        log_returns = []
        for i in range(1, len(bars)):
            if bars[i-1].close > 0 and bars[i].close > 0:
                log_ret = math.log(bars[i].close / bars[i-1].close)
                log_returns.append(log_ret)
        
        if len(log_returns) < 2:
            return None
        
        # Per-bar standard deviation
        mean = sum(log_returns) / len(log_returns)
        variance = sum((r - mean) ** 2 for r in log_returns) / (len(log_returns) - 1)
        vol_per_bar = math.sqrt(variance) if variance >= 0 else None
        if vol_per_bar is None:
            return None

        # Annualize: vol_annual = vol_per_bar * sqrt(periods_per_year)
        # Trading year: 252 days * 6.5h * 3600s = 5896800 seconds
        periods_per_year = (252 * 6.5 * 3600) / max(1, bar_duration_seconds)
        annualize_factor = math.sqrt(periods_per_year)
        return vol_per_bar * annualize_factor
    
    def compute_outcomes_from_bars_sync(
        self,
        bars: List[BarData],
        provider: str,
        symbol: str,
        bar_duration_seconds: int,
        horizons: List[int],
    ) -> List[OutcomeResult]:
        """Synchronous version for testing - compute outcomes from bar list.
        
        This is useful for unit tests where we don't need DB access.
        """
        bars_by_ts = {b.timestamp_ms: b for b in bars}
        results = []
        
        for bar in bars:
            for horizon in horizons:
                result = self._compute_single_outcome(
                    bar=bar,
                    bars_by_ts=bars_by_ts,
                    all_bars=bars,
                    provider=provider,
                    symbol=symbol,
                    bar_duration_seconds=bar_duration_seconds,
                    horizon_seconds=horizon,
                )
                results.append(result)
        
        return results
