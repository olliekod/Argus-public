"""
Argus Indicators Module
=======================

Pure, deterministic technical indicators for live, replay, and backtest.

This module provides two interfaces for each indicator:
1. **Batch functions** — stateless, take full sequence, return full result sequence
2. **Incremental state classes** — stateful, update one value at a time

PARITY GUARANTEE: Incremental outputs match batch outputs exactly for the
same input series. This is enforced by tests.

WARMUP SEMANTICS: Both batch and incremental return `None` for positions
where insufficient data exists.

NO RANDOMNESS. NO WALL-CLOCK. Fully deterministic.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, NamedTuple, Optional, Sequence


# ═══════════════════════════════════════════════════════════════════════════
# Data Types
# ═══════════════════════════════════════════════════════════════════════════


class BarTuple(NamedTuple):
    """Lightweight bar representation for indicator calculations."""
    timestamp: float  # bar open time (UTC epoch)
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True, slots=True)
class MACDResult:
    """MACD indicator output."""
    macd_line: float
    signal_line: float
    histogram: float


class InsufficientDataError(Exception):
    """Raised when insufficient data for indicator calculation."""
    pass


# ═══════════════════════════════════════════════════════════════════════════
# EMA (Exponential Moving Average)
# ═══════════════════════════════════════════════════════════════════════════


def ema_batch(prices: Sequence[float], period: int) -> List[Optional[float]]:
    """
    Compute EMA for entire price series.
    
    Returns list of same length as prices.
    First (period-1) values are None (warmup).
    
    Uses Wilder's smoothing: alpha = 2 / (period + 1)
    Initial EMA = SMA of first `period` prices.
    """
    if period < 1:
        raise ValueError("period must be >= 1")
    
    n = len(prices)
    result: List[Optional[float]] = [None] * n
    
    if n < period:
        return result
    
    # Initial SMA
    sma = sum(prices[:period]) / period
    result[period - 1] = sma
    
    # EMA from period onwards
    alpha = 2.0 / (period + 1)
    ema = sma
    for i in range(period, n):
        ema = alpha * prices[i] + (1 - alpha) * ema
        result[i] = ema
    
    return result


class EMAState:
    """
    Incremental EMA calculator.
    
    Parameters
    ----------
    period : int
        EMA period.
    """
    __slots__ = ("_period", "_alpha", "_ema", "_warmup", "_warmup_sum")
    
    def __init__(self, period: int) -> None:
        if period < 1:
            raise ValueError("period must be >= 1")
        self._period = period
        self._alpha = 2.0 / (period + 1)
        self._ema: Optional[float] = None
        self._warmup: int = 0
        self._warmup_sum: float = 0.0
    
    def update(self, price: float) -> Optional[float]:
        """
        Update EMA with new price.
        
        Returns None during warmup (first period-1 values).
        """
        if self._ema is None:
            # Still in warmup
            self._warmup_sum += price
            self._warmup += 1
            if self._warmup >= self._period:
                # Initialize with SMA
                self._ema = self._warmup_sum / self._period
                return self._ema
            return None
        else:
            self._ema = self._alpha * price + (1 - self._alpha) * self._ema
            return self._ema
    
    def reset(self) -> None:
        """Reset state for new session."""
        self._ema = None
        self._warmup = 0
        self._warmup_sum = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# RSI (Relative Strength Index — Wilder's smoothed)
# ═══════════════════════════════════════════════════════════════════════════


def rsi_batch(prices: Sequence[float], period: int = 14) -> List[Optional[float]]:
    """
    Compute RSI for entire price series using Wilder's smoothing.
    
    Returns list of same length as prices.
    First `period` values are None (need period+1 prices for period changes).
    """
    if period < 1:
        raise ValueError("period must be >= 1")
    
    n = len(prices)
    result: List[Optional[float]] = [None] * n
    
    if n < period + 1:
        return result
    
    # Calculate initial gains/losses
    gains: List[float] = []
    losses: List[float] = []
    for i in range(1, period + 1):
        change = prices[i] - prices[i - 1]
        if change > 0:
            gains.append(change)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(-change)
    
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    
    # First RSI
    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - (100.0 / (1.0 + rs))
    
    # Subsequent RSIs using Wilder's smoothing
    for i in range(period + 1, n):
        change = prices[i] - prices[i - 1]
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        
        if avg_loss == 0:
            result[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return result


class RSIState:
    """
    Incremental RSI calculator using Wilder's smoothing.
    
    Parameters
    ----------
    period : int
        RSI period (default 14).
    """
    __slots__ = ("_period", "_prev_price", "_warmup", "_gains", "_losses",
                 "_avg_gain", "_avg_loss")
    
    def __init__(self, period: int = 14) -> None:
        if period < 1:
            raise ValueError("period must be >= 1")
        self._period = period
        self._prev_price: Optional[float] = None
        self._warmup: int = 0
        self._gains: List[float] = []
        self._losses: List[float] = []
        self._avg_gain: Optional[float] = None
        self._avg_loss: Optional[float] = None
    
    def update(self, price: float) -> Optional[float]:
        """
        Update RSI with new price.
        
        Returns None during warmup (first `period` prices).
        """
        if self._prev_price is None:
            self._prev_price = price
            return None
        
        change = price - self._prev_price
        self._prev_price = price
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        
        if self._avg_gain is None:
            # Still collecting initial period
            self._gains.append(gain)
            self._losses.append(loss)
            self._warmup += 1
            
            if self._warmup >= self._period:
                self._avg_gain = sum(self._gains) / self._period
                self._avg_loss = sum(self._losses) / self._period
                self._gains = []  # Free memory
                self._losses = []
                return self._compute_rsi()
            return None
        else:
            # Wilder's smoothing
            self._avg_gain = (self._avg_gain * (self._period - 1) + gain) / self._period
            self._avg_loss = (self._avg_loss * (self._period - 1) + loss) / self._period
            return self._compute_rsi()
    
    def _compute_rsi(self) -> float:
        if self._avg_loss == 0:
            return 100.0
        rs = self._avg_gain / self._avg_loss
        return 100.0 - (100.0 / (1.0 + rs))
    
    def reset(self) -> None:
        """Reset state for new session."""
        self._prev_price = None
        self._warmup = 0
        self._gains = []
        self._losses = []
        self._avg_gain = None
        self._avg_loss = None


# ═══════════════════════════════════════════════════════════════════════════
# VWAP (Volume Weighted Average Price)
# ═══════════════════════════════════════════════════════════════════════════


def vwap_batch(bars: Sequence[BarTuple]) -> List[Optional[float]]:
    """
    Compute session VWAP for bar series.
    
    Uses typical price = (high + low + close) / 3.
    Assumes all bars are in same session. For multi-session,
    split bars by session before calling.
    
    Returns None for bars with zero cumulative volume.
    """
    n = len(bars)
    result: List[Optional[float]] = [None] * n
    
    cum_vol = 0.0
    cum_vp = 0.0
    
    for i, bar in enumerate(bars):
        typical_price = (bar.high + bar.low + bar.close) / 3.0
        cum_vp += typical_price * bar.volume
        cum_vol += bar.volume
        
        if cum_vol > 0:
            result[i] = cum_vp / cum_vol
    
    return result


class VWAPState:
    """
    Incremental VWAP calculator.
    
    Call reset() at session boundaries.
    """
    __slots__ = ("_cum_vol", "_cum_vp")
    
    def __init__(self) -> None:
        self._cum_vol: float = 0.0
        self._cum_vp: float = 0.0
    
    def update(self, bar: BarTuple) -> Optional[float]:
        """
        Update VWAP with new bar.
        
        Returns None if cumulative volume is zero.
        """
        typical_price = (bar.high + bar.low + bar.close) / 3.0
        self._cum_vp += typical_price * bar.volume
        self._cum_vol += bar.volume
        
        if self._cum_vol > 0:
            return self._cum_vp / self._cum_vol
        return None
    
    def reset(self) -> None:
        """Reset for new session."""
        self._cum_vol = 0.0
        self._cum_vp = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# MACD (Moving Average Convergence Divergence)
# ═══════════════════════════════════════════════════════════════════════════


def macd_batch(
    prices: Sequence[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> List[Optional[MACDResult]]:
    """
    Compute MACD for entire price series.
    
    Returns list of MACDResult. None during warmup.
    Warmup = slow - 1 + signal - 1 = slow + signal - 2 prices.
    """
    if fast < 1 or slow < 1 or signal < 1:
        raise ValueError("periods must be >= 1")
    if fast >= slow:
        raise ValueError("fast period must be < slow period")
    
    n = len(prices)
    result: List[Optional[MACDResult]] = [None] * n
    
    # Compute EMAs
    fast_ema = ema_batch(prices, fast)
    slow_ema = ema_batch(prices, slow)
    
    # MACD line = fast EMA - slow EMA
    macd_line: List[Optional[float]] = [None] * n
    for i in range(n):
        if fast_ema[i] is not None and slow_ema[i] is not None:
            macd_line[i] = fast_ema[i] - slow_ema[i]
    
    # Signal line = EMA of MACD line
    # Need to compute EMA only over non-None MACD values
    macd_values = [m for m in macd_line if m is not None]
    if len(macd_values) < signal:
        return result
    
    signal_ema = ema_batch(macd_values, signal)
    
    # Map signal EMA back to original indices
    macd_start = next(i for i, m in enumerate(macd_line) if m is not None)
    
    for i, sig in enumerate(signal_ema):
        if sig is not None:
            orig_idx = macd_start + i
            macd_val = macd_line[orig_idx]
            result[orig_idx] = MACDResult(
                macd_line=macd_val,
                signal_line=sig,
                histogram=macd_val - sig,
            )
    
    return result


class MACDState:
    """
    Incremental MACD calculator.
    
    Parameters
    ----------
    fast : int
        Fast EMA period (default 12).
    slow : int
        Slow EMA period (default 26).
    signal : int
        Signal line EMA period (default 9).
    """
    __slots__ = ("_fast_ema", "_slow_ema", "_signal_ema")
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        if fast < 1 or slow < 1 or signal < 1:
            raise ValueError("periods must be >= 1")
        if fast >= slow:
            raise ValueError("fast period must be < slow period")
        self._fast_ema = EMAState(fast)
        self._slow_ema = EMAState(slow)
        self._signal_ema = EMAState(signal)
    
    def update(self, price: float) -> Optional[MACDResult]:
        """
        Update MACD with new price.
        
        Returns None during warmup.
        """
        fast_val = self._fast_ema.update(price)
        slow_val = self._slow_ema.update(price)
        
        if fast_val is None or slow_val is None:
            return None
        
        macd_line = fast_val - slow_val
        signal_val = self._signal_ema.update(macd_line)
        
        if signal_val is None:
            return None
        
        return MACDResult(
            macd_line=macd_line,
            signal_line=signal_val,
            histogram=macd_line - signal_val,
        )
    
    def reset(self) -> None:
        """Reset for new session."""
        self._fast_ema.reset()
        self._slow_ema.reset()
        self._signal_ema.reset()


# ═══════════════════════════════════════════════════════════════════════════
# Log Return
# ═══════════════════════════════════════════════════════════════════════════


def log_return(price_prev: float, price_curr: float) -> float:
    """
    Compute single-period log return.
    
    Returns math.log(price_curr / price_prev).
    Raises ValueError if either price <= 0.
    """
    if price_prev <= 0 or price_curr <= 0:
        raise ValueError("prices must be positive")
    return math.log(price_curr / price_prev)


def log_returns_batch(prices: Sequence[float]) -> List[Optional[float]]:
    """
    Compute log returns for price series.
    
    First value is None (no prior price).
    """
    n = len(prices)
    result: List[Optional[float]] = [None] * n
    
    for i in range(1, n):
        if prices[i - 1] > 0 and prices[i] > 0:
            result[i] = math.log(prices[i] / prices[i - 1])
    
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Rolling Volatility
# ═══════════════════════════════════════════════════════════════════════════


def rolling_vol_batch(
    returns: Sequence[float],
    window: int,
    annualize_factor: float,
) -> List[Optional[float]]:
    """
    Compute rolling annualized volatility from log returns.
    
    Returns None for first (window-1) values.
    
    Parameters
    ----------
    returns : sequence of float
        Log returns.
    window : int
        Rolling window size.
    annualize_factor : float
        Annualization multiplier (e.g., sqrt(252) for daily, sqrt(365.25*24*60) for 1-min).
    """
    if window < 2:
        raise ValueError("window must be >= 2")
    
    n = len(returns)
    result: List[Optional[float]] = [None] * n
    
    if n < window:
        return result
    
    # Use deque for efficient rolling window
    buf: Deque[float] = deque(maxlen=window)
    
    for i, ret in enumerate(returns):
        buf.append(ret)
        
        if len(buf) >= window:
            mean = sum(buf) / window
            variance = sum((r - mean) ** 2 for r in buf) / (window - 1)
            result[i] = math.sqrt(variance) * annualize_factor
    
    return result


class RollingVolState:
    """
    Incremental rolling volatility calculator.
    
    Parameters
    ----------
    window : int
        Rolling window size.
    annualize_factor : float
        Annualization multiplier.
    """
    __slots__ = ("_window", "_annualize", "_buf")
    
    def __init__(self, window: int, annualize_factor: float) -> None:
        if window < 2:
            raise ValueError("window must be >= 2")
        self._window = window
        self._annualize = annualize_factor
        self._buf: Deque[float] = deque(maxlen=window)
    
    def update(self, log_return: float) -> Optional[float]:
        """
        Update with new log return.
        
        Returns None during warmup.
        """
        self._buf.append(log_return)
        
        if len(self._buf) < self._window:
            return None
        
        mean = sum(self._buf) / self._window
        variance = sum((r - mean) ** 2 for r in self._buf) / (self._window - 1)
        return math.sqrt(variance) * self._annualize
    
    def reset(self) -> None:
        """Reset for new session."""
        self._buf.clear()


# ═══════════════════════════════════════════════════════════════════════════
# ATR (Average True Range)
# ═══════════════════════════════════════════════════════════════════════════


def _true_range(high: float, low: float, close: float, prev_close: float) -> float:
    """Compute true range."""
    return max(
        high - low,
        abs(high - prev_close),
        abs(low - prev_close),
    )


def atr_batch(bars: Sequence[BarTuple], period: int = 14) -> List[Optional[float]]:
    """
    Compute ATR for bar series using Wilder's smoothing.
    
    Returns None for first `period` bars.
    """
    if period < 1:
        raise ValueError("period must be >= 1")
    
    n = len(bars)
    result: List[Optional[float]] = [None] * n
    
    if n < period + 1:
        return result
    
    # Collect first `period` true ranges (need period+1 bars for period TRs)
    trs: List[float] = []
    for i in range(1, period + 1):
        tr = _true_range(
            bars[i].high, bars[i].low, bars[i].close,
            bars[i - 1].close,
        )
        trs.append(tr)
    
    # Initial ATR = average of first `period` TRs
    atr = sum(trs) / period
    result[period] = atr
    
    # Subsequent ATRs using Wilder's smoothing
    for i in range(period + 1, n):
        tr = _true_range(
            bars[i].high, bars[i].low, bars[i].close,
            bars[i - 1].close,
        )
        atr = (atr * (period - 1) + tr) / period
        result[i] = atr
    
    return result


class ATRState:
    """
    Incremental ATR calculator using Wilder's smoothing.
    
    Parameters
    ----------
    period : int
        ATR period (default 14).
    """
    __slots__ = ("_period", "_prev_close", "_warmup", "_trs", "_atr")
    
    def __init__(self, period: int = 14) -> None:
        if period < 1:
            raise ValueError("period must be >= 1")
        self._period = period
        self._prev_close: Optional[float] = None
        self._warmup: int = 0
        self._trs: List[float] = []
        self._atr: Optional[float] = None
    
    def update(self, high: float, low: float, close: float) -> Optional[float]:
        """
        Update ATR with new bar data.
        
        Returns None during warmup.
        """
        if self._prev_close is None:
            self._prev_close = close
            return None
        
        tr = _true_range(high, low, close, self._prev_close)
        self._prev_close = close
        
        if self._atr is None:
            # Still in warmup
            self._trs.append(tr)
            self._warmup += 1
            
            if self._warmup >= self._period:
                self._atr = sum(self._trs) / self._period
                self._trs = []  # Free memory
                return self._atr
            return None
        else:
            # Wilder's smoothing
            self._atr = (self._atr * (self._period - 1) + tr) / self._period
            return self._atr
    
    def reset(self) -> None:
        """Reset for new session."""
        self._prev_close = None
        self._warmup = 0
        self._trs = []
        self._atr = None
