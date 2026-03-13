"""
Tests for Argus Indicators Module.

Verifies:
1. Batch and incremental outputs match exactly
2. Warmup behavior is consistent
3. Determinism (same input → same output)

Run with: python -m pytest tests/test_indicators.py -v
"""

import math
import pytest
from typing import List, Optional

from src.core.indicators import (
    # Batch functions
    ema_batch,
    rsi_batch,
    vwap_batch,
    macd_batch,
    log_returns_batch,
    rolling_vol_batch,
    atr_batch,
    # Incremental state classes
    EMAState,
    RSIState,
    VWAPState,
    MACDState,
    RollingVolState,
    ATRState,
    # Types
    BarTuple,
    MACDResult,
    log_return,
)


# ═══════════════════════════════════════════════════════════════════════════
# Deterministic Fixture Data
# ═══════════════════════════════════════════════════════════════════════════

# Price series with known pattern for reproducible tests
FIXTURE_PRICES = [
    100.0, 101.5, 99.8, 102.3, 101.0,
    103.5, 104.2, 102.8, 105.0, 104.5,
    106.0, 105.5, 107.2, 108.0, 106.5,
    109.0, 108.5, 110.0, 109.5, 111.0,
    110.5, 112.0, 111.5, 113.0, 112.5,
    114.0, 113.5, 115.0, 114.5, 116.0,
    115.5, 117.0, 116.5, 118.0, 117.5,
    119.0, 118.5, 120.0, 119.5, 121.0,
]

# Bar data for VWAP/ATR tests
FIXTURE_BARS = [
    BarTuple(timestamp=1700000000 + i * 60, open=p, high=p + 0.5, low=p - 0.3, close=p + 0.1, volume=1000 + i * 10)
    for i, p in enumerate(FIXTURE_PRICES)
]


# ═══════════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════════

def assert_close(a: Optional[float], b: Optional[float], tol: float = 1e-10) -> None:
    """Assert two values are close or both None."""
    if a is None and b is None:
        return
    if a is None or b is None:
        pytest.fail(f"Mismatch: one is None, other is {a or b}")
    if abs(a - b) > tol:
        pytest.fail(f"Values differ: {a} vs {b}, diff={abs(a - b)}")


def assert_parity(
    batch_results: List[Optional[float]],
    incremental_results: List[Optional[float]],
    name: str,
    tol: float = 1e-10,
) -> None:
    """Assert batch and incremental results match at every position."""
    assert len(batch_results) == len(incremental_results), (
        f"{name}: length mismatch {len(batch_results)} vs {len(incremental_results)}"
    )
    for i, (b, inc) in enumerate(zip(batch_results, incremental_results)):
        if b is None and inc is None:
            continue
        if b is None or inc is None:
            pytest.fail(f"{name}[{i}]: one is None, other is not (batch={b}, inc={inc})")
        if abs(b - inc) > tol:
            pytest.fail(f"{name}[{i}]: mismatch batch={b}, inc={inc}, diff={abs(b - inc)}")


# ═══════════════════════════════════════════════════════════════════════════
# EMA Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestEMA:
    def test_batch_incremental_parity(self):
        """Batch and incremental EMA must match exactly."""
        period = 10
        batch = ema_batch(FIXTURE_PRICES, period)
        
        state = EMAState(period)
        incremental = [state.update(p) for p in FIXTURE_PRICES]
        
        assert_parity(batch, incremental, "EMA")
    
    def test_warmup(self):
        """First (period-1) values should be None."""
        period = 5
        batch = ema_batch(FIXTURE_PRICES, period)
        
        for i in range(period - 1):
            assert batch[i] is None, f"EMA[{i}] should be None during warmup"
        assert batch[period - 1] is not None, f"EMA[{period-1}] should have value"
    
    def test_determinism(self):
        """Same input produces same output."""
        r1 = ema_batch(FIXTURE_PRICES, 10)
        r2 = ema_batch(FIXTURE_PRICES, 10)
        assert r1 == r2
    
    def test_invalid_period(self):
        with pytest.raises(ValueError):
            ema_batch(FIXTURE_PRICES, 0)
        with pytest.raises(ValueError):
            EMAState(0)


class TestRSI:
    def test_batch_incremental_parity(self):
        """Batch and incremental RSI must match exactly."""
        period = 14
        batch = rsi_batch(FIXTURE_PRICES, period)
        
        state = RSIState(period)
        incremental = [state.update(p) for p in FIXTURE_PRICES]
        
        assert_parity(batch, incremental, "RSI")
    
    def test_warmup(self):
        """First `period` values should be None."""
        period = 14
        batch = rsi_batch(FIXTURE_PRICES, period)
        
        for i in range(period):
            assert batch[i] is None, f"RSI[{i}] should be None during warmup"
        assert batch[period] is not None, f"RSI[{period}] should have value"
    
    def test_bounds(self):
        """RSI should be between 0 and 100."""
        batch = rsi_batch(FIXTURE_PRICES, 14)
        for val in batch:
            if val is not None:
                assert 0 <= val <= 100, f"RSI out of bounds: {val}"
    
    def test_determinism(self):
        r1 = rsi_batch(FIXTURE_PRICES, 14)
        r2 = rsi_batch(FIXTURE_PRICES, 14)
        assert r1 == r2


class TestVWAP:
    def test_batch_incremental_parity(self):
        """Batch and incremental VWAP must match exactly."""
        batch = vwap_batch(FIXTURE_BARS)
        
        state = VWAPState()
        incremental = [state.update(bar) for bar in FIXTURE_BARS]
        
        assert_parity(batch, incremental, "VWAP")
    
    def test_first_bar_has_value(self):
        """VWAP should have value from first bar (if volume > 0)."""
        batch = vwap_batch(FIXTURE_BARS)
        assert batch[0] is not None
    
    def test_reset(self):
        """Reset should clear state for new session."""
        state = VWAPState()
        for bar in FIXTURE_BARS[:10]:
            state.update(bar)
        
        val_before_reset = state.update(FIXTURE_BARS[10])
        
        state.reset()
        state2 = VWAPState()
        
        # After reset, should behave like fresh state
        val_after_reset = state.update(FIXTURE_BARS[0])
        val_fresh = state2.update(FIXTURE_BARS[0])
        
        assert_close(val_after_reset, val_fresh)


class TestMACD:
    def test_batch_incremental_parity(self):
        """Batch and incremental MACD must match exactly."""
        batch = macd_batch(FIXTURE_PRICES, fast=12, slow=26, signal=9)
        
        state = MACDState(fast=12, slow=26, signal=9)
        incremental = [state.update(p) for p in FIXTURE_PRICES]
        
        # Compare each component
        for i, (b, inc) in enumerate(zip(batch, incremental)):
            if b is None and inc is None:
                continue
            if b is None or inc is None:
                pytest.fail(f"MACD[{i}]: mismatch None state")
            assert_close(b.macd_line, inc.macd_line, tol=1e-10)
            assert_close(b.signal_line, inc.signal_line, tol=1e-10)
            assert_close(b.histogram, inc.histogram, tol=1e-10)
    
    def test_warmup(self):
        """MACD needs slow + signal - 2 warmup bars."""
        fast, slow, signal = 12, 26, 9
        expected_warmup = slow + signal - 2  # 33
        
        batch = macd_batch(FIXTURE_PRICES, fast, slow, signal)
        
        # All values before warmup should be None
        for i in range(min(len(FIXTURE_PRICES), expected_warmup)):
            assert batch[i] is None, f"MACD[{i}] should be None during warmup"
    
    def test_invalid_params(self):
        with pytest.raises(ValueError):
            macd_batch(FIXTURE_PRICES, fast=26, slow=12, signal=9)  # fast >= slow


class TestLogReturns:
    def test_single_return(self):
        ret = log_return(100.0, 110.0)
        expected = math.log(110.0 / 100.0)
        assert abs(ret - expected) < 1e-10
    
    def test_batch(self):
        batch = log_returns_batch(FIXTURE_PRICES)
        assert batch[0] is None  # First value has no prior
        assert batch[1] is not None
        
        # Verify computation
        expected = math.log(FIXTURE_PRICES[1] / FIXTURE_PRICES[0])
        assert abs(batch[1] - expected) < 1e-10
    
    def test_invalid_price(self):
        with pytest.raises(ValueError):
            log_return(0.0, 100.0)
        with pytest.raises(ValueError):
            log_return(100.0, -1.0)


class TestRollingVol:
    def test_batch_incremental_parity(self):
        """Batch and incremental rolling vol must match exactly."""
        returns = log_returns_batch(FIXTURE_PRICES)
        # Filter out None for rolling vol
        valid_returns = [r for r in returns if r is not None]
        
        window = 10
        annualize = math.sqrt(365.25 * 24 * 60)  # 1-minute bars
        
        batch = rolling_vol_batch(valid_returns, window, annualize)
        
        state = RollingVolState(window, annualize)
        incremental = [state.update(r) for r in valid_returns]
        
        assert_parity(batch, incremental, "RollingVol")
    
    def test_warmup(self):
        """First (window-1) values should be None."""
        window = 10
        returns = [0.01] * 20
        batch = rolling_vol_batch(returns, window, 1.0)
        
        for i in range(window - 1):
            assert batch[i] is None
        assert batch[window - 1] is not None


class TestATR:
    def test_batch_incremental_parity(self):
        """Batch and incremental ATR must match exactly."""
        period = 14
        batch = atr_batch(FIXTURE_BARS, period)
        
        state = ATRState(period)
        incremental = [state.update(bar.high, bar.low, bar.close) for bar in FIXTURE_BARS]
        
        assert_parity(batch, incremental, "ATR")
    
    def test_warmup(self):
        """First `period` values should be None."""
        period = 14
        batch = atr_batch(FIXTURE_BARS, period)
        
        for i in range(period):
            assert batch[i] is None, f"ATR[{i}] should be None during warmup"
        assert batch[period] is not None, f"ATR[{period}] should have value"
    
    def test_positive(self):
        """ATR should always be non-negative."""
        batch = atr_batch(FIXTURE_BARS, 14)
        for val in batch:
            if val is not None:
                assert val >= 0, f"ATR should be non-negative: {val}"


# ═══════════════════════════════════════════════════════════════════════════
# Comprehensive Parity Test
# ═══════════════════════════════════════════════════════════════════════════

class TestComprehensiveParity:
    """Run all parity checks with longer series to catch edge cases."""
    
    @pytest.fixture
    def long_prices(self):
        """Generate 200 prices with realistic variation."""
        import random
        random.seed(42)  # Deterministic!
        prices = [100.0]
        for _ in range(199):
            change = random.gauss(0, 1)
            prices.append(max(1.0, prices[-1] * (1 + change * 0.01)))
        return prices
    
    def test_ema_long_series(self, long_prices):
        for period in [5, 10, 20, 50]:
            batch = ema_batch(long_prices, period)
            state = EMAState(period)
            incremental = [state.update(p) for p in long_prices]
            assert_parity(batch, incremental, f"EMA({period})")
    
    def test_rsi_long_series(self, long_prices):
        for period in [7, 14, 21]:
            batch = rsi_batch(long_prices, period)
            state = RSIState(period)
            incremental = [state.update(p) for p in long_prices]
            assert_parity(batch, incremental, f"RSI({period})")
    
    def test_macd_long_series(self, long_prices):
        batch = macd_batch(long_prices, 12, 26, 9)
        state = MACDState(12, 26, 9)
        incremental = [state.update(p) for p in long_prices]
        
        for i, (b, inc) in enumerate(zip(batch, incremental)):
            if b is None and inc is None:
                continue
            if b is None or inc is None:
                pytest.fail(f"MACD[{i}]: parity failure")
            assert_close(b.macd_line, inc.macd_line)
            assert_close(b.signal_line, inc.signal_line)
            assert_close(b.histogram, inc.histogram)
