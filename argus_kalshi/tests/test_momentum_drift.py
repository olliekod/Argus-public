# Created by Oliver Meihls

import pytest
from argus_kalshi.kalshi_probability import estimate_momentum


def test_momentum_flat_prices():
    prices = [1000.0] * 60
    assert estimate_momentum(prices) == pytest.approx(0.0, abs=1e-9)


def test_momentum_uptrend():
    # Consistent 0.1%/step uptrend over 30 prices
    prices = [1000.0 * (1.001 ** i) for i in range(30)]
    m = estimate_momentum(prices)
    assert m > 0.0005  # clearly positive


def test_momentum_downtrend():
    prices = [1000.0 * (0.999 ** i) for i in range(30)]
    m = estimate_momentum(prices)
    assert m < -0.0005  # clearly negative


def test_momentum_too_few_prices():
    assert estimate_momentum([]) == 0.0
    assert estimate_momentum([1000.0]) == 0.0
    assert estimate_momentum([1000.0, 1001.0]) == 0.0


def test_momentum_window_respected():
    # Only the last `window` prices should matter
    old_prices = [500.0] * 100     # old downtrend — should be ignored
    new_prices = [1000.0 * (1.001 ** i) for i in range(30)]  # uptrend
    m = estimate_momentum(old_prices + new_prices, window=30)
    assert m > 0.0  # uptrend wins
