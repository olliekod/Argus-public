# Created by Oliver Meihls

# Tests for kalshi_probability — settlement probability model.
#
# Covers:
# - Deterministic outputs for fixed inputs
# - Case 1: time_to_settle > 60 s
# - Case 2: time_to_settle ≤ 60 s (inside final window)
# - m_req computation correctness
# - Volatility estimator behavior with missing seconds
# - Edge cases (zero vol, zero time, extreme strikes)

from __future__ import annotations

import math
import pytest

from argus_kalshi.kalshi_probability import (
    _norm_cdf,
    compute_probability,
    estimate_volatility,
)
from argus_kalshi.models import BtcWindowState


#  norm_cdf sanity

def test_norm_cdf_center() -> None:
    assert _norm_cdf(0.0) == pytest.approx(0.5)


def test_norm_cdf_tails() -> None:
    assert _norm_cdf(3.0) == pytest.approx(0.99865, abs=1e-4)
    assert _norm_cdf(-3.0) == pytest.approx(0.00135, abs=1e-4)


#  Volatility estimator

def test_vol_constant_prices() -> None:
    # Constant prices → zero volatility.
    prices = [100.0] * 60
    assert estimate_volatility(prices) == 0.0


def test_vol_two_prices() -> None:
    # Two prices → single log-return, variance is 0 with n-1=0 denom guard.
    # With only 1 log-return, the sample variance uses n-1=0 → degenerate.
    # Our implementation uses n-1 so with n=1 it returns 0.
    prices = [100.0, 101.0]
    vol = estimate_volatility(prices)
    # With only one return, variance = 0 (n-1 denominator with n=1).
    assert vol == 0.0


def test_vol_known_returns() -> None:
    # Fabricated prices with known log-returns.
    # Prices: 100, 100*e^0.01, 100*e^0.02 → log-returns = [0.01, 0.01]
    p0 = 100.0
    p1 = p0 * math.exp(0.01)
    p2 = p1 * math.exp(0.01)
    vol = estimate_volatility([p0, p1, p2])
    # Both log-returns are 0.01, so variance = 0, vol = 0.
    assert vol == pytest.approx(0.0, abs=1e-10)


def test_vol_with_variation() -> None:
    # Prices with actual variation produce positive vol.
    p0 = 100.0
    p1 = p0 * math.exp(0.01)
    p2 = p1 * math.exp(-0.02)
    p3 = p2 * math.exp(0.005)
    vol = estimate_volatility([p0, p1, p2, p3])
    assert vol > 0


def test_vol_robust_to_short_list() -> None:
    # Fewer than 2 prices → zero vol.
    assert estimate_volatility([]) == 0.0
    assert estimate_volatility([100.0]) == 0.0


#  Case 1: time_to_settle > 60 s (full window in the future)

def test_prob_at_the_money_high_vol() -> None:
    # ATM with non-zero vol → ~0.5.
    p = compute_probability(
        strike=50000.0,
        current_price=50000.0,
        sigma=0.001,
        time_to_settle_s=120.0,
    )
    assert p == pytest.approx(0.5, abs=0.01)


def test_prob_deep_itm() -> None:
    # Price well above strike → P(YES) ≈ 1.
    p = compute_probability(
        strike=40000.0,
        current_price=60000.0,
        sigma=0.0005,
        time_to_settle_s=120.0,
    )
    assert p > 0.99


def test_prob_deep_otm() -> None:
    # Price well below strike → P(YES) ≈ 0.
    p = compute_probability(
        strike=60000.0,
        current_price=40000.0,
        sigma=0.0005,
        time_to_settle_s=120.0,
    )
    assert p < 0.01


def test_prob_deterministic() -> None:
    # Same inputs always produce the same output.
    kwargs = dict(
        strike=65000.0,
        current_price=64500.0,
        sigma=0.0003,
        time_to_settle_s=90.0,
    )
    p1 = compute_probability(**kwargs)
    p2 = compute_probability(**kwargs)
    assert p1 == p2


#  Case 2: inside the 60-second window

def test_inside_window_observed_dominates() -> None:
    # When most of the window is observed, the result is nearly deterministic.
    # 55 seconds observed at 65000, 5 remaining, strike=65000
    ws = BtcWindowState(
        last_60_sum=55 * 65000.0,
        last_60_avg=65000.0,
        count=55,
        timestamp=0.0,
    )
    p = compute_probability(
        strike=65000.0,
        current_price=65000.0,
        sigma=0.0003,
        time_to_settle_s=5.0,
        window_state=ws,
    )
    # With 55/60 seconds locked in at the strike and current price at
    # strike, probability should be close to 0.5.
    assert 0.3 < p < 0.7


def test_inside_window_already_won() -> None:
    # Observed sum so high that remaining can't lose.
    # 59 seconds observed at 70000, strike=60000, 1 second remaining
    ws = BtcWindowState(
        last_60_sum=59 * 70000.0,
        last_60_avg=70000.0,
        count=59,
        timestamp=0.0,
    )
    p = compute_probability(
        strike=60000.0,
        current_price=70000.0,
        sigma=0.001,
        time_to_settle_s=1.0,
        window_state=ws,
    )
    # m_req = (60*60000 - 59*70000) / 1 = 3600000 - 4130000 = -530000
    # Current price 70000 >> m_req → P ≈ 1
    assert p > 0.99


def test_inside_window_already_lost() -> None:
    # Observed sum so low that remaining can't win.
    # 59 seconds at 50000, strike=65000, 1 second remaining
    ws = BtcWindowState(
        last_60_sum=59 * 50000.0,
        last_60_avg=50000.0,
        count=59,
        timestamp=0.0,
    )
    p = compute_probability(
        strike=65000.0,
        current_price=50000.0,
        sigma=0.001,
        time_to_settle_s=1.0,
        window_state=ws,
    )
    # m_req = (60*65000 - 59*50000) / 1 = 3900000 - 2950000 = 950000
    # Current price 50000 << 950000 → P ≈ 0
    assert p < 0.01


def test_m_req_computation() -> None:
    # Verify the m_req = (60*K - S_obs) / τ formula manually.
    K = 65000.0
    s_obs = 55 * 64000.0   # 55 seconds at 64000
    tau = 5.0
    m_req = (60 * K - s_obs) / tau
    # m_req = (3900000 - 3520000) / 5 = 76000

    ws = BtcWindowState(
        last_60_sum=s_obs,
        last_60_avg=64000.0,
        count=55,
        timestamp=0.0,
    )
    p = compute_probability(
        strike=K,
        current_price=64000.0,
        sigma=0.001,
        time_to_settle_s=tau,
        window_state=ws,
    )
    # Current price (64000) < m_req (76000) → P(YES) should be low.
    assert p < 0.15


#  Edge cases

def test_zero_time_with_window() -> None:
    # time_to_settle=0 with window → deterministic based on avg.
    ws = BtcWindowState(
        last_60_sum=60 * 66000.0,
        last_60_avg=66000.0,
        count=60,
        timestamp=0.0,
    )
    p = compute_probability(
        strike=65000.0,
        current_price=66000.0,
        sigma=0.001,
        time_to_settle_s=0.0,
        window_state=ws,
    )
    assert p == 1.0  # avg=66000 ≥ strike=65000


def test_zero_vol_atm() -> None:
    # Zero volatility at the money → P=0.5 (edge of the CDF).
    # With sigma clamped to 1e-12, the z-score is huge or tiny.
    # ATM: z ≈ 0, so P ≈ 0.5
    p = compute_probability(
        strike=50000.0,
        current_price=50000.0,
        sigma=0.0,
        time_to_settle_s=120.0,
    )
    assert p == pytest.approx(0.5, abs=0.01)


def test_negative_time() -> None:
    # Negative time_to_settle → fallback behavior.
    p = compute_probability(
        strike=50000.0,
        current_price=50000.0,
        sigma=0.001,
        time_to_settle_s=-10.0,
    )
    # No window state → returns 0.5.
    assert p == 0.5


def test_prob_bounded_0_1() -> None:
    # P(YES) is always in [0, 1].
    import random
    random.seed(123)
    for _ in range(100):
        strike = random.uniform(30000, 80000)
        price = random.uniform(30000, 80000)
        sigma = random.uniform(0.0001, 0.01)
        tts = random.uniform(0, 300)
        p = compute_probability(strike, price, sigma, tts)
        assert 0.0 <= p <= 1.0


#  Drift parameter

def test_drift_shifts_p_yes_upward():
    # Positive drift should increase p_yes vs zero drift.
    base = compute_probability(
        strike=100.0, current_price=98.0, sigma=0.0001,
        time_to_settle_s=120.0, drift=0.0,
    )
    with_drift = compute_probability(
        strike=100.0, current_price=98.0, sigma=0.0001,
        time_to_settle_s=120.0, drift=0.001,  # uptrend
    )
    assert with_drift > base


def test_drift_shifts_p_yes_downward():
    base = compute_probability(
        strike=100.0, current_price=102.0, sigma=0.0001,
        time_to_settle_s=120.0, drift=0.0,
    )
    with_drift = compute_probability(
        strike=100.0, current_price=102.0, sigma=0.0001,
        time_to_settle_s=120.0, drift=-0.001,  # downtrend
    )
    assert with_drift < base


def test_zero_drift_unchanged():
    # drift=0 must be identical to old behaviour (backward compat).
    p1 = compute_probability(100.0, 99.0, 0.0001, 90.0, drift=0.0)
    p2 = compute_probability(100.0, 99.0, 0.0001, 90.0)  # no drift kwarg
    assert p1 == pytest.approx(p2, abs=1e-9)
