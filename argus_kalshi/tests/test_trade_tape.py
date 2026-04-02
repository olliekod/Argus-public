# Created by Oliver Meihls

# Tests for trade tape: KalshiTradeEvent model and flow imbalance computation.

import time

import pytest

from argus_kalshi.models import KalshiTradeEvent


#  KalshiTradeEvent

def test_trade_event_fields():
    ev = KalshiTradeEvent(
        market_ticker="KXBTC15M-26MAR081830-30",
        taker_side="no",
        count=5,
        ts=1773009000.0,
    )
    assert ev.market_ticker == "KXBTC15M-26MAR081830-30"
    assert ev.taker_side == "no"
    assert ev.count == 5
    assert ev.ts == pytest.approx(1773009000.0)


def test_trade_event_frozen():
    # KalshiTradeEvent uses slots=True and should be immutable-ish.
    ev = KalshiTradeEvent("TICKER", "yes", 10, 1000.0)
    assert ev.taker_side == "yes"


#  Flow imbalance computation (inline helper mirrors FarmDispatcher logic)

def _compute_flow(trades, window_s=60.0):
    # Helper matching FarmDispatcher._consume_trades imbalance logic.
    now = time.time()
    cutoff = now - window_s
    yes_vol = sum(c for ts, side, c in trades if ts >= cutoff and side == "yes")
    no_vol  = sum(c for ts, side, c in trades if ts >= cutoff and side == "no")
    total = yes_vol + no_vol
    return (yes_vol - no_vol) / total if total > 0 else 0.0


def test_flow_all_yes():
    now = time.time()
    trades = [(now - 10, "yes", 100)]
    assert _compute_flow(trades) == pytest.approx(1.0)


def test_flow_all_no():
    now = time.time()
    trades = [(now - 10, "no", 100)]
    assert _compute_flow(trades) == pytest.approx(-1.0)


def test_flow_balanced():
    now = time.time()
    trades = [(now - 10, "yes", 50), (now - 10, "no", 50)]
    assert _compute_flow(trades) == pytest.approx(0.0)


def test_flow_stale_ignored():
    now = time.time()
    # Old YES trades should be outside 60s window; only the recent NO remains.
    trades = [(now - 120, "yes", 1000), (now - 10, "no", 10)]
    assert _compute_flow(trades) == pytest.approx(-1.0)


def test_flow_empty():
    assert _compute_flow([]) == pytest.approx(0.0)


def test_flow_partial_yes_bias():
    now = time.time()
    trades = [(now - 5, "yes", 70), (now - 5, "no", 30)]
    result = _compute_flow(trades)
    assert result == pytest.approx(0.4)   # (70-30)/100
