"""
Tests for the Conservative Execution Model.

Validates fill logic, rejection rules, spread fills, and the ledger.
"""

from __future__ import annotations

import pytest

from src.analysis.execution_model import (
    ExecutionConfig,
    ExecutionLedger,
    ExecutionModel,
    FillResult,
    Quote,
    RejectReason,
)


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def model() -> ExecutionModel:
    return ExecutionModel()


@pytest.fixture
def liquid_quote() -> Quote:
    """A healthy, liquid quote."""
    return Quote(
        bid=1.20,
        ask=1.35,
        bid_size=50,
        ask_size=40,
        quote_ts_ms=1_700_000_000_000,
        symbol="SPY_P_450",
    )


@pytest.fixture
def sim_ts() -> int:
    return 1_700_000_000_000


# ═══════════════════════════════════════════════════════════════════════════
# Basic fill tests
# ═══════════════════════════════════════════════════════════════════════════

class TestBasicFills:
    def test_sell_fills_at_bid_minus_slippage(self, model, liquid_quote, sim_ts):
        result = model.attempt_fill(liquid_quote, "SELL", 1, sim_ts)
        assert result.filled is True
        assert result.raw_price == 1.20
        assert result.fill_price == 1.20 - 0.02  # default slippage
        assert result.side == "SELL"
        assert result.source == "simulated"

    def test_buy_fills_at_ask_plus_slippage(self, model, liquid_quote, sim_ts):
        result = model.attempt_fill(liquid_quote, "BUY", 1, sim_ts)
        assert result.filled is True
        assert result.raw_price == 1.35
        assert result.fill_price == 1.35 + 0.02  # default slippage

    def test_fill_quantity_preserved(self, model, liquid_quote, sim_ts):
        result = model.attempt_fill(liquid_quote, "SELL", 5, sim_ts)
        assert result.filled is True
        assert result.quantity == 5

    def test_commission_per_contract(self, model, liquid_quote, sim_ts):
        result = model.attempt_fill(liquid_quote, "BUY", 3, sim_ts)
        assert result.filled is True
        assert result.commission == pytest.approx(0.65 * 3)

    def test_custom_slippage(self, sim_ts, liquid_quote):
        cfg = ExecutionConfig(slippage_per_contract=0.05)
        model = ExecutionModel(cfg)
        result = model.attempt_fill(liquid_quote, "SELL", 1, sim_ts)
        assert result.fill_price == pytest.approx(1.20 - 0.05)

    def test_fill_price_floored_at_zero(self, sim_ts):
        q = Quote(bid=0.01, ask=0.05, quote_ts_ms=sim_ts)
        cfg = ExecutionConfig(slippage_per_contract=0.10, max_spread_pct=10.0)
        model = ExecutionModel(cfg)
        result = model.attempt_fill(q, "SELL", 1, sim_ts)
        assert result.filled is True
        assert result.fill_price == 0.0  # floored (0.01 - 0.10 → 0)


# ═══════════════════════════════════════════════════════════════════════════
# Rejection tests
# ═══════════════════════════════════════════════════════════════════════════

class TestRejections:
    def test_zero_bid_rejected(self, model, sim_ts):
        q = Quote(bid=0.0, ask=1.00, quote_ts_ms=sim_ts)
        result = model.attempt_fill(q, "SELL", 1, sim_ts)
        assert result.filled is False
        assert result.reject_reason == RejectReason.ZERO_BID

    def test_crossed_market_rejected(self, model, sim_ts):
        q = Quote(bid=1.50, ask=1.20, quote_ts_ms=sim_ts)
        result = model.attempt_fill(q, "BUY", 1, sim_ts)
        assert result.filled is False
        assert result.reject_reason == RejectReason.CROSSED

    def test_wide_spread_rejected(self, sim_ts):
        cfg = ExecutionConfig(max_spread_pct=0.10)
        model = ExecutionModel(cfg)
        # Spread = 0.80, mid = 1.40 → 57% > 10%
        q = Quote(bid=1.00, ask=1.80, quote_ts_ms=sim_ts)
        result = model.attempt_fill(q, "BUY", 1, sim_ts)
        assert result.filled is False
        assert result.reject_reason == RejectReason.SPREAD_WIDE

    def test_stale_quote_rejected(self, model):
        old_ts = 1_700_000_000_000
        sim_ts = old_ts + 300_000  # 5 minutes later
        q = Quote(bid=1.00, ask=1.10, quote_ts_ms=old_ts)
        result = model.attempt_fill(q, "BUY", 1, sim_ts)
        assert result.filled is False
        assert result.reject_reason == RejectReason.STALE_QUOTE

    def test_insufficient_bid_size_sell(self, sim_ts):
        cfg = ExecutionConfig(min_bid_size=10)
        model = ExecutionModel(cfg)
        q = Quote(bid=1.00, ask=1.10, bid_size=5, ask_size=20, quote_ts_ms=sim_ts)
        result = model.attempt_fill(q, "SELL", 1, sim_ts)
        assert result.filled is False
        assert result.reject_reason == RejectReason.INSUFFICIENT_SIZE

    def test_insufficient_ask_size_buy(self, sim_ts):
        cfg = ExecutionConfig(min_ask_size=10)
        model = ExecutionModel(cfg)
        q = Quote(bid=1.00, ask=1.10, bid_size=20, ask_size=5, quote_ts_ms=sim_ts)
        result = model.attempt_fill(q, "BUY", 1, sim_ts)
        assert result.filled is False
        assert result.reject_reason == RejectReason.INSUFFICIENT_SIZE

    def test_partial_fill_not_allowed_by_default(self, sim_ts):
        model = ExecutionModel()
        q = Quote(bid=1.00, ask=1.10, bid_size=2, ask_size=20, quote_ts_ms=sim_ts)
        result = model.attempt_fill(q, "SELL", 5, sim_ts)
        assert result.filled is False
        assert result.reject_reason == RejectReason.INSUFFICIENT_SIZE

    def test_size_zero_skips_check(self, model, sim_ts):
        """Size=0 means unknown → skip size check."""
        q = Quote(bid=1.00, ask=1.10, bid_size=0, ask_size=0, quote_ts_ms=sim_ts)
        result = model.attempt_fill(q, "SELL", 100, sim_ts)
        assert result.filled is True

    def test_stale_check_skipped_when_ts_zero(self, model):
        """quote_ts_ms=0 means unknown → skip staleness check."""
        q = Quote(bid=1.00, ask=1.10, quote_ts_ms=0)
        result = model.attempt_fill(q, "BUY", 1, sim_ts_ms=1_700_000_999_000)
        assert result.filled is True


# ═══════════════════════════════════════════════════════════════════════════
# Spread fill tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSpreadFills:
    def test_fill_spread_both_legs(self, model, sim_ts):
        short_q = Quote(bid=1.50, ask=1.65, quote_ts_ms=sim_ts, symbol="SHORT")
        long_q = Quote(bid=0.30, ask=0.40, quote_ts_ms=sim_ts, symbol="LONG")
        result = model.fill_spread(short_q, long_q, 2, sim_ts)
        assert result["filled"] is True
        # short fill_price = 1.50 - 0.02 = 1.48
        # long fill_price = 0.40 + 0.02 = 0.42
        # net credit = (1.48 - 0.42) * 2 * 100 = 212.00
        assert result["net_credit"] == pytest.approx(212.00, abs=0.01)
        assert result["total_commission"] > 0

    def test_spread_rejected_if_short_leg_fails(self, model, sim_ts):
        short_q = Quote(bid=0.0, ask=1.00, quote_ts_ms=sim_ts)  # zero bid
        long_q = Quote(bid=0.30, ask=0.40, quote_ts_ms=sim_ts)
        result = model.fill_spread(short_q, long_q, 1, sim_ts)
        assert result["filled"] is False
        # reject_reason contains the fill's reject_detail (e.g. "bid=0.0")
        assert result.get("reject_reason", "") != ""

    def test_spread_rejected_if_long_leg_fails(self, sim_ts):
        cfg = ExecutionConfig(max_spread_pct=0.05)
        model = ExecutionModel(cfg)
        short_q = Quote(bid=1.50, ask=1.52, quote_ts_ms=sim_ts)  # tight
        long_q = Quote(bid=0.10, ask=0.80, quote_ts_ms=sim_ts)   # 160% spread
        result = model.fill_spread(short_q, long_q, 1, sim_ts)
        assert result["filled"] is False

    def test_close_spread(self, model, sim_ts):
        short_q = Quote(bid=0.50, ask=0.60, quote_ts_ms=sim_ts)
        long_q = Quote(bid=0.10, ask=0.15, quote_ts_ms=sim_ts)
        result = model.close_spread(short_q, long_q, 1, sim_ts)
        assert result["filled"] is True
        # Buy to close short: 0.60 + 0.02 = 0.62
        # Sell to close long: 0.10 - 0.02 = 0.08
        # net debit = (0.62 - 0.08) * 1 * 100 = 54.00
        assert result["net_debit"] == pytest.approx(54.00, abs=0.01)


# ═══════════════════════════════════════════════════════════════════════════
# Ledger tests
# ═══════════════════════════════════════════════════════════════════════════

class TestLedger:
    def test_ledger_tracks_fills_and_rejects(self, model, sim_ts):
        good_q = Quote(bid=1.00, ask=1.10, quote_ts_ms=sim_ts)
        bad_q = Quote(bid=0.0, ask=1.00, quote_ts_ms=sim_ts)

        model.attempt_fill(good_q, "SELL", 1, sim_ts)
        model.attempt_fill(bad_q, "SELL", 1, sim_ts)
        model.attempt_fill(good_q, "BUY", 2, sim_ts)

        assert model.ledger.fills_count == 2
        assert model.ledger.rejects_count == 1
        assert model.ledger.fill_rate == pytest.approx(2 / 3)

    def test_ledger_accumulates_commission(self, model, sim_ts):
        q = Quote(bid=1.00, ask=1.10, quote_ts_ms=sim_ts)
        model.attempt_fill(q, "BUY", 3, sim_ts)
        model.attempt_fill(q, "SELL", 2, sim_ts)
        assert model.ledger.total_commission == pytest.approx(0.65 * 5)

    def test_ledger_summary(self, model, sim_ts):
        q = Quote(bid=1.00, ask=1.10, quote_ts_ms=sim_ts)
        model.attempt_fill(q, "BUY", 1, sim_ts)
        summary = model.ledger.summary()
        assert summary["fills"] == 1
        assert summary["rejects"] == 0
        assert "total_commission" in summary

    def test_reject_breakdown(self, model, sim_ts):
        model.attempt_fill(Quote(bid=0, ask=1, quote_ts_ms=sim_ts), "SELL", 1, sim_ts)
        model.attempt_fill(Quote(bid=1.5, ask=1.2, quote_ts_ms=sim_ts), "BUY", 1, sim_ts)
        model.attempt_fill(Quote(bid=0, ask=1, quote_ts_ms=sim_ts), "SELL", 1, sim_ts)

        breakdown = model.ledger._reject_breakdown()
        assert breakdown.get("ZERO_BID", 0) == 2
        assert breakdown.get("CROSSED", 0) == 1

    def test_reset_clears_ledger(self, model, sim_ts):
        q = Quote(bid=1.00, ask=1.10, quote_ts_ms=sim_ts)
        model.attempt_fill(q, "BUY", 1, sim_ts)
        assert model.ledger.fills_count == 1
        model.reset()
        assert model.ledger.fills_count == 0
