"""
Tests for orderbook.OrderBook.

Covers:
  - Snapshot application
  - Delta application (add, remove, update levels)
  - Sequence gap detection
  - Best bid/ask derivation
  - Implied prices
  - Edge cases (empty book, single level, level removal)
"""

from __future__ import annotations

import pytest

from argus_kalshi.orderbook import OrderBook, _fp_to_centicx, _centicx_to_fp


# ---------------------------------------------------------------------------
#  Fixed-point helpers
# ---------------------------------------------------------------------------

def test_fp_to_centicx() -> None:
    assert _fp_to_centicx("1.00") == 100
    assert _fp_to_centicx("0.50") == 50
    assert _fp_to_centicx("123.45") == 12345
    assert _fp_to_centicx("0.01") == 1


def test_centicx_to_fp() -> None:
    assert _centicx_to_fp(100) == "1.00"
    assert _centicx_to_fp(50) == "0.50"
    assert _centicx_to_fp(12345) == "123.45"
    assert _centicx_to_fp(1) == "0.01"


# ---------------------------------------------------------------------------
#  Snapshot
# ---------------------------------------------------------------------------

def test_snapshot_basic() -> None:
    book = OrderBook(market_ticker="TEST-MARKET")
    snapshot = {
        "yes": [[55, "10.00"], [50, "20.00"], [45, "5.00"]],
        "no": [[40, "15.00"], [35, "8.00"]],
    }
    book.apply_snapshot(snapshot, seq=1)

    assert book.valid
    assert book.last_seq == 1
    assert book.best_yes_bid_cents == 55
    assert book.best_no_bid_cents == 40


def test_snapshot_replaces_existing() -> None:
    book = OrderBook(market_ticker="TEST")
    book.apply_snapshot({"yes": [[60, "5.00"]], "no": [[30, "3.00"]]}, seq=1)
    assert book.best_yes_bid_cents == 60

    book.apply_snapshot({"yes": [[70, "2.00"]], "no": [[25, "1.00"]]}, seq=10)
    assert book.best_yes_bid_cents == 70
    assert book.last_seq == 10


def test_snapshot_empty_sides() -> None:
    book = OrderBook(market_ticker="TEST")
    book.apply_snapshot({"yes": [], "no": []}, seq=1)
    assert book.valid
    assert book.best_yes_bid_cents == 0
    assert book.best_no_bid_cents == 0


# ---------------------------------------------------------------------------
#  Deltas
# ---------------------------------------------------------------------------

def test_delta_add_level() -> None:
    book = OrderBook(market_ticker="TEST")
    book.apply_snapshot({"yes": [[50, "10.00"]], "no": [[40, "5.00"]]}, seq=1)

    ok = book.apply_delta(
        {"side": "yes", "price": 55, "delta": "5.00"},
        seq=2,
    )
    assert ok
    assert book.best_yes_bid_cents == 55
    assert book.last_seq == 2


def test_delta_remove_level() -> None:
    book = OrderBook(market_ticker="TEST")
    book.apply_snapshot({"yes": [[50, "10.00"], [55, "5.00"]], "no": []}, seq=1)

    # Remove the best bid.
    ok = book.apply_delta(
        {"side": "yes", "price": 55, "delta": "-5.00"},
        seq=2,
    )
    assert ok
    assert book.best_yes_bid_cents == 50


def test_delta_update_quantity() -> None:
    book = OrderBook(market_ticker="TEST")
    book.apply_snapshot({"yes": [[50, "10.00"]], "no": []}, seq=1)

    ok = book.apply_delta(
        {"side": "yes", "price": 50, "delta": "5.00"},
        seq=2,
    )
    assert ok
    # Quantity should now be 10.00 + 5.00 = 15.00 (1500 centi-cx).
    assert book.yes_bids._levels[50] == 1500


def test_delta_list_of_entries() -> None:
    """Apply multiple delta entries in a single message."""
    book = OrderBook(market_ticker="TEST")
    book.apply_snapshot({"yes": [[50, "10.00"]], "no": [[40, "5.00"]]}, seq=1)

    deltas = [
        {"side": "yes", "price": 55, "delta": "3.00"},
        {"side": "no", "price": 45, "delta": "2.00"},
    ]
    ok = book.apply_delta(deltas, seq=2)
    assert ok
    assert book.best_yes_bid_cents == 55
    assert book.best_no_bid_cents == 45


# ---------------------------------------------------------------------------
#  Sequence handling
# ---------------------------------------------------------------------------

def test_non_monotonic_seq_invalidates() -> None:
    book = OrderBook(market_ticker="TEST")
    book.apply_snapshot({"yes": [[50, "10.00"]], "no": []}, seq=5)
    assert book.valid

    # Duplicate / out-of-order seq should invalidate.
    ok = book.apply_delta(
        {"side": "yes", "price": 55, "delta": "1.00"},
        seq=5,
    )
    assert not ok
    assert not book.valid


def test_delta_on_invalid_book_fails() -> None:
    book = OrderBook(market_ticker="TEST")
    # Never had a snapshot → not valid.
    ok = book.apply_delta(
        {"side": "yes", "price": 50, "delta": "1.00"},
        seq=1,
    )
    assert not ok


def test_sequential_deltas_ok() -> None:
    book = OrderBook(market_ticker="TEST")
    book.apply_snapshot({"yes": [[50, "10.00"]], "no": []}, seq=1)

    for i in range(2, 12):
        ok = book.apply_delta(
            {"side": "yes", "price": 50, "delta": "1.00"},
            seq=i,
        )
        assert ok

    assert book.valid
    assert book.last_seq == 11


def test_forward_seq_jump_is_allowed() -> None:
    """Seq can jump forward when a shared subscription interleaves other tickers."""
    book = OrderBook(market_ticker="TEST")
    book.apply_snapshot({"yes": [[50, "10.00"]], "no": []}, seq=1)

    ok = book.apply_delta(
        {"side": "yes", "price": 55, "delta": "1.00"},
        seq=7,
    )
    assert ok
    assert book.valid
    assert book.last_seq == 7
    assert book.best_yes_bid_cents == 55


# ---------------------------------------------------------------------------
#  Implied prices
# ---------------------------------------------------------------------------

def test_implied_yes_ask() -> None:
    """implied_yes_ask = 100 - best_no_bid."""
    book = OrderBook(market_ticker="TEST")
    book.apply_snapshot({"yes": [[55, "1.00"]], "no": [[40, "1.00"]]}, seq=1)
    assert book.implied_yes_ask_cents == 60  # 100 - 40


def test_implied_no_ask() -> None:
    """implied_no_ask = 100 - best_yes_bid."""
    book = OrderBook(market_ticker="TEST")
    book.apply_snapshot({"yes": [[55, "1.00"]], "no": [[40, "1.00"]]}, seq=1)
    assert book.implied_no_ask_cents == 45  # 100 - 55


def test_implied_prices_empty_book() -> None:
    """Empty book → asks are 100 (widest possible)."""
    book = OrderBook(market_ticker="TEST")
    book.apply_snapshot({"yes": [], "no": []}, seq=1)
    assert book.implied_yes_ask_cents == 100
    assert book.implied_no_ask_cents == 100


def test_spread() -> None:
    book = OrderBook(market_ticker="TEST")
    book.apply_snapshot({"yes": [[55, "1.00"]], "no": [[40, "1.00"]]}, seq=1)
    # yes_ask=60, yes_bid=55 → spread=5
    assert book.spread_cents == 5


# ---------------------------------------------------------------------------
#  Edge cases
# ---------------------------------------------------------------------------

def test_remove_nonexistent_level() -> None:
    """Removing a level that doesn't exist is a no-op."""
    book = OrderBook(market_ticker="TEST")
    book.apply_snapshot({"yes": [[50, "10.00"]], "no": []}, seq=1)

    ok = book.apply_delta(
        {"side": "yes", "price": 99, "delta": "-5.00"},
        seq=2,
    )
    assert ok  # Should not crash.
    assert book.best_yes_bid_cents == 50


def test_invalidate_method() -> None:
    book = OrderBook(market_ticker="TEST")
    book.apply_snapshot({"yes": [[50, "1.00"]], "no": []}, seq=1)
    assert book.valid
    book.invalidate()
    assert not book.valid


def test_summary() -> None:
    book = OrderBook(market_ticker="TEST-SUMM")
    book.apply_snapshot({"yes": [[55, "1.00"]], "no": [[40, "1.00"]]}, seq=5)
    s = book.summary()
    assert s["ticker"] == "TEST-SUMM"
    assert s["yes_bid"] == 55
    assert s["no_bid"] == 40
    assert s["yes_ask"] == 60
    assert s["no_ask"] == 45
    assert s["seq"] == 5
    assert s["valid"] is True


def test_levels_snapshot_descending() -> None:
    """levels_snapshot() returns levels in descending price order."""
    book = OrderBook(market_ticker="TEST")
    book.apply_snapshot({
        "yes": [[30, "1.00"], [50, "2.00"], [40, "3.00"]],
        "no": [],
    }, seq=1)
    levels = book.yes_bids.levels_snapshot()
    prices = [p for p, _ in levels]
    assert prices == [50, 40, 30]


def test_has_snapshot_flag() -> None:
    """has_snapshot distinguishes 'never initialised' from 'invalidated'."""
    book = OrderBook(market_ticker="TEST")
    assert not book.has_snapshot
    assert not book.valid

    book.apply_snapshot({"yes": [[50, "1.00"]], "no": []}, seq=1)
    assert book.has_snapshot
    assert book.valid

    book.invalidate()
    assert book.has_snapshot  # was initialised, then invalidated
    assert not book.valid


def test_non_monotonic_seq_preserves_has_snapshot() -> None:
    """An out-of-order seq invalidates the book but has_snapshot remains True."""
    book = OrderBook(market_ticker="TEST")
    book.apply_snapshot({"yes": [[50, "1.00"]], "no": []}, seq=1)
    book.apply_delta({"side": "yes", "price": 50, "delta": "1.00"}, seq=1)
    assert not book.valid
    assert book.has_snapshot  # snapshot was applied before the gap
