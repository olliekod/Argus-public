"""
Tests for Kalshi subpenny pricing (March 2026 migration).

Ensures we parse _dollars fields when present and fall back to legacy cents.
"""

from __future__ import annotations

import pytest

from argus_kalshi.kalshi_subpenny import (
    parse_count_centicx,
    parse_level_price,
    parse_price_cents,
    parse_qty_centicx,
    parse_snapshot_level,
)


def test_parse_price_cents_prefers_dollars() -> None:
    """When both keys exist, _dollars wins."""
    assert parse_price_cents(
        {"yes_bid": 55, "yes_bid_dollars": "0.1234"},
        "yes_bid",
        "yes_bid_dollars",
    ) == 12  # 0.1234 * 100 rounded
    assert parse_price_cents(
        {"last_price": 99, "last_price_dollars": "0.9950"},
        "last_price",
        "last_price_dollars",
    ) == 100  # rounds to 100


def test_parse_price_cents_fallback_to_cents() -> None:
    """When only legacy key exists, use it."""
    assert parse_price_cents(
        {"yes_bid": 55},
        "yes_bid",
        "yes_bid_dollars",
    ) == 55
    assert parse_price_cents(
        {"last_price": 0},
        "last_price",
        "last_price_dollars",
        default=0,
    ) == 0


def test_parse_price_cents_dollars_only() -> None:
    """After March 5 2026 only _dollars may be present."""
    assert parse_price_cents(
        {"yes_bid_dollars": "0.5500"},
        "yes_bid",
        "yes_bid_dollars",
    ) == 55
    assert parse_price_cents(
        {"yes_ask_dollars": "0.0099"},
        "yes_ask",
        "yes_ask_dollars",
        default=0,
    ) == 1


def test_parse_price_cents_default() -> None:
    """Missing keys return default."""
    assert parse_price_cents({}, "yes_bid", "yes_bid_dollars", default=0) == 0
    assert parse_price_cents({}, "price", "price_dollars", default=50) == 50


def test_parse_level_price_int() -> None:
    """Legacy snapshot level: int cents."""
    assert parse_level_price(55) == 55
    assert parse_level_price(0) == 0


def test_parse_level_price_str_dollars() -> None:
    """Level as fixed-point dollars string."""
    assert parse_level_price("0.5500") == 55
    assert parse_level_price("0.12") == 12


def test_parse_level_price_dict() -> None:
    """Level as dict with price or price_dollars."""
    assert parse_level_price({"price": 60}) == 60
    assert parse_level_price({"price_dollars": "0.6000"}) == 60
    # When both present, price_dollars wins; 0.4150 * 100 -> 42
    assert parse_level_price({"price": 40, "price_dollars": "0.4150"}) == 42


def test_orderbook_snapshot_dollars_only_produces_correct_cents() -> None:
    """
    After March 5 2026 Kalshi may send only _dollars. OrderBook.apply_snapshot
    with levels that use only price_dollars (no legacy cents) must produce
    correct best bid cents and thus correct OrderbookState downstream.
    """
    from argus_kalshi.orderbook import OrderBook

    book = OrderBook(market_ticker="TEST")
    # Snapshot with dollars-only levels: yes best at 0.45, no best at 0.55
    snapshot = {
        "yes": [["0.4500", "10.00"], ["0.4400", "5.00"]],
        "no": [["0.5500", "8.00"], ["0.5600", "3.00"]],
    }
    book.apply_snapshot(snapshot, seq=1)
    assert book.valid is True
    assert book.best_yes_bid_cents == 45
    # Best no bid is highest no price (0.56 > 0.55)
    assert book.best_no_bid_cents == 56
    assert book.implied_yes_ask_cents == 100 - 56
    assert book.implied_no_ask_cents == 100 - 45


# ---------------------------------------------------------------------------
#  Fixed-point count (March 2026)
# ---------------------------------------------------------------------------


def test_parse_count_centicx_prefers_fp() -> None:
    """When both keys exist, count_fp wins."""
    assert parse_count_centicx({"count": 10, "count_fp": "1.55"}, "count", "count_fp") == 155
    assert parse_count_centicx({"count": 5, "count_fp": "10.00"}, "count", "count_fp") == 1000


def test_parse_count_centicx_fallback() -> None:
    """Legacy count (int) -> centicx."""
    assert parse_count_centicx({"count": 10}, "count", "count_fp") == 1000
    assert parse_count_centicx({"count": 1}, "count", "count_fp", default=0) == 100


def test_parse_count_centicx_legacy_fp_string() -> None:
    """Legacy key with fp string (e.g. delta "5.00")."""
    assert parse_count_centicx({"delta": "5.00"}, "delta", "delta_fp") == 500
    assert parse_count_centicx({"delta": "-3.00"}, "delta", "delta_fp") == -300


def test_parse_qty_centicx() -> None:
    """Orderbook level qty: str (fp) or int (legacy)."""
    assert parse_qty_centicx("5.00") == 500
    assert parse_qty_centicx(5) == 500


def test_parse_qty_centicx_dict() -> None:
    """Orderbook level qty as dict with quantity_fp/quantity."""
    assert parse_qty_centicx({"quantity_fp": "5.00", "quantity": 5}) == 500
    assert parse_qty_centicx({"quantity": 3}) == 300


def test_parse_snapshot_level_list() -> None:
    """Legacy list format [price, qty]."""
    assert parse_snapshot_level([55, "10.00"]) == (55, 1000)
    assert parse_snapshot_level([50, 5]) == (50, 500)
    assert parse_snapshot_level(["0.55", "5.00"]) == (55, 500)


def test_parse_snapshot_level_dict() -> None:
    """Subpenny object format per level."""
    assert parse_snapshot_level(
        {"price_dollars": "0.55", "quantity_fp": "10.00"}
    ) == (55, 1000)
    assert parse_snapshot_level({"price": 45, "quantity": 3}) == (45, 300)
