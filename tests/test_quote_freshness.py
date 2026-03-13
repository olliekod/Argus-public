"""
Tests for quote freshness utilities.

Validates receipt-time-based freshness checks, Greeks timestamp resolution,
and staleness summary statistics.
"""

from __future__ import annotations

import time
import pytest

from src.core.quote_freshness import (
    effective_greeks_timestamp,
    freshness_summary,
    is_greeks_fresh,
    is_quote_fresh,
    now_ms,
    quote_age_ms,
)


# ═══════════════════════════════════════════════════════════════════════════
# is_quote_fresh
# ═══════════════════════════════════════════════════════════════════════════

class TestIsQuoteFresh:
    def test_fresh_quote(self):
        ref = 1_700_000_100_000
        recv = 1_700_000_050_000  # 50 seconds old
        assert is_quote_fresh(recv, max_age_ms=120_000, reference_ms=ref) is True

    def test_stale_quote(self):
        ref = 1_700_000_300_000
        recv = 1_700_000_000_000  # 300 seconds old
        assert is_quote_fresh(recv, max_age_ms=120_000, reference_ms=ref) is False

    def test_zero_recv_ts_is_not_fresh(self):
        assert is_quote_fresh(0, max_age_ms=120_000, reference_ms=1_700_000_000_000) is False

    def test_negative_recv_ts_is_not_fresh(self):
        assert is_quote_fresh(-1, max_age_ms=120_000) is False

    def test_exact_boundary_is_fresh(self):
        ref = 1_700_000_120_000
        recv = 1_700_000_000_000  # exactly 120s old
        assert is_quote_fresh(recv, max_age_ms=120_000, reference_ms=ref) is True

    def test_one_ms_over_boundary_is_stale(self):
        ref = 1_700_000_120_001
        recv = 1_700_000_000_000  # 120001ms old
        assert is_quote_fresh(recv, max_age_ms=120_000, reference_ms=ref) is False

    def test_custom_max_age(self):
        ref = 1_700_000_010_000
        recv = 1_700_000_000_000  # 10s old
        assert is_quote_fresh(recv, max_age_ms=5_000, reference_ms=ref) is False
        assert is_quote_fresh(recv, max_age_ms=15_000, reference_ms=ref) is True

    def test_defaults_to_now(self):
        # A timestamp from 1 second ago should be fresh
        recv = int(time.time() * 1000) - 1000
        assert is_quote_fresh(recv, max_age_ms=120_000) is True


# ═══════════════════════════════════════════════════════════════════════════
# quote_age_ms
# ═══════════════════════════════════════════════════════════════════════════

class TestQuoteAgeMs:
    def test_valid_age(self):
        ref = 1_700_000_100_000
        recv = 1_700_000_050_000
        assert quote_age_ms(recv, reference_ms=ref) == 50_000

    def test_zero_recv_returns_negative_one(self):
        assert quote_age_ms(0, reference_ms=1_700_000_000_000) == -1


# ═══════════════════════════════════════════════════════════════════════════
# effective_greeks_timestamp
# ═══════════════════════════════════════════════════════════════════════════

class TestEffectiveGreeksTimestamp:
    def test_prefers_event_ts(self):
        event = {"event_ts_ms": 1_700_000_050_000, "recv_ts_ms": 1_700_000_060_000}
        ts, source = effective_greeks_timestamp(event)
        assert ts == 1_700_000_050_000
        assert source == "event"

    def test_falls_back_to_recv_ts(self):
        event = {"event_ts_ms": 0, "recv_ts_ms": 1_700_000_060_000}
        ts, source = effective_greeks_timestamp(event)
        assert ts == 1_700_000_060_000
        assert source == "receipt"

    def test_missing_event_ts(self):
        event = {"recv_ts_ms": 1_700_000_060_000}
        ts, source = effective_greeks_timestamp(event)
        assert ts == 1_700_000_060_000
        assert source == "receipt"

    def test_both_zero(self):
        event = {"event_ts_ms": 0, "recv_ts_ms": 0}
        ts, source = effective_greeks_timestamp(event)
        assert ts == 0
        assert source == "none"

    def test_empty_event(self):
        ts, source = effective_greeks_timestamp({})
        assert ts == 0
        assert source == "none"

    def test_custom_keys(self):
        event = {"greeks_ts": 1_700_000_050_000, "local_ts": 1_700_000_060_000}
        ts, source = effective_greeks_timestamp(
            event, event_ts_key="greeks_ts", recv_ts_key="local_ts"
        )
        assert ts == 1_700_000_050_000
        assert source == "event"


# ═══════════════════════════════════════════════════════════════════════════
# is_greeks_fresh
# ═══════════════════════════════════════════════════════════════════════════

class TestIsGreeksFresh:
    def test_fresh_with_event_ts(self):
        ref = 1_700_000_100_000
        event = {"event_ts_ms": 1_700_000_050_000, "recv_ts_ms": 1_700_000_060_000}
        assert is_greeks_fresh(event, max_age_ms=120_000, reference_ms=ref) is True

    def test_stale_with_event_ts(self):
        ref = 1_700_000_300_000
        event = {"event_ts_ms": 1_700_000_000_000}
        assert is_greeks_fresh(event, max_age_ms=120_000, reference_ms=ref) is False

    def test_no_timestamps(self):
        assert is_greeks_fresh({}, max_age_ms=120_000, reference_ms=1_700_000_000_000) is False


# ═══════════════════════════════════════════════════════════════════════════
# freshness_summary
# ═══════════════════════════════════════════════════════════════════════════

class TestFreshnessSummary:
    def test_basic_summary(self):
        ref = 1_700_000_100_000
        recv_list = [
            1_700_000_050_000,  # 50s old
            1_700_000_070_000,  # 30s old
            1_700_000_090_000,  # 10s old
        ]
        s = freshness_summary(recv_list, reference_ms=ref)
        assert s["count"] == 3
        assert s["valid_count"] == 3
        assert s["age_p50_ms"] == 30_000
        assert s["age_max_ms"] == 50_000
        assert s["stale_count_120s"] == 0

    def test_some_stale(self):
        ref = 1_700_000_300_000
        recv_list = [
            1_700_000_000_000,  # 300s old — stale
            1_700_000_250_000,  # 50s old — fresh
        ]
        s = freshness_summary(recv_list, reference_ms=ref)
        assert s["stale_count_120s"] == 1

    def test_empty_list(self):
        s = freshness_summary([], reference_ms=1_700_000_000_000)
        assert s["count"] == 0
        assert s["valid_count"] == 0
        assert s["age_p50_ms"] is None

    def test_zeroes_excluded(self):
        ref = 1_700_000_100_000
        recv_list = [0, 0, 1_700_000_090_000]
        s = freshness_summary(recv_list, reference_ms=ref)
        assert s["count"] == 3
        assert s["valid_count"] == 1
        assert s["age_p50_ms"] == 10_000


# ═══════════════════════════════════════════════════════════════════════════
# now_ms utility
# ═══════════════════════════════════════════════════════════════════════════

class TestNowMs:
    def test_returns_positive_int(self):
        t = now_ms()
        assert isinstance(t, int)
        assert t > 0

    def test_reasonable_range(self):
        t = now_ms()
        # Should be after 2020-01-01 and before 2030-01-01
        assert t > 1_577_836_800_000
        assert t < 1_893_456_000_000
