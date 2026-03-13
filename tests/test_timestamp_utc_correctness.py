"""
Tests for Alpaca UTC / timestamp correctness (10.1).

Verifies:
- RFC3339 parsing always produces UTC epoch-ms
- Naive timestamps treated as UTC
- DST boundary case: spring-forward and fall-back produce correct session tags
- Session classification uses ET conversion consistently
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone

from src.connectors.alpaca_client import _parse_rfc3339_to_ms
from src.core.outcome_engine import _timestamp_to_ms
from src.core.sessions import get_session_regime, _et_from_ts_ms


class TestRFC3339Parsing:
    """Validate _parse_rfc3339_to_ms produces correct UTC epoch-ms."""

    def test_z_suffix(self):
        ms = _parse_rfc3339_to_ms("2024-01-15T14:30:00Z")
        assert isinstance(ms, int)
        assert ms == 1705329000000

    def test_plus_offset(self):
        ms = _parse_rfc3339_to_ms("2024-01-15T14:30:00+00:00")
        assert ms == 1705329000000

    def test_non_utc_offset_converted(self):
        """Timestamp with offset is converted to UTC epoch-ms."""
        ms = _parse_rfc3339_to_ms("2024-01-15T09:30:00-05:00")
        assert ms == 1705329000000  # 9:30 AM EST = 14:30 UTC

    def test_fractional_seconds(self):
        ms = _parse_rfc3339_to_ms("2024-01-15T14:30:00.500Z")
        assert ms == 1705329000500


class TestTimestampToMs:
    """Validate _timestamp_to_ms treats naive as UTC."""

    def test_iso_with_z(self):
        ms = _timestamp_to_ms("2024-01-15T14:30:00Z")
        assert ms == 1705329000000

    def test_naive_treated_as_utc(self):
        """Naive datetime strings (no tz) must be treated as UTC."""
        ms = _timestamp_to_ms("2024-01-15 14:30:00")
        assert ms == 1705329000000

    def test_iso_with_offset(self):
        ms = _timestamp_to_ms("2024-01-15T09:30:00-05:00")
        assert ms == 1705329000000


class TestDSTBoundarySessionClassification:
    """Verify session classification handles DST transitions correctly.

    US Eastern Time:
    - EST (standard): UTC-5  (Nov first Sun → Mar second Sun)
    - EDT (daylight): UTC-4  (Mar second Sun → Nov first Sun)

    2024 spring-forward: March 10, 2024 02:00 EST → 03:00 EDT
    2024 fall-back:      November 3, 2024 02:00 EDT → 01:00 EST
    """

    def test_spring_forward_rth(self):
        """March 11, 2024 (day after spring-forward): 14:30 UTC = 10:30 EDT → RTH."""
        # 2024-03-11T14:30:00Z = 10:30 AM EDT (after spring-forward)
        ts_ms = int(datetime(2024, 3, 11, 14, 30, tzinfo=timezone.utc).timestamp() * 1000)
        session = get_session_regime("EQUITIES", ts_ms)
        assert session == "RTH", f"Expected RTH, got {session}"

    def test_spring_forward_pre(self):
        """March 11, 2024: 13:00 UTC = 09:00 EDT → PRE (not RTH)."""
        ts_ms = int(datetime(2024, 3, 11, 13, 0, tzinfo=timezone.utc).timestamp() * 1000)
        session = get_session_regime("EQUITIES", ts_ms)
        assert session == "PRE", f"Expected PRE, got {session}"

    def test_fall_back_rth(self):
        """November 4, 2024 (day after fall-back): 15:30 UTC = 10:30 EST → RTH."""
        ts_ms = int(datetime(2024, 11, 4, 15, 30, tzinfo=timezone.utc).timestamp() * 1000)
        session = get_session_regime("EQUITIES", ts_ms)
        assert session == "RTH", f"Expected RTH, got {session}"

    def test_fall_back_closed(self):
        """November 4, 2024: 02:00 UTC = 21:00 EST (prev day) → CLOSED."""
        ts_ms = int(datetime(2024, 11, 4, 2, 0, tzinfo=timezone.utc).timestamp() * 1000)
        session = get_session_regime("EQUITIES", ts_ms)
        assert session == "CLOSED", f"Expected CLOSED, got {session}"

    def test_est_vs_edt_boundary_shift(self):
        """Verify that the same UTC hour maps to different sessions across DST.

        14:30 UTC:
        - During EST (winter): 14:30 - 5 = 9:30 ET → RTH open
        - During EDT (summer): 14:30 - 4 = 10:30 ET → RTH
        Both should be RTH, but 13:30 UTC:
        - During EST: 13:30 - 5 = 8:30 ET → PRE
        - During EDT: 13:30 - 4 = 9:30 ET → RTH
        """
        # January (EST): 13:30 UTC = 8:30 AM EST → PRE
        jan_ts = int(datetime(2024, 1, 15, 13, 30, tzinfo=timezone.utc).timestamp() * 1000)
        jan_session = get_session_regime("EQUITIES", jan_ts)
        assert jan_session == "PRE"

        # July (EDT): 13:30 UTC = 9:30 AM EDT → RTH
        jul_ts = int(datetime(2024, 7, 15, 13, 30, tzinfo=timezone.utc).timestamp() * 1000)
        jul_session = get_session_regime("EQUITIES", jul_ts)
        assert jul_session == "RTH"


class TestETConversionConsistency:
    """Verify _et_from_ts_ms is consistent with session boundaries."""

    def test_et_from_ts_ms_returns_eastern(self):
        ts_ms = int(datetime(2024, 6, 15, 14, 30, tzinfo=timezone.utc).timestamp() * 1000)
        dt_et = _et_from_ts_ms(ts_ms)
        assert dt_et.hour == 10
        assert dt_et.minute == 30

    def test_et_from_ts_ms_winter(self):
        ts_ms = int(datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc).timestamp() * 1000)
        dt_et = _et_from_ts_ms(ts_ms)
        assert dt_et.hour == 9
        assert dt_et.minute == 30
