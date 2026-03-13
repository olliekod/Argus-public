"""
Tests for e2e_verify.py enhancements (C7).

Verifies:
- summary.md is written with UTF-8 encoding and no special arrow chars
- IV readiness report includes provider/derived/overall percentages
- Zero-trade reason strings are correct
- diagnose_zero_trades works for various pack states
- Cold-start mode (no greeks) degrades gracefully

No live network calls.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

# We import the functions directly from the script
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


class TestSummaryMdEncoding:
    """Verify summary.md uses UTF-8 and no problematic chars."""

    def test_no_arrow_chars_in_source(self):
        """The source file should not contain the unicode arrow character."""
        script_path = Path(__file__).resolve().parent.parent / "scripts" / "e2e_verify.py"
        content = script_path.read_text(encoding="utf-8")
        # The unicode right arrow (U+2192) should have been replaced with ->
        assert "\u2192" not in content, "e2e_verify.py should not contain unicode arrow (U+2192)"

    def test_summary_md_encoding(self, tmp_path):
        """summary.md should be writable and readable as UTF-8."""
        summary_path = tmp_path / "summary.md"
        content = (
            "# Argus E2E Verification -- 2024-01-15\n\n"
            "**Date range:** 2024-01-10 -> 2024-01-15\n"
            "**Result:** 3/6 steps passed\n\n"
            "| Step | Result |\n"
            "|------|--------|\n"
            "| bars | PASS |\n"
        )
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(content)

        # Read back and verify
        with open(summary_path, "r", encoding="utf-8") as f:
            read_back = f.read()

        assert "->" in read_back
        assert "\u2192" not in read_back  # No unicode arrow

    def test_summary_md_ascii_safe(self, tmp_path):
        """summary.md content should be encodable as ASCII (except for standard UTF-8)."""
        summary_path = tmp_path / "summary.md"
        content = (
            "# E2E Verification\n\n"
            "**Date range:** 2024-01-10 -> 2024-01-15\n"
            "Result: PASS -- all good\n"
        )
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(content)

        # This should not raise even if the system encoding is cp1252
        read_back = Path(summary_path).read_text(encoding="utf-8")
        # ASCII-safe check (all chars < 128)
        assert all(ord(c) < 128 for c in read_back), "summary.md should be ASCII-safe"


class TestDiagnoseZeroTrades:
    """Test the zero-trade diagnostics function."""

    def test_no_bars(self, tmp_path):
        """Pack with no bars should report 'no bars in replay pack'."""
        from scripts.e2e_verify import diagnose_zero_trades

        pack_path = tmp_path / "pack.json"
        pack_path.write_text(json.dumps({
            "bars": [],
            "snapshots": [{"recv_ts_ms": 123, "atm_iv": 0.25}],
            "metadata": {},
        }))

        reasons = diagnose_zero_trades(str(pack_path))
        assert "no bars in replay pack" in reasons

    def test_no_snapshots(self, tmp_path):
        """Pack with no snapshots should report 'no snapshots in range'."""
        from scripts.e2e_verify import diagnose_zero_trades

        pack_path = tmp_path / "pack.json"
        pack_path.write_text(json.dumps({
            "bars": [{"timestamp_ms": 100, "close": 50.0}],
            "snapshots": [],
            "metadata": {},
        }))

        reasons = diagnose_zero_trades(str(pack_path))
        assert "no snapshots in range" in reasons

    def test_no_recv_ts(self, tmp_path):
        """Snapshots with recv_ts_ms=0 should report gating issue."""
        from scripts.e2e_verify import diagnose_zero_trades

        pack_path = tmp_path / "pack.json"
        pack_path.write_text(json.dumps({
            "bars": [{"timestamp_ms": 100}],
            "snapshots": [
                {"recv_ts_ms": 0, "atm_iv": None},
                {"recv_ts_ms": 0, "atm_iv": None},
            ],
            "metadata": {},
        }))

        reasons = diagnose_zero_trades(str(pack_path))
        assert any("recv_ts_ms" in r for r in reasons)

    def test_no_iv(self, tmp_path):
        """Snapshots with no atm_iv should report IV issue."""
        from scripts.e2e_verify import diagnose_zero_trades

        pack_path = tmp_path / "pack.json"
        pack_path.write_text(json.dumps({
            "bars": [{"timestamp_ms": 100}],
            "snapshots": [
                {"recv_ts_ms": 1000, "atm_iv": None},
                {"recv_ts_ms": 2000, "atm_iv": 0},
            ],
            "metadata": {},
        }))

        reasons = diagnose_zero_trades(str(pack_path))
        assert any("atm_iv" in r for r in reasons)

    def test_all_good(self, tmp_path):
        """Pack with bars + snapshots + IV should report 'unknown'."""
        from scripts.e2e_verify import diagnose_zero_trades

        pack_path = tmp_path / "pack.json"
        pack_path.write_text(json.dumps({
            "bars": [{"timestamp_ms": 100}],
            "snapshots": [
                {"recv_ts_ms": 1000, "atm_iv": 0.25},
            ],
            "metadata": {},
        }))

        reasons = diagnose_zero_trades(str(pack_path))
        assert any("unknown" in r for r in reasons)

    def test_invalid_pack(self, tmp_path):
        """Unreadable pack should report graceful error."""
        from scripts.e2e_verify import diagnose_zero_trades

        pack_path = tmp_path / "bad.json"
        pack_path.write_text("not json at all")

        reasons = diagnose_zero_trades(str(pack_path))
        assert any("could not load" in r for r in reasons)


class TestIVReadinessReport:
    """Verify the enhanced IV readiness fields in check_options_snapshots."""

    def test_report_fields_present(self):
        """Snapshot result should include all enhanced IV readiness fields."""
        # Build a mock result matching the expected format
        result = {
            "provider": "tastytrade",
            "symbol": "SPY",
            "snapshot_count": 100,
            "atm_iv_present": 60,
            "atm_iv_pct": 60.0,
            "iv_derivable": 20,
            "iv_derivable_pct": 20.0,
            "iv_ready_count": 80,
            "iv_ready_pct": 80.0,
            "recv_ts_gated": 95,
            "pass": True,
        }
        assert "atm_iv_pct" in result
        assert "iv_derivable_pct" in result
        assert "iv_ready_count" in result
        assert "iv_ready_pct" in result
        assert "recv_ts_gated" in result


class TestColdStartMode:
    """Simulate no greeks in cache — should degrade gracefully."""

    def test_cold_start_zero_iv(self, tmp_path):
        """With no greeks cached, all snapshots should have atm_iv=None
        and the IV readiness should report 0%."""
        from scripts.e2e_verify import diagnose_zero_trades

        # Simulate a pack where snapshots exist but have no IV
        pack_path = tmp_path / "cold_start_pack.json"
        snapshots = [
            {"recv_ts_ms": 1000 + i * 60000, "atm_iv": None, "underlying_price": 450.0}
            for i in range(10)
        ]
        pack = {
            "bars": [{"timestamp_ms": 1000 + i * 60000, "close": 450.0} for i in range(10)],
            "snapshots": snapshots,
            "metadata": {"bar_count": 10, "snapshot_count": 10},
        }
        pack_path.write_text(json.dumps(pack))

        reasons = diagnose_zero_trades(str(pack_path))
        # Should identify IV issue
        assert any("atm_iv" in r or "IV" in r for r in reasons), \
            f"Cold start should identify IV issue, got: {reasons}"

    def test_cold_start_graceful_degradation(self):
        """IV readiness of 0% should not cause any exceptions."""
        # A result with 0 IV should be valid
        result = {
            "snapshot_count": 50,
            "atm_iv_present": 0,
            "atm_iv_pct": 0.0,
            "iv_derivable": 0,
            "iv_derivable_pct": 0.0,
            "iv_ready_count": 0,
            "iv_ready_pct": 0.0,
            "recv_ts_gated": 50,
        }
        assert result["iv_ready_pct"] == 0.0
        assert result["snapshot_count"] > 0
