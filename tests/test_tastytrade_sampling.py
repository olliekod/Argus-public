"""
Unit tests for deterministic option-symbol sampling, spot fallback, and Bearer prefix.

Verify that given a fixed spot and normalized chain fixture, the sampled
symbols list remains stable across invocations.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tastytrade_health_audit import _select_sampled_contracts, select_spot
from src.connectors.tastytrade_rest import ensure_bearer_prefix


# ---------------------------------------------------------------------------
# Fixture data: synthetic normalized chain
# ---------------------------------------------------------------------------

def _make_chain(
    underlying: str = "SPY",
    expiries: list[str] | None = None,
    strikes: list[float] | None = None,
) -> list[dict]:
    """Build a synthetic normalized chain for testing."""
    today = date.today()
    if expiries is None:
        expiries = [
            (today + timedelta(days=d)).isoformat()
            for d in [3, 7, 14, 30, 60]
        ]
    if strikes is None:
        strikes = [
            490, 492, 494, 496, 498,
            500, 502, 504, 506, 508,
            510, 512, 514, 516, 518,
        ]

    chain = []
    for exp in expiries:
        for strike in strikes:
            for right in ("C", "P"):
                sym = f".{underlying}{exp.replace('-', '')}{right}{int(strike * 1000):08d}"
                chain.append({
                    "provider": "tastytrade",
                    "underlying": underlying,
                    "option_symbol": sym,
                    "expiry": exp,
                    "right": right,
                    "strike": strike,
                    "multiplier": 100,
                    "currency": "USD",
                    "exchange": None,
                    "meta": {"streamer_symbol": sym},
                })
    chain.sort(
        key=lambda item: (
            item.get("expiry") or "",
            item.get("strike") if item.get("strike") is not None else -1.0,
            item.get("right") or "",
            item.get("option_symbol") or "",
        )
    )
    return chain


# ---------------------------------------------------------------------------
# Tests: deterministic sampling via _select_sampled_contracts
# ---------------------------------------------------------------------------

class TestSelectSampledContracts:
    def test_determinism(self):
        """Same input must always produce the same output."""
        chain = _make_chain()
        now = datetime(2025, 12, 1, tzinfo=timezone.utc)
        r1 = _select_sampled_contracts(chain, 500.0, now)
        r2 = _select_sampled_contracts(chain, 500.0, now)
        assert [c["option_symbol"] for c in r1] == [c["option_symbol"] for c in r2]
        assert len(r1) > 0

    def test_reversed_input_same_output(self):
        """Order of input contracts must not matter."""
        chain = _make_chain()
        now = datetime(2025, 12, 1, tzinfo=timezone.utc)
        r1 = _select_sampled_contracts(chain, 500.0, now)
        r2 = _select_sampled_contracts(list(reversed(chain)), 500.0, now)
        assert [c["option_symbol"] for c in r1] == [c["option_symbol"] for c in r2]

    def test_respects_expiry_count(self):
        chain = _make_chain()
        now = datetime(2025, 12, 1, tzinfo=timezone.utc)
        result = _select_sampled_contracts(chain, 500.0, now, expiry_count=1)
        expiries = {c["expiry"] for c in result}
        assert len(expiries) == 1

    def test_respects_strike_window(self):
        chain = _make_chain()
        now = datetime(2025, 12, 1, tzinfo=timezone.utc)
        result = _select_sampled_contracts(chain, 500.0, now, expiry_count=1, strike_window=1)
        # Window of 1 = center +/- 1 = up to 3 strikes * 2 rights = 6
        assert len(result) <= 6

    def test_no_spot_uses_median(self):
        chain = _make_chain()
        now = datetime(2025, 12, 1, tzinfo=timezone.utc)
        result = _select_sampled_contracts(chain, None, now)
        assert len(result) > 0

    def test_empty_chain(self):
        now = datetime(2025, 12, 1, tzinfo=timezone.utc)
        assert _select_sampled_contracts([], 100.0, now) == []

    def test_expired_contracts_excluded(self):
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        chain = _make_chain(expiries=[yesterday])
        now = datetime.now(timezone.utc)
        result = _select_sampled_contracts(chain, 500.0, now)
        assert result == []

    def test_symbols_are_unique(self):
        chain = _make_chain()
        now = datetime(2025, 12, 1, tzinfo=timezone.utc)
        result = _select_sampled_contracts(chain, 500.0, now)
        syms = [c["option_symbol"] for c in result]
        assert len(syms) == len(set(syms))

    def test_max_contracts_respected(self):
        chain = _make_chain()
        now = datetime(2025, 12, 1, tzinfo=timezone.utc)
        result = _select_sampled_contracts(chain, 500.0, now, max_contracts=10)
        assert len(result) <= 10

    def test_stability_across_runs(self):
        chain = _make_chain()
        now = datetime(2025, 12, 1, tzinfo=timezone.utc)
        first = [c["option_symbol"] for c in _select_sampled_contracts(chain, 505.0, now)]
        for _ in range(9):
            assert [c["option_symbol"] for c in _select_sampled_contracts(chain, 505.0, now)] == first


# ---------------------------------------------------------------------------
# Tests: select_spot fallback hierarchy
# ---------------------------------------------------------------------------

class TestSelectSpot:
    def test_dxlink_preferred(self):
        spot = select_spot(
            {"bidPrice": 99.0, "askPrice": 101.0, "eventTime": 1000.0, "_recv_ts": 1001.0},
            123.0,
            [90.0, 100.0, 110.0],
        )
        assert spot["spot_source"] == "dxlink"
        assert spot["spot_value"] == 100.0

    def test_cli_fallback(self):
        spot = select_spot(None, 123.0, [90.0, 100.0, 110.0])
        assert spot["spot_source"] == "cli"
        assert spot["spot_value"] == 123.0

    def test_median_strike_last_resort(self):
        spot = select_spot(None, None, [90.0, 100.0, 110.0])
        assert spot["spot_source"] == "median_strike"
        assert spot["spot_value"] == 100.0
        assert "WARNING" in spot["warning"]


# ---------------------------------------------------------------------------
# Tests: ensure_bearer_prefix
# ---------------------------------------------------------------------------

class TestEnsureBearerPrefix:
    def test_adds_prefix(self):
        assert ensure_bearer_prefix("tok-123") == "Bearer tok-123"

    def test_idempotent(self):
        assert ensure_bearer_prefix("Bearer tok-123") == "Bearer tok-123"

    def test_empty_string(self):
        assert ensure_bearer_prefix("") == "Bearer "
