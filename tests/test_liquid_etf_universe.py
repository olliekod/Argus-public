from datetime import datetime, timezone

from scripts.tastytrade_health_audit import _select_sampled_contracts, select_spot
from src.core.liquid_etf_universe import LIQUID_ETF_UNIVERSE, get_liquid_etf_universe


def _build_contracts() -> list[dict]:
    contracts = []
    for expiry in ("2026-01-16", "2026-02-20", "2026-03-20"):
        for strike in range(95, 106):
            for right in ("C", "P"):
                contracts.append(
                    {
                        "underlying": "SPY",
                        "expiry": expiry,
                        "strike": float(strike),
                        "right": right,
                        "option_symbol": f".SPY{expiry.replace('-', '')}{right}{strike:08d}",
                    }
                )
    return contracts


def test_liquid_etf_universe_ordering_alphabetical():
    universe = get_liquid_etf_universe()
    assert universe == sorted(universe)
    assert tuple(universe) == LIQUID_ETF_UNIVERSE


def test_spot_hierarchy_dxlink_then_cli_then_median():
    spot = select_spot({"bidPrice": 99.0, "askPrice": 101.0, "eventTime": 1000.0, "_recv_ts": 1001.0}, 123.0, [90.0, 100.0, 110.0])
    assert spot["spot_source"] == "dxlink"
    assert spot["spot_value"] == 100.0

    spot = select_spot(None, 123.0, [90.0, 100.0, 110.0])
    assert spot["spot_source"] == "cli"
    assert spot["spot_value"] == 123.0

    spot = select_spot(None, None, [90.0, 100.0, 110.0])
    assert spot["spot_source"] == "median_strike"
    assert spot["spot_value"] == 100.0
    assert "WARNING" in spot["warning"]


def test_sampling_selection_stable_and_expected_symbols():
    contracts = _build_contracts()
    now = datetime(2025, 12, 1, tzinfo=timezone.utc)

    sample1 = _select_sampled_contracts(contracts, spot_value=100.0, now_utc=now, expiry_count=2, strike_window=1, max_contracts=12)
    sample2 = _select_sampled_contracts(list(reversed(contracts)), spot_value=100.0, now_utc=now, expiry_count=2, strike_window=1, max_contracts=12)

    symbols1 = [c["option_symbol"] for c in sample1]
    symbols2 = [c["option_symbol"] for c in sample2]
    assert symbols1 == symbols2
    assert symbols1[:3] == [
        ".SPY20260116C00000099",
        ".SPY20260116P00000099",
        ".SPY20260116C00000100",
    ]
    assert len(symbols1) == 12
