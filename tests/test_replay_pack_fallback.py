from src.tools.replay_pack import _merge_snapshots_primary_with_fallback


def _snap(ts: int, provider: str):
    return {
        "recv_ts_ms": ts,
        "timestamp_ms": ts,
        "provider": provider,
        "atm_iv": 0.2,
        "underlying_price": 100.0,
        "quotes_json": "{}",
        "symbol": "SPY",
        "n_strikes": 10,
    }


def test_merge_prefers_primary_and_fills_gap_from_secondary():
    primary = [_snap(0, "tastytrade"), _snap(10 * 60_000, "tastytrade")]
    secondary = [_snap(4 * 60_000, "public"), _snap(5 * 60_000, "public")]

    merged, filled = _merge_snapshots_primary_with_fallback(primary, secondary, gap_ms=3 * 60_000)

    assert filled == 2
    assert [s["provider"] for s in merged] == ["tastytrade", "public", "public", "tastytrade"]


def test_merge_uses_secondary_only_when_no_primary():
    merged, filled = _merge_snapshots_primary_with_fallback([], [_snap(60_000, "public")], gap_ms=120_000)
    assert filled == 1
    assert len(merged) == 1
    assert merged[0]["provider"] == "public"
