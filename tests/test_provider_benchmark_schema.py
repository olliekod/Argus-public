from scripts.provider_benchmark import _build_scorecards


def test_provider_benchmark_scorecard_shape():
    bars = [
        {"provider": "yahoo", "symbol": "SPY", "request_latency_ms": 100.0, "bar_age_sec": 20.0, "success": True},
        {"provider": "yahoo", "symbol": "QQQ", "request_latency_ms": 120.0, "bar_age_sec": 25.0, "success": True},
    ]
    options = [
        {
            "provider": "tastytrade_dxlink",
            "missing_rate": 0.1,
            "handshake_latency": 500.0,
            "lag_p95": 1000.0,
            "spread_p95": 40.0,
            "total_events": 100,
            "events_missing_ts": 5,
            "greeks_presence_rate": 0.0,
        }
    ]
    out = _build_scorecards(bars, options, greeks_enabled=True)
    assert "BarsScorecard" in out
    assert "OptionsQuoteScorecard" in out
    assert "OptionsGreeksScorecard" in out
    assert "Composite" in out
    assert out["Composite"]["status"] in {"complete", "partial"}
