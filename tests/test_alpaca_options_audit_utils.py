from datetime import datetime, timezone

from scripts.alpaca_option_chain_snapshot import (
    AuditThresholds,
    TradableConfig,
    compute_spread_bps,
    evaluate_tradable_pass_fail,
    is_tradable_candidate,
    parse_rfc3339_to_datetime,
    percentile,
    quote_age_seconds,
)


def test_parse_rfc3339_to_datetime_handles_nanoseconds_and_z():
    dt = parse_rfc3339_to_datetime("2024-01-01T12:34:56.123456789Z")
    assert dt.tzinfo is not None
    assert dt == datetime(2024, 1, 1, 12, 34, 56, 123456, tzinfo=timezone.utc)


def test_compute_spread_bps():
    assert compute_spread_bps(10.0, 10.5) == 0.5 / 10.25 * 10000
    assert compute_spread_bps(0, 10.5) is None
    assert compute_spread_bps(10.0, 0) is None
    assert compute_spread_bps(None, 10.0) is None


def test_percentile_helper():
    values = [1, 2, 3, 4, 5]
    assert percentile(values, 50) == 3
    assert percentile(values, 0) == 1
    assert percentile(values, 100) == 5


def test_quote_age_seconds():
    now = datetime(2024, 1, 1, 12, 0, 1, tzinfo=timezone.utc)
    ts = "2024-01-01T12:00:00.000000Z"
    assert quote_age_seconds(ts, now) == 1.0


def test_is_tradable_candidate_filters_by_age_and_spread_and_moneyness():
    now = datetime(2024, 1, 1, 12, 0, 10, tzinfo=timezone.utc)
    cfg = TradableConfig(max_quote_age_s=5, max_spread_bps=300, max_moneyness=0.10)
    contract = {"status": "active", "tradable": True, "strike_price": 100, "symbol": "TEST"}
    snapshot = {
        "latestQuote": {
            "t": "2024-01-01T12:00:08.000000Z",
            "bp": 10.0,
            "ap": 10.05,
        }
    }

    result = is_tradable_candidate(contract, snapshot, 100, now, cfg)
    assert result.qualifies is True

    stale_snapshot = {
        "latestQuote": {
            "t": "2024-01-01T12:00:00.000000Z",
            "bp": 10.0,
            "ap": 10.05,
        }
    }
    result = is_tradable_candidate(contract, stale_snapshot, 100, now, cfg)
    assert result.qualifies is False

    wide_snapshot = {
        "latestQuote": {
            "t": "2024-01-01T12:00:08.000000Z",
            "bp": 1.0,
            "ap": 2.0,
        }
    }
    result = is_tradable_candidate(contract, wide_snapshot, 100, now, cfg)
    assert result.qualifies is False

    far_contract = {"status": "active", "tradable": True, "strike_price": 130, "symbol": "FAR"}
    result = is_tradable_candidate(far_contract, snapshot, 100, now, cfg)
    assert result.qualifies is False


def test_evaluate_tradable_pass_fail():
    thresholds = AuditThresholds(quote_age_p99=3.0, spread_bps_p95=200)
    tradable_report = {
        "counts": {"qualifying_contracts": 40},
        "quote_age_percentiles": {"p99": 2.5},
        "spread_bps_percentiles": {"p95": 150},
    }

    passed, reasons = evaluate_tradable_pass_fail(tradable_report, thresholds, min_tradable_count=30)
    assert passed is True
    assert reasons == []

    failing_report = {
        "counts": {"qualifying_contracts": 10},
        "quote_age_percentiles": {"p99": 4.5},
        "spread_bps_percentiles": {"p95": 250},
    }
    passed, reasons = evaluate_tradable_pass_fail(failing_report, thresholds, min_tradable_count=30)
    assert passed is False
    assert any("min_tradable_count" in reason for reason in reasons)
