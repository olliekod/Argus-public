"""
Unit tests for DXLink frame parser.

Uses recorded JSON fixtures from tests/fixtures/dxlink/ to verify
frame-to-event normalization without network access.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.connectors.tastytrade_dxlink_parser import (
    AuthState,
    DXLinkFrame,
    DXLinkMessageType,
    GreeksEvent,
    QuoteEvent,
    build_auth_frame,
    build_channel_request,
    build_feed_setup,
    build_feed_subscription,
    build_keepalive,
    build_setup_frame,
    classify_frame,
    is_auth_state,
    is_channel_opened,
    is_error,
    is_feed_config,
    is_keepalive,
    parse_feed_data,
    parse_raw_json,
)

FIXTURES = Path(__file__).parent / "fixtures" / "dxlink"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# classify_frame
# ---------------------------------------------------------------------------

class TestClassifyFrame:
    def test_auth_state_unauthorized(self):
        raw = _load("setup_response.json")
        frame = classify_frame(raw)
        assert frame.msg_type == "AUTH_STATE"
        assert frame.channel == 0

    def test_auth_state_authorized(self):
        raw = _load("auth_authorized.json")
        frame = classify_frame(raw)
        assert is_auth_state(frame, AuthState.AUTHORIZED)

    def test_channel_opened(self):
        raw = _load("channel_opened.json")
        frame = classify_frame(raw)
        assert is_channel_opened(frame, 1)
        assert not is_channel_opened(frame, 2)

    def test_feed_config(self):
        raw = _load("feed_config.json")
        frame = classify_frame(raw)
        assert is_feed_config(frame)

    def test_keepalive(self):
        raw = _load("keepalive.json")
        frame = classify_frame(raw)
        assert is_keepalive(frame)

    def test_error(self):
        raw = _load("error.json")
        frame = classify_frame(raw)
        assert is_error(frame)


# ---------------------------------------------------------------------------
# parse_feed_data – Quote events
# ---------------------------------------------------------------------------

class TestParseFeedDataQuotes:
    def test_two_quotes(self):
        raw = _load("feed_data_quote.json")
        frame = classify_frame(raw)
        events = parse_feed_data(frame)
        assert len(events) == 2
        assert all(isinstance(e, QuoteEvent) for e in events)

        spy = events[0]
        assert spy.event_symbol == "SPY"
        assert spy.bid_price == 500.10
        assert spy.ask_price == 500.15
        assert spy.bid_size == 100
        assert spy.ask_size == 200
        assert spy.bid_time == 1700000000000
        assert spy.sequence == 12345

        ibit = events[1]
        assert ibit.event_symbol == "IBIT"
        assert ibit.bid_price == 52.30
        assert ibit.ask_price == 52.35

    def test_empty_data(self):
        frame = DXLinkFrame(msg_type="FEED_DATA", channel=1, raw={"type": "FEED_DATA", "data": []})
        assert parse_feed_data(frame) == []


# ---------------------------------------------------------------------------
# parse_feed_data – Greeks events
# ---------------------------------------------------------------------------

class TestParseFeedDataGreeks:
    def test_greeks(self):
        raw = _load("feed_data_greeks.json")
        frame = classify_frame(raw)
        events = parse_feed_data(frame)
        assert len(events) == 1
        assert isinstance(events[0], GreeksEvent)
        g = events[0]
        assert g.event_symbol == ".SPY250221C500"
        assert g.price == 5.20
        assert g.volatility == 0.18
        assert g.delta == 0.55
        assert g.gamma == 0.03
        assert g.theta == -0.05
        assert g.rho == 0.02
        assert g.vega == 0.15


# ---------------------------------------------------------------------------
# parse_feed_data – Mixed events
# ---------------------------------------------------------------------------

class TestParseFeedDataMixed:
    def test_mixed_quote_and_greeks(self):
        raw = _load("feed_data_mixed.json")
        frame = classify_frame(raw)
        events = parse_feed_data(frame)
        assert len(events) == 2
        assert isinstance(events[0], QuoteEvent)
        assert isinstance(events[1], GreeksEvent)


# ---------------------------------------------------------------------------
# parse_raw_json edge cases
# ---------------------------------------------------------------------------

class TestParseRawJson:
    def test_valid(self):
        result = parse_raw_json('{"type": "KEEPALIVE"}')
        assert result == {"type": "KEEPALIVE"}

    def test_invalid(self):
        result = parse_raw_json("not json")
        assert result == {}

    def test_none(self):
        result = parse_raw_json(None)
        assert result == {}


# ---------------------------------------------------------------------------
# NaN / None handling in numeric fields
# ---------------------------------------------------------------------------

class TestNanHandling:
    def test_nan_bid_price(self):
        raw = {
            "type": "FEED_DATA",
            "channel": 1,
            "data": [
                "Quote",
                ["eventSymbol", "bidPrice", "askPrice"],
                ["TEST", "NaN", 1.5],
            ],
        }
        frame = classify_frame(raw)
        events = parse_feed_data(frame)
        assert len(events) == 1
        assert events[0].bid_price is None
        assert events[0].ask_price == 1.5

    def test_none_values(self):
        raw = {
            "type": "FEED_DATA",
            "channel": 1,
            "data": [
                "Quote",
                ["eventSymbol", "bidPrice", "askPrice"],
                ["TEST", None, None],
            ],
        }
        frame = classify_frame(raw)
        events = parse_feed_data(frame)
        assert len(events) == 1
        assert events[0].bid_price is None
        assert events[0].ask_price is None


# ---------------------------------------------------------------------------
# Feed config – Greeks rejected
# ---------------------------------------------------------------------------

class TestFeedConfigGreeksRejected:
    def test_quote_only_config(self):
        raw = _load("feed_config_quote_only.json")
        frame = classify_frame(raw)
        accepted = frame.raw.get("acceptedEventTypes", [])
        assert "Quote" in accepted
        assert "Greeks" not in accepted


# ---------------------------------------------------------------------------
# Outbound frame builders
# ---------------------------------------------------------------------------

class TestFrameBuilders:
    def test_setup_frame(self):
        f = build_setup_frame(keepalive_timeout=30, version="0.1")
        assert f["type"] == "SETUP"
        assert f["channel"] == 0
        assert f["keepaliveTimeout"] == 30
        assert f["version"] == "0.1"

    def test_auth_frame(self):
        f = build_auth_frame("tok-abc")
        assert f["type"] == "AUTH"
        assert f["channel"] == 0
        assert f["token"] == "tok-abc"

    def test_channel_request(self):
        f = build_channel_request(1)
        assert f["type"] == "CHANNEL_REQUEST"
        assert f["channel"] == 1
        assert f["service"] == "FEED"
        assert f["parameters"]["contract"] == "AUTO"

    def test_feed_setup_default(self):
        f = build_feed_setup(1)
        assert f["type"] == "FEED_SETUP"
        assert f["channel"] == 1
        assert "Quote" in f["acceptEventTypes"]
        assert "Greeks" in f["acceptEventTypes"]

    def test_feed_setup_custom(self):
        f = build_feed_setup(1, event_types=["Quote"])
        assert f["acceptEventTypes"] == ["Quote"]

    def test_feed_subscription(self):
        f = build_feed_subscription(1, ["SPY", "IBIT"], ["Quote"])
        assert f["type"] == "FEED_SUBSCRIPTION"
        assert f["channel"] == 1
        assert len(f["add"]) == 2
        assert f["add"][0] == {"type": "Quote", "symbol": "SPY"}
        assert f["add"][1] == {"type": "Quote", "symbol": "IBIT"}

    def test_feed_subscription_multi_type(self):
        f = build_feed_subscription(1, ["SPY"], ["Quote", "Greeks"])
        assert len(f["add"]) == 2
        types = {entry["type"] for entry in f["add"]}
        assert types == {"Quote", "Greeks"}

    def test_keepalive(self):
        f = build_keepalive()
        assert f["type"] == "KEEPALIVE"
        assert f["channel"] == 0
