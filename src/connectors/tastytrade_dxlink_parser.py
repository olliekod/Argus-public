"""
DXLink Protocol Frame Parser
=============================

Isolated, pure-function parser for DXLink WebSocket JSON frames.
No network or state-machine logic lives here – only deserialization
of incoming frames into typed Python dataclasses.

DXLink protocol reference (v0.1):
  Frame types: SETUP, AUTH_STATE, CHANNEL_OPENED, CHANNEL_CLOSED,
               FEED_CONFIG, FEED_DATA, KEEPALIVE, ERROR

FEED_DATA carries compact arrays keyed by ``eventType`` with a header
row followed by one or more data rows.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger("argus.dxlink.parser")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DXLinkMessageType(str, Enum):
    SETUP = "SETUP"
    AUTH_STATE = "AUTH_STATE"
    CHANNEL_OPENED = "CHANNEL_OPENED"
    CHANNEL_CLOSED = "CHANNEL_CLOSED"
    FEED_CONFIG = "FEED_CONFIG"
    FEED_DATA = "FEED_DATA"
    KEEPALIVE = "KEEPALIVE"
    ERROR = "ERROR"


class AuthState(str, Enum):
    AUTHORIZED = "AUTHORIZED"
    UNAUTHORIZED = "UNAUTHORIZED"


# ---------------------------------------------------------------------------
# Data classes for parsed events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DXLinkFrame:
    """Minimal envelope for any DXLink frame."""
    msg_type: str
    channel: int
    raw: Dict[str, Any]


@dataclass(frozen=True)
class QuoteEvent:
    """Parsed DXLink Quote event."""
    event_symbol: str
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None
    bid_time: Optional[int] = None
    ask_time: Optional[int] = None
    sequence: Optional[int] = None
    receipt_time: Optional[int] = None


@dataclass(frozen=True)
class GreeksEvent:
    """Parsed DXLink Greeks event."""
    event_symbol: str
    price: Optional[float] = None
    volatility: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    rho: Optional[float] = None
    vega: Optional[float] = None
    timestamp: Optional[int] = None
    receipt_time: Optional[int] = None


# ---------------------------------------------------------------------------
# Parser helpers
# ---------------------------------------------------------------------------

def parse_raw_json(text: str) -> Dict[str, Any]:
    """Deserialize a raw WebSocket text frame into a dict."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning("Failed to decode DXLink frame: %s", exc)
        return {}


def classify_frame(raw: Dict[str, Any]) -> DXLinkFrame:
    """Wrap a raw dict into a :class:`DXLinkFrame`."""
    return DXLinkFrame(
        msg_type=raw.get("type", "UNKNOWN"),
        channel=raw.get("channel", 0),
        raw=raw,
    )


def is_auth_state(frame: DXLinkFrame, expected: AuthState) -> bool:
    return (
        frame.msg_type == DXLinkMessageType.AUTH_STATE
        and frame.raw.get("state") == expected.value
    )


def is_channel_opened(frame: DXLinkFrame, channel: Optional[int] = None) -> bool:
    if frame.msg_type != DXLinkMessageType.CHANNEL_OPENED:
        return False
    if channel is not None and frame.channel != channel:
        return False
    return True


def is_feed_config(frame: DXLinkFrame) -> bool:
    return frame.msg_type == DXLinkMessageType.FEED_CONFIG


def is_keepalive(frame: DXLinkFrame) -> bool:
    return frame.msg_type == DXLinkMessageType.KEEPALIVE


def is_error(frame: DXLinkFrame) -> bool:
    return frame.msg_type == DXLinkMessageType.ERROR


# ---------------------------------------------------------------------------
# FEED_DATA parsing
# ---------------------------------------------------------------------------

def _zip_header_data(header: Sequence[str], data_row: Sequence[Any]) -> Dict[str, Any]:
    """Combine a header list and a data row into a dict."""
    result: Dict[str, Any] = {}
    for i, key in enumerate(header):
        if i < len(data_row):
            result[key] = data_row[i]
    return result


def _parse_quote_dict(d: Dict[str, Any], receipt_time: Optional[int] = None) -> QuoteEvent:
    # Selection logic for timestamps: prefer specific bid/ask time, then general event/time.
    # We must be careful as TT often sends 0 for bidTime/askTime on underlyings.
    raw_time = d.get("bidTime")
    if raw_time is None or raw_time == 0:
        raw_time = d.get("askTime")
    if raw_time is None or raw_time == 0:
        raw_time = (
            d.get("time") or 
            d.get("eventTime") or 
            d.get("event_time") or 
            d.get("exchangeTime") or 
            d.get("indexTime")
        )
        
    ts = _safe_int(raw_time)
    
    return QuoteEvent(
        event_symbol=d.get("eventSymbol") or d.get("symbol") or "",
        bid_price=_safe_float(d.get("bidPrice")),
        ask_price=_safe_float(d.get("askPrice")),
        bid_size=_safe_float(d.get("bidSize")),
        ask_size=_safe_float(d.get("askSize")),
        bid_time=ts,
        ask_time=ts,
        sequence=_safe_int(d.get("sequence")),
        receipt_time=receipt_time,
    )
    


def _parse_greeks_dict(d: Dict[str, Any], receipt_time: Optional[int] = None) -> GreeksEvent:
    return GreeksEvent(
        event_symbol=d.get("eventSymbol") or d.get("symbol") or "",
        price=_safe_float(d.get("price")),
        volatility=_safe_float(d.get("volatility")),
        delta=_safe_float(d.get("delta")),
        gamma=_safe_float(d.get("gamma")),
        theta=_safe_float(d.get("theta")),
        rho=_safe_float(d.get("rho")),
        vega=_safe_float(d.get("vega")),
        timestamp=_safe_int(d.get("timestamp") or d.get("time")),
        receipt_time=receipt_time,
    )


def _safe_float(v: Any) -> Optional[float]:
    if v is None or v == "NaN":
        return None
    try:
        f = float(v)
        return None if f != f else f  # NaN check
    except (TypeError, ValueError):
        return None


def _safe_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def parse_feed_data(frame: DXLinkFrame, receipt_time: Optional[int] = None) -> List[QuoteEvent | GreeksEvent]:
    """Extract typed events from a FEED_DATA frame.

    Supports both:
    1. **Compact Array** (Standard DXLink):
       `["Quote", ["eventSymbol", ...], ["SPY", ...]]`
    2. **Verbose Dictionary** (Tastytrade specific):
       `[{"eventType": "Quote", "eventSymbol": "SPY", ...}]`
    """
    raw_data = frame.raw.get("data", [])
    if not isinstance(raw_data, list):
        return []

    events: List[QuoteEvent | GreeksEvent] = []
    idx = 0
    while idx < len(raw_data):
        item = raw_data[idx]

        # Case 2: Verbose Dictionary
        if isinstance(item, dict):
            event_type = item.get("eventType")
            if event_type == "Quote":
                events.append(_parse_quote_dict(item, receipt_time))
            elif event_type == "Greeks":
                events.append(_parse_greeks_dict(item, receipt_time))
            idx += 1
            continue

        # Case 1: Compact Array (expects event-type name string)
        if not isinstance(item, str):
            idx += 1
            continue

        event_type = item
        idx += 1

        # Next must be the header array
        if idx >= len(raw_data) or not isinstance(raw_data[idx], list):
            break
        header = raw_data[idx]
        idx += 1

        # Read data rows until we hit a non-list
        while idx < len(raw_data) and isinstance(raw_data[idx], list):
            row = raw_data[idx]
            d = _zip_header_data(header, row)
            if event_type == "Quote":
                events.append(_parse_quote_dict(d, receipt_time))
            elif event_type == "Greeks":
                events.append(_parse_greeks_dict(d, receipt_time))
            idx += 1

    return events


# ---------------------------------------------------------------------------
# Outbound frame builders (convenience)
# ---------------------------------------------------------------------------

def build_setup_frame(
    *,
    keepalive_timeout: int = 60,
    version: str = "0.1",
    aggregation_period: float = 0.1,
) -> Dict[str, Any]:
    return {
        "type": "SETUP",
        "channel": 0,
        "keepaliveTimeout": keepalive_timeout,
        "acceptDataFormat": "json",
        "acceptAggregationPeriod": aggregation_period,
        "acceptEventFlavor": "instrument-id",
        "version": version,
    }


def build_auth_frame(token: str) -> Dict[str, Any]:
    return {
        "type": "AUTH",
        "channel": 0,
        "token": token,
    }


def build_channel_request(channel: int = 1) -> Dict[str, Any]:
    return {
        "type": "CHANNEL_REQUEST",
        "channel": channel,
        "service": "FEED",
        "parameters": {"contract": "AUTO"},
    }


def build_feed_setup(
    channel: int = 1,
    event_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "type": "FEED_SETUP",
        "channel": channel,
        "acceptEventTypes": event_types or ["Quote", "Greeks"],
    }


def build_feed_subscription(
    channel: int,
    symbols: List[str],
    event_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    types = event_types or ["Quote"]
    add_list = [{"type": t, "symbol": s} for s in symbols for t in types]
    return {
        "type": "FEED_SUBSCRIPTION",
        "channel": channel,
        "add": add_list,
    }


def build_keepalive() -> Dict[str, Any]:
    return {"type": "KEEPALIVE", "channel": 0}
