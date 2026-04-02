# Created by Oliver Meihls

# Market selection helpers for BTC/ETH/SOL 15m, hourly and range contracts.

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

from .logging_utils import ComponentLogger

log = ComponentLogger("market_selectors")

_SERIES_MAP = {
    # BTC: 15m (M and 15M naming), hourly (D, H), range
    "KXBTCM": ("BTC", 15, False),
    "KXBTC15M": ("BTC", 15, False),
    "KXBTCD": ("BTC", 60, False),
    "KXBTCH": ("BTC", 60, False),
    "KXBTC": ("BTC", 0, True),
    # ETH
    "KXETHM": ("ETH", 15, False),
    "KXETH15M": ("ETH", 15, False),
    "KXETHD": ("ETH", 60, False),
    "KXETHH": ("ETH", 60, False),
    "KXETH": ("ETH", 0, True),
    # SOL
    "KXSOLM": ("SOL", 15, False),
    "KXSOL15M": ("SOL", 15, False),
    "KXSOLD": ("SOL", 60, False),
    "KXSOLH": ("SOL", 60, False),
    "KXSOL": ("SOL", 0, True),
}


def classify_series(series_ticker: str) -> Optional[tuple[str, int, bool]]:
    # Return (asset, window_minutes, is_range) for known series.
    if not series_ticker:
        return None
    return _SERIES_MAP.get(series_ticker.upper())


def settlement_timestamp(settlement_time_iso: str) -> Optional[float]:
    # Parse a Kalshi settlement/close ISO timestamp to epoch seconds.
    if not settlement_time_iso:
        return None
    try:
        return datetime.fromisoformat(settlement_time_iso.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def time_to_settlement_seconds(
    settlement_time_iso: str,
    *,
    now_ts: Optional[float] = None,
) -> Optional[float]:
    # Return seconds until settlement for a market, or None if unknown.
    settle_ts = settlement_timestamp(settlement_time_iso)
    if settle_ts is None:
        return None
    current_ts = now_ts if now_ts is not None else datetime.now(timezone.utc).timestamp()
    return settle_ts - current_ts


def hold_entry_horizon_seconds(
    *,
    window_minutes: int,
    is_range: bool,
    max_entry_minutes_to_expiry: int,
    range_max_entry_minutes_to_expiry: int,
) -> int:
    # Return the effective hold-to-expiry entry horizon for a market.
    #
    # The farm should only open non-scalping positions close to expiry:
    # 15m contracts in roughly the last 3 minutes, 60m contracts in roughly
    # the last 12 minutes, and longer range-style contracts in roughly the
    # last 12 minutes as well. Configured limits act as upper bounds.
    if is_range:
        default_minutes = 12
        configured_minutes = range_max_entry_minutes_to_expiry
    elif window_minutes <= 15:
        default_minutes = 3
        configured_minutes = max_entry_minutes_to_expiry
    elif window_minutes <= 60:
        default_minutes = 12
        configured_minutes = max_entry_minutes_to_expiry
    else:
        default_minutes = max(12, int(round(window_minutes * 0.2)))
        configured_minutes = max_entry_minutes_to_expiry

    if configured_minutes > 0:
        return min(default_minutes, configured_minutes) * 60
    return default_minutes * 60


def filter_supported_markets(
    markets: List[Dict],
    *,
    assets: List[str],
    window_minutes: List[int],
    include_range_markets: bool,
    title_regex: Optional[str] = None,
    require_open: bool = True,
) -> List[Dict]:
    import re

    compiled_re = re.compile(title_regex, re.IGNORECASE) if title_regex else None
    assets_set = {a.upper() for a in assets}
    allowed_windows = set(window_minutes)
    out: List[Dict] = []

    for m in markets:
        if require_open:
            status = str(m.get("status", "")).lower()
            if status and status not in ("open", "active"):
                continue

        # API list response may omit series_ticker; derive from event_ticker (e.g. "KXBTC15M-26FEB25" -> "KXBTC15M").
        series = str(m.get("series_ticker", "")).upper()
        if not series:
            event = str(m.get("event_ticker", "")).strip()
            if event and "-" in event:
                series = event.split("-")[0].upper()
        info = classify_series(series)
        if not info:
            continue
        asset, win_min, is_range = info
        if asset not in assets_set:
            continue
        if is_range and not include_range_markets:
            continue
        if (not is_range) and win_min not in allowed_windows:
            continue

        if compiled_re:
            text = f"{m.get('title', '')} {m.get('subtitle', '')}"
            if not compiled_re.search(text):
                continue

        out.append(m)

    if not out and markets:
        log.warning(f"No supported markets found after filtering {len(markets)} candidates")

    return out


# Backward-compatible wrappers for existing tests/callers.
def is_btc_related(market: Dict) -> bool:
    series = str(market.get("series_ticker", "")).upper()
    if series.startswith("KXBTC"):
        return True
    text = " ".join(str(market.get(k, "")) for k in ("ticker", "title", "subtitle"))
    return "BTC" in text.upper() or "BITCOIN" in text.upper()


def is_15min_window(market: Dict) -> bool:
    info = classify_series(str(market.get("series_ticker", "")))
    if info:
        return info[1] == 15
    text = " ".join(str(market.get(k, "")) for k in ("title", "subtitle", "rules_primary", "rules"))
    return "15" in text and "MIN" in text.upper()


def filter_btc_15min_markets(markets: List[Dict], *, title_regex: Optional[str] = None, require_open: bool = True) -> List[Dict]:
    return [
        m for m in filter_supported_markets(
            markets,
            assets=["BTC"],
            window_minutes=[15],
            include_range_markets=False,
            title_regex=title_regex,
            require_open=require_open,
        )
    ]
