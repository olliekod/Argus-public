"""
Kalshi subpenny pricing and fixed-point count support (March 2026 migration).

Pricing (March 5 deprecation):
  - Legacy: "price": 12  (integer cents)
  - New:    "price_dollars": "0.1200"  (fixed-point string, ≥4 decimals)

Fixed-point counts (March 5 deprecation):
  - Legacy: "count": 10  (integer whole contracts)
  - New:    "count_fp": "10.00"  (fixed-point string, 0–2 decimals, whole for now)

We parse to integer cents / centi-contracts internally. Per Kalshi guidance,
multiply _fp by 100 and cast to int for centi-contracts (integer arithmetic).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union


def parse_price_cents(
    payload: Dict[str, Any],
    cents_key: str,
    dollars_key: str,
    default: int = 0,
) -> int:
    """Parse price from API payload: prefer dollars key, fallback to cents.

    Args:
        payload: Dict that may contain cents_key (int) and/or dollars_key (str).
        cents_key: Legacy key, e.g. "yes_bid", "last_price".
        dollars_key: New key, e.g. "yes_bid_dollars", "last_price_dollars".
        default: Value if neither key is present.

    Returns:
        Price in integer cents (0–100 for contract prices).
        When dollars_key is used, rounds float(dollars) * 100 to nearest int.
    """
    if dollars_key in payload:
        raw = payload[dollars_key]
        if raw is None:
            return default
        return round(float(str(raw).strip()) * 100)
    if cents_key in payload:
        raw = payload[cents_key]
        if raw is None:
            return default
        return int(raw)
    return default


def parse_level_price(level_price: Union[int, str, Dict[str, Any]]) -> int:
    """Parse a single price from an orderbook level (snapshot or delta entry).

    Level price may be:
    - int: legacy cents.
    - str: fixed-point dollars (e.g. "0.5500").
    - dict: {"price": 55} or {"price_dollars": "0.5500"} or both.

    Returns:
        Price in integer cents.
    """
    if isinstance(level_price, dict):
        return parse_price_cents(level_price, "price", "price_dollars", default=0)
    if isinstance(level_price, str):
        return round(float(level_price.strip()) * 100)
    return int(level_price)


# ---------------------------------------------------------------------------
#  Fixed-point count / quantity (March 2026 migration)
# ---------------------------------------------------------------------------


def parse_count_centicx(
    payload: Dict[str, Any],
    count_key: str,
    count_fp_key: str,
    default: int = 0,
) -> int:
    """Parse contract count from API payload: prefer *_fp, fallback to integer.

    Args:
        payload: Dict that may contain count_key (int whole contracts)
                 and/or count_fp_key (str fixed-point, e.g. "10.00").
        count_key: Legacy key, e.g. "count", "filled_count".
        count_fp_key: New key, e.g. "count_fp", "filled_count_fp".
        default: Value if neither key is present.

    Returns:
        Centi-contracts (1 whole contract = 100 centicx).
        When count_fp is used: round(float(s) * 100).
        When count (legacy) is used: int(count) * 100.
    """
    if count_fp_key in payload:
        raw = payload[count_fp_key]
        if raw is None:
            return default
        return int(round(float(str(raw).strip()) * 100))
    if count_key in payload:
        raw = payload[count_key]
        if raw is None:
            return default
        s = str(raw).strip()
        if "." in s:
            return int(round(float(s) * 100))  # fp string in legacy key
        return int(raw) * 100  # whole contracts -> centicx
    return default


def parse_count_whole(
    payload: Dict[str, Any],
    count_key: str,
    count_fp_key: str,
    default: int = 0,
) -> int:
    """Parse contract count as whole contracts (for OrderUpdate.quantity_contracts etc.).

    Prefers count_fp when present; falls back to legacy integer count.
    """
    if count_fp_key in payload:
        raw = payload[count_fp_key]
        if raw is None:
            return default
        return int(round(float(str(raw).strip())))
    if count_key in payload:
        raw = payload[count_key]
        if raw is None:
            return default
        return int(raw)
    return default


def parse_qty_centicx(qty_val: Union[int, str, float, Dict[str, Any], None]) -> int:
    """Parse quantity from orderbook level or delta: str (fp), int (legacy whole),
    or dict with quantity_fp/quantity keys.

    Args:
        qty_val: Str fixed-point (e.g. "5.00"), int whole contracts (legacy),
                 or dict {"quantity_fp": "5.00"} / {"quantity": 5}.

    Returns:
        Centi-contracts.
    """
    if qty_val is None:
        return 0
    if isinstance(qty_val, dict):
        return parse_count_centicx(qty_val, "quantity", "quantity_fp", default=0)
    if isinstance(qty_val, str):
        return int(round(float(qty_val.strip()) * 100))
    return int(qty_val) * 100  # legacy whole contracts -> centicx


def parse_snapshot_level(level: Union[list, Dict[str, Any]]) -> Optional[tuple[int, int]]:
    """Parse a single orderbook snapshot level to (price_cents, qty_centicx).

    Supports:
    - Legacy list [price, qty] e.g. [55, 10] or [55, "10.00"]
    - Subpenny list [price_dollars_str, qty_fp_str] e.g. ["0.55", "10.00"]
    - Object format {"price_dollars": "0.55", "quantity_fp": "10.00"} or
      {"price": 55, "quantity": 10}

    Returns (price_cents, qty_centicx) or None if unparseable.
    """
    try:
        if isinstance(level, dict):
            price_cents = parse_price_cents(level, "price", "price_dollars", default=0)
            qty_val = level.get("quantity_fp", level.get("quantity"))
            qty = parse_qty_centicx(qty_val) if qty_val is not None else 0
            return (price_cents, qty) if price_cents >= 0 else None
        if isinstance(level, (list, tuple)) and len(level) >= 2:
            price_cents = parse_level_price(level[0])
            qty = parse_qty_centicx(level[1])
            return (price_cents, qty) if price_cents >= 0 else None
    except (TypeError, ValueError, KeyError):
        pass
    return None
