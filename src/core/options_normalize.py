"""Option chain normalization utilities.

Multiplier handling
-------------------
The ``shares_per_contract`` (multiplier) is extracted from the API response
and defaults to 100 when absent.  Non-standard multipliers (≠ 100) may
occur after corporate actions (stock splits, mergers, special dividends).
Such contracts are flagged in ``meta["non_standard_multiplier"]`` so
downstream consumers can apply conservative handling.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable

from src.connectors.tastytrade_rest import parse_rfc3339_nano

logger = logging.getLogger(__name__)


def _first_present(values: Iterable[Any]) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _parse_expiry(value: Any) -> str | None:
    if not value:
        return None
    if isinstance(value, str) and ("T" in value or "Z" in value or "+" in value):
        try:
            return parse_rfc3339_nano(value).date().isoformat()
        except ValueError:
            return None
    if isinstance(value, str):
        return value
    return None


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_default(value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_tastytrade_nested_chain(raw: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Normalize tastytrade nested option chain response into a flat list."""
    if not raw or not isinstance(raw, dict):
        return []

    chains: list[Dict[str, Any]] = []
    data = raw.get("data")
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        chains = data.get("items", [])
    elif isinstance(data, dict) and isinstance(data.get("expirations"), list):
        chains = [data]
    elif isinstance(raw.get("expirations"), list):
        chains = [raw]
    else:
        return []

    normalized: list[Dict[str, Any]] = []

    for chain in chains:
        if not isinstance(chain, dict):
            continue

        underlying = _first_present(
            [
                chain.get("underlying-symbol"),
                chain.get("underlying_symbol"),
                chain.get("root-symbol"),
                chain.get("root_symbol"),
                chain.get("symbol"),
                chain.get("underlying"),
            ]
        )
        chain_type = _first_present(
            [
                chain.get("option-chain-type"),
                chain.get("option_chain_type"),
                chain.get("chain-type"),
            ]
        )
        shares_per_contract = _int_or_default(
            _first_present(
                [
                    chain.get("shares-per-contract"),
                    chain.get("shares_per_contract"),
                ]
            ),
            100,
        )

        expirations = chain.get("expirations") or []
        for expiration in expirations:
            if not isinstance(expiration, dict):
                continue

            expiry_raw = _first_present(
                [
                    expiration.get("expiration-date"),
                    expiration.get("expiration"),
                    expiration.get("expiration-date-time"),
                    expiration.get("date"),
                ]
            )
            expiry = _parse_expiry(expiry_raw)

            strikes = (
                expiration.get("strikes")
                or expiration.get("strike-prices")
                or expiration.get("strike-price-list")
                or []
            )

            for strike in strikes:
                if not isinstance(strike, dict):
                    continue

                strike_price = _first_present(
                    [
                        strike.get("strike-price"),
                        strike.get("strike"),
                        strike.get("price"),
                        strike.get("strike_price"),
                    ]
                )
                strike_value = _float_or_none(strike_price)

                for right_label, option_key, streamer_key in (
                    ("C", "call", "call-streamer-symbol"),
                    ("P", "put", "put-streamer-symbol"),
                ):
                    option_value = strike.get(option_key)
                    if not option_value:
                        continue

                    option_symbol = option_value
                    option_streamer = strike.get(streamer_key) or strike.get(
                        streamer_key.replace("-", "_")
                    )

                    if isinstance(option_value, dict):
                        option_symbol = _first_present(
                            [
                                option_value.get("streamer-symbol"),
                                option_value.get("symbol"),
                                option_value.get("occ-symbol"),
                            ]
                        )
                        option_streamer = option_streamer or option_value.get(
                            "streamer-symbol"
                        )

                    meta = {
                        "streamer_symbol": option_streamer,
                        "chain_type": chain_type,
                    }
                    meta = {key: value for key, value in meta.items() if value is not None}

                    # Flag non-standard multipliers (corporate actions)
                    if shares_per_contract != 100:
                        meta["non_standard_multiplier"] = True
                        logger.warning(
                            "Non-standard multiplier %d for %s %s %s %.2f%s — "
                            "possible corporate action; verify before trading",
                            shares_per_contract,
                            underlying,
                            expiry or "?",
                            right_label,
                            strike_value if strike_value is not None else 0.0,
                            f" ({option_symbol})" if option_symbol else "",
                        )

                    normalized.append(
                        {
                            "provider": "tastytrade",
                            "underlying": underlying,
                            "option_symbol": option_symbol,
                            "expiry": expiry,
                            "right": right_label,
                            "strike": strike_value,
                            "multiplier": shares_per_contract,
                            "currency": "USD",
                            "exchange": None,
                            "meta": meta,
                        }
                    )

    normalized.sort(
        key=lambda item: (
            item.get("expiry") or "",
            item.get("strike") if item.get("strike") is not None else -1.0,
            item.get("right") or "",
            item.get("option_symbol") or "",
        )
    )
    return normalized
