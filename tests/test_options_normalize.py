from src.core.options_normalize import normalize_tastytrade_nested_chain


def test_normalize_tastytrade_nested_chain_ordering_and_fields():
    raw = {
        "data": {
            "underlying-symbol": "IBIT",
            "currency": "USD",
            "expirations": [
                {
                    "expiration-date": "2024-01-19",
                    "strikes": [
                        {
                            "strike-price": "450",
                            "call": {
                                "streamer-symbol": ".IBIT240119C450",
                                "multiplier": 100,
                                "product-type": "Equity Option",
                            },
                            "put": {
                                "streamer-symbol": ".IBIT240119P450",
                                "multiplier": 100,
                                "product-type": "Equity Option",
                            },
                        },
                        {
                            "strike-price": "440",
                            "call": {
                                "streamer-symbol": ".IBIT240119C440",
                                "multiplier": 100,
                                "product-type": "Equity Option",
                            },
                        },
                    ],
                },
                {
                    "expiration-date": "2024-01-12T00:00:00.000000000Z",
                    "strikes": [
                        {
                            "strike-price": "430",
                            "call": {"streamer-symbol": ".IBIT240112C430"},
                            "put": {"streamer-symbol": ".IBIT240112P430"},
                        }
                    ],
                },
            ],
        }
    }

    normalized = normalize_tastytrade_nested_chain(raw)

    assert len(normalized) == 5
    assert normalized[0]["expiry"] == "2024-01-12"
    assert normalized[-1]["expiry"] == "2024-01-19"
    assert normalized[0]["provider"] == "tastytrade"
    assert normalized[0]["underlying"] == "IBIT"
    assert normalized[0]["right"] in {"C", "P"}
    assert isinstance(normalized[0]["strike"], float)
    assert isinstance(normalized[0]["multiplier"], int)
    assert all(item["currency"] == "USD" for item in normalized)

    expected_order = sorted(
        normalized,
        key=lambda item: (
            item.get("expiry") or "",
            item.get("strike") if item.get("strike") is not None else -1.0,
            item.get("right") or "",
            item.get("option_symbol") or "",
        ),
    )
    assert normalized == expected_order


def test_normalize_tastytrade_nested_chain_nested_payload():
    raw = {
        "data": {
            "items": [
                {
                    "underlying-symbol": "SPY",
                    "option-chain-type": "Standard",
                    "shares-per-contract": 100,
                    "expirations": [
                        {
                            "expiration-date": "2026-02-09",
                            "strikes": [
                                {
                                    "strike-price": "500.0",
                                    "call": "SPY   260209C00500000",
                                    "put": "SPY   260209P00500000",
                                }
                            ],
                        }
                    ],
                }
            ]
        }
    }

    normalized = normalize_tastytrade_nested_chain(raw)

    assert len(normalized) == 2
    expirations = {item["expiry"] for item in normalized if item.get("expiry")}
    strikes = {item["strike"] for item in normalized if item.get("strike") is not None}
    assert expirations == {"2026-02-09"}
    assert strikes == {500.0}
    assert normalized[0]["right"] == "C"
    assert normalized[1]["right"] == "P"


def test_normalize_tastytrade_nested_chain_flat_chain():
    raw = {
        "underlying-symbol": "SPY",
        "expirations": [
            {
                "expiration-date": "2025-03-21",
                "strikes": [
                    {"strike-price": "450.0", "call": "SPY   250321C00450000"}
                ],
            }
        ],
    }

    normalized = normalize_tastytrade_nested_chain(raw)

    assert len(normalized) == 1
    assert normalized[0]["expiry"] == "2025-03-21"
