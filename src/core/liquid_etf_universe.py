"""Liquid ETF universe constants and helpers."""

from __future__ import annotations

LIQUID_ETF_UNIVERSE: tuple[str, ...] = tuple(
    sorted(
        {
            "SPY",
            "QQQ",
            "IWM",
            "DIA",
            "TLT",
            "GLD",
            "XLF",
            "XLK",
            "XLE",
            "SMH",
        }
    )
)


def get_liquid_etf_universe() -> list[str]:
    """Return the liquid ETF universe in deterministic alphabetical order."""
    return list(LIQUID_ETF_UNIVERSE)
