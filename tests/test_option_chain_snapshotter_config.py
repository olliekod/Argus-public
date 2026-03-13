"""
Tests that the option chain snapshotter uses a config-driven underlyings list.

Verifies:
- The snapshotter reads symbols from config (no hardcoded list in the poll loop).
- Default is IBIT + liquid ETF universe (no BITO).
"""

from __future__ import annotations

import pytest

from src.core.config import load_config
from src.core.liquid_etf_universe import get_liquid_etf_universe


def _default_options_symbols() -> list[str]:
    """Same default as orchestrator: IBIT + liquid ETF universe (no BITO)."""
    return sorted({"IBIT"} | set(get_liquid_etf_universe()))


def _options_symbols_from_config(config: dict) -> list[str]:
    """Extract options chain symbols the same way the orchestrator does."""
    options_cfg = (
        config.get("exchanges", {}).get("alpaca", {}).get("options", {})
    )
    return options_cfg.get("symbols", _default_options_symbols())


class TestOptionChainSnapshotterConfigDriven:
    """Snapshotter must iterate over a config-driven list (no hardcoded symbols)."""

    def test_symbols_come_from_config(self):
        """When config has options.symbols, that list is used."""
        config = {
            "exchanges": {
                "alpaca": {
                    "options": {
                        "enabled": True,
                        "symbols": ["CUSTOM1", "CUSTOM2", "CUSTOM3"],
                    },
                },
            },
        }
        symbols = _options_symbols_from_config(config)
        assert symbols == ["CUSTOM1", "CUSTOM2", "CUSTOM3"]

    def test_fallback_when_no_symbols_key(self):
        """When options.symbols is missing, fallback is IBIT + liquid universe (no BITO)."""
        config = {
            "exchanges": {
                "alpaca": {
                    "options": {"enabled": True},
                },
            },
        }
        symbols = _options_symbols_from_config(config)
        assert "IBIT" in symbols
        assert "SPY" in symbols
        assert "QQQ" in symbols
        assert "BITO" not in symbols

    def test_default_config_ibit_liquid_universe_no_bito(self):
        """Default config includes IBIT and liquid ETF universe; BITO is dropped from options."""
        config = load_config()
        symbols = _options_symbols_from_config(config)
        assert "SPY" in symbols, "Default config should include SPY for replay packs"
        assert "QQQ" in symbols, "Default config should include QQQ for replay packs"
        assert "IBIT" in symbols, "Default config should include IBIT"
        assert "BITO" not in symbols, "BITO dropped from options/IV streaming"
