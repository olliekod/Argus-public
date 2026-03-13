"""
Argus Data Source Policy
========================

Canonical helper that reads the ``data_sources`` section from
``config/config.yaml`` and returns provider selections used by
replay packs, experiment runners, and strategies.

The policy enforces a single source of truth so that no CLI flag
or script needs to hard-code provider names.

Usage::

    from src.core.data_sources import get_data_source_policy

    policy = get_data_source_policy()
    policy.bars_primary           # "alpaca"
    policy.options_snapshots_primary  # "tastytrade"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.core.config import load_config

logger = logging.getLogger("argus.data_sources")

# ─── Defaults (mirror config/config.yaml) ────────────────────────────────

_DEFAULT_BARS_PRIMARY = "alpaca"
_DEFAULT_OUTCOMES_FROM = "bars_primary"
_DEFAULT_OPTIONS_SNAPSHOTS_PRIMARY = "tastytrade"
_DEFAULT_OPTIONS_SNAPSHOTS_SECONDARY: List[str] = ["public"]
_DEFAULT_OPTIONS_STREAM_PRIMARY = "tastytrade_dxlink"
_DEFAULT_BARS_SECONDARY: List[str] = ["yahoo"]

# Alpaca is bars and outcomes only; it does not provide IV/greeks/option data.
# Options snapshot primary must be tastytrade or public, never alpaca.
_ALLOWED_OPTIONS_SNAPSHOT_PROVIDERS = {"tastytrade", "public"}


@dataclass(frozen=True)
class DataSourcePolicy:
    """Immutable snapshot of the Argus data-source policy."""

    bars_primary: str = _DEFAULT_BARS_PRIMARY
    outcomes_from: str = _DEFAULT_OUTCOMES_FROM
    options_snapshots_primary: str = _DEFAULT_OPTIONS_SNAPSHOTS_PRIMARY
    options_snapshots_secondary: List[str] = field(
        default_factory=lambda: list(_DEFAULT_OPTIONS_SNAPSHOTS_SECONDARY)
    )
    options_stream_primary: str = _DEFAULT_OPTIONS_STREAM_PRIMARY
    bars_secondary: List[str] = field(
        default_factory=lambda: list(_DEFAULT_BARS_SECONDARY)
    )

    @property
    def bars_provider(self) -> str:
        return self.bars_primary

    @property
    def options_snapshot_provider(self) -> str:
        return self.options_snapshots_primary

    def is_secondary_options_provider(self, provider: str) -> bool:
        return provider in self.options_snapshots_secondary

    def snapshot_providers(self, include_secondary: bool = False) -> List[str]:
        providers = [self.options_snapshots_primary]
        if include_secondary:
            providers.extend(self.options_snapshots_secondary)
        return providers


def _parse_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def get_data_source_policy(config: Optional[Dict[str, Any]] = None) -> DataSourcePolicy:
    """Load the data-source policy from config."""
    if config is None:
        try:
            config = load_config()
        except Exception:
            logger.warning(
                "Could not load config.yaml; using built-in data-source defaults"
            )
            return DataSourcePolicy()

    ds = config.get("data_sources", {})
    if not ds:
        logger.info("No data_sources section in config; using defaults")
        return DataSourcePolicy()

    policy = DataSourcePolicy(
        bars_primary=ds.get("bars_primary", _DEFAULT_BARS_PRIMARY),
        outcomes_from=ds.get("outcomes_from", _DEFAULT_OUTCOMES_FROM),
        options_snapshots_primary=ds.get(
            "options_snapshots_primary", _DEFAULT_OPTIONS_SNAPSHOTS_PRIMARY
        ),
        options_snapshots_secondary=_parse_list(
            ds.get("options_snapshots_secondary", _DEFAULT_OPTIONS_SNAPSHOTS_SECONDARY)
        ),
        options_stream_primary=ds.get(
            "options_stream_primary", _DEFAULT_OPTIONS_STREAM_PRIMARY
        ),
        bars_secondary=_parse_list(ds.get("bars_secondary", _DEFAULT_BARS_SECONDARY)),
    )

    if policy.options_snapshots_primary not in _ALLOWED_OPTIONS_SNAPSHOT_PROVIDERS:
        logger.warning(
            "Unknown options_snapshots_primary=%s; expected one of %s",
            policy.options_snapshots_primary,
            sorted(_ALLOWED_OPTIONS_SNAPSHOT_PROVIDERS),
        )

    return policy
