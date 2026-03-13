"""Collector mode guard helpers for trading entrypoints."""

from __future__ import annotations

import os


class CollectorModeViolation(RuntimeError):
    """Raised when trade-execution code is invoked in collector mode."""


def guard_collector_mode() -> None:
    """Fail-fast if any module attempts trade execution in collector mode."""
    if os.environ.get("ARGUS_MODE", "collector").lower() == "collector":
        raise CollectorModeViolation(
            "Trade execution attempted while ARGUS_MODE=collector. "
            "Collector mode must never execute trades."
        )
