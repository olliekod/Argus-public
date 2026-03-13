"""
Strategy Registry
==================

Minimal registry that tracks candidate strategies (those that survived
all deploy gates) with their parameters and metadata.

The registry is the bridge between the StrategyEvaluator (which produces
rankings) and the AllocationEngine (which consumes forecasts).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("argus.strategy_registry")


@dataclass
class StrategyEntry:
    """A single strategy entry in the registry.

    Attributes
    ----------
    strategy_id : str
        Unique identifier (e.g. ``VRP_v1``).
    strategy_class : str
        Fully-qualified class name (e.g. ``VRPCreditSpreadStrategy``).
    params : dict
        Strategy parameters used in the evaluation.
    run_id : str
        Experiment run ID that produced this candidate.
    composite_score : float
        Composite score from StrategyEvaluator.
    sharpe : float
        Sharpe ratio from evaluation.
    dsr : float
        Deflated Sharpe Ratio (0 if not computed).
    meta : dict
        Additional metadata (kill reasons, cost sensitivity, etc.).
    """
    strategy_id: str
    strategy_class: str
    params: Dict[str, Any] = field(default_factory=dict)
    run_id: str = ""
    composite_score: float = 0.0
    sharpe: float = 0.0
    dsr: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)


class StrategyRegistry:
    """Registry of candidate strategies that survived deploy gates.

    Usage::

        registry = StrategyRegistry()

        # Populate from evaluator rankings
        registry.load_from_rankings(rankings, min_dsr=0.95)

        # Iterate candidates
        for entry in registry.candidates:
            print(entry.strategy_id, entry.composite_score)
    """

    def __init__(self) -> None:
        self._entries: Dict[str, StrategyEntry] = {}

    def register(self, entry: StrategyEntry) -> None:
        """Register a strategy entry.

        Overwrites any existing entry with the same ``strategy_id``.
        """
        self._entries[entry.strategy_id] = entry
        logger.info(
            "Registered strategy %s (class=%s, score=%.4f, dsr=%.4f)",
            entry.strategy_id, entry.strategy_class,
            entry.composite_score, entry.dsr,
        )

    def remove(self, strategy_id: str) -> bool:
        """Remove a strategy by ID.  Returns True if found."""
        if strategy_id in self._entries:
            del self._entries[strategy_id]
            logger.info("Removed strategy %s", strategy_id)
            return True
        return False

    def get(self, strategy_id: str) -> Optional[StrategyEntry]:
        """Get a strategy entry by ID, or None."""
        return self._entries.get(strategy_id)

    @property
    def candidates(self) -> List[StrategyEntry]:
        """All registered candidates, sorted by composite score descending."""
        return sorted(
            self._entries.values(),
            key=lambda e: (-e.composite_score, e.strategy_id),
        )

    @property
    def count(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    def load_from_rankings(
        self,
        rankings: List[Dict[str, Any]],
        min_composite_score: float = 0.0,
        min_dsr: float = 0.0,
        exclude_killed: bool = True,
    ) -> int:
        """Populate registry from StrategyEvaluator rankings output.

        Parameters
        ----------
        rankings : list of dict
            The ``rankings`` list from StrategyEvaluator output.
        min_composite_score : float
            Minimum composite score to include (default 0).
        min_dsr : float
            Minimum DSR to include (default 0 = no filter).
        exclude_killed : bool
            Whether to exclude strategies with kill reasons (default True).

        Returns
        -------
        int
            Number of strategies registered.
        """
        count = 0
        for rec in rankings:
            if exclude_killed and rec.get("killed", False):
                continue

            score = rec.get("composite_score", 0.0)
            if score < min_composite_score:
                continue

            dsr_val = rec.get("dsr", rec.get("metrics", {}).get("dsr", 0.0))
            if dsr_val < min_dsr:
                continue

            entry = StrategyEntry(
                strategy_id=rec.get("strategy_id", "UNKNOWN"),
                strategy_class=rec.get("strategy_class", ""),
                params=rec.get("strategy_params", {}),
                run_id=rec.get("run_id", ""),
                composite_score=score,
                sharpe=rec.get("metrics", {}).get("sharpe", 0.0),
                dsr=dsr_val,
                meta={
                    "rank": rec.get("rank"),
                    "regime_scores": rec.get("regime_scores", {}),
                },
            )
            self.register(entry)
            count += 1

        logger.info(
            "Loaded %d candidates from %d rankings (min_score=%.2f, min_dsr=%.2f)",
            count, len(rankings), min_composite_score, min_dsr,
        )
        return count

    def to_list(self) -> List[Dict[str, Any]]:
        """Serialize all entries to a list of dicts."""
        return [
            {
                "strategy_id": e.strategy_id,
                "strategy_class": e.strategy_class,
                "params": e.params,
                "run_id": e.run_id,
                "composite_score": e.composite_score,
                "sharpe": e.sharpe,
                "dsr": e.dsr,
                "meta": e.meta,
            }
            for e in self.candidates
        ]
