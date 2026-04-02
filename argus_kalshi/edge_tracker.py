# Created by Oliver Meihls

# Track expected-vs-realized edge retention per context key.
#
# Maintains rolling windows of expected edge (at signal time) and realized edge
# (at settlement) for each context key.  Provides retention ratios and weight
# multipliers so the farm can down-weight contexts where edge is decaying.

from __future__ import annotations

import threading
from collections import defaultdict, deque
from typing import Any, Deque, Dict

from .config import KalshiConfig


class EdgeTracker:
    # Thread-safe rolling-window edge retention tracker.

    def __init__(self, cfg: KalshiConfig) -> None:
        self._enabled = cfg.enable_edge_tracking
        self._window = cfg.edge_tracking_window_settles
        self._min_samples = cfg.edge_tracking_min_samples
        self._max_keys = int(getattr(cfg, "edge_tracking_max_keys", 10000))
        self._decay_threshold = cfg.edge_retention_decay_threshold
        self._decay_multiplier = cfg.edge_retention_decay_multiplier
        self._dropped_new_keys = 0

        self._expected: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=self._window)
        )
        self._realized: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=self._window)
        )
        self._lock = threading.Lock()

    # ── Recording ────────────────────────────────────────────────────────

    def record_entry(
        self, context_key: str, expected_edge: float, entry_price_cents: int
    ) -> None:
        # Store expected edge at signal time.
        if not self._enabled:
            return
        with self._lock:
            if context_key not in self._expected and len(self._expected) >= self._max_keys:
                self._dropped_new_keys += 1
                return
            self._expected[context_key].append(expected_edge)

    def record_settlement(
        self,
        context_key: str,
        net_pnl_usd: float,
        entry_price_cents: int,
        quantity: int,
    ) -> None:
        # Compute and store realized edge from settlement outcome.
        if not self._enabled:
            return
        if entry_price_cents <= 0 or quantity <= 0:
            return
        cost = (entry_price_cents / 100.0) * quantity
        realized = net_pnl_usd / cost
        realized = max(-1.0, min(2.0, realized))
        with self._lock:
            if context_key not in self._realized and len(self._realized) >= self._max_keys:
                self._dropped_new_keys += 1
                return
            self._realized[context_key].append(realized)

    # ── Queries ──────────────────────────────────────────────────────────

    def get_retention_ratio(self, context_key: str) -> float:
        # Return realized_avg / expected_avg, or 1.0 if insufficient data.
        with self._lock:
            realized = self._realized.get(context_key)
            expected = self._expected.get(context_key)
            if (
                not realized
                or not expected
                or len(realized) < self._min_samples
            ):
                return 1.0
            expected_avg = sum(expected) / len(expected)
            if expected_avg <= 0.0:
                return 1.0
            realized_avg = sum(realized) / len(realized)
            return realized_avg / expected_avg

    def get_weight_multiplier(self, context_key: str) -> float:
        # Return 1.0 normally; decay_multiplier if retention < threshold.
        ratio = self.get_retention_ratio(context_key)
        if ratio < self._decay_threshold:
            return self._decay_multiplier
        return 1.0

    def diagnostics(self) -> Dict[str, Any]:
        # Summary stats across all tracked context keys.
        with self._lock:
            keys = set(self._expected.keys()) | set(self._realized.keys())

        if not keys:
            return {
                "total_keys": 0,
                "max_keys": self._max_keys,
                "dropped_new_keys": self._dropped_new_keys,
                "keys_poor_retention": 0,
                "avg_retention": 1.0,
            }

        ratios = [self.get_retention_ratio(k) for k in keys]
        poor = sum(1 for r in ratios if r < self._decay_threshold)
        avg = sum(ratios) / len(ratios) if ratios else 1.0

        return {
            "total_keys": len(keys),
            "max_keys": self._max_keys,
            "dropped_new_keys": self._dropped_new_keys,
            "keys_poor_retention": poor,
            "avg_retention": round(avg, 4),
        }
