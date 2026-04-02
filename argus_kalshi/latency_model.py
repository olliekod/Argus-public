# Created by Oliver Meihls

from __future__ import annotations

import random
from collections import defaultdict, deque
from typing import Deque, Dict, Tuple

from .simulation import ScenarioProfile


class EmpiricalLatencyModel:
    # Rolling empirical latency sampler keyed by (family, order_style, scenario).

    def __init__(self, max_samples: int = 4000, seed: int = 0) -> None:
        self._samples: Dict[Tuple[str, str, str], Deque[float]] = defaultdict(
            lambda: deque(maxlen=max_samples)
        )
        self._rng = random.Random(seed)

    def observe(self, family: str, order_style: str, scenario: str, latency_ms: float) -> None:
        if latency_ms <= 0:
            return
        key = (family or "Other", order_style or "aggressive", scenario or "base")
        self._samples[key].append(float(latency_ms))

    def sample_latency_s(
        self,
        family: str,
        order_style: str,
        profile: ScenarioProfile,
    ) -> float:
        key = (family or "Other", order_style or "aggressive", profile.name)
        series = self._samples.get(key)
        if series and len(series) >= 25:
            # Bootstrap from observed distribution, then clamp to a sane band.
            ms = self._rng.choice(tuple(series))
            lo = max(1.0, profile.latency_min_ms * 0.5)
            hi = max(lo + 1.0, profile.latency_max_ms * 2.0)
            ms = max(lo, min(hi, ms))
            return ms / 1000.0

        # Fallback to scenario profile range.
        lo = max(0, profile.latency_min_ms)
        hi = max(lo, profile.latency_max_ms)
        if hi <= 0:
            return 0.0
        return self._rng.uniform(lo, hi) / 1000.0

