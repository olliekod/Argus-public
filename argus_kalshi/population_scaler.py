# Created by Oliver Meihls

# Staged bot population scaling with gate checks for the Kalshi farm.

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import KalshiConfig


@dataclass(frozen=True, slots=True)
class ScaleGateResult:
    passed: bool
    concentration_ok: bool
    edge_retention_ok: bool
    expectancy_ok: bool
    drawdown_ok: bool
    crowding_ok: bool
    runtime_ok: bool
    details: Dict[str, Any]


@dataclass(frozen=True, slots=True)
class ScaleEvent:
    action: str  # "scale_up", "hold", "scale_down"
    from_stage: int
    to_stage: int
    from_multiplier: float
    to_multiplier: float
    reason: str
    timestamp: float


_SCALE_DOWN_THRESHOLD = 3  # consecutive failures before scale-down


class PopulationScaler:

    def __init__(self, cfg: KalshiConfig) -> None:
        self._cfg = cfg
        self._lock = threading.Lock()

        schedule: List[float] = list(
            getattr(cfg, "bot_population_scale_schedule", [1.0])
        )
        self._schedule: List[float] = schedule if schedule else [1.0]

        self._current_stage: int = 0
        self._consecutive_passes: int = 0
        self._consecutive_fails: int = 0
        self._last_scale_ts: float = 0.0
        self._events: deque[ScaleEvent] = deque(maxlen=50)

    # -- properties ----------------------------------------------------------

    @property
    def current_stage(self) -> int:
        with self._lock:
            return self._current_stage

    @property
    def current_multiplier(self) -> float:
        with self._lock:
            return self._schedule[self._current_stage]

    def effective_bot_count(self, base_count: int) -> int:
        return max(1, int(base_count * self.current_multiplier))

    # -- gate evaluation ------------------------------------------------------

    def evaluate_gate(
        self,
        *,
        concentration_share: float,
        edge_retention: float,
        expectancy_usd: float,
        drawdown_pct: float,
        crowding_stable: bool,
        dispatch_p95_ms: float,
        now_ts: float,
    ) -> ScaleGateResult:
        cfg = self._cfg

        max_concentration = getattr(cfg, "bot_population_scale_max_concentration", 0.25)
        min_edge = getattr(cfg, "bot_population_scale_min_edge_retention", 0.70)
        min_expectancy = getattr(cfg, "bot_population_scale_min_expectancy", 0.0)
        max_drawdown = getattr(cfg, "bot_population_scale_max_drawdown", 0.05)
        min_window_h = getattr(cfg, "bot_population_scale_min_window_hours", 1.0)

        concentration_ok = concentration_share <= max_concentration
        edge_retention_ok = edge_retention >= min_edge
        expectancy_ok = expectancy_usd >= min_expectancy
        drawdown_ok = drawdown_pct <= max_drawdown
        crowding_ok = crowding_stable
        runtime_ok = (now_ts - self._last_scale_ts) >= min_window_h * 3600 if self._last_scale_ts > 0 else True

        passed = all([
            concentration_ok,
            edge_retention_ok,
            expectancy_ok,
            drawdown_ok,
            crowding_ok,
            runtime_ok,
        ])

        return ScaleGateResult(
            passed=passed,
            concentration_ok=concentration_ok,
            edge_retention_ok=edge_retention_ok,
            expectancy_ok=expectancy_ok,
            drawdown_ok=drawdown_ok,
            crowding_ok=crowding_ok,
            runtime_ok=runtime_ok,
            details={
                "concentration_share": concentration_share,
                "edge_retention": edge_retention,
                "expectancy_usd": expectancy_usd,
                "drawdown_pct": drawdown_pct,
                "crowding_stable": crowding_stable,
                "dispatch_p95_ms": dispatch_p95_ms,
                "max_concentration": max_concentration,
                "min_edge": min_edge,
                "min_expectancy": min_expectancy,
                "max_drawdown": max_drawdown,
            },
        )

    # -- scaling logic --------------------------------------------------------

    def attempt_scale(
        self, gate_result: ScaleGateResult, now_ts: float
    ) -> Optional[ScaleEvent]:
        cfg = self._cfg
        if not getattr(cfg, "bot_population_scale_enabled", False):
            return None

        require_passes = int(
            getattr(cfg, "bot_population_scale_require_passes", 3)
        )
        cooldown_h = float(
            getattr(cfg, "bot_population_scale_cooldown_hours", 1.0)
        )
        max_step = float(
            getattr(cfg, "bot_population_scale_max_step", 0.20)
        )

        with self._lock:
            if gate_result.passed:
                self._consecutive_passes += 1
                self._consecutive_fails = 0
            else:
                self._consecutive_passes = 0
                self._consecutive_fails += 1

            # -- scale-down on persistent degradation -------------------------
            if (
                self._consecutive_fails >= _SCALE_DOWN_THRESHOLD
                and self._current_stage > 0
            ):
                prev_stage = self._current_stage
                self._current_stage -= 1
                self._consecutive_fails = 0
                self._last_scale_ts = now_ts
                evt = ScaleEvent(
                    action="scale_down",
                    from_stage=prev_stage,
                    to_stage=self._current_stage,
                    from_multiplier=self._schedule[prev_stage],
                    to_multiplier=self._schedule[self._current_stage],
                    reason=f"{_SCALE_DOWN_THRESHOLD} consecutive gate failures",
                    timestamp=now_ts,
                )
                self._events.append(evt)
                return evt

            # -- cooldown guard -----------------------------------------------
            if self._last_scale_ts > 0 and (now_ts - self._last_scale_ts) < cooldown_h * 3600:
                return None

            # -- scale-up attempt ---------------------------------------------
            if not gate_result.passed:
                return None

            if self._consecutive_passes < require_passes:
                return None

            if self._current_stage >= len(self._schedule) - 1:
                return None  # already at max

            next_stage = self._current_stage + 1
            cur_mult = self._schedule[self._current_stage]
            next_mult = self._schedule[next_stage]

            if (next_mult - cur_mult) > max_step:
                return None  # step too large

            prev_stage = self._current_stage
            self._current_stage = next_stage
            self._consecutive_passes = 0
            self._last_scale_ts = now_ts
            evt = ScaleEvent(
                action="scale_up",
                from_stage=prev_stage,
                to_stage=self._current_stage,
                from_multiplier=cur_mult,
                to_multiplier=next_mult,
                reason=f"{require_passes} consecutive passes",
                timestamp=now_ts,
            )
            self._events.append(evt)
            return evt

    # -- manual override ------------------------------------------------------

    def force_stage(self, stage: int) -> None:
        with self._lock:
            if stage < 0 or stage >= len(self._schedule):
                raise ValueError(
                    f"stage {stage} out of range [0, {len(self._schedule) - 1}]"
                )
            self._current_stage = stage
            self._consecutive_passes = 0
            self._consecutive_fails = 0

    # -- diagnostics ----------------------------------------------------------

    def diagnostics(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "enabled": getattr(self._cfg, "bot_population_scale_enabled", False),
                "current_stage": self._current_stage,
                "current_multiplier": self._schedule[self._current_stage],
                "schedule": list(self._schedule),
                "consecutive_passes": self._consecutive_passes,
                "consecutive_fails": self._consecutive_fails,
                "last_scale_ts": self._last_scale_ts,
                "recent_events": [
                    {
                        "action": e.action,
                        "from_stage": e.from_stage,
                        "to_stage": e.to_stage,
                        "from_multiplier": e.from_multiplier,
                        "to_multiplier": e.to_multiplier,
                        "reason": e.reason,
                        "timestamp": e.timestamp,
                    }
                    for e in self._events
                ],
            }
