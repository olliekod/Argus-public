# Created by Oliver Meihls

# Soft context-policy engine for Kalshi bot farm allocation.
#
# Replaces hard family drops with weighted core/challenger/explore sleeves.
# Each context key tracks rolling settlement evidence and computes a weight
# multiplier (0.5 to 1.5) used by the farm dispatcher to scale position sizing.
#
# Thread-safe: all mutable state guarded by threading.Lock.

from __future__ import annotations

import json
import time
import threading
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional

from .config import KalshiConfig

# Context key builder

def momentum_bucket(drift: float, scale: float = 1e-4) -> str:
    # Classify drift into directional regime: up / dn / flat.
    #
    # scale = half of scalp_directional_drift_scale (0.0002 default).
    # At 1e-4/s price moves ~0.6%/min — a meaningful trend signal.
    # Direction-agnostic: the context policy learns which side wins in
    # which regime without any hard-coded directional preference.
    if drift > scale:
        return "up"
    if drift < -scale:
        return "dn"
    return "flat"


def build_context_key(
    family: str,
    side: str,
    edge_bucket: str,
    price_bucket: str,
    strike_distance_bucket: str = "na",
    near_money: bool = False,
    momentum: str = "flat",
) -> str:
    nm = "nm" if near_money else "far"
    return f"{family}|{side}|{edge_bucket}|{price_bucket}|{strike_distance_bucket}|{nm}|{momentum}"


# Policy version for serialization compatibility

_POLICY_VERSION = 1


# ContextPolicyEngine

class ContextPolicyEngine:
    # Rolling-window context weight engine with Core/Challenger/Explore sleeves.

    def __init__(self, cfg: KalshiConfig) -> None:
        self._cfg = cfg
        self._lock = threading.Lock()
        # key -> deque of net_pnl_usd floats
        self._windows: Dict[str, Deque[float]] = {}
        # key -> "core" | "challenger" | "explore"
        self._lanes: Dict[str, str] = {}
        self._maxlen = cfg.context_policy_window_settles
        self._max_keys = int(getattr(cfg, "context_policy_max_keys", 10000))
        self._dropped_new_keys = 0
        # Lane-share controller state (based on recent get_weight calls).
        self._lane_call_window: Deque[str] = deque(
            maxlen=int(getattr(cfg, "context_policy_share_window_calls", 20000))
        )
        self._lane_call_counts: Dict[str, int] = {"core": 0, "challenger": 0, "explore": 0}

    def record_settlement(self, context_key: str, net_pnl_usd: float) -> None:
        with self._lock:
            if context_key not in self._windows:
                if len(self._windows) >= self._max_keys:
                    self._dropped_new_keys += 1
                    return
                self._windows[context_key] = deque(maxlen=self._maxlen)
            self._windows[context_key].append(net_pnl_usd)
            # Re-classify after each settlement
            self._classify_locked(context_key)

    def get_weight(self, context_key: str) -> float:
        if not self._cfg.enable_context_policy:
            return 1.0
        with self._lock:
            weight, lane = self._compute_weight_locked(context_key)
            self._record_lane_call_locked(lane)
            return weight

    def classify_context(self, context_key: str) -> str:
        with self._lock:
            return self._lanes.get(context_key, "unknown")

    def save_policy(self, path: str) -> None:
        with self._lock:
            data = {
                "version": _POLICY_VERSION,
                "timestamp": time.time(),
                "keys": {},
            }
            for key, window in self._windows.items():
                data["keys"][key] = {
                    "settlements": list(window),
                    "lane": self._lanes.get(key, "unknown"),
                    "weight": self._compute_weight_locked(key),
                }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_policy(self, path: str) -> bool:
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return False
        if data.get("version") != _POLICY_VERSION:
            return False
        with self._lock:
            for key, info in data.get("keys", {}).items():
                settlements = info.get("settlements", [])
                window: Deque[float] = deque(maxlen=self._maxlen)
                window.extend(settlements[-self._maxlen:])
                self._windows[key] = window
                lane = info.get("lane", "challenger")
                if lane in ("core", "challenger", "explore"):
                    self._lanes[key] = lane
                else:
                    self._lanes[key] = "challenger"
        return True

    def diagnostics(self) -> Dict[str, Any]:
        with self._lock:
            core_count = sum(1 for v in self._lanes.values() if v == "core")
            challenger_count = sum(1 for v in self._lanes.values() if v == "challenger")
            explore_count = sum(1 for v in self._lanes.values() if v == "explore")
            total_keys = len(self._windows)
            total_settles = sum(len(w) for w in self._windows.values())
            observed_shares = self._observed_lane_shares_locked()
            lane_factors = {
                lane: self._share_control_multiplier_locked(lane, observed_shares)
                for lane in ("core", "challenger", "explore")
            }
            return {
                "enabled": self._cfg.enable_context_policy,
                "total_keys": total_keys,
                "max_keys": self._max_keys,
                "dropped_new_keys": self._dropped_new_keys,
                "core_keys": core_count,
                "challenger_keys": challenger_count,
                "explore_keys": explore_count,
                "unknown_keys": total_keys - core_count - challenger_count - explore_count,
                "total_settlements": total_settles,
                "lane_targets": self._lane_target_shares_locked(),
                "observed_lane_shares": observed_shares,
                "lane_control_multipliers": lane_factors,
            }

    # -- internal helpers (caller holds lock) --

    def _classify_locked(self, key: str) -> None:
        window = self._windows.get(key)
        if not window or len(window) < self._cfg.context_policy_min_samples:
            return  # leave as-is (unknown or previous)
        expectancy = sum(window) / len(window)
        if expectancy >= self._cfg.context_policy_promote_threshold_usd:
            self._lanes[key] = "core"
        elif expectancy <= self._cfg.context_policy_demote_threshold_usd:
            self._lanes[key] = "explore"
        else:
            self._lanes[key] = "challenger"

    def _compute_weight_locked(self, key: str) -> tuple[float, str]:
        window = self._windows.get(key)
        if not window:
            return 1.0, "unknown"

        n = len(window)
        min_samples = self._cfg.context_policy_min_samples
        lane = self._lanes.get(key, "unknown")
        if lane == "unknown" and n >= min_samples:
            self._classify_locked(key)
            lane = self._lanes.get(key, "unknown")

        if lane == "core":
            raw = self._cfg.context_policy_core_weight_max
        elif lane == "challenger":
            raw = self._cfg.context_policy_challenger_weight
        elif lane == "explore":
            raw = self._cfg.context_policy_explore_weight
        else:
            return 1.0, "unknown"

        # Shrinkage toward 1.0 for low-sample contexts
        if n < min_samples:
            shrinkage = self._cfg.context_policy_shrinkage
            frac = n / min_samples
            alpha = frac * (1.0 - shrinkage) + shrinkage * 0.0
            # alpha goes from 0 (n=0) to (1-shrinkage) at n=min_samples
            # With shrinkage=0.7 and n=min_samples: alpha=0.3
            # Actually: blend = alpha * raw + (1-alpha) * 1.0
            alpha = frac * (1.0 - shrinkage)
            weight = alpha * raw + (1.0 - alpha) * 1.0
        else:
            weight = raw
        if self._cfg.context_policy_share_control_enabled:
            weight *= self._share_control_multiplier_locked(
                lane,
                self._observed_lane_shares_locked(),
            )
        return max(0.0, weight), lane

    def _record_lane_call_locked(self, lane: str) -> None:
        if lane not in self._lane_call_counts:
            return
        if len(self._lane_call_window) >= self._lane_call_window.maxlen:
            old = self._lane_call_window[0]
            if old in self._lane_call_counts:
                self._lane_call_counts[old] = max(0, self._lane_call_counts[old] - 1)
        self._lane_call_window.append(lane)
        self._lane_call_counts[lane] += 1

    def _lane_target_shares_locked(self) -> Dict[str, float]:
        raw = {
            "core": max(0.0, float(self._cfg.context_policy_core_target_share)),
            "challenger": max(0.0, float(self._cfg.context_policy_challenger_target_share)),
            "explore": max(0.0, float(self._cfg.context_policy_explore_target_share)),
        }
        total = sum(raw.values())
        if total <= 0:
            return {"core": 1.0, "challenger": 0.0, "explore": 0.0}
        return {k: v / total for k, v in raw.items()}

    def _observed_lane_shares_locked(self) -> Dict[str, float]:
        total = float(sum(self._lane_call_counts.values()))
        if total <= 0.0:
            return {"core": 0.0, "challenger": 0.0, "explore": 0.0}
        return {
            lane: float(self._lane_call_counts.get(lane, 0)) / total
            for lane in ("core", "challenger", "explore")
        }

    def _share_control_multiplier_locked(
        self,
        lane: str,
        observed_shares: Dict[str, float],
    ) -> float:
        if not self._cfg.context_policy_share_control_enabled:
            return 1.0
        targets = self._lane_target_shares_locked()
        target = float(targets.get(lane, 0.0))
        observed = float(observed_shares.get(lane, 0.0))
        gain = float(self._cfg.context_policy_share_control_gain)
        min_mult = float(self._cfg.context_policy_share_control_min_mult)
        max_mult = float(self._cfg.context_policy_share_control_max_mult)
        if target <= 0.0:
            return min_mult
        if observed <= 0.0:
            return max_mult
        ratio = target / observed
        mult = 1.0 + gain * (ratio - 1.0)
        if mult < min_mult:
            mult = min_mult
        if mult > max_mult:
            mult = max_mult
        return mult


# DriftGuard

class DriftGuard:
    # Detect promoted contexts whose expectancy drifts negative and auto-demote.

    def __init__(self, cfg: KalshiConfig) -> None:
        self._cfg = cfg
        self._lock = threading.Lock()
        # key -> deque of net_pnl_usd (comparison window)
        self._windows: Dict[str, Deque[float]] = {}
        # key -> count of consecutive negative windows evaluated
        self._consecutive_neg: Dict[str, int] = {}
        # key -> True if currently demoted by drift
        self._demoted: Dict[str, bool] = {}
        self._maxlen = cfg.drift_guard_window_settles
        self._max_keys = int(getattr(cfg, "drift_guard_max_keys", 10000))
        self._dropped_new_keys = 0
        # diagnostics counters
        self._drift_alerts = 0
        self._auto_demotions = 0

    def record_settlement(self, context_key: str, net_pnl_usd: float) -> None:
        if not self._cfg.enable_drift_guard:
            return
        with self._lock:
            if context_key not in self._windows:
                if len(self._windows) >= self._max_keys:
                    self._dropped_new_keys += 1
                    return
                self._windows[context_key] = deque(maxlen=self._maxlen)
            self._windows[context_key].append(net_pnl_usd)

    def check_drift(self, context_key: str) -> Optional[str]:
        if not self._cfg.enable_drift_guard:
            return None
        with self._lock:
            window = self._windows.get(context_key)
            if not window or len(window) < self._maxlen:
                return None
            expectancy = sum(window) / len(window)
            if expectancy <= self._cfg.drift_guard_negative_threshold_usd:
                self._consecutive_neg[context_key] = (
                    self._consecutive_neg.get(context_key, 0) + 1
                )
                self._drift_alerts += 1
                if (self._consecutive_neg[context_key]
                        >= self._cfg.drift_guard_consecutive_negative):
                    self._demoted[context_key] = True
                    self._auto_demotions += 1
                    return "demote"
            else:
                self._consecutive_neg[context_key] = 0
                self._demoted[context_key] = False
            return None

    def get_demote_multiplier(self, context_key: str) -> float:
        if not self._cfg.enable_drift_guard:
            return 1.0
        with self._lock:
            if self._demoted.get(context_key, False):
                return self._cfg.drift_guard_demote_multiplier
            return 1.0

    def diagnostics(self) -> Dict[str, Any]:
        with self._lock:
            demoted_keys = [k for k, v in self._demoted.items() if v]
            return {
                "enabled": self._cfg.enable_drift_guard,
                "tracked_keys": len(self._windows),
                "max_keys": self._max_keys,
                "dropped_new_keys": self._dropped_new_keys,
                "drift_alerts": self._drift_alerts,
                "auto_demotions": self._auto_demotions,
                "currently_demoted": len(demoted_keys),
                "demoted_keys": demoted_keys[:20],  # cap for readability
            }


# AdaptiveCapEngine

class AdaptiveCapEngine:
    # Tighten caps when concentration + negative expectancy persist on a key.

    def __init__(self, cfg: KalshiConfig) -> None:
        self._cfg = cfg
        self._lock = threading.Lock()
        # key -> deque of (net_pnl_usd, concentration_share) tuples
        self._windows: Dict[str, Deque[tuple]] = {}
        # key -> timestamp when cooldown started (0 = not cooling)
        self._cooldown_start: Dict[str, float] = {}
        self._maxlen = cfg.adaptive_cap_min_samples
        self._max_keys = int(getattr(cfg, "adaptive_cap_max_keys", 10000))
        self._dropped_new_keys = 0
        # diagnostics counters
        self._cap_events = 0
        self._cooldown_events = 0

    def record_settlement(
        self,
        context_key: str,
        net_pnl_usd: float,
        concentration_share: float,
    ) -> None:
        if not self._cfg.enable_adaptive_caps:
            return
        with self._lock:
            if context_key not in self._windows:
                if len(self._windows) >= self._max_keys:
                    self._dropped_new_keys += 1
                    return
                self._windows[context_key] = deque(maxlen=self._maxlen)
            self._windows[context_key].append((net_pnl_usd, concentration_share))
            self._maybe_tighten_locked(context_key)

    def get_cap_multiplier(self, context_key: str, now_ts: float) -> float:
        if not self._cfg.enable_adaptive_caps:
            return 1.0
        with self._lock:
            if self._is_cooled_down_locked(context_key, now_ts):
                return self._cfg.adaptive_cap_tightening_mult
            return 1.0

    def is_cooled_down(self, context_key: str, now_ts: float) -> bool:
        if not self._cfg.enable_adaptive_caps:
            return False
        with self._lock:
            return self._is_cooled_down_locked(context_key, now_ts)

    def diagnostics(self) -> Dict[str, Any]:
        with self._lock:
            now = time.time()
            active_cooldowns = sum(
                1 for k, ts in self._cooldown_start.items()
                if ts > 0 and self._is_cooled_down_locked(k, now)
            )
            return {
                "enabled": self._cfg.enable_adaptive_caps,
                "tracked_keys": len(self._windows),
                "max_keys": self._max_keys,
                "dropped_new_keys": self._dropped_new_keys,
                "adaptive_cap_events": self._cap_events,
                "key_cooldown_events": self._cooldown_events,
                "active_cooldowns": active_cooldowns,
            }

    # -- internal helpers (caller holds lock) --

    def _is_cooled_down_locked(self, key: str, now_ts: float) -> bool:
        start = self._cooldown_start.get(key, 0.0)
        if start <= 0:
            return False
        cooldown_s = self._cfg.adaptive_cap_cooldown_minutes * 60.0
        return (now_ts - start) < cooldown_s

    def _maybe_tighten_locked(self, key: str) -> None:
        window = self._windows.get(key)
        if not window or len(window) < self._cfg.adaptive_cap_min_samples:
            return
        pnls = [entry[0] for entry in window]
        concentrations = [entry[1] for entry in window]
        expectancy = sum(pnls) / len(pnls)
        avg_concentration = sum(concentrations) / len(concentrations)
        if (expectancy <= self._cfg.adaptive_cap_negative_threshold_usd
                and avg_concentration > 0.0):
            self._cap_events += 1
            if self._cooldown_start.get(key, 0.0) <= 0:
                self._cooldown_start[key] = time.time()
                self._cooldown_events += 1
