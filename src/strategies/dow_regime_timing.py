# Created by Oliver Meihls

# Day-of-Week + Regime Timing Gate Strategy.
#
# Emits FILTER signals that act as deterministic gating context for
# downstream strategy routing (no trades).

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseStrategy
from ..core.events import BarEvent
from ..core.regimes import (
    MarketRegimeEvent,
    SymbolRegimeEvent,
    DQ_GAP_WINDOW,
    DQ_REPAIRED_INPUT,
)
from ..core.sessions import (
    get_equities_rth_window_minutes,
    is_within_last_n_minutes,
)
from ..core.signals import (
    SignalEvent,
    compute_signal_id,
    DIRECTION_NEUTRAL,
    SIGNAL_TYPE_FILTER,
)

logger = logging.getLogger("argus.strategies.dow_regime_timing")


class DowRegimeTimingGateStrategy(BaseStrategy):
    # Deterministic gating strategy based on session, regimes, and DOW.

    @property
    def strategy_id(self) -> str:
        return "DOW_REGIME_TIMING_V1"

    @property
    def default_config(self) -> Dict[str, Any]:
        return {
            "gates": {
                "SELL_PUT_SPREAD": {
                    "symbols": ["IBIT", "BITO"],
                    "market": "EQUITIES",
                    "enable_market_scope": True,
                    "allow_pre_market": False,
                    "allow_post_market": False,
                    "avoid_last_n_minutes_rth": 15,
                },
            },
            "dow_weights": {
                "Mon": 0.9,
                "Tue": 1.0,
                "Wed": 1.0,
                "Thu": 0.95,
                "Fri": 0.85,
            },
            "gate_score_base": 1.0,
            "gate_score_threshold": 0.5,
            "gate_score_penalties": {
                "market_missing": 1.0,
                "symbol_missing": 1.0,
                "not_warm": 0.8,
                "session_closed": 1.0,
                "session_pre": 0.6,
                "session_post": 0.6,
                "rth_last_n": 0.7,
                "vol_spike": 0.6,
                "dq_gap_window": 0.7,
                "dq_repaired_input": 0.7,
            },
            "vol_spike_avoid": True,
            "dq_avoid_flags": ["GAP_WINDOW", "REPAIRED_INPUT"],
        }

    def __init__(self, bus, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(bus, config=config)
        self._last_gate_state: Dict[Tuple[str, str], Tuple[float, bool, str]] = {}
        self._gates_evaluated = 0
        self._gates_emitted = 0
        self._gates_suppressed = 0

    def evaluate(
        self,
        bar: BarEvent,
        symbol_regime: Optional[SymbolRegimeEvent],
        market_regime: Optional[MarketRegimeEvent],
    ) -> Optional[SignalEvent]:
        return None

    def _on_bar(self, bar: BarEvent) -> None:
        self._bars_processed += 1

        symbol_regime = self._symbol_regimes.get(bar.symbol)
        market = self._get_market_for_symbol(bar.symbol)
        market_regime = self._market_regimes.get(market)

        signals = self._evaluate_gates(bar, symbol_regime, market_regime, market)
        for signal in signals:
            self._emit_signal(signal)

    def _evaluate_gates(
        self,
        bar: BarEvent,
        symbol_regime: Optional[SymbolRegimeEvent],
        market_regime: Optional[MarketRegimeEvent],
        market: str,
    ) -> List[SignalEvent]:
        signals: List[SignalEvent] = []
        gate_config = self._config.get("gates", {})
        for gate_name, gate_cfg in gate_config.items():
            symbols = set(gate_cfg.get("symbols", []))
            if bar.symbol in symbols:
                scope = f"SYMBOL:{bar.symbol}"
                signal = self._build_gate_signal(
                    bar=bar,
                    symbol_regime=symbol_regime,
                    market_regime=market_regime,
                    scope=scope,
                    gate_name=gate_name,
                    gate_cfg=gate_cfg,
                )
                if signal is not None:
                    signals.append(signal)
            if gate_cfg.get("enable_market_scope", False) and market == gate_cfg.get("market"):
                scope = f"MARKET:{market}"
                signal = self._build_gate_signal(
                    bar=bar,
                    symbol_regime=symbol_regime,
                    market_regime=market_regime,
                    scope=scope,
                    gate_name=gate_name,
                    gate_cfg=gate_cfg,
                )
                if signal is not None:
                    signals.append(signal)
        return signals

    def _build_gate_signal(
        self,
        bar: BarEvent,
        symbol_regime: Optional[SymbolRegimeEvent],
        market_regime: Optional[MarketRegimeEvent],
        scope: str,
        gate_name: str,
        gate_cfg: Dict[str, Any],
    ) -> Optional[SignalEvent]:
        self._gates_evaluated += 1
        timestamp_ms = int(bar.timestamp * 1000)

        session = market_regime.session_regime if market_regime else "UNKNOWN"
        vol = symbol_regime.vol_regime if symbol_regime else "UNKNOWN"
        trend = symbol_regime.trend_regime if symbol_regime else "UNKNOWN"
        risk = market_regime.risk_regime if market_regime else "UNKNOWN"

        dq_flags = 0
        if market_regime:
            dq_flags |= market_regime.data_quality_flags
        if symbol_regime and not scope.startswith("MARKET:"):
            dq_flags |= symbol_regime.data_quality_flags

        reasons: List[str] = []
        hard_avoid = False

        is_market_scope = scope.startswith("MARKET:")

        if market_regime is None:
            reasons.append("MARKET_MISSING")
            hard_avoid = True
        if not is_market_scope:
            if symbol_regime is None:
                reasons.append("SYMBOL_MISSING")
                hard_avoid = True
            elif not symbol_regime.is_warm:
                reasons.append("NOT_WARM")
                hard_avoid = True

        allow_pre = gate_cfg.get("allow_pre_market", False)
        allow_post = gate_cfg.get("allow_post_market", False)

        if market_regime:
            if session == "CLOSED":
                reasons.append("SESSION_CLOSED")
                hard_avoid = True
            if session == "PRE" and not allow_pre:
                reasons.append("SESSION_PRE")
                hard_avoid = True
            if session == "POST" and not allow_post:
                reasons.append("SESSION_POST")
                hard_avoid = True

            last_n = int(gate_cfg.get("avoid_last_n_minutes_rth", 0))
            if session == "RTH" and last_n > 0:
                rth_window = get_equities_rth_window_minutes()
                if is_within_last_n_minutes(timestamp_ms, rth_window, last_n):
                    reasons.append("RTH_LAST_N")
                    hard_avoid = True

        if symbol_regime and not is_market_scope and self._config.get("vol_spike_avoid", True):
            if vol == "VOL_SPIKE":
                reasons.append("VOL_SPIKE")
                hard_avoid = True

        dq_avoid = set(self._config.get("dq_avoid_flags", []))
        if dq_flags and not is_market_scope:
            if (dq_flags & DQ_GAP_WINDOW) and "GAP_WINDOW" in dq_avoid:
                reasons.append("DQ_GAP_WINDOW")
                hard_avoid = True
            if (dq_flags & DQ_REPAIRED_INPUT) and "REPAIRED_INPUT" in dq_avoid:
                reasons.append("DQ_REPAIRED_INPUT")
                hard_avoid = True

        dow_weight = self._get_dow_weight(timestamp_ms)
        score = float(self._config.get("gate_score_base", 1.0)) * dow_weight
        penalties = self._config.get("gate_score_penalties", {})
        for reason in reasons:
            score -= float(penalties.get(reason.lower(), 0.0))
        score = max(0.0, min(1.0, score))

        threshold = float(self._config.get("gate_score_threshold", 0.5))
        gate_allow = not hard_avoid and score >= threshold
        if not gate_allow and "ALLOW" not in reasons:
            if score < threshold and not hard_avoid:
                reasons.append("SCORE_BELOW_THRESHOLD")

        if not reasons:
            reasons.append("ALLOW")

        explain = "|".join(reasons)
        features_snapshot = {
            "gate_name": gate_name,
            "gate_scope": scope,
            "gate_allow": 1.0 if gate_allow else 0.0,
            "gate_score": score,
        }
        regime_snapshot = {
            "session": session,
            "vol": vol,
            "trend": trend,
            "risk": risk,
        }

        state_key = (gate_name, scope)
        state_val = (score, gate_allow, explain)
        last_state = self._last_gate_state.get(state_key)
        if last_state == state_val:
            self._gates_suppressed += 1
            return None

        self._last_gate_state[state_key] = state_val
        self._gates_emitted += 1

        return SignalEvent(
            timestamp_ms=timestamp_ms,
            strategy_id=self.strategy_id,
            config_hash=self._config_hash,
            symbol=scope,
            direction=DIRECTION_NEUTRAL,
            signal_type=SIGNAL_TYPE_FILTER,
            timeframe=bar.bar_duration,
            confidence=1.0,
            data_quality_flags=dq_flags,
            regime_snapshot=regime_snapshot,
            features_snapshot=features_snapshot,
            explain=explain,
            idempotency_key=compute_signal_id(
                self.strategy_id, self._config_hash, scope, timestamp_ms
            ),
        )

    def _get_dow_weight(self, timestamp_ms: int) -> float:
        dt = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)
        dow_key = dt.strftime("%a")
        return float(self._config.get("dow_weights", {}).get(dow_key, 1.0))

    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        status.update(
            {
                "gates_evaluated": self._gates_evaluated,
                "gates_emitted": self._gates_emitted,
                "gates_suppressed": self._gates_suppressed,
                "last_gate_states": len(self._last_gate_state),
            }
        )
        return status


# Replay-compatible adapter

_DOW_REPLAY_DEFAULT_CONFIG: Dict[str, Any] = {
    "gates": {
        "SELL_PUT_SPREAD": {
            "symbols": ["IBIT", "BITO"],
            "allow_pre_market": False,
            "allow_post_market": False,
        },
    },
    "dow_weights": {"Mon": 0.9, "Tue": 1.0, "Wed": 1.0, "Thu": 0.95, "Fri": 0.85},
    "gate_score_base": 1.0,
    "gate_score_threshold": 0.5,
    "gate_score_penalties": {
        "session_closed": 1.0,
        "session_pre": 0.6,
        "session_post": 0.6,
        "vol_spike": 0.6,
        "score_below_threshold": 0.0,
    },
    "vol_spike_avoid": True,
}


class DowRegimeTimingGateReplayStrategy:
    # Replay-compatible adapter for DowRegimeTimingGateStrategy.
    #
    # This is a FILTER-only strategy — it evaluates timing gates but never
    # generates trade intents.  Research loop and ExperimentRunner can use
    # this to assess gate behavior over historical data.

    def __init__(self, params: Dict[str, Any]) -> None:
        self._params = params
        self._config: Dict[str, Any] = {**_DOW_REPLAY_DEFAULT_CONFIG, **params}
        self._gates_evaluated = 0
        self._gates_allowed = 0
        self._last_gate_results: Dict[str, Any] = {}

    @property
    def strategy_id(self) -> str:
        return "DOW_REGIME_TIMING_REPLAY_V1"

    def on_bar(
        self,
        bar,
        sim_ts_ms: int,
        session_regime: str,
        visible_outcomes,
        *,
        visible_regimes=None,
        visible_snapshots=None,
    ) -> None:
        # Evaluate gates based on available regime data.
        self._gates_evaluated += 1

        dow_weights = self._config.get("dow_weights", {})
        dt = datetime.fromtimestamp(sim_ts_ms / 1000.0, tz=timezone.utc)
        dow_key = dt.strftime("%a")
        dow_weight = float(dow_weights.get(dow_key, 1.0))

        gate_allow = True
        reasons: List[str] = []

        if session_regime == "CLOSED":
            gate_allow = False
            reasons.append("SESSION_CLOSED")
        elif session_regime == "PRE" and not self._config.get("gates", {}).get(
            "SELL_PUT_SPREAD", {}
        ).get("allow_pre_market", False):
            gate_allow = False
            reasons.append("SESSION_PRE")
        elif session_regime == "POST" and not self._config.get("gates", {}).get(
            "SELL_PUT_SPREAD", {}
        ).get("allow_post_market", False):
            gate_allow = False
            reasons.append("SESSION_POST")

        # Check vol regime from visible_regimes
        if visible_regimes:
            symbol = getattr(bar, "symbol", "")
            sym_regime = visible_regimes.get(symbol, {})
            if (
                sym_regime.get("vol_regime") == "VOL_SPIKE"
                and self._config.get("vol_spike_avoid", True)
            ):
                gate_allow = False
                reasons.append("VOL_SPIKE")

        score = self._config.get("gate_score_base", 1.0) * dow_weight
        penalties = self._config.get("gate_score_penalties", {})
        for reason in reasons:
            score -= float(penalties.get(reason.lower(), 0.0))
        score = max(0.0, min(1.0, score))

        threshold = float(self._config.get("gate_score_threshold", 0.5))
        if gate_allow and score < threshold:
            gate_allow = False
            reasons.append("SCORE_BELOW_THRESHOLD")

        if gate_allow:
            self._gates_allowed += 1

        self._last_gate_results = {
            "gate_allow": gate_allow,
            "score": score,
            "dow_weight": dow_weight,
            "reasons": reasons,
            "session": session_regime,
        }

    def generate_intents(self, sim_ts_ms: int) -> List:
        # Filter strategy — never generates trade intents.
        return []

    def on_fill(self, intent, fill) -> None:
        pass

    def on_reject(self, intent, fill) -> None:
        pass

    def finalize(self) -> Dict[str, Any]:
        return {
            "gates_evaluated": self._gates_evaluated,
            "gates_allowed": self._gates_allowed,
            "last_gate_results": self._last_gate_results,
        }
