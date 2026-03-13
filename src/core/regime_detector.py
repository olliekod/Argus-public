"""
Argus Regime Detector (Phase 2)
===============================

Deterministic regime classification from BarEvents.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

from .bus import EventBus
from .events import (
    BarEvent,
    ComponentHeartbeatEvent,
    ExternalMetricEvent,
    QuoteEvent,
    TOPIC_EXTERNAL_METRICS,
    TOPIC_MARKET_BARS,
    TOPIC_MARKET_QUOTES,
    TOPIC_REGIMES_SYMBOL,
    TOPIC_REGIMES_MARKET,
    TOPIC_SYSTEM_COMPONENT_HEARTBEAT,
)
from .indicators import ATRState, EMAState, RSIState, RollingVolState
from .regimes import (
    SymbolRegimeEvent,
    MarketRegimeEvent,
    DQ_NONE,
    DQ_REPAIRED_INPUT,
    DQ_GAP_WINDOW,
    DQ_STALE_INPUT,
    DEFAULT_REGIME_THRESHOLDS,
    compute_config_hash,
    get_market_for_symbol,
)
from .sessions import get_session_regime

logger = logging.getLogger("argus.regime_detector")

ATR_EPSILON = 1e-8
EMA_FAST_PERIOD = 12
EMA_SLOW_PERIOD = 26
RSI_PERIOD = 14
ATR_PERIOD = 14
VOL_WINDOW = 20
ANNUALIZE_1M = 725.0


@dataclass
class SymbolState:
    symbol: str
    timeframe: int
    ema_fast: EMAState
    ema_slow: EMAState
    rsi: RSIState
    atr: ATRState
    vol: RollingVolState

    prev_ema_fast: Optional[float] = None
    prev_ema_slope: Optional[float] = None
    prev_close: Optional[float] = None

    last_bar_ts_ms: Optional[int] = None
    last_update_ts_ms: Optional[int] = None
    last_gap_ms: int = 0
    gap_flag_remaining: int = 0

    vol_history: Deque[float] = None
    spread_history: Deque[float] = None
    volume_history: Deque[float] = None

    bars_processed: int = 0

    prev_vol_regime: Optional[str] = None
    prev_trend_regime: Optional[str] = None
    bars_since_vol_change: int = 10_000
    bars_since_trend_change: int = 10_000

    def __post_init__(self):
        if self.vol_history is None:
            self.vol_history = deque(maxlen=50)
        if self.spread_history is None:
            self.spread_history = deque(maxlen=50)
        if self.volume_history is None:
            self.volume_history = deque(maxlen=50)


def create_symbol_state(symbol: str, timeframe: int = 60) -> SymbolState:
    return SymbolState(
        symbol=symbol,
        timeframe=timeframe,
        ema_fast=EMAState(EMA_FAST_PERIOD),
        ema_slow=EMAState(EMA_SLOW_PERIOD),
        rsi=RSIState(RSI_PERIOD),
        atr=ATRState(ATR_PERIOD),
        vol=RollingVolState(VOL_WINDOW, ANNUALIZE_1M),
        vol_history=deque(maxlen=50),
        spread_history=deque(maxlen=50),
        volume_history=deque(maxlen=50),
    )


class RegimeDetector:
    def __init__(
        self,
        bus: EventBus,
        db: Optional[Any] = None,
        thresholds: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._bus = bus
        self._db = db
        self._lock = threading.Lock()
        self._start_time = time.time()

        self._thresholds = {**DEFAULT_REGIME_THRESHOLDS, **(thresholds or {})}
        self._config_hash = compute_config_hash(self._thresholds)

        self._symbol_states: Dict[str, SymbolState] = {}
        self._last_market_regimes: Dict[str, MarketRegimeEvent] = {}
        self._quote_history: Dict[str, Deque[Tuple[int, float, float]]] = {}

        self._bars_received = 0
        self._symbol_events_emitted = 0
        self._market_events_emitted = 0
        self._warmup_skips = 0
        self._gaps_detected = 0

        # Risk Regime state (aggregated). IBIT = crypto/risk-on proxy (correlates with BTC).
        self._risk_basket = self._thresholds.get("risk_basket", ["SPY", "IBIT", "TLT", "GLD"])
        self._defensive_symbols = frozenset(
            self._thresholds.get("risk_basket_defensive", ["TLT", "GLD"])
        )
        self._risk_state = "UNKNOWN"
        self._risk_metrics: Dict[str, Any] = {}

        bus.subscribe(TOPIC_MARKET_BARS, self._on_bar)
        bus.subscribe(TOPIC_MARKET_QUOTES, self._on_quote)
        bus.subscribe(TOPIC_EXTERNAL_METRICS, self._on_external_metric)
        logger.info("RegimeDetector initialized — config_hash=%s", self._config_hash)

    def _on_quote(self, event: QuoteEvent) -> None:
        if not self._thresholds.get("quote_liquidity_enabled", False):
            return
        recv_ts_ms = int((event.receive_time or event.timestamp) * 1000)
        history = self._quote_history.setdefault(
            event.symbol, deque(maxlen=int(self._thresholds.get("quote_history_maxlen", 512)))
        )
        history.append((recv_ts_ms, float(event.bid), float(event.ask)))

    async def warmup_from_db(self, n_bars: int = 60) -> Dict[str, int]:
        if self._db is None:
            logger.warning("warmup_from_db called but no db reference")
            return {}
        result = {}
        cursor = await self._db.fetch_all(
            """
            SELECT DISTINCT symbol, source, bar_duration
            FROM market_bars
            ORDER BY timestamp DESC
            LIMIT 100
            """
        )
        seen = set()
        for row in cursor:
            key = (row["symbol"], row["source"], row["bar_duration"])
            if key in seen:
                continue
            seen.add(key)
            bars = await self._db.get_recent_bars(row["source"], row["symbol"], row["bar_duration"], n_bars)
            if not bars:
                continue
            state = self._get_or_create_state(row["symbol"], row["bar_duration"])
            for bar_dict in bars:
                self._update_indicators_from_bar_dict(state, bar_dict)
            result[row["symbol"]] = len(bars)
        return result

    def _update_indicators_from_bar_dict(self, state: SymbolState, bar: Dict) -> None:
        close = bar["close"]
        high = bar["high"]
        low = bar["low"]
        state.ema_fast.update(close)
        state.ema_slow.update(close)
        state.rsi.update(close)
        state.atr.update(high, low, close)
        if state.prev_close is not None and state.prev_close > 0:
            vol = state.vol.update(math.log(close / state.prev_close))
            if vol is not None:
                state.vol_history.append(vol)
        state.prev_close = close
        state.bars_processed += 1

    def _on_bar(self, event: BarEvent) -> None:
        with self._lock:
            self._bars_received += 1
            symbol = event.symbol
            timeframe = event.bar_duration
            ts_ms = int(event.timestamp * 1000)

            state = self._get_or_create_state(symbol, timeframe)

            dq_flags = DQ_NONE
            if event.repaired:
                dq_flags |= DQ_REPAIRED_INPUT
            dq_flags |= self._check_gap(state, ts_ms, timeframe)

            self._update_indicators(state, event, ts_ms)
            self._emit_symbol_regime(state, ts_ms, dq_flags)

            market = get_market_for_symbol(symbol)
            self._emit_market_regime(market, timeframe, ts_ms, dq_flags)

    def _get_or_create_state(self, symbol: str, timeframe: int) -> SymbolState:
        key = f"{symbol}:{timeframe}"
        if key not in self._symbol_states:
            self._symbol_states[key] = create_symbol_state(symbol, timeframe)
        return self._symbol_states[key]

    def _reset_state_windows(self, state: SymbolState) -> None:
        state.ema_fast = EMAState(EMA_FAST_PERIOD)
        state.ema_slow = EMAState(EMA_SLOW_PERIOD)
        state.rsi = RSIState(RSI_PERIOD)
        state.atr = ATRState(ATR_PERIOD)
        state.vol = RollingVolState(VOL_WINDOW, ANNUALIZE_1M)
        state.prev_ema_fast = None
        state.prev_ema_slope = None
        state.prev_close = None
        state.vol_history.clear()
        state.spread_history.clear()
        state.volume_history.clear()
        state.bars_processed = 0
        state.prev_vol_regime = None
        state.prev_trend_regime = None
        state.bars_since_vol_change = 10_000
        state.bars_since_trend_change = 10_000

    def _check_gap(self, state: SymbolState, ts_ms: int, timeframe: int) -> int:
        flags = DQ_NONE
        if state.gap_flag_remaining > 0:
            state.gap_flag_remaining -= 1
            flags |= DQ_GAP_WINDOW

        state.last_gap_ms = 0
        if state.last_bar_ts_ms is not None:
            gap_ms = ts_ms - state.last_bar_ts_ms
            state.last_gap_ms = max(0, gap_ms)
            expected_ts = state.last_bar_ts_ms + (timeframe * 1000)
            tolerance = self._thresholds.get("gap_tolerance_bars", 1) * timeframe * 1000
            if ts_ms > expected_ts + tolerance:
                self._gaps_detected += 1
                flags |= DQ_GAP_WINDOW
                state.gap_flag_remaining = self._thresholds.get("gap_flag_duration_bars", 2)

                reset_thresh = int(self._thresholds.get("gap_reset_window_threshold_ms", 0) or 0)
                if reset_thresh > 0 and gap_ms >= reset_thresh:
                    self._reset_state_windows(state)

                decay_thresh = int(self._thresholds.get("gap_confidence_decay_threshold_ms", 0) or 0)
                decay_bars = int(self._thresholds.get("gap_warmth_decay_bars", 0) or 0)
                if decay_thresh > 0 and decay_bars > 0 and gap_ms >= decay_thresh:
                    state.bars_processed = max(0, state.bars_processed - decay_bars)

        state.last_bar_ts_ms = ts_ms
        state.last_update_ts_ms = ts_ms
        return flags

    def _select_quote_spread(self, symbol: str, asof_ts_ms: int) -> Optional[float]:
        history = self._quote_history.get(symbol)
        if not history:
            return None
        latest = None
        for recv_ts_ms, bid, ask in reversed(history):
            if recv_ts_ms <= asof_ts_ms:
                latest = (bid, ask)
                break
        if latest is None:
            return None
        bid, ask = latest
        if bid <= 0 or ask < bid:
            return None
        mid = (bid + ask) / 2.0
        if mid <= 0:
            return None
        return (ask - bid) / mid

    def _update_indicators(self, state: SymbolState, bar: BarEvent, ts_ms: int) -> None:
        close = bar.close
        high = bar.high
        low = bar.low
        if state.ema_fast._ema is not None:
            state.prev_ema_fast = state.ema_fast._ema

        state.ema_fast.update(close)
        state.ema_slow.update(close)
        state.rsi.update(close)
        state.atr.update(high, low, close)

        if state.prev_close is not None and state.prev_close > 0:
            vol = state.vol.update(math.log(close / state.prev_close))
            if vol is not None:
                state.vol_history.append(vol)

        quote_spread = None
        if self._thresholds.get("quote_liquidity_enabled", False):
            quote_spread = self._select_quote_spread(state.symbol, ts_ms)

        if quote_spread is not None:
            state.spread_history.append(quote_spread)
        else:
            mid = (high + low) / 2.0
            state.spread_history.append((high - low) / mid if mid > 0 else 0.0)

        state.volume_history.append(float(getattr(bar, "volume", 0) or 0))
        state.prev_close = close
        state.bars_processed += 1

    def _emit_symbol_regime(self, state: SymbolState, ts_ms: int, dq_flags: int) -> None:
        warmup_bars = self._thresholds.get("warmup_bars", 30)
        ema_fast = state.ema_fast._ema
        ema_slow = state.ema_slow._ema
        atr = state.atr._atr

        is_warm = (
            state.bars_processed >= warmup_bars
            and ema_fast is not None
            and ema_slow is not None
            and atr is not None
            and atr > ATR_EPSILON
            and state.rsi._avg_gain is not None
            and len(state.vol_history) >= 5
        )
        if not is_warm:
            self._warmup_skips += 1
            dq_flags |= DQ_STALE_INPUT

        atr = atr if atr and atr > ATR_EPSILON else 1.0
        ema_fast = ema_fast if ema_fast is not None else (state.prev_close or 0.0)
        ema_slow = ema_slow if ema_slow is not None else (state.prev_close or 0.0)
        close = state.prev_close or 1.0

        rsi = 50.0
        if state.rsi._avg_gain is not None and state.rsi._avg_loss is not None:
            if state.rsi._avg_loss == 0:
                rsi = 100.0
            else:
                rs = state.rsi._avg_gain / state.rsi._avg_loss
                rsi = 100.0 - (100.0 / (1.0 + rs))

        atr_pct = atr / close if close > 0 else 0.0
        ema_slope = 0.0
        if state.prev_ema_fast is not None and atr > ATR_EPSILON:
            ema_slope = (ema_fast - state.prev_ema_fast) / atr
        trend_accel = 0.0 if state.prev_ema_slope is None else (ema_slope - state.prev_ema_slope)
        state.prev_ema_slope = ema_slope

        trend_strength = abs(ema_fast - ema_slow) / atr if atr > ATR_EPSILON else 0.0
        if self._thresholds.get("trend_accel_classification_enabled", False):
            trend_strength += abs(trend_accel)

        vol_z = 0.0
        if len(state.vol_history) >= 5:
            vol_list = list(state.vol_history)
            current_vol = vol_list[-1]
            mean_vol = sum(vol_list) / len(vol_list)
            if len(vol_list) > 1:
                var = sum((v - mean_vol) ** 2 for v in vol_list) / (len(vol_list) - 1)
                std_vol = math.sqrt(var) if var > 0 else 1.0
                vol_z = (current_vol - mean_vol) / std_vol if std_vol > 0 else 0.0

        vol_regime, vol_trigger = self._classify_vol_regime(vol_z, state)
        trend_regime, trend_trigger = self._classify_trend_regime(ema_slope, trend_strength, is_warm, state)

        current_spread = state.spread_history[-1] if state.spread_history else 0.0
        current_volume = state.volume_history[-1] if state.volume_history else 0.0
        volume_pctile = 50.0
        if len(state.volume_history) >= 5 and current_volume > 0:
            vol_list = sorted(state.volume_history)
            rank = sum(1 for v in vol_list if v <= current_volume)
            volume_pctile = (rank / len(vol_list)) * 100.0

        liq_regime, spread_pct_val, vol_pctile_val = self._classify_liquidity_regime(current_spread, volume_pctile)

        confidence = 1.0 if is_warm else 0.5
        if dq_flags & DQ_REPAIRED_INPUT:
            confidence *= 0.9
        if dq_flags & DQ_GAP_WINDOW:
            confidence *= 0.8
        gap_decay_thresh = int(self._thresholds.get("gap_confidence_decay_threshold_ms", 0) or 0)
        if gap_decay_thresh > 0 and state.last_gap_ms >= gap_decay_thresh:
            confidence *= float(self._thresholds.get("gap_confidence_decay_multiplier", 1.0))

        event = SymbolRegimeEvent(
            symbol=state.symbol,
            timeframe=state.timeframe,
            timestamp_ms=ts_ms,
            vol_regime=vol_regime,
            trend_regime=trend_regime,
            liquidity_regime=liq_regime,
            atr=atr,
            atr_pct=atr_pct,
            vol_z=vol_z,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            ema_slope=ema_slope,
            rsi=rsi,
            spread_pct=current_spread,
            volume_pctile=volume_pctile,
            trend_accel=trend_accel,
            confidence=confidence,
            is_warm=is_warm,
            data_quality_flags=dq_flags,
            config_hash=self._config_hash,
        )
        self._update_risk_state(state.symbol, event)
        self._bus.publish(TOPIC_REGIMES_SYMBOL, event)
        self._symbol_events_emitted += 1

        if vol_trigger:
            logger.info(
                "regime_transition symbol=%s kind=vol old=%s new=%s vol_z=%.6f trigger=%s",
                state.symbol,
                state.prev_vol_regime,
                vol_regime,
                vol_z,
                vol_trigger,
            )
        if trend_trigger:
            logger.info(
                "regime_transition symbol=%s kind=trend old=%s new=%s ema_slope=%.6f trend_strength=%.6f trigger=%s",
                state.symbol,
                state.prev_trend_regime,
                trend_regime,
                ema_slope,
                trend_strength,
                trend_trigger,
            )

        state.bars_since_vol_change = state.bars_since_vol_change + 1
        state.bars_since_trend_change = state.bars_since_trend_change + 1
        if state.prev_vol_regime != vol_regime:
            state.bars_since_vol_change = 0
        if state.prev_trend_regime != trend_regime:
            state.bars_since_trend_change = 0
        state.prev_vol_regime = vol_regime
        state.prev_trend_regime = trend_regime

    def _classify_vol_regime(self, vol_z: float, state: SymbolState) -> Tuple[str, Optional[str]]:
        spike_z = self._thresholds.get("vol_spike_z", 2.5)
        high_z = self._thresholds.get("vol_high_z", 1.0)
        low_z = self._thresholds.get("vol_low_z", -0.5)

        def base(v: float) -> str:
            if v > spike_z:
                return "VOL_SPIKE"
            if v > high_z:
                return "VOL_HIGH"
            if v < low_z:
                return "VOL_LOW"
            return "VOL_NORMAL"

        proposed = base(vol_z)
        prev = state.prev_vol_regime
        trigger = None

        if self._thresholds.get("vol_hysteresis_enabled", False) and prev:
            band = float(self._thresholds.get("vol_hysteresis_band", 0.0) or 0.0)
            _HYSTERESIS_EPS = 1e-6
            spike_exit = spike_z - band
            high_exit = high_z - band
            low_exit = low_z + band

            if prev == "VOL_SPIKE" and vol_z >= spike_exit - _HYSTERESIS_EPS:
                proposed = "VOL_SPIKE"
            elif prev == "VOL_HIGH" and high_exit - _HYSTERESIS_EPS <= vol_z < spike_z:
                proposed = "VOL_HIGH"
            elif prev == "VOL_LOW" and vol_z <= low_exit + _HYSTERESIS_EPS:
                proposed = "VOL_LOW"

            if proposed != prev:
                trigger = f"hysteresis_cross(vol_z={vol_z:.4f})"

        min_dwell = int(self._thresholds.get("min_dwell_bars", 0) or 0)
        if prev and proposed != prev and state.bars_since_vol_change < min_dwell:
            return prev, None
        return proposed, trigger if proposed != prev else None

    def _classify_trend_regime(
        self,
        ema_slope: float,
        trend_strength: float,
        is_warm: bool,
        state: SymbolState,
    ) -> Tuple[str, Optional[str]]:
        if not is_warm:
            return "RANGE", None

        slope_thresh = self._thresholds.get("trend_slope_threshold", 0.5)
        strength_thresh = self._thresholds.get("trend_strength_threshold", 1.0)
        prev = state.prev_trend_regime

        def base(slope: float, strength: float) -> str:
            if strength > strength_thresh:
                if slope > slope_thresh:
                    return "TREND_UP"
                if slope < -slope_thresh:
                    return "TREND_DOWN"
            return "RANGE"

        proposed = base(ema_slope, trend_strength)
        trigger = None

        if self._thresholds.get("trend_hysteresis_enabled", False) and prev:
            slope_band = float(self._thresholds.get("trend_hysteresis_slope_band", 0.0) or 0.0)
            strength_band = float(self._thresholds.get("trend_hysteresis_strength_band", 0.0) or 0.0)
            slope_exit = max(0.0, slope_thresh - slope_band)
            strength_exit = max(0.0, strength_thresh - strength_band)

            if prev == "TREND_UP" and ema_slope >= slope_exit and trend_strength >= strength_exit:
                proposed = "TREND_UP"
            elif prev == "TREND_DOWN" and ema_slope <= -slope_exit and trend_strength >= strength_exit:
                proposed = "TREND_DOWN"

            if proposed != prev:
                trigger = (
                    f"hysteresis_cross(ema_slope={ema_slope:.4f},trend_strength={trend_strength:.4f})"
                )

        min_dwell = int(self._thresholds.get("min_dwell_bars", 0) or 0)
        if prev and proposed != prev and state.bars_since_trend_change < min_dwell:
            return prev, None
        return proposed, trigger if proposed != prev else None

    def _classify_liquidity_regime(self, spread_pct: float, volume_pctile: float) -> Tuple[str, float, float]:
        dried_pct = self._thresholds.get("liq_spread_dried_pct", 0.50)
        low_pct = self._thresholds.get("liq_spread_low_pct", 0.20)
        high_pct = self._thresholds.get("liq_spread_high_pct", 0.05)

        if spread_pct > dried_pct:
            return "LIQ_DRIED", spread_pct, volume_pctile
        if spread_pct > low_pct or volume_pctile < self._thresholds.get("liq_volume_low_pctile", 25):
            return "LIQ_LOW", spread_pct, volume_pctile
        if spread_pct < high_pct and volume_pctile > self._thresholds.get("liq_volume_high_pctile", 75):
            return "LIQ_HIGH", spread_pct, volume_pctile
        return "LIQ_NORMAL", spread_pct, volume_pctile

    def _update_risk_state(self, symbol: str, event: SymbolRegimeEvent) -> None:
        """Update global risk state from risk basket (SPY, IBIT, TLT, GLD).

        Risk-on proxies (SPY, IBIT): trend up + calm vol → +1; trend down or vol spike → -1.
        Defensive (TLT, GLD): trend up (flight to safety) → -1; trend down → +1.
        Aggregate votes to set RISK_ON / RISK_OFF / NEUTRAL. IBIT ties regime to crypto/equity correlation.
        """
        if symbol not in self._risk_basket:
            return

        # Update symbol-specific risk component
        self._risk_metrics[symbol] = {
            "vol_regime": event.vol_regime,
            "trend_regime": event.trend_regime,
            "ema_slope": event.ema_slope,
            "confidence": event.confidence,
            "timestamp_ms": event.timestamp_ms,
        }

        votes: List[int] = []
        min_conf = float(self._thresholds.get("risk_basket_min_confidence", 0.5))

        for sym in self._risk_basket:
            data = self._risk_metrics.get(sym)
            if not data or data.get("confidence", 0) < min_conf:
                continue
            vol = data.get("vol_regime", "UNKNOWN")
            trend = data.get("trend_regime", "UNKNOWN")
            vol_ok = vol in ("VOL_LOW", "VOL_NORMAL")

            if sym in self._defensive_symbols:
                # TLT/GLD: up = flight to safety = risk-off vote
                if trend == "TREND_UP":
                    votes.append(-1)
                elif trend == "TREND_DOWN":
                    votes.append(1)
                else:
                    votes.append(0)
            else:
                # SPY, IBIT, etc.: risk-on proxy (same as legacy SPY logic)
                if vol == "VOL_SPIKE" or trend == "TREND_DOWN":
                    votes.append(-1)
                elif trend == "TREND_UP" and vol_ok:
                    votes.append(1)
                else:
                    votes.append(0)

        if not votes:
            self._risk_state = "UNKNOWN"
            return
        total = sum(votes)
        if total > 0:
            self._risk_state = "RISK_ON"
        elif total < 0:
            self._risk_state = "RISK_OFF"
        else:
            self._risk_state = "NEUTRAL"

    def _emit_market_regime(self, market: str, timeframe: int, ts_ms: int, dq_flags: int) -> None:
        session = self._get_session_regime(market, ts_ms)
        
        from .regimes import canonical_metrics_json
        m_json = canonical_metrics_json(self._risk_metrics)

        event = MarketRegimeEvent(
            market=market,
            timeframe=timeframe,
            timestamp_ms=ts_ms,
            session_regime=session,
            risk_regime=self._risk_state,
            confidence=1.0,
            data_quality_flags=dq_flags,
            config_hash=self._config_hash,
            metrics_json=m_json,
        )
        self._bus.publish(TOPIC_REGIMES_MARKET, event)
        self._market_events_emitted += 1
        self._last_market_regimes[market] = event

    def set_external_metric(self, key: str, value: Any) -> None:
        """Inject an external metric into the risk metrics dict.

        This is the integration point for features that are computed
        outside the per-bar indicator pipeline (e.g. GlobalRiskFlow
        from daily ETF bars).  The value will appear in ``metrics_json``
        on subsequent market regime events.

        Args:
            key: Metric name (e.g. ``"global_risk_flow"``).
            value: Metric value (must be JSON-serialisable).
        """
        with self._lock:
            self._risk_metrics[key] = value

    def _on_external_metric(self, event: ExternalMetricEvent) -> None:
        """Handle an ExternalMetricEvent from the event bus."""
        v = getattr(event, "v", 1)
        if v > 1:
            logger.warning(
                "Received ExternalMetricEvent with unknown schema version v=%d. "
                "Ignoring to prevent unexpected behavior.", v
            )
            return

        self.set_external_metric(event.key, event.value)
        logger.debug(
            "external_metric key=%s value=%s ts_ms=%d",
            event.key, event.value, event.timestamp_ms,
        )

    def _get_session_regime(self, market: str, ts_ms: int) -> str:
        return get_session_regime(market, ts_ms)

    def get_symbol_regime(self, symbol: str, timeframe: int = 60) -> Optional[str]:
        key = f"{symbol}:{timeframe}"
        with self._lock:
            state = self._symbol_states.get(key)
            if state is None:
                return None
            vol_z = 0.0
            if len(state.vol_history) >= 5:
                vol_list = list(state.vol_history)
                current_vol = vol_list[-1]
                mean_vol = sum(vol_list) / len(vol_list)
                var = sum((v - mean_vol) ** 2 for v in vol_list) / max(1, len(vol_list) - 1)
                std_vol = math.sqrt(var) if var > 0 else 1.0
                vol_z = (current_vol - mean_vol) / std_vol if std_vol > 0 else 0.0
            regime, _ = self._classify_vol_regime(vol_z, state)
            return regime

    def get_all_regimes(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {
                key: {
                    "bars_processed": state.bars_processed,
                    "is_warm": state.bars_processed >= self._thresholds.get("warmup_bars", 30),
                    "last_ts_ms": state.last_bar_ts_ms,
                }
                for key, state in self._symbol_states.items()
            }

    def get_market_regime(self, market: str) -> Optional[MarketRegimeEvent]:
        with self._lock:
            return self._last_market_regimes.get(market)

    def emit_heartbeat(self) -> ComponentHeartbeatEvent:
        now = time.time()
        with self._lock:
            bars = self._bars_received
            symbol_events = self._symbol_events_emitted
            market_events = self._market_events_emitted
            warmup_skips = self._warmup_skips
            gaps = self._gaps_detected

        uptime = now - self._start_time
        health = "ok" if bars > 0 or uptime < 120 else "down"
        hb = ComponentHeartbeatEvent(
            component="regime_detector",
            uptime_seconds=round(uptime, 1),
            events_processed=bars,
            health=health,
            extra={
                "symbol_events_emitted": symbol_events,
                "market_events_emitted": market_events,
                "warmup_skips_total": warmup_skips,
                "gaps_detected_total": gaps,
                "config_hash": self._config_hash,
            },
        )
        self._bus.publish(TOPIC_SYSTEM_COMPONENT_HEARTBEAT, hb)
        return hb

    def get_status(self) -> Dict[str, Any]:
        from .status import build_status

        now = time.time()
        with self._lock:
            bars = self._bars_received
            symbol_events = self._symbol_events_emitted
            market_events = self._market_events_emitted
            warmup_skips = self._warmup_skips
            gaps = self._gaps_detected
            symbols = list(self._symbol_states.keys())

        uptime = now - self._start_time
        status = "ok" if bars > 0 else ("ok" if uptime < 120 else "down")
        return build_status(
            name="regime_detector",
            type="internal",
            status=status,
            request_count=bars,
            extras={
                "bars_received": bars,
                "symbol_events_emitted": symbol_events,
                "market_events_emitted": market_events,
                "warmup_skips_total": warmup_skips,
                "gaps_detected_total": gaps,
                "config_hash": self._config_hash,
                "symbols_tracked": len(symbols),
                "uptime_seconds": round(uptime, 1),
            },
        )
