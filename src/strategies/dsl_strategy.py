"""
DSL Strategy — Manifest-driven strategy execution engine.
==========================================================

Turns a :class:`~src.core.manifests.StrategyManifest` into a live
:class:`BaseStrategy` that evaluates indicator-based logic trees on
each incoming bar.

The DSL supports boolean (AND/OR/NOT), comparison (GT/LT/GE/LE/EQ/NE),
crossover (CROSS_ABOVE/CROSS_BELOW), and regime filter (IN_REGIME/
NOT_IN_REGIME) operators.

**Determinism guarantee**: Given identical bar data and regime state,
the strategy will always produce identical signals.  No wall-clock,
no randomness, no mutable global state.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

from .base import BaseStrategy
from ..core.events import BarEvent
from ..core.indicators import (
    ATRState,
    EMAState,
    MACDState,
    RSIState,
    VWAPState,
    BarTuple,
)
from ..core.manifests import (
    HADES_INDICATOR_CATALOG,
    REGIME_FILTER_CATALOG,
    VALID_LOGIC_OPS,
    ManifestValidationError,
    StrategyManifest,
)
from ..core.regimes import MarketRegimeEvent, SymbolRegimeEvent
from ..core.signals import (
    DIRECTION_LONG,
    DIRECTION_SHORT,
    SIGNAL_TYPE_ENTRY,
    SIGNAL_TYPE_EXIT,
    SignalEvent,
)

logger = logging.getLogger("argus.strategies.dsl_strategy")


# ═══════════════════════════════════════════════════════════════════════════
# Indicator State Registry
# ═══════════════════════════════════════════════════════════════════════════

class _IndicatorBank:
    """Per-symbol collection of incremental indicator state machines.

    Manages warmup and provides a flat ``name -> value`` mapping consumed
    by the DSL evaluator.  All indicators are registered once from the
    manifest's ``signals`` list and then updated on every bar.
    """

    __slots__ = (
        "_emas", "_rsi", "_macd", "_vwap", "_atr",
        "_prev_close", "_log_return_buf", "_close_buf",
        "_values", "_prev_values", "_params",
    )

    def __init__(self, signals: List[str], params: Dict[str, Any]) -> None:
        self._params = params
        self._values: Dict[str, Optional[float]] = {}
        self._prev_values: Dict[str, Optional[float]] = {}

        # EMA family
        self._emas: Dict[str, EMAState] = {}
        # RSI
        self._rsi: Optional[RSIState] = None
        # MACD
        self._macd: Optional[MACDState] = None
        # VWAP
        self._vwap: Optional[VWAPState] = None
        # ATR
        self._atr: Optional[ATRState] = None
        # Rolling vol helpers
        self._prev_close: Optional[float] = None
        self._log_return_buf: Deque[float] = deque(maxlen=int(params.get("rolling_vol_window", 20)))
        # Close buffer for slope / trend_accel
        self._close_buf: Deque[float] = deque(maxlen=60)

        self._init_indicators(signals, params)

    def _init_indicators(self, signals: List[str], params: Dict[str, Any]) -> None:
        for sig in signals:
            if sig == "ema":
                for suffix in self._ema_suffixes(params):
                    period = int(suffix)
                    self._emas[f"ema_{period}"] = EMAState(period)
                    self._values[f"ema_{period}"] = None
                # If no explicit periods, default to 12 and 26
                if not self._emas:
                    self._emas["ema_12"] = EMAState(12)
                    self._emas["ema_26"] = EMAState(26)
                    self._values["ema_12"] = None
                    self._values["ema_26"] = None
            elif sig == "rsi":
                period = int(params.get("rsi_period", 14))
                self._rsi = RSIState(period)
                self._values["rsi"] = None
            elif sig == "macd":
                fast = int(params.get("macd_fast", 12))
                slow = int(params.get("macd_slow", 26))
                signal = int(params.get("macd_signal", 9))
                self._macd = MACDState(fast, slow, signal)
                self._values["macd_line"] = None
                self._values["macd_signal"] = None
                self._values["macd_histogram"] = None
            elif sig == "vwap":
                self._vwap = VWAPState()
                self._values["vwap"] = None
            elif sig == "atr":
                period = int(params.get("atr_period", 14))
                self._atr = ATRState(period)
                self._values["atr"] = None
            elif sig == "rolling_vol":
                self._values["rolling_vol"] = None
            elif sig == "bollinger_bands":
                self._values["bb_upper"] = None
                self._values["bb_middle"] = None
                self._values["bb_lower"] = None
            elif sig == "vol_z":
                self._values["vol_z"] = None
            elif sig == "ema_slope":
                self._values["ema_slope"] = None
            elif sig == "spread_pct":
                self._values["spread_pct"] = None
            elif sig == "volume_pctile":
                self._values["volume_pctile"] = None
            elif sig == "trend_accel":
                self._values["trend_accel"] = None

    @staticmethod
    def _ema_suffixes(params: Dict[str, Any]) -> List[str]:
        """Extract EMA period suffixes from parameters."""
        suffixes: List[str] = []
        for key, val in params.items():
            if key.startswith("ema_") and key[4:].isdigit():
                suffixes.append(key[4:])
            elif key == "ema_period":
                suffixes.append(str(int(val)))
        if not suffixes:
            # Check for ema_fast / ema_slow style
            if "ema_fast" in params:
                suffixes.append(str(int(params["ema_fast"])))
            if "ema_slow" in params:
                suffixes.append(str(int(params["ema_slow"])))
        return suffixes

    def update(self, bar: BarEvent) -> Dict[str, Optional[float]]:
        """Feed one bar into all indicators and return the current snapshot."""
        close = bar.close
        self._close_buf.append(close)

        # Snapshot previous values for crossover detection
        self._prev_values = dict(self._values)

        # EMA
        for name, state in self._emas.items():
            self._values[name] = state.update(close)

        # RSI
        if self._rsi is not None:
            self._values["rsi"] = self._rsi.update(close)

        # MACD
        if self._macd is not None:
            result = self._macd.update(close)
            if result is not None:
                self._values["macd_line"] = result.macd_line
                self._values["macd_signal"] = result.signal_line
                self._values["macd_histogram"] = result.histogram

        # VWAP
        if self._vwap is not None:
            bt = BarTuple(
                timestamp=bar.timestamp,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
            )
            self._values["vwap"] = self._vwap.update(bt)

        # ATR
        if self._atr is not None:
            self._values["atr"] = self._atr.update(bar.high, bar.low, bar.close)

        # Rolling volatility
        if "rolling_vol" in self._values:
            if self._prev_close is not None and close > 0 and self._prev_close > 0:
                lr = math.log(close / self._prev_close)
                self._log_return_buf.append(lr)
                window = self._log_return_buf.maxlen
                if len(self._log_return_buf) >= window:
                    mean = sum(self._log_return_buf) / window
                    var = sum((r - mean) ** 2 for r in self._log_return_buf) / (window - 1)
                    ann_factor = math.sqrt(365.25 * 24 * 60)  # 1-min bars
                    self._values["rolling_vol"] = math.sqrt(var) * ann_factor
            self._prev_close = close

        # EMA slope (rate of change of primary EMA)
        if "ema_slope" in self._values:
            primary_ema = self._pick_primary_ema()
            prev_ema = self._prev_values.get(primary_ema)
            curr_ema = self._values.get(primary_ema)
            if curr_ema is not None and prev_ema is not None and prev_ema != 0:
                self._values["ema_slope"] = (curr_ema - prev_ema) / prev_ema
            else:
                self._values["ema_slope"] = None

        # Bollinger Bands (20-period SMA +/- 2 std)
        if "bb_upper" in self._values:
            bb_period = int(self._params.get("bb_period", 20))
            if len(self._close_buf) >= bb_period:
                recent = list(self._close_buf)[-bb_period:]
                sma = sum(recent) / bb_period
                std = math.sqrt(sum((c - sma) ** 2 for c in recent) / bb_period)
                mult = float(self._params.get("bb_std", 2.0))
                self._values["bb_middle"] = sma
                self._values["bb_upper"] = sma + mult * std
                self._values["bb_lower"] = sma - mult * std

        # Vol Z-score
        if "vol_z" in self._values and self._log_return_buf:
            window = len(self._log_return_buf)
            if window >= 2:
                mean = sum(self._log_return_buf) / window
                std = math.sqrt(
                    sum((r - mean) ** 2 for r in self._log_return_buf) / (window - 1)
                )
                if std > 0:
                    latest_ret = self._log_return_buf[-1]
                    self._values["vol_z"] = (latest_ret - mean) / std

        # Trend acceleration (second derivative of close via close_buf)
        if "trend_accel" in self._values:
            buf = self._close_buf
            if len(buf) >= 3:
                d1 = buf[-1] - buf[-2]
                d2 = buf[-2] - buf[-3]
                self._values["trend_accel"] = d1 - d2

        # Add raw close, high, low, volume to values for comparisons
        self._values["close"] = close
        self._values["high"] = bar.high
        self._values["low"] = bar.low
        self._values["open"] = bar.open
        self._values["volume"] = bar.volume

        return dict(self._values)

    def _pick_primary_ema(self) -> str:
        """Return the key of the first available EMA."""
        for name in sorted(self._emas.keys()):
            return name
        return "ema_12"

    @property
    def prev_values(self) -> Dict[str, Optional[float]]:
        return dict(self._prev_values)

    @property
    def is_warm(self) -> bool:
        """True when all core indicators have produced at least one value."""
        for key, val in self._values.items():
            if key in ("close", "high", "low", "open", "volume"):
                continue
            if val is None:
                return False
        return True


# ═══════════════════════════════════════════════════════════════════════════
# DSL Parser — Recursive logic tree evaluator
# ═══════════════════════════════════════════════════════════════════════════

class DSLEvalError(Exception):
    """Non-fatal evaluation error for a single bar."""


class DSLParser:
    """Evaluates a logic tree node against indicator values and regimes.

    All comparisons are strictly typed (float vs float).  Missing
    indicator data causes the entire tree to evaluate to ``False``
    (conservative default), never ``True``.
    """

    @staticmethod
    def evaluate(
        node: Dict[str, Any],
        values: Dict[str, Optional[float]],
        prev_values: Dict[str, Optional[float]],
        regimes: Dict[str, str],
    ) -> bool:
        """Recursively evaluate a logic node.

        Parameters
        ----------
        node : dict
            A logic node with at minimum an ``op`` key.
        values : dict
            Current indicator name -> float mapping.
        prev_values : dict
            Previous-bar indicator name -> float mapping (for crossovers).
        regimes : dict
            Current regime name -> regime value mapping.

        Returns
        -------
        bool
            True if the condition is met, False otherwise.
            Returns False for any evaluation error (missing data, bad types).
        """
        if not isinstance(node, dict):
            return False

        op = node.get("op")
        if op is None:
            return False

        try:
            if op == "AND":
                return (
                    DSLParser.evaluate(node["left"], values, prev_values, regimes)
                    and DSLParser.evaluate(node["right"], values, prev_values, regimes)
                )
            elif op == "OR":
                return (
                    DSLParser.evaluate(node["left"], values, prev_values, regimes)
                    or DSLParser.evaluate(node["right"], values, prev_values, regimes)
                )
            elif op == "NOT":
                operand = node.get("operand") or node.get("left") or node.get("right")
                if operand is None:
                    return False
                return not DSLParser.evaluate(operand, values, prev_values, regimes)
            elif op in ("GT", "LT", "GE", "LE", "EQ", "NE"):
                return DSLParser._eval_comparison(op, node, values)
            elif op in ("CROSS_ABOVE", "CROSS_BELOW"):
                return DSLParser._eval_crossover(op, node, values, prev_values)
            elif op == "IN_REGIME":
                return DSLParser._eval_regime(node, regimes, negate=False)
            elif op == "NOT_IN_REGIME":
                return DSLParser._eval_regime(node, regimes, negate=True)
            else:
                logger.warning("Unknown DSL operator: %s", op)
                return False
        except (KeyError, TypeError, ValueError) as exc:
            logger.debug("DSL eval error at op=%s: %s", op, exc)
            return False

    @staticmethod
    def _resolve_operand(
        operand: Any, values: Dict[str, Optional[float]]
    ) -> Optional[float]:
        """Resolve an operand to a float value.

        Operand can be:
        - A string (indicator name to look up)
        - A number (literal)
        - A nested logic node (not supported for comparisons — returns None)
        """
        if isinstance(operand, (int, float)):
            return float(operand)
        if isinstance(operand, str):
            val = values.get(operand)
            if val is None:
                return None
            return float(val)
        if isinstance(operand, dict) and "op" not in operand:
            # Could be a parameter reference like {"indicator": "ema", "period": 12}
            indicator = operand.get("indicator", "")
            period = operand.get("period", "")
            key = f"{indicator}_{period}" if period else indicator
            val = values.get(key)
            return float(val) if val is not None else None
        return None

    @staticmethod
    def _eval_comparison(
        op: str,
        node: Dict[str, Any],
        values: Dict[str, Optional[float]],
    ) -> bool:
        left = DSLParser._resolve_operand(node.get("left"), values)
        right = DSLParser._resolve_operand(node.get("right"), values)
        if left is None or right is None:
            return False

        if op == "GT":
            return left > right
        elif op == "LT":
            return left < right
        elif op == "GE":
            return left >= right
        elif op == "LE":
            return left <= right
        elif op == "EQ":
            return abs(left - right) < 1e-10
        elif op == "NE":
            return abs(left - right) >= 1e-10
        return False

    @staticmethod
    def _eval_crossover(
        op: str,
        node: Dict[str, Any],
        values: Dict[str, Optional[float]],
        prev_values: Dict[str, Optional[float]],
    ) -> bool:
        """Detect a crossover event between two series.

        CROSS_ABOVE: left was <= right on previous bar, and left > right now.
        CROSS_BELOW: left was >= right on previous bar, and left < right now.
        """
        left_curr = DSLParser._resolve_operand(node.get("left"), values)
        right_curr = DSLParser._resolve_operand(node.get("right"), values)
        left_prev = DSLParser._resolve_operand(node.get("left"), prev_values)
        right_prev = DSLParser._resolve_operand(node.get("right"), prev_values)

        if any(v is None for v in (left_curr, right_curr, left_prev, right_prev)):
            return False

        if op == "CROSS_ABOVE":
            return left_prev <= right_prev and left_curr > right_curr
        elif op == "CROSS_BELOW":
            return left_prev >= right_prev and left_curr < right_curr
        return False

    @staticmethod
    def _eval_regime(
        node: Dict[str, Any],
        regimes: Dict[str, str],
        negate: bool,
    ) -> bool:
        """Evaluate an IN_REGIME / NOT_IN_REGIME condition.

        Node format: {"op": "IN_REGIME", "condition": {"vol_regime": "VOL_LOW"}}
        or:          {"op": "IN_REGIME", "condition": {"vol_regime": ["VOL_LOW", "VOL_NORMAL"]}}
        """
        condition = node.get("condition", {})
        if not isinstance(condition, dict):
            return False

        for regime_type, expected in condition.items():
            current = regimes.get(regime_type)
            if current is None:
                return negate  # Missing regime = conservative

            if isinstance(expected, list):
                match = current in expected
            else:
                match = current == str(expected)

            if negate:
                if match:
                    return False
            else:
                if not match:
                    return False

        return True


# ═══════════════════════════════════════════════════════════════════════════
# DSLStrategy
# ═══════════════════════════════════════════════════════════════════════════

class DSLStrategy(BaseStrategy):
    """Strategy driven by a :class:`StrategyManifest` logic tree.

    On each bar, the indicator bank is updated, then the entry and exit
    logic trees are evaluated.  If entry conditions are met a signal is
    emitted; if exit conditions are met an exit signal is emitted.

    Parameters
    ----------
    bus : EventBus
        The Argus pub/sub event bus.
    manifest : StrategyManifest
        The validated strategy manifest to execute.
    config : dict, optional
        Override config (merged with manifest parameters).
    """

    def __init__(
        self,
        bus: Any,
        manifest: StrategyManifest,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Validate manifest before anything else
        try:
            manifest.validate()
        except ManifestValidationError:
            logger.error(
                "DSLStrategy received invalid manifest: %s", manifest.name
            )
            raise

        self._manifest = manifest
        self._manifest_hash = manifest.compute_hash()

        # Merge manifest parameters into strategy config
        merged_config = {
            "signals": manifest.signals,
            "entry_logic": manifest.entry_logic,
            "exit_logic": manifest.exit_logic,
            "direction": manifest.direction,
            "regime_filters": manifest.regime_filters,
            "risk_per_trade_pct": manifest.risk_per_trade_pct,
            "holding_period": manifest.holding_period,
            "universe": manifest.universe,
            "timeframe": manifest.timeframe,
        }
        # Flatten parameter ranges to their midpoints for initial config
        for key, val in manifest.parameters.items():
            if isinstance(val, dict) and "min" in val and "max" in val:
                merged_config[key] = (val["min"] + val["max"]) / 2.0
            else:
                merged_config[key] = val

        if config:
            merged_config.update(config)

        # Per-symbol indicator banks (created lazily)
        self._indicator_banks: Dict[str, _IndicatorBank] = {}

        super().__init__(bus, config=merged_config)

        logger.info(
            "DSLStrategy initialized: name=%s hash=%s signals=%s direction=%s",
            manifest.name,
            self._manifest_hash,
            manifest.signals,
            manifest.direction,
        )

    @property
    def strategy_id(self) -> str:
        safe_name = self._manifest.name.replace(" ", "_").replace("-", "_")
        return f"DSL_{safe_name}_{self._manifest_hash[:8]}"

    @property
    def default_config(self) -> Dict[str, Any]:
        return {
            "signals": [],
            "entry_logic": {"op": "AND", "left": {"op": "GT", "left": "rsi", "right": 50}, "right": {"op": "GT", "left": "rsi", "right": 50}},
            "exit_logic": {"op": "LT", "left": "rsi", "right": 30},
            "direction": "LONG",
            "regime_filters": {},
            "risk_per_trade_pct": 0.02,
            "holding_period": "intraday",
            "universe": ["IBIT"],
            "timeframe": 60,
        }

    @property
    def manifest(self) -> StrategyManifest:
        return self._manifest

    def _get_indicator_bank(self, symbol: str) -> _IndicatorBank:
        """Get or create the indicator bank for a symbol."""
        if symbol not in self._indicator_banks:
            self._indicator_banks[symbol] = _IndicatorBank(
                signals=self._config["signals"],
                params=self._config,
            )
        return self._indicator_banks[symbol]

    def _build_regime_snapshot(
        self,
        symbol_regime: Optional[SymbolRegimeEvent],
        market_regime: Optional[MarketRegimeEvent],
    ) -> Dict[str, str]:
        """Build a flat regime dict for DSL evaluation."""
        regimes: Dict[str, str] = {}

        if symbol_regime is not None:
            regimes["vol_regime"] = str(
                getattr(symbol_regime, "vol_regime", "UNKNOWN")
            )
            regimes["trend_regime"] = str(
                getattr(symbol_regime, "trend_regime", "UNKNOWN")
            )
            regimes["liquidity_regime"] = str(
                getattr(symbol_regime, "liquidity_regime", "UNKNOWN")
            )

        if market_regime is not None:
            regimes["session_regime"] = str(
                getattr(market_regime, "session_regime", "UNKNOWN")
            )
            regimes["risk_regime"] = str(
                getattr(market_regime, "risk_regime", "UNKNOWN")
            )

        return regimes

    def _check_regime_filters(self, regimes: Dict[str, str]) -> bool:
        """Check that current regimes pass the manifest's regime filters.

        Returns True if all filters pass (or no filters configured).
        """
        filters = self._config.get("regime_filters", {})
        if not filters:
            return True

        for regime_type, allowed_values in filters.items():
            current = regimes.get(regime_type)
            if current is None:
                # Missing regime data — conservative: skip bar
                return False
            if isinstance(allowed_values, list):
                if current not in allowed_values:
                    return False
            else:
                if current != str(allowed_values):
                    return False
        return True

    def evaluate(
        self,
        bar: BarEvent,
        symbol_regime: Optional[SymbolRegimeEvent],
        market_regime: Optional[MarketRegimeEvent],
    ) -> Optional[SignalEvent]:
        """Evaluate the manifest logic trees for this bar.

        Steps:
        1. Filter by universe (skip symbols not in manifest universe).
        2. Update indicator bank with bar data.
        3. Wait for indicator warmup.
        4. Check regime filters.
        5. Evaluate entry logic tree.
        6. Evaluate exit logic tree.
        7. Emit signal if conditions are met.
        """
        # 1. Universe filter
        universe = self._config.get("universe", [])
        if universe and bar.symbol not in universe:
            return None

        # 2. Update indicators
        bank = self._get_indicator_bank(bar.symbol)
        values = bank.update(bar)
        prev_values = bank.prev_values

        # 3. Warmup check — don't evaluate until indicators are ready
        if not bank.is_warm:
            return None

        # 4. Build regime snapshot and check filters
        regimes = self._build_regime_snapshot(symbol_regime, market_regime)
        if not self._check_regime_filters(regimes):
            self._signals_suppressed += 1
            return None

        # 5. Evaluate entry logic
        entry_logic = self._config.get("entry_logic", {})
        exit_logic = self._config.get("exit_logic", {})

        entry_signal = DSLParser.evaluate(entry_logic, values, prev_values, regimes)
        exit_signal = DSLParser.evaluate(exit_logic, values, prev_values, regimes)

        # 6. Determine what signal to emit (entry takes priority)
        direction = self._config.get("direction", "LONG")
        if entry_signal:
            signal_direction = (
                DIRECTION_LONG if direction == "LONG" else DIRECTION_SHORT
            )
            # Build features snapshot from indicator values
            features = {
                k: round(v, 8)
                for k, v in values.items()
                if v is not None and isinstance(v, (int, float))
            }
            return self._create_signal(
                bar=bar,
                direction=signal_direction,
                signal_type=SIGNAL_TYPE_ENTRY,
                confidence=1.0,
                regime_snapshot=regimes,
                features_snapshot=features,
                explain=f"DSL_ENTRY|manifest={self._manifest_hash[:8]}",
            )

        if exit_signal:
            # Exit signals use opposite direction
            exit_direction = (
                DIRECTION_SHORT if direction == "LONG" else DIRECTION_LONG
            )
            features = {
                k: round(v, 8)
                for k, v in values.items()
                if v is not None and isinstance(v, (int, float))
            }
            return self._create_signal(
                bar=bar,
                direction=exit_direction,
                signal_type=SIGNAL_TYPE_EXIT,
                confidence=1.0,
                regime_snapshot=regimes,
                features_snapshot=features,
                explain=f"DSL_EXIT|manifest={self._manifest_hash[:8]}",
            )

        return None

    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        status.update({
            "manifest_name": self._manifest.name,
            "manifest_hash": self._manifest_hash,
            "manifest_direction": self._manifest.direction,
            "manifest_signals": self._manifest.signals,
            "universe": self._manifest.universe,
            "indicator_banks": len(self._indicator_banks),
            "banks_warm": sum(
                1 for bank in self._indicator_banks.values() if bank.is_warm
            ),
        })
        return status

    @classmethod
    def from_manifest(
        cls,
        bus: Any,
        manifest: StrategyManifest,
        config: Optional[Dict[str, Any]] = None,
    ) -> DSLStrategy:
        """Factory method for creating a DSLStrategy from a manifest."""
        return cls(bus=bus, manifest=manifest, config=config)

    @classmethod
    def from_dict(
        cls,
        bus: Any,
        data: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> DSLStrategy:
        """Factory method for creating from a raw dict (e.g., from JSON)."""
        manifest = StrategyManifest.from_dict(data)
        return cls(bus=bus, manifest=manifest, config=config)

    @classmethod
    def from_json(
        cls,
        bus: Any,
        json_str: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> DSLStrategy:
        """Factory method for creating from a JSON string."""
        manifest = StrategyManifest.from_json(json_str)
        return cls(bus=bus, manifest=manifest, config=config)
