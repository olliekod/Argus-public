"""
Strategy Manifest Schemas
=========================

Type-safe manifest structures for the Pantheon Intelligence Engine.

A :class:`StrategyManifest` is the structured artifact produced by Prometheus
and consumed by the Hades backtest engine.  It contains everything needed to
configure, replay, and evaluate a strategy without human intervention.

:class:`AresCritique` captures Ares's adversarial analysis.
:class:`AthenaVerdict` captures Athena's final adjudication.

All models use dataclasses with strict validation so that invalid manifests
are rejected before reaching Hades.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

MANIFEST_SCHEMA_VERSION = 1

# Indicators that are actually implemented in the Hades engine / core
HADES_INDICATOR_CATALOG = frozenset({
    "ema",
    "rsi",
    "macd",
    "vwap",
    "atr",
    "rolling_vol",
    "bollinger_bands",
    "vol_z",
    "ema_slope",
    "spread_pct",
    "volume_pctile",
    "trend_accel",
})

# Regime types available for filtering
REGIME_FILTER_CATALOG = frozenset({
    "vol_regime",
    "trend_regime",
    "liquidity_regime",
    "session_regime",
    "risk_regime",
})

# Valid directions for strategy entry/exit
VALID_DIRECTIONS = frozenset({"LONG", "SHORT", "NEUTRAL"})

# Valid entry types
VALID_ENTRY_TYPES = frozenset({"MARKET", "LIMIT", "STOP", "CONDITIONAL"})

# Valid logic operators for the DSL
VALID_LOGIC_OPS = frozenset({
    "AND", "OR", "NOT",
    "GT", "LT", "GE", "LE", "EQ", "NE",
    "CROSS_ABOVE", "CROSS_BELOW",
    "IN_REGIME", "NOT_IN_REGIME",
})

# Human-readable one-liner descriptions of each indicator
# Used by ContextInjector to ground Prometheus in what each signal actually measures
HADES_INDICATOR_DESCRIPTIONS: Dict[str, str] = {
    "rsi":            "Relative Strength Index (0–100). Momentum oscillator. >70 = overbought, <30 = oversold.",
    "ema":            "Exponential Moving Average. Trend-following price smoothing. Use CROSS_ABOVE/BELOW for signals.",
    "macd":           "MACD histogram. Trend + momentum. Positive = bullish momentum, negative = bearish.",
    "vwap":           "Volume-Weighted Average Price. Intraday fair value anchor for institutional orders.",
    "atr":            "Average True Range. Measures price volatility magnitude (not direction). Good for stops/exits.",
    "rolling_vol":    "Rolling realized volatility (std dev of returns). NOT implied volatility. Good for VRP strategies.",
    "bollinger_bands":"Bollinger Bands width/position. Signals mean-reversion when price is outside bands.",
    "vol_z":          "Volatility Z-score. How many std deviations current vol is from its rolling mean.",
    "ema_slope":      "Rate of change of EMA. Positive = trend accelerating, negative = decelerating.",
    "spread_pct":     "Bid-ask spread as percentage. Proxy for liquidity. High spread = illiquid conditions.",
    "volume_pctile":  "Volume percentile (0–100) vs recent history. >80 = unusually high volume. Liquidity proxy.",
    "trend_accel":    "Trend acceleration. Second derivative of price direction. Positive = strengthening trend.",
}


# ═══════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════

class ManifestStatus(str, Enum):
    """Lifecycle status of a strategy manifest."""
    DRAFT = "DRAFT"                 # Prometheus initial output
    CRITIQUED = "CRITIQUED"         # After Ares review
    REVISED = "REVISED"             # After Prometheus revision
    ADJUDICATED = "ADJUDICATED"     # After Athena verdict
    PROMOTED = "PROMOTED"           # Approved for Hades backtest
    REJECTED = "REJECTED"           # Killed by Athena or Ares blockers


class CritiqueSeverity(str, Enum):
    """Severity of an Ares critique finding."""
    BLOCKER = "BLOCKER"       # Stops the case — must be resolved
    ADVISORY = "ADVISORY"     # Allows revision — should be addressed
    RESOLVED = "RESOLVED"     # Previously raised, now resolved


class CritiqueCategory(str, Enum):
    """Category of an Ares critique finding."""
    OVERFITTING = "OVERFITTING"
    LOOK_AHEAD_BIAS = "LOOK_AHEAD_BIAS"
    DATA_LEAKAGE = "DATA_LEAKAGE"
    PARAMETER_FRAGILITY = "PARAMETER_FRAGILITY"
    REGIME_DEPENDENCY = "REGIME_DEPENDENCY"
    EXECUTION_RISK = "EXECUTION_RISK"
    REGULATORY_RISK = "REGULATORY_RISK"
    LIQUIDITY_RISK = "LIQUIDITY_RISK"
    DRAWDOWN_RISK = "DRAWDOWN_RISK"
    SURVIVORSHIP_BIAS = "SURVIVORSHIP_BIAS"
    INSUFFICIENT_SAMPLE = "INSUFFICIENT_SAMPLE"
    STRATEGY = "STRATEGY"
    OTHER = "OTHER"


# ═══════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════

class ManifestValidationError(Exception):
    """Raised when a strategy manifest fails structural validation."""


def _validate_non_empty_string(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ManifestValidationError(
            f"{field_name} must be a non-empty string, got {value!r}"
        )


def _validate_signals(signals: List[str]) -> None:
    if not signals:
        raise ManifestValidationError("signals must contain at least one indicator")
    unknown = set(signals) - HADES_INDICATOR_CATALOG
    if unknown:
        raise ManifestValidationError(
            f"Unknown indicators not in Hades catalog: {sorted(unknown)}. "
            f"Available: {sorted(HADES_INDICATOR_CATALOG)}"
        )


def _validate_parameters(parameters: Dict[str, Any]) -> None:
    if not isinstance(parameters, dict):
        raise ManifestValidationError("parameters must be a dict")
    for key, spec in parameters.items():
        if not isinstance(key, str):
            raise ManifestValidationError(f"Parameter key must be str, got {type(key)}")
        if isinstance(spec, dict):
            # Range spec: {"min": x, "max": y, "step": z}
            if "min" in spec and "max" in spec:
                if spec["min"] > spec["max"]:
                    raise ManifestValidationError(
                        f"Parameter '{key}' has min > max: {spec['min']} > {spec['max']}"
                    )


def _validate_logic_node(node: Dict[str, Any], path: str = "root") -> None:
    """Recursively validate a logic tree node."""
    if not isinstance(node, dict):
        raise ManifestValidationError(
            f"Logic node at '{path}' must be a dict, got {type(node)}"
        )
    op = node.get("op")
    if op is None:
        raise ManifestValidationError(
            f"Logic node at '{path}' missing 'op' field"
        )
    if op not in VALID_LOGIC_OPS:
        raise ManifestValidationError(
            f"Unknown logic operator '{op}' at '{path}'. "
            f"Valid: {sorted(VALID_LOGIC_OPS)}"
        )
    # Recursively validate children
    for child_key in ("left", "right", "operand", "condition"):
        child = node.get(child_key)
        if isinstance(child, dict) and "op" in child:
            _validate_logic_node(child, f"{path}.{child_key}")
    children = node.get("children", [])
    for i, child in enumerate(children):
        if isinstance(child, dict) and "op" in child:
            _validate_logic_node(child, f"{path}.children[{i}]")


# ═══════════════════════════════════════════════════════════════════════════
# StrategyManifest
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class StrategyManifest:
    """Structured strategy proposal generated by Prometheus.

    This is the primary artifact of the Pantheon research loop.
    It must be structurally valid for direct consumption by the
    Hades backtest engine (replay harness + experiment runner).

    Fields
    ------
    name : str
        Human-readable strategy name (e.g., "Vol-Adjusted Momentum").
    objective : str
        What the strategy aims to capture (e.g., "momentum premium in low-vol").
    signals : List[str]
        Hades-native indicators to use (must be in HADES_INDICATOR_CATALOG).
    entry_logic : Dict[str, Any]
        DSL-compatible logic tree for entry conditions.
    exit_logic : Dict[str, Any]
        DSL-compatible logic tree for exit conditions.
    parameters : Dict[str, Any]
        Tuning ranges for backtest optimization.
        Each key maps to either a fixed value or a range spec:
        ``{"min": float, "max": float, "step": float}``.
    direction : str
        Primary direction: LONG, SHORT, or NEUTRAL.
    universe : List[str]
        Target symbols (e.g., ["IBIT", "SPY", "BTCUSDT"]).
    regime_filters : Dict[str, List[str]]
        Regime conditions under which the strategy operates.
        Keys are regime types (vol_regime, trend_regime, etc.),
        values are acceptable regime states.
    timeframe : int
        Bar duration in seconds (default 60 for 1-min bars).
    holding_period : str
        Expected holding period (e.g., "intraday", "1-5 days").
    risk_per_trade_pct : float
        Maximum risk per trade as fraction of equity.
    """

    name: str
    objective: str
    signals: List[str]
    entry_logic: Dict[str, Any]
    exit_logic: Dict[str, Any]
    parameters: Dict[str, Any]
    direction: str = "LONG"
    universe: List[str] = field(default_factory=lambda: ["IBIT"])
    regime_filters: Dict[str, List[str]] = field(default_factory=dict)
    timeframe: int = 60
    holding_period: str = "intraday"
    risk_per_trade_pct: float = 0.02
    case_id: str = ""  # Pantheon research case ID for provenance tracking
    status: ManifestStatus = ManifestStatus.DRAFT
    version: int = MANIFEST_SCHEMA_VERSION

    def validate(self) -> None:
        """Validate all fields. Raises ManifestValidationError on failure."""
        _validate_non_empty_string(self.name, "name")
        _validate_non_empty_string(self.objective, "objective")
        _validate_signals(self.signals)
        _validate_logic_node(self.entry_logic, "entry_logic")
        _validate_logic_node(self.exit_logic, "exit_logic")
        _validate_parameters(self.parameters)

        if self.direction not in VALID_DIRECTIONS:
            raise ManifestValidationError(
                f"direction must be one of {sorted(VALID_DIRECTIONS)}, "
                f"got {self.direction!r}"
            )

        if not self.universe:
            raise ManifestValidationError("universe must contain at least one symbol")

        for regime_type in self.regime_filters:
            if regime_type not in REGIME_FILTER_CATALOG:
                raise ManifestValidationError(
                    f"Unknown regime filter '{regime_type}'. "
                    f"Valid: {sorted(REGIME_FILTER_CATALOG)}"
                )

        if self.timeframe <= 0:
            raise ManifestValidationError(
                f"timeframe must be positive, got {self.timeframe}"
            )

        if not (0.0 < self.risk_per_trade_pct <= 1.0):
            raise ManifestValidationError(
                f"risk_per_trade_pct must be in (0, 1], got {self.risk_per_trade_pct}"
            )

    def compute_hash(self) -> str:
        """Compute deterministic hash for manifest content identity.

        Excludes lifecycle metadata (status, case_id, version) so that
        the hash remains stable across status transitions.
        """
        d = self.to_dict()
        # Exclude lifecycle/metadata fields from identity hash
        for key in ("status", "case_id", "version"):
            d.pop(key, None)
        canonical = json.dumps(d, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        d: Dict[str, Any] = {
            "name": self.name,
            "objective": self.objective,
            "signals": sorted(self.signals),
            "entry_logic": self.entry_logic,
            "exit_logic": self.exit_logic,
            "parameters": self.parameters,
            "direction": self.direction,
            "universe": sorted(self.universe),
            "regime_filters": {
                k: sorted(v) for k, v in sorted(self.regime_filters.items())
            },
            "timeframe": self.timeframe,
            "holding_period": self.holding_period,
            "risk_per_trade_pct": self.risk_per_trade_pct,
            "case_id": self.case_id,
            "status": self.status.value,
            "version": self.version,
        }
        return d

    def to_json(self) -> str:
        """Serialize to deterministic JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StrategyManifest:
        """Deserialize from dict. Raises ManifestValidationError on bad data."""
        try:
            # Coerce direction: LLMs sometimes hallucinate a nested dict/logic
            # object for bidirectional strategies (e.g. mean reversion).  Rather
            # than hard-failing the entire pipeline, we normalise to "NEUTRAL"
            # with a warning so the case can continue to Ares / Athena review.
            raw_direction = data.get("direction", "LONG")
            if not isinstance(raw_direction, str):
                logger.warning(
                    "StrategyManifest: 'direction' field is %s (expected str) — "
                    "coercing to 'NEUTRAL'. The LLM should use a plain string.",
                    type(raw_direction).__name__,
                )
                raw_direction = "NEUTRAL"
            direction = str(raw_direction).strip().upper()
            # Accept minor capitalisation variants produced by the LLM
            _DIR_ALIASES = {"LONG": "LONG", "SHORT": "SHORT", "NEUTRAL": "NEUTRAL",
                            "BUY": "LONG", "SELL": "SHORT", "BOTH": "NEUTRAL"}
            direction = _DIR_ALIASES.get(direction, direction)

            manifest = cls(
                name=str(data["name"]),
                objective=str(data["objective"]),
                signals=list(data["signals"]),
                entry_logic=dict(data["entry_logic"]),
                exit_logic=dict(data["exit_logic"]),
                parameters=dict(data["parameters"]),
                direction=direction,
                universe=list(data.get("universe", ["IBIT"])),
                regime_filters={
                    str(k): list(v) for k, v in data.get("regime_filters", {}).items()
                },
                timeframe=int(data.get("timeframe", 60)),
                holding_period=str(data.get("holding_period", "intraday")),
                risk_per_trade_pct=float(data.get("risk_per_trade_pct", 0.02)),
                case_id=str(data.get("case_id", "")),
                status=ManifestStatus(data.get("status", "DRAFT")),
                version=int(data.get("version", MANIFEST_SCHEMA_VERSION)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ManifestValidationError(
                f"Failed to parse StrategyManifest: {exc}"
            ) from exc
        return manifest

    @classmethod
    def from_json(cls, json_str: str) -> StrategyManifest:
        """Parse from JSON string. Raises ManifestValidationError on bad JSON."""
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ManifestValidationError(
                f"Invalid JSON in manifest: {exc}"
            ) from exc
        return cls.from_dict(data)

    def to_backtest_config(self) -> Dict[str, Any]:
        """Convert to a config dict compatible with the Hades experiment runner.

        Returns a dict that can be used as ``strategy_params`` in
        :class:`~src.analysis.research_loop_config.StrategySpec`.
        """
        return {
            "strategy_class": f"PantheonGenerated_{self.name.replace(' ', '_')}",
            "params": {
                **{k: v for k, v in self.parameters.items()
                   if not isinstance(v, dict)},
                "signals": self.signals,
                "entry_logic": self.entry_logic,
                "exit_logic": self.exit_logic,
                "direction": self.direction,
                "regime_filters": self.regime_filters,
                "risk_per_trade_pct": self.risk_per_trade_pct,
                "holding_period": self.holding_period,
            },
            "sweep": {
                k: v for k, v in self.parameters.items()
                if isinstance(v, dict) and "min" in v and "max" in v
            },
            "universe": self.universe,
            "timeframe": self.timeframe,
        }


# ═══════════════════════════════════════════════════════════════════════════
# AresCritique
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CritiqueFinding:
    """A single finding from Ares's adversarial analysis."""
    category: CritiqueCategory
    severity: CritiqueSeverity
    description: str
    evidence: str = ""
    recommendation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "description": self.description,
            "evidence": self.evidence,
            "recommendation": self.recommendation,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CritiqueFinding:
        raw_cat = str(data.get("category", "")).strip().upper()
        try:
            category = CritiqueCategory(raw_cat)
        except ValueError:
            matched = False
            for valid_cat in CritiqueCategory:
                if valid_cat.value in raw_cat:
                    category = valid_cat
                    matched = True
                    break
            if not matched:
                category = CritiqueCategory.OTHER
                logger.warning("AresCritique: Unknown category '%s' mapped to OTHER", raw_cat)
            else:
                logger.warning("AresCritique: Malformed category '%s' mapped to %s", raw_cat, category.value)
                
        raw_sev = str(data.get("severity", "ADVISORY")).strip().upper()
        try:
            severity = CritiqueSeverity(raw_sev)
        except ValueError:
            matched = False
            for valid_sev in CritiqueSeverity:
                if valid_sev.value in raw_sev:
                    severity = valid_sev
                    matched = True
                    break
            if not matched:
                severity = CritiqueSeverity.ADVISORY
                logger.warning("AresCritique: Unknown severity '%s' mapped to ADVISORY", raw_sev)
            else:
                logger.warning("AresCritique: Malformed severity '%s' mapped to %s", raw_sev, severity.value)

        return cls(
            category=category,
            severity=severity,
            description=str(data.get("description", "")),
            evidence=str(data.get("evidence", "")),
            recommendation=str(data.get("recommendation", "")),
        )


@dataclass
class AresCritique:
    """Structured adversarial critique from Ares.

    Contains prioritized findings split into blockers and advisories.
    A manifest with unresolved blockers cannot be promoted.
    """
    manifest_hash: str
    findings: List[CritiqueFinding] = field(default_factory=list)
    summary: str = ""

    @property
    def blockers(self) -> List[CritiqueFinding]:
        return [
            f for f in self.findings
            if f.severity == CritiqueSeverity.BLOCKER
        ]

    @property
    def advisories(self) -> List[CritiqueFinding]:
        return [
            f for f in self.findings
            if f.severity == CritiqueSeverity.ADVISORY
        ]

    @property
    def resolved(self) -> List[CritiqueFinding]:
        return [
            f for f in self.findings
            if f.severity == CritiqueSeverity.RESOLVED
        ]

    @property
    def has_blockers(self) -> bool:
        return len(self.blockers) > 0

    def validate(self) -> None:
        """Ensure critique has at least three failure vectors analyzed."""
        if len(self.findings) < 3:
            raise ManifestValidationError(
                f"Ares must analyze at least 3 failure vectors, "
                f"found {len(self.findings)}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manifest_hash": self.manifest_hash,
            "findings": [f.to_dict() for f in self.findings],
            "summary": self.summary,
            "blocker_count": len(self.blockers),
            "advisory_count": len(self.advisories),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AresCritique:
        return cls(
            manifest_hash=str(data["manifest_hash"]),
            findings=[
                CritiqueFinding.from_dict(f) for f in data.get("findings", [])
            ],
            summary=str(data.get("summary", "")),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> AresCritique:
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ManifestValidationError(
                f"Invalid JSON in critique: {exc}"
            ) from exc
        return cls.from_dict(data)


# ═══════════════════════════════════════════════════════════════════════════
# AthenaVerdict
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AthenaVerdict:
    """Final adjudication from Athena.

    Contains a confidence score, the validated research packet
    (cleaned StrategyManifest), and the rationale for the decision.
    """
    confidence: float                     # 0.0–1.0
    decision: str                         # "PROMOTE" or "REJECT"
    rationale: str
    research_packet: Optional[Dict[str, Any]] = None  # Validated manifest dict
    unresolved_blockers: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)  # Conditions for promotion

    # Scoring rubric breakdown
    rubric_scores: Dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        """Ensure verdict is structurally sound.

        For PROMOTE decisions, also validates that research_packet is
        a parseable StrategyManifest (or at least has required keys).
        """
        if not (0.0 <= self.confidence <= 1.0):
            raise ManifestValidationError(
                f"confidence must be in [0, 1], got {self.confidence}"
            )
        if self.decision not in ("PROMOTE", "REJECT"):
            raise ManifestValidationError(
                f"decision must be PROMOTE or REJECT, got {self.decision!r}"
            )
        if self.decision == "PROMOTE" and self.research_packet is None:
            raise ManifestValidationError(
                "PROMOTE decision requires a non-null research_packet"
            )
        # Validate research_packet is a real manifest, not a wrapped label
        if self.decision == "PROMOTE" and self.research_packet is not None:
            rp = self.research_packet
            # If the LLM wrapped the manifest in a label key, unwrap it
            if len(rp) == 1 and "name" not in rp:
                inner_key = next(iter(rp))
                inner = rp[inner_key]
                if isinstance(inner, dict) and "name" in inner:
                    logger.warning(
                        "AthenaVerdict: research_packet was wrapped in key '%s' — unwrapping.",
                        inner_key,
                    )
                    self.research_packet = inner
        _validate_non_empty_string(self.rationale, "rationale")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "confidence": round(self.confidence, 4),
            "decision": self.decision,
            "rationale": self.rationale,
            "research_packet": self.research_packet,
            "unresolved_blockers": self.unresolved_blockers,
            "conditions": self.conditions,
            "rubric_scores": {
                k: round(v, 4) for k, v in self.rubric_scores.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AthenaVerdict:
        """Deserialize from dict with fallback logic for missing/hallucinated fields."""
        try:
            # ── 1. Normalize decision string (LLMs love hallucinating variants) ──
            raw_decision = str(data.get("decision", "") or data.get("verdict", ""))
            _PROMOTE_ALIASES = {"promote", "approved", "approve", "pass", "go", "yes", "accept"}
            _REJECT_ALIASES  = {"reject", "rejected", "decline", "declined", "no", "fail", "block"}
            normalized = raw_decision.strip().lower().rstrip(".")
            if normalized in _PROMOTE_ALIASES:
                if raw_decision != "PROMOTE":
                    logger.warning("AthenaVerdict: normalizing decision %r → PROMOTE", raw_decision)
                decision = "PROMOTE"
            elif normalized in _REJECT_ALIASES:
                if raw_decision != "REJECT":
                    logger.warning("AthenaVerdict: normalizing decision %r → REJECT", raw_decision)
                decision = "REJECT"
            else:
                decision = ""  # will be inferred later

            # ── 2. Extract Rubric Scores from multiple possible keys ──
            _KEY_MAP = {
                "theoretical soundness": "theoretical_soundness",
                "theoreticalsoundness":  "theoretical_soundness",
                "theoretical_soundness": "theoretical_soundness",
                "critique resolution":   "critique_resolution",
                "critiqueresolution":     "critique_resolution",
                "critique_resolution":   "critique_resolution",
                "parameter fragility":   "critique_resolution",
                "testability":           "testability",
                "risk management":       "risk_management",
                "riskmanagement":        "risk_management",
                "risk_management":       "risk_management",
                "drawdown risk mitigation": "risk_management",
                "regulatory compliance": "risk_management",
                "novelty":               "novelty",
                "liquidity risk":        "novelty",
                "sufficient sample size": "novelty",
            }

            raw_rubric: Dict[str, Any] = {}
            for candidate_key in ("rubric_scores", "score", "scores", "rubric"):
                val = data.get(candidate_key)
                if isinstance(val, dict) and val:
                    raw_rubric = val
                    if candidate_key != "rubric_scores":
                        logger.warning(
                            "AthenaVerdict: rubric scores found under '%s' key, not 'rubric_scores'",
                            candidate_key,
                        )
                    break

            rubric_scores: Dict[str, float] = {}
            for k, v in raw_rubric.items():
                if isinstance(v, (int, float)):
                    canonical = _KEY_MAP.get(k.lower().strip(), k.lower().replace(" ", "_"))
                    rubric_scores[canonical] = float(v)

            # Last-resort: pull canonical keys from top-level data
            if not rubric_scores:
                for canon in ("theoretical_soundness", "critique_resolution",
                               "testability", "risk_management", "novelty"):
                    if isinstance(data.get(canon), (int, float)):
                        rubric_scores[canon] = float(data[canon])

            # Fallback: build rubric from 'findings' array with category/rating pairs.
            # The LLM sometimes outputs [{"category": "OVERFITTING", "rating": 5}, ...]
            # Map finding categories to canonical rubric keys using _KEY_MAP.
            # Map finding categories to canonical rubric keys using _KEY_MAP.
            if not rubric_scores:
                findings_raw = data.get("findings") or data.get("reasoning")
                if isinstance(findings_raw, list) and findings_raw:
                    for finding in findings_raw:
                        if isinstance(finding, dict):
                            cat = str(finding.get("category", "")).lower().strip()
                            rating = finding.get("rating")
                            if cat and isinstance(rating, (int, float)):
                                canonical = _KEY_MAP.get(cat, cat.lower().replace(" ", "_"))
                                # Normalize: ratings on 0-5 scale → 0.0-1.0
                                rubric_scores[canonical] = min(float(rating) / 5.0, 1.0)
                    if rubric_scores:
                        logger.warning(
                            "AthenaVerdict: built rubric_scores from findings[].rating: %s",
                            rubric_scores,
                        )

            # ── 3. Extract or Calculate Confidence ──
            confidence = data.get("confidence")

            # Detect total_score pattern (e.g. Athena scores 3/6 blockers resolved)
            total_score_raw = data.get("total_score")
            if confidence is None and isinstance(total_score_raw, (int, float)):
                # Normalize to [0, 1]. Heuristic: max is 6 (typical rubric items).
                # Clamp so 6/6 = 1.0 and 3/6 = 0.5.
                confidence = min(float(total_score_raw) / 6.0, 1.0)
                logger.warning(
                    "AthenaVerdict: detected 'total_score=%s' — normalized confidence to %.4f",
                    total_score_raw, confidence,
                )

            # Detect overall_quality pattern (0–100 scale, e.g. {"overall_quality": 80})
            overall_quality_raw = data.get("overall_quality")
            if confidence is None and isinstance(overall_quality_raw, (int, float)):
                confidence = min(max(float(overall_quality_raw) / 100.0, 0.0), 1.0)
                logger.warning(
                    "AthenaVerdict: detected 'overall_quality=%s' — normalized confidence to %.4f",
                    overall_quality_raw, confidence,
                )

            if confidence is None:
                # Compute from weighted rubric
                theoretical = rubric_scores.get("theoretical_soundness", 0.0)
                resolution  = rubric_scores.get("critique_resolution", 0.0)
                testability = rubric_scores.get("testability", 0.0)
                risk        = rubric_scores.get("risk_management", 0.0)
                novelty     = rubric_scores.get("novelty", 0.0)
                confidence  = (
                    (theoretical * 0.25) +
                    (resolution  * 0.25) +
                    (testability * 0.20) +
                    (risk        * 0.15) +
                    (novelty     * 0.15)
                )

                # Fallback if standard keys were missing but we have other scores
                if confidence == 0.0 and rubric_scores:
                    vals = list(rubric_scores.values())
                    max_val = max(vals)
                    if max_val > 1.0:
                        if max_val <= 3.0:
                            scale = 3.0
                        elif max_val <= 5.0:
                            scale = 5.0
                        elif max_val <= 10.0:
                            scale = 10.0
                        else:
                            scale = 100.0
                        vals = [min(v / scale, 1.0) for v in vals]
                    confidence = sum(vals) / len(vals)
                    logger.warning("AthenaVerdict calculated fallback confidence from non-standard rubric keys: %.4f", confidence)
                else:
                    logger.warning("AthenaVerdict missing top-level 'confidence'. Calculated: %.4f", confidence)
            else:
                confidence = float(confidence)

            # ── 4. Build rationale from multiple fallback keys ──
            rationale_raw = (
                data.get("rationale")
                or data.get("summary")
                or data.get("reasoning")
                or data.get("comments")
                or ""
            )
            if isinstance(rationale_raw, list):
                rationale_raw = " ".join(
                    (item.get("comment") or item.get("description") or str(item))
                    if isinstance(item, dict) else str(item)
                    for item in rationale_raw
                ).strip()
            rationale = str(rationale_raw).strip() or "No rationale provided."

            # ── 5. Infer decision from context if still missing ──
            if not decision:
                unresolved = list(data.get("unresolved_blockers", []))
                if unresolved:
                    decision = "REJECT"
                    logger.warning(
                        "AthenaVerdict: no decision field found, but unresolved_blockers present — defaulting to REJECT"
                    )
                elif confidence >= 0.6:
                    decision = "PROMOTE"
                    logger.warning(
                        "AthenaVerdict: no decision field, confidence=%.4f — defaulting to PROMOTE",
                        confidence,
                    )
                else:
                    decision = "REJECT"
                    logger.warning(
                        "AthenaVerdict: no decision field, confidence=%.4f — defaulting to REJECT",
                        confidence,
                    )

            return cls(
                confidence=confidence,
                decision=decision,
                rationale=rationale,
                research_packet=data.get("research_packet"),
                unresolved_blockers=list(data.get("unresolved_blockers", [])),
                conditions=list(data.get("conditions", [])),
                rubric_scores=rubric_scores,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ManifestValidationError(
                f"Failed to parse AthenaVerdict from data keys {list(data.keys())}: {exc}"
            ) from exc

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> AthenaVerdict:
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ManifestValidationError(
                f"Invalid JSON in verdict: {exc}"
            ) from exc
        return cls.from_dict(data)


# ═══════════════════════════════════════════════════════════════════════════
# JSON Extraction Helper
# ═══════════════════════════════════════════════════════════════════════════

_extract_json_logger = logging.getLogger(__name__ + ".extract_json")


def extract_json_from_response(text: str) -> Optional[Dict[str, Any]]:
    """Extract a JSON object from an LLM response that may contain prose.

    Tries, in order:
    1. If the text is pure JSON (e.g. from Ollama format=json mode), parse it and
       try to extract 'manifest', 'critique', or 'verdict' keys if it wrapped them.
    2. Fenced ``<manifest>...</manifest>`` or ``<critique>...</critique>``
       or ``<verdict>...</verdict>`` tags
    3. Fenced ```json ... ``` blocks
    4. First balanced ``{...}`` in the text

    Returns None if no valid JSON found.
    """
    text = text.strip()
    
    # 1. Try pure JSON first (common when using constrained JSON generation)
    if text.startswith("{") and text.endswith("}"):
        try:
            parsed = json.loads(text)
            # If the LLM wrapped the object in {"thought": "...", "manifest": "{...}"}
            for candidate_key in ("manifest", "critique", "verdict"):
                if candidate_key in parsed:
                    inner = parsed[candidate_key]
                    if isinstance(inner, str):
                        try:
                            # Sometimes the LLM double-encodes the inner JSON
                            return json.loads(inner)
                        except json.JSONDecodeError:
                            pass
                    elif isinstance(inner, dict):
                        return inner
            return parsed
        except json.JSONDecodeError:
            pass

    # 2. Try tagged blocks
    for tag in ("manifest", "critique", "verdict"):
        pattern = rf"<{tag}>\s*(\{{.*?\}})\s*</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            candidate = match.group(1)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as exc:
                _extract_json_logger.warning(
                    "<%s> tag found but JSON is invalid (%s). "
                    "First 300 chars: %r",
                    tag, exc, candidate[:300]
                )
                continue

    # 2. Try fenced code blocks
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # 3. Try first balanced braces
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                candidate = text[start:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    start = -1
                    continue

    return None
