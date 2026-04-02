# Created by Oliver Meihls

# Tests for Strategy Manifest schemas and validation.

from __future__ import annotations

import json

import pytest

from src.core.manifests import (
    AresCritique,
    AthenaVerdict,
    CritiqueCategory,
    CritiqueFinding,
    CritiqueSeverity,
    ManifestStatus,
    ManifestValidationError,
    StrategyManifest,
    extract_json_from_response,
)


# Helpers

def _valid_manifest_dict() -> dict:
    # Return a minimal valid manifest dict.
    return {
        "name": "Vol-Adjusted Momentum",
        "objective": "Capture momentum premium in low-volatility regimes",
        "signals": ["ema", "rsi", "atr"],
        "entry_logic": {
            "op": "AND",
            "children": [
                {"op": "GT", "left": "rsi", "right": 60},
                {"op": "IN_REGIME", "left": "vol_regime", "right": "VOL_LOW"},
            ],
        },
        "exit_logic": {
            "op": "OR",
            "children": [
                {"op": "LT", "left": "rsi", "right": 40},
                {"op": "NOT_IN_REGIME", "left": "vol_regime", "right": "VOL_LOW"},
            ],
        },
        "parameters": {
            "rsi_period": {"min": 10, "max": 20, "step": 2},
            "ema_fast": {"min": 5, "max": 15, "step": 1},
            "ema_slow": {"min": 20, "max": 50, "step": 5},
        },
        "direction": "LONG",
        "universe": ["IBIT", "SPY"],
        "regime_filters": {
            "vol_regime": ["VOL_LOW", "VOL_NORMAL"],
            "trend_regime": ["TREND_UP"],
        },
        "timeframe": 60,
        "holding_period": "intraday",
        "risk_per_trade_pct": 0.02,
    }


def _valid_manifest() -> StrategyManifest:
    # Return a valid StrategyManifest instance.
    return StrategyManifest.from_dict(_valid_manifest_dict())


# StrategyManifest tests

class TestStrategyManifest:
    def test_valid_manifest_parses(self):
        m = _valid_manifest()
        assert m.name == "Vol-Adjusted Momentum"
        assert "ema" in m.signals
        assert m.direction == "LONG"

    def test_valid_manifest_validates(self):
        m = _valid_manifest()
        m.validate()  # Should not raise

    def test_roundtrip_dict(self):
        m = _valid_manifest()
        d = m.to_dict()
        m2 = StrategyManifest.from_dict(d)
        assert m2.name == m.name
        assert m2.signals == sorted(m.signals)
        assert m2.direction == m.direction

    def test_roundtrip_json(self):
        m = _valid_manifest()
        j = m.to_json()
        m2 = StrategyManifest.from_json(j)
        assert m2.name == m.name
        assert m2.objective == m.objective

    def test_compute_hash_deterministic(self):
        m1 = _valid_manifest()
        m2 = _valid_manifest()
        assert m1.compute_hash() == m2.compute_hash()
        assert len(m1.compute_hash()) == 16

    def test_to_backtest_config(self):
        m = _valid_manifest()
        bc = m.to_backtest_config()
        assert "strategy_class" in bc
        assert "params" in bc
        assert "sweep" in bc
        assert "universe" in bc
        assert bc["timeframe"] == 60
        # Sweep should only contain range-spec parameters
        for key, val in bc["sweep"].items():
            assert "min" in val and "max" in val

    def test_status_default_is_draft(self):
        m = _valid_manifest()
        assert m.status == ManifestStatus.DRAFT

    def test_status_serialization(self):
        m = _valid_manifest()
        m.status = ManifestStatus.PROMOTED
        d = m.to_dict()
        assert d["status"] == "PROMOTED"
        m2 = StrategyManifest.from_dict(d)
        assert m2.status == ManifestStatus.PROMOTED


class TestManifestValidation:
    def test_empty_name_fails(self):
        d = _valid_manifest_dict()
        d["name"] = ""
        m = StrategyManifest.from_dict(d)
        with pytest.raises(ManifestValidationError, match="name"):
            m.validate()

    def test_empty_objective_fails(self):
        d = _valid_manifest_dict()
        d["objective"] = "  "
        m = StrategyManifest.from_dict(d)
        with pytest.raises(ManifestValidationError, match="objective"):
            m.validate()

    def test_unknown_signal_fails(self):
        d = _valid_manifest_dict()
        d["signals"] = ["ema", "nonexistent_indicator"]
        m = StrategyManifest.from_dict(d)
        with pytest.raises(ManifestValidationError, match="nonexistent_indicator"):
            m.validate()

    def test_empty_signals_fails(self):
        d = _valid_manifest_dict()
        d["signals"] = []
        m = StrategyManifest.from_dict(d)
        with pytest.raises(ManifestValidationError, match="at least one"):
            m.validate()

    def test_invalid_direction_fails(self):
        d = _valid_manifest_dict()
        d["direction"] = "UP"
        m = StrategyManifest.from_dict(d)
        with pytest.raises(ManifestValidationError, match="direction"):
            m.validate()

    def test_empty_universe_fails(self):
        d = _valid_manifest_dict()
        d["universe"] = []
        m = StrategyManifest.from_dict(d)
        with pytest.raises(ManifestValidationError, match="universe"):
            m.validate()

    def test_invalid_regime_filter_fails(self):
        d = _valid_manifest_dict()
        d["regime_filters"] = {"nonexistent_regime": ["X"]}
        m = StrategyManifest.from_dict(d)
        with pytest.raises(ManifestValidationError, match="nonexistent_regime"):
            m.validate()

    def test_negative_timeframe_fails(self):
        d = _valid_manifest_dict()
        d["timeframe"] = -1
        m = StrategyManifest.from_dict(d)
        with pytest.raises(ManifestValidationError, match="timeframe"):
            m.validate()

    def test_risk_out_of_range_fails(self):
        d = _valid_manifest_dict()
        d["risk_per_trade_pct"] = 1.5
        m = StrategyManifest.from_dict(d)
        with pytest.raises(ManifestValidationError, match="risk_per_trade_pct"):
            m.validate()

    def test_zero_risk_fails(self):
        d = _valid_manifest_dict()
        d["risk_per_trade_pct"] = 0.0
        m = StrategyManifest.from_dict(d)
        with pytest.raises(ManifestValidationError, match="risk_per_trade_pct"):
            m.validate()

    def test_logic_missing_op_fails(self):
        d = _valid_manifest_dict()
        d["entry_logic"] = {"left": "rsi", "right": 50}  # Missing "op"
        m = StrategyManifest.from_dict(d)
        with pytest.raises(ManifestValidationError, match="op"):
            m.validate()

    def test_logic_unknown_op_fails(self):
        d = _valid_manifest_dict()
        d["entry_logic"] = {"op": "INVALID_OP", "left": "rsi", "right": 50}
        m = StrategyManifest.from_dict(d)
        with pytest.raises(ManifestValidationError, match="INVALID_OP"):
            m.validate()

    def test_parameter_min_gt_max_fails(self):
        d = _valid_manifest_dict()
        d["parameters"] = {"bad_param": {"min": 100, "max": 10, "step": 1}}
        m = StrategyManifest.from_dict(d)
        with pytest.raises(ManifestValidationError, match="min > max"):
            m.validate()

    def test_invalid_json_string_fails(self):
        with pytest.raises(ManifestValidationError, match="Invalid JSON"):
            StrategyManifest.from_json("not valid json {{{")

    def test_missing_required_field_fails(self):
        d = _valid_manifest_dict()
        del d["signals"]
        with pytest.raises(ManifestValidationError, match="Failed to parse"):
            StrategyManifest.from_dict(d)


# AresCritique tests

class TestAresCritique:
    def test_valid_critique_parses(self):
        c = AresCritique(
            manifest_hash="abc123",
            findings=[
                CritiqueFinding(
                    category=CritiqueCategory.OVERFITTING,
                    severity=CritiqueSeverity.BLOCKER,
                    description="Too many parameters for the data window.",
                ),
                CritiqueFinding(
                    category=CritiqueCategory.REGIME_DEPENDENCY,
                    severity=CritiqueSeverity.ADVISORY,
                    description="Strategy only tested in VOL_LOW regime.",
                ),
                CritiqueFinding(
                    category=CritiqueCategory.EXECUTION_RISK,
                    severity=CritiqueSeverity.ADVISORY,
                    description="Assumes market orders fill at mid.",
                ),
            ],
            summary="Strategy has overfitting risk. Revise parameter space.",
        )
        assert c.has_blockers
        assert len(c.blockers) == 1
        assert len(c.advisories) == 2

    def test_critique_validation_minimum_findings(self):
        c = AresCritique(
            manifest_hash="abc123",
            findings=[
                CritiqueFinding(
                    category=CritiqueCategory.OVERFITTING,
                    severity=CritiqueSeverity.BLOCKER,
                    description="One issue.",
                ),
            ],
        )
        with pytest.raises(ManifestValidationError, match="at least 3"):
            c.validate()

    def test_critique_with_three_findings_validates(self):
        c = AresCritique(
            manifest_hash="abc123",
            findings=[
                CritiqueFinding(CritiqueCategory.OVERFITTING, CritiqueSeverity.BLOCKER, "A"),
                CritiqueFinding(CritiqueCategory.REGIME_DEPENDENCY, CritiqueSeverity.ADVISORY, "B"),
                CritiqueFinding(CritiqueCategory.EXECUTION_RISK, CritiqueSeverity.ADVISORY, "C"),
            ],
        )
        c.validate()  # Should not raise

    def test_critique_roundtrip(self):
        c = AresCritique(
            manifest_hash="hash123",
            findings=[
                CritiqueFinding(CritiqueCategory.OVERFITTING, CritiqueSeverity.BLOCKER, "Desc1", "Ev1", "Rec1"),
                CritiqueFinding(CritiqueCategory.DATA_LEAKAGE, CritiqueSeverity.ADVISORY, "Desc2"),
                CritiqueFinding(CritiqueCategory.EXECUTION_RISK, CritiqueSeverity.ADVISORY, "Desc3"),
            ],
            summary="Test summary",
        )
        j = c.to_json()
        c2 = AresCritique.from_json(j)
        assert c2.manifest_hash == "hash123"
        assert len(c2.findings) == 3
        assert c2.findings[0].evidence == "Ev1"


# AthenaVerdict tests

class TestAthenaVerdict:
    def test_valid_promote_verdict(self):
        v = AthenaVerdict(
            confidence=0.75,
            decision="PROMOTE",
            rationale="Strategy is sound with adequate risk controls.",
            research_packet=_valid_manifest_dict(),
            rubric_scores={
                "theoretical_soundness": 0.8,
                "critique_resolution": 0.7,
                "testability": 0.9,
                "risk_management": 0.6,
                "novelty": 0.7,
            },
        )
        v.validate()  # Should not raise

    def test_promote_without_packet_fails(self):
        v = AthenaVerdict(
            confidence=0.75,
            decision="PROMOTE",
            rationale="Looks good",
            research_packet=None,
        )
        with pytest.raises(ManifestValidationError, match="research_packet"):
            v.validate()

    def test_invalid_confidence_fails(self):
        v = AthenaVerdict(
            confidence=1.5,
            decision="REJECT",
            rationale="Rejected",
        )
        with pytest.raises(ManifestValidationError, match="confidence"):
            v.validate()

    def test_invalid_decision_fails(self):
        v = AthenaVerdict(
            confidence=0.5,
            decision="MAYBE",
            rationale="Unsure",
        )
        with pytest.raises(ManifestValidationError, match="decision"):
            v.validate()

    def test_empty_rationale_fails(self):
        v = AthenaVerdict(
            confidence=0.5,
            decision="REJECT",
            rationale="",
        )
        with pytest.raises(ManifestValidationError, match="rationale"):
            v.validate()

    def test_verdict_roundtrip(self):
        v = AthenaVerdict(
            confidence=0.72,
            decision="PROMOTE",
            rationale="Strategy approved after debate.",
            research_packet=_valid_manifest_dict(),
            conditions=["Add stop-loss at 2 ATR"],
            rubric_scores={"theoretical_soundness": 0.8, "critique_resolution": 0.7},
        )
        j = v.to_json()
        v2 = AthenaVerdict.from_json(j)
        assert v2.confidence == 0.72
        assert v2.decision == "PROMOTE"
        assert len(v2.conditions) == 1

    def test_overall_quality_fallback(self):
        # When LLM outputs overall_quality (0-100) instead of confidence (0-1).
        v = AthenaVerdict.from_dict({
            "overall_quality": 80,
            "summary": "Strategy is solid.",
        })
        assert abs(v.confidence - 0.80) < 0.01
        assert v.decision == "PROMOTE"  # 0.80 >= 0.6 → inferred PROMOTE

    def test_findings_ratings_to_rubric(self):
        # When LLM outputs findings with category/rating instead of rubric_scores.
        v = AthenaVerdict.from_dict({
            "overall_quality": 80,
            "findings": [
                {"category": "OVERFITTING", "rating": 5, "description": "Resolved."},
                {"category": "REGIME_DEPENDENCY", "rating": 5, "description": "Resolved."},
                {"category": "DRAWDOWN_RISK", "rating": 3, "description": "Advisory."},
                {"category": "EXECUTION_RISK", "rating": 3, "description": "Advisory."},
            ],
            "summary": "Strategy addressed major issues.",
        })
        # overall_quality=80 → confidence=0.80
        assert abs(v.confidence - 0.80) < 0.01
        assert v.decision == "PROMOTE"
        # Rubric built from findings
        assert len(v.rubric_scores) == 4
        assert v.rubric_scores["overfitting"] == 1.0  # 5/5
        assert abs(v.rubric_scores["drawdown_risk"] - 0.60) < 0.01  # 3/5

    def test_exact_llm_verdict_format(self):
        # Exact format observed from LLM in production runs.
        v = AthenaVerdict.from_dict({
            "overall_quality": 80,
            "findings": [
                {
                    "category": "OVERFITTING",
                    "rating": 5,
                    "description": "The strategy has successfully resolved the issue.",
                },
                {
                    "category": "REGIME_DEPENDENCY",
                    "rating": 5,
                    "description": "Removed reliance on high-volatility regimes.",
                },
                {
                    "category": "DRAWDOWN_RISK",
                    "rating": 3,
                    "description": "Incorporated spread_pct and RSI for more robust exit.",
                },
                {
                    "category": "EXECUTION_RISK",
                    "rating": 3,
                    "description": "Used vwap in conjunction with ATR.",
                },
            ],
            "summary": "The revised strategy has effectively addressed the OVERFITTING "
                       "and REGIME_DEPENDENCY issues while mitigating DRAWDOWN_RISK "
                       "and EXECUTION_RISK.",
        })
        assert v.decision == "PROMOTE"
        assert abs(v.confidence - 0.80) < 0.01
        assert v.rationale.startswith("The revised strategy")

    def test_exact_llm_verdict_format_2(self):
        # Exact format observed from LLM in production runs (verdict + reasoning array + summary).
        v = AthenaVerdict.from_dict({
            "verdict": "REJECTED",
            "reasoning": [
                {
                    "category": "REGIME_DEPENDENCY",
                    "severity": "BLOCKER",
                    "description": "The strategy still relies on high volatility regimes.",
                    "recommendation": "Further refine the dynamic regime detection.",
                }
            ],
            "summary": "The VRP put spread strategy has made progress but significant issues remain.",
        })
        assert v.decision == "REJECT"
        assert v.confidence == 0.0
        assert v.rationale == "The VRP put spread strategy has made progress but significant issues remain."

    def test_rationale_fallback_to_array_with_description(self):
        # When reasoning array has 'description' but no 'comment' and no summary is provided.
        v = AthenaVerdict.from_dict({
            "decision": "REJECT",
            "reasoning": [
                {"category": "FOO", "description": "Bar."},
                {"category": "BAZ", "description": "Qux."}
            ]
        })
        assert v.rationale == "Bar. Qux."


# JSON extraction tests

class TestJsonExtraction:
    def test_extract_from_manifest_tags(self):
        text = '<thought>reasoning</thought>\n<manifest>{"name": "test"}</manifest>'
        result = extract_json_from_response(text)
        assert result == {"name": "test"}

    def test_extract_from_critique_tags(self):
        text = '<thought>attack</thought>\n<critique>{"findings": []}</critique>'
        result = extract_json_from_response(text)
        assert result == {"findings": []}

    def test_extract_from_verdict_tags(self):
        text = '<thought>judge</thought>\n<verdict>{"confidence": 0.8}</verdict>'
        result = extract_json_from_response(text)
        assert result == {"confidence": 0.8}

    def test_extract_from_fenced_block(self):
        text = 'Here is my analysis:\n```json\n{"name": "test"}\n```\nDone.'
        result = extract_json_from_response(text)
        assert result == {"name": "test"}

    def test_extract_raw_json(self):
        text = 'Some text {"key": "value"} more text'
        result = extract_json_from_response(text)
        assert result == {"key": "value"}

    def test_no_json_returns_none(self):
        text = "Just plain text with no JSON objects."
        result = extract_json_from_response(text)
        assert result is None

    def test_nested_json_extracted(self):
        data = {"outer": {"inner": [1, 2, 3]}, "flag": True}
        text = f"Response: {json.dumps(data)}"
        result = extract_json_from_response(text)
        assert result == data
