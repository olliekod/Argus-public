"""Tests for Pantheon Intelligence Engine — roles, context injection, and orchestrator integration."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from src.agent.argus_orchestrator import (
    ArgusOrchestrator,
    CaseStage,
    UPGRADE_MODEL,
)
from src.agent.delphi import DelphiToolRegistry
from src.agent.pantheon.roles import (
    ARES,
    ATHENA,
    ESCALATION_CLAUDE,
    ESCALATION_LOCAL_14B,
    ESCALATION_LOCAL_32B,
    PROMETHEUS,
    ContextInjector,
    PantheonRole,
    build_stage_prompt,
    get_role_for_stage,
    parse_critique_response,
    parse_manifest_response,
    parse_verdict_response,
)
from src.agent.runtime_controller import RuntimeController
from src.agent.zeus import RuntimeMode, ZeusPolicyEngine
from src.core.config import ZeusConfig
from src.core.manifests import (
    HADES_INDICATOR_CATALOG,
    ManifestValidationError,
    ManifestStatus,
    StrategyManifest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeResourceManager:
    gpu_enabled: bool = True


def _make_stack(tmp_path, **zeus_kwargs):
    db_path = str(tmp_path / "argus.db")
    zeus_cfg = ZeusConfig(governance_db_path=db_path, **zeus_kwargs)
    zeus = ZeusPolicyEngine(zeus_cfg)
    delphi = DelphiToolRegistry(zeus, role_tool_allowlist={"Argus": {"*"}})
    rm = _FakeResourceManager()
    runtime = RuntimeController(zeus, rm)
    return zeus, delphi, runtime


def _valid_manifest_json() -> str:
    """A valid strategy manifest as JSON."""
    return json.dumps({
        "name": "Test Momentum",
        "objective": "Capture momentum in uptrends",
        "signals": ["ema", "rsi", "atr"],
        "entry_logic": {
            "op": "AND",
            "children": [
                {"op": "GT", "left": "rsi", "right": 55},
                {"op": "IN_REGIME", "left": "vol_regime", "right": "VOL_LOW"},
            ],
        },
        "exit_logic": {"op": "LT", "left": "rsi", "right": 45},
        "parameters": {
            "rsi_period": {"min": 10, "max": 20, "step": 2},
            "ema_fast": 12,
        },
        "direction": "LONG",
        "universe": ["IBIT"],
        "regime_filters": {"vol_regime": ["VOL_LOW", "VOL_NORMAL"]},
        "timeframe": 60,
        "holding_period": "intraday",
        "risk_per_trade_pct": 0.02,
    })


def _manifest_hash_for_test() -> str:
    """Compute the hash of the canonical test manifest so critiques match."""
    m = parse_manifest_response(f"<manifest>{_valid_manifest_json()}</manifest>")
    return m.compute_hash()


def _valid_critique_json(manifest_hash: str = "") -> str:
    """A valid Ares critique as JSON."""
    if not manifest_hash:
        manifest_hash = _manifest_hash_for_test()
    return json.dumps({
        "manifest_hash": manifest_hash,
        "findings": [
            {
                "category": "OVERFITTING",
                "severity": "ADVISORY",
                "description": "RSI period range could lead to curve-fitting.",
                "evidence": "Only 10 parameter combinations tested.",
                "recommendation": "Expand out-of-sample window.",
            },
            {
                "category": "REGIME_DEPENDENCY",
                "severity": "BLOCKER",
                "description": "Strategy only operates in VOL_LOW.",
                "evidence": "Regime filter excludes 60% of market time.",
                "recommendation": "Test across all vol regimes.",
            },
            {
                "category": "EXECUTION_RISK",
                "severity": "ADVISORY",
                "description": "Market orders may experience slippage.",
                "evidence": "No slippage model in parameters.",
                "recommendation": "Add slippage sensitivity sweep.",
            },
        ],
        "summary": "Moderate risk. Regime dependency is a blocker.",
    })


def _valid_verdict_json() -> str:
    """A valid Athena verdict as JSON."""
    return json.dumps({
        "confidence": 0.72,
        "decision": "PROMOTE",
        "rationale": "Strategy is sound after revision. Regime filter expanded.",
        "research_packet": {
            "name": "Test Momentum",
            "objective": "Capture momentum in uptrends",
            "signals": ["ema", "rsi", "atr"],
            "entry_logic": {"op": "GT", "left": "rsi", "right": 55},
            "exit_logic": {"op": "LT", "left": "rsi", "right": 45},
            "parameters": {"rsi_period": {"min": 10, "max": 20, "step": 2}},
            "direction": "LONG",
            "universe": ["IBIT"],
        },
        "unresolved_blockers": [],
        "conditions": ["Monitor regime transitions during backtest"],
        "rubric_scores": {
            "theoretical_soundness": 0.8,
            "critique_resolution": 0.7,
            "testability": 0.85,
            "risk_management": 0.6,
            "novelty": 0.65,
        },
    })


# ---------------------------------------------------------------------------
# PantheonRole tests
# ---------------------------------------------------------------------------

class TestPantheonRole:
    def test_prometheus_defined(self):
        assert PROMETHEUS.name == "Prometheus"
        assert PROMETHEUS.escalation_priority == ESCALATION_LOCAL_32B
        assert "signals" in str(PROMETHEUS.output_schema)

    def test_ares_defined(self):
        assert ARES.name == "Ares"
        assert ARES.escalation_priority == ESCALATION_LOCAL_32B
        assert "findings" in str(ARES.output_schema)

    def test_athena_defined(self):
        assert ATHENA.name == "Athena"
        assert ATHENA.escalation_priority == ESCALATION_CLAUDE
        assert "confidence" in str(ATHENA.output_schema)

    def test_role_system_prompt_builds(self):
        ctx = ContextInjector()
        prompt = PROMETHEUS.build_system_prompt(ctx)
        assert "Prometheus" in prompt
        assert "Strategy Manifest" in prompt
        # Check that indicator catalog is injected
        for ind in ["ema", "rsi", "atr"]:
            assert ind in prompt

    def test_ares_system_prompt_contains_attack_vectors(self):
        ctx = ContextInjector()
        prompt = ARES.build_system_prompt(ctx)
        assert "Overfitting" in prompt or "OVERFITTING" in prompt
        assert "look-ahead" in prompt.lower() or "LOOK_AHEAD" in prompt
        assert "wash" in prompt.lower() or "WASH" in prompt

    def test_athena_system_prompt_contains_rubric(self):
        ctx = ContextInjector()
        prompt = ATHENA.build_system_prompt(ctx)
        assert "theoretical_soundness" in prompt
        assert "critique_resolution" in prompt
        assert "testability" in prompt
        assert "0.6" in prompt  # threshold


# ---------------------------------------------------------------------------
# ContextInjector tests
# ---------------------------------------------------------------------------

class TestContextInjector:
    def test_default_indicators(self):
        ctx = ContextInjector()
        assert set(ctx.available_indicators) == set(HADES_INDICATOR_CATALOG)

    def test_set_regime_context(self):
        ctx = ContextInjector()
        ctx.set_regime_context(vol_regime="VOL_HIGH", trend_regime="TREND_DOWN")
        assert ctx.regime_context["vol_regime"] == "VOL_HIGH"
        assert ctx.regime_context["trend_regime"] == "TREND_DOWN"

    def test_format_regime_block_with_data(self):
        ctx = ContextInjector()
        ctx.set_regime_context(vol_regime="VOL_LOW")
        block = ctx.format_regime_block()
        assert "VOL_LOW" in block
        assert "Current Market Regime" in block

    def test_format_regime_block_without_data(self):
        ctx = ContextInjector()
        block = ctx.format_regime_block()
        assert "UNKNOWN" in block

    def test_add_failure_log(self):
        ctx = ContextInjector()
        ctx.add_failure_log("case_001", "Overfitting detected", "TestStrat")
        assert len(ctx.failure_logs) == 1
        block = ctx.format_failure_logs()
        assert "Overfitting" in block
        assert "case_001" in block

    def test_failure_log_cap_at_20(self):
        ctx = ContextInjector()
        for i in range(25):
            ctx.add_failure_log(f"case_{i}", f"reason_{i}")
        assert len(ctx.failure_logs) == 20

    def test_format_full_context(self):
        ctx = ContextInjector()
        ctx.set_regime_context(vol_regime="VOL_SPIKE")
        ctx.add_failure_log("case_x", "Bad strategy")
        full = ctx.format_full_context()
        assert "VOL_SPIKE" in full
        assert "Available Hades Indicators" in full
        assert "Available Regime Filters" in full
        assert "Bad strategy" in full

    def test_regime_context_injected_into_prompt(self):
        ctx = ContextInjector()
        ctx.set_regime_context(vol_regime="VOL_SPIKE", trend_regime="TREND_DOWN")
        prompt = PROMETHEUS.build_system_prompt(ctx)
        assert "VOL_SPIKE" in prompt
        assert "TREND_DOWN" in prompt


# ---------------------------------------------------------------------------
# Stage mapping tests
# ---------------------------------------------------------------------------

class TestStageMapping:
    def test_stage_1_returns_prometheus(self):
        assert get_role_for_stage(1).name == "Prometheus"

    def test_stage_2_returns_ares(self):
        assert get_role_for_stage(2).name == "Ares"

    def test_stage_3_returns_prometheus(self):
        assert get_role_for_stage(3).name == "Prometheus"

    def test_stage_4_returns_ares(self):
        assert get_role_for_stage(4).name == "Ares"

    def test_stage_5_returns_athena(self):
        assert get_role_for_stage(5).name == "Athena"

    def test_invalid_stage_raises(self):
        with pytest.raises(ValueError, match="No role"):
            get_role_for_stage(99)


# ---------------------------------------------------------------------------
# build_stage_prompt tests
# ---------------------------------------------------------------------------

class TestBuildStagePrompt:
    def test_stage_1_prompt_contains_objective(self):
        ctx = ContextInjector()
        messages = build_stage_prompt(1, "Find momentum strategy", ctx, [])
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "Find momentum strategy" in messages[1]["content"]
        assert "Prometheus" in messages[0]["content"]

    def test_stage_2_prompt_contains_proposal(self):
        ctx = ContextInjector()
        artifacts = [{"role": "Prometheus", "stage": 1, "content": "My proposal text"}]
        messages = build_stage_prompt(2, "Test objective", ctx, artifacts)
        assert "My proposal text" in messages[1]["content"]
        assert "Ares" in messages[0]["content"]

    def test_stage_3_prompt_contains_critique(self):
        ctx = ContextInjector()
        artifacts = [
            {"role": "Prometheus", "stage": 1, "content": "Original proposal"},
            {"role": "Ares", "stage": 2, "content": "Critique content"},
        ]
        messages = build_stage_prompt(3, "Test objective", ctx, artifacts)
        assert "Original proposal" in messages[1]["content"]
        assert "Critique content" in messages[1]["content"]

    def test_stage_5_prompt_contains_full_debate(self):
        ctx = ContextInjector()
        artifacts = [
            {"role": "Prometheus", "stage": 1, "content": "Proposal"},
            {"role": "Ares", "stage": 2, "content": "Critique"},
            {"role": "Prometheus", "stage": 3, "content": "Revision"},
            {"role": "Ares", "stage": 4, "content": "Final attack"},
        ]
        messages = build_stage_prompt(5, "Test", ctx, artifacts)
        assert "Proposal" in messages[1]["content"]
        assert "Final attack" in messages[1]["content"]
        assert "Athena" in messages[0]["content"]

    def test_stage_prompts_include_indicator_catalog(self):
        ctx = ContextInjector()
        messages = build_stage_prompt(1, "Test", ctx, [])
        system = messages[0]["content"]
        for indicator in ["ema", "rsi", "atr", "macd"]:
            assert indicator in system


# ---------------------------------------------------------------------------
# Response parser tests
# ---------------------------------------------------------------------------

class TestParseManifestResponse:
    def test_parse_valid_manifest_in_tags(self):
        response = (
            "<thought>I'll create a momentum strategy.</thought>\n"
            f"<manifest>{_valid_manifest_json()}</manifest>"
        )
        manifest = parse_manifest_response(response)
        assert manifest.name == "Test Momentum"
        assert "ema" in manifest.signals

    def test_parse_manifest_in_code_block(self):
        response = (
            "Here is my strategy:\n"
            f"```json\n{_valid_manifest_json()}\n```"
        )
        manifest = parse_manifest_response(response)
        assert manifest.name == "Test Momentum"

    def test_parse_invalid_json_raises(self):
        response = "<thought>thinking</thought>\n<manifest>not valid json</manifest>"
        with pytest.raises(ManifestValidationError):
            parse_manifest_response(response)

    def test_parse_no_json_raises(self):
        response = "I couldn't generate a strategy. Here are my thoughts."
        with pytest.raises(ManifestValidationError, match="no valid JSON"):
            parse_manifest_response(response)

    def test_parse_manifest_with_invalid_signals_raises(self):
        bad = json.loads(_valid_manifest_json())
        bad["signals"] = ["nonexistent"]
        response = f"<manifest>{json.dumps(bad)}</manifest>"
        with pytest.raises(ManifestValidationError, match="nonexistent"):
            parse_manifest_response(response)


class TestParseCritiqueResponse:
    def test_parse_valid_critique(self):
        expected_hash = _manifest_hash_for_test()
        response = (
            "<thought>Analyzing the proposal...</thought>\n"
            f"<critique>{_valid_critique_json()}</critique>"
        )
        critique = parse_critique_response(response, expected_hash)
        assert len(critique.findings) == 3
        assert critique.has_blockers
        assert len(critique.blockers) == 1

    def test_parse_critique_sets_manifest_hash(self):
        data = json.loads(_valid_critique_json())
        del data["manifest_hash"]
        response = f"<critique>{json.dumps(data)}</critique>"
        critique = parse_critique_response(response, "my_hash")
        assert critique.manifest_hash == "my_hash"

    def test_parse_no_critique_raises(self):
        with pytest.raises(ManifestValidationError, match="no valid JSON"):
            parse_critique_response("Just text, no JSON.")


class TestParseVerdictResponse:
    def test_parse_valid_verdict(self):
        response = (
            "<thought>After reviewing the debate...</thought>\n"
            f"<verdict>{_valid_verdict_json()}</verdict>"
        )
        verdict = parse_verdict_response(response)
        assert verdict.confidence == 0.72
        assert verdict.decision == "PROMOTE"
        assert verdict.research_packet is not None
        assert "theoretical_soundness" in verdict.rubric_scores

    def test_parse_reject_verdict(self):
        data = json.loads(_valid_verdict_json())
        data["decision"] = "REJECT"
        data["confidence"] = 0.3
        data["research_packet"] = None
        data["rationale"] = "Too many unresolved blockers."
        response = f"<verdict>{json.dumps(data)}</verdict>"
        verdict = parse_verdict_response(response)
        assert verdict.decision == "REJECT"
        assert verdict.confidence == 0.3


# ---------------------------------------------------------------------------
# Orchestrator integration tests — structured research
# ---------------------------------------------------------------------------

class TestOrchestratorStructuredResearch:
    """Tests for the Pantheon-integrated research handler in ArgusOrchestrator."""

    @pytest.mark.asyncio
    async def test_structured_research_runs_all_stages(self, tmp_path):
        """Full debate produces output with all 5 stages."""
        zeus, delphi, runtime = _make_stack(tmp_path)

        stage_counter = {"count": 0}

        async def _structured_llm(messages, model):
            stage_counter["count"] += 1
            system = messages[0]["content"] if messages else ""

            if "You are Athena" in system:
                return (
                    f"<thought>Adjudicating.</thought>\n"
                    f"<verdict>{_valid_verdict_json()}</verdict>"
                )
            elif "You are Ares" in system and "FINAL ATTACK" in system:
                return (
                    f"<thought>Final attack.</thought>\n"
                    f"<critique>{_valid_critique_json()}</critique>"
                )
            elif "You are Ares" in system:
                return (
                    f"<thought>Attacking the proposal.</thought>\n"
                    f"<critique>{_valid_critique_json()}</critique>"
                )
            elif "You are Prometheus" in system and "REVISING" in system:
                return (
                    f"<thought>Revising based on critique.</thought>\n"
                    f"<manifest>{_valid_manifest_json()}</manifest>"
                )
            elif "You are Prometheus" in system:
                return (
                    f"<thought>Creating initial proposal.</thought>\n"
                    f"<manifest>{_valid_manifest_json()}</manifest>"
                )
            return "Generic response"

        orch = ArgusOrchestrator(zeus, delphi, runtime, llm_call=_structured_llm)
        response = await orch.chat("research a momentum strategy")

        assert "Stage 1" in response
        assert "Stage 2" in response
        assert "Stage 3" in response
        assert "Stage 4" in response
        assert "Stage 5" in response
        assert "Prometheus" in response
        assert "Ares" in response
        assert "Athena" in response
        assert stage_counter["count"] == 5

    @pytest.mark.asyncio
    async def test_structured_research_promote_verdict(self, tmp_path):
        """A PROMOTE verdict includes backtest config in output."""
        zeus, delphi, runtime = _make_stack(tmp_path)

        async def _promoting_llm(messages, model):
            system = messages[0]["content"] if messages else ""
            if "You are Athena" in system:
                return f"<verdict>{_valid_verdict_json()}</verdict>"
            elif "You are Ares" in system:
                return f"<critique>{_valid_critique_json()}</critique>"
            elif "You are Prometheus" in system:
                return f"<manifest>{_valid_manifest_json()}</manifest>"
            return "response"

        orch = ArgusOrchestrator(zeus, delphi, runtime, llm_call=_promoting_llm)
        response = await orch.chat("research strategy proposal")

        assert "PROMOTE" in response
        assert "0.72" in response
        assert "Backtest Configuration" in response

    @pytest.mark.asyncio
    async def test_structured_research_reject_verdict(self, tmp_path):
        """A REJECT verdict is properly reported."""
        zeus, delphi, runtime = _make_stack(tmp_path)

        reject_verdict = json.dumps({
            "confidence": 0.3,
            "decision": "REJECT",
            "rationale": "Too many unresolved blockers remain.",
            "research_packet": None,
            "unresolved_blockers": ["Regime dependency not addressed"],
            "conditions": [],
            "rubric_scores": {
                "theoretical_soundness": 0.4,
                "critique_resolution": 0.2,
                "testability": 0.5,
                "risk_management": 0.3,
                "novelty": 0.3,
            },
        })

        async def _rejecting_llm(messages, model):
            system = messages[0]["content"] if messages else ""
            if "You are Athena" in system:
                return f"<verdict>{reject_verdict}</verdict>"
            elif "You are Ares" in system:
                return f"<critique>{_valid_critique_json()}</critique>"
            elif "You are Prometheus" in system:
                return f"<manifest>{_valid_manifest_json()}</manifest>"
            return "response"

        orch = ArgusOrchestrator(zeus, delphi, runtime, llm_call=_rejecting_llm)
        response = await orch.chat("research strategy proposal")

        assert "REJECT" in response
        assert "0.30" in response

    @pytest.mark.asyncio
    async def test_invalid_prometheus_json_triggers_escalation(self, tmp_path):
        """When Prometheus returns invalid JSON, orchestrator escalates to 32B."""
        zeus, delphi, runtime = _make_stack(tmp_path)

        calls: List[str] = []

        async def _failing_then_succeeding_llm(messages, model):
            calls.append(model)
            system = messages[0]["content"] if messages else ""

            if "You are Athena" in system:
                return f"<verdict>{_valid_verdict_json()}</verdict>"
            elif "You are Ares" in system:
                return f"<critique>{_valid_critique_json()}</critique>"
            elif "You are Prometheus" in system and "REVISING" not in system:
                # First call with 32B (since Prometheus already gets 32B)
                # fails, retry should use 32B
                if calls.count(model) <= 1 and model == UPGRADE_MODEL:
                    return "I can't format JSON properly."
                return f"<manifest>{_valid_manifest_json()}</manifest>"
            elif "You are Prometheus" in system:
                return f"<manifest>{_valid_manifest_json()}</manifest>"
            return "response"

        orch = ArgusOrchestrator(zeus, delphi, runtime, llm_call=_failing_then_succeeding_llm)
        response = await orch.chat("research a strategy idea")

        # Should still complete all stages
        assert "Stage 5" in response

    @pytest.mark.asyncio
    async def test_invalid_prometheus_json_logs_parse_warning(self, tmp_path):
        """If manifest parse fails even after escalation, a warning is logged."""
        zeus, delphi, runtime = _make_stack(tmp_path)

        async def _always_bad_prometheus_llm(messages, model):
            system = messages[0]["content"] if messages else ""
            if "You are Athena" in system:
                return f"<verdict>{_valid_verdict_json()}</verdict>"
            elif "You are Ares" in system:
                return f"<critique>{_valid_critique_json()}</critique>"
            elif "You are Prometheus" in system:
                return "I refuse to output JSON. Here's my reasoning instead."
            return "response"

        orch = ArgusOrchestrator(zeus, delphi, runtime, llm_call=_always_bad_prometheus_llm)
        response = await orch.chat("research volatility strategy")

        # Should still complete but with parse warnings
        assert "Parse Warning" in response

    @pytest.mark.asyncio
    async def test_context_injector_used_in_research(self, tmp_path):
        """Verify the orchestrator's context injector is used in research prompts."""
        zeus, delphi, runtime = _make_stack(tmp_path)

        captured_systems: List[str] = []

        async def _capture_llm(messages, model):
            system = messages[0]["content"] if messages else ""
            captured_systems.append(system)
            if "You are Athena" in system:
                return f"<verdict>{_valid_verdict_json()}</verdict>"
            elif "You are Ares" in system:
                return f"<critique>{_valid_critique_json()}</critique>"
            elif "You are Prometheus" in system:
                return f"<manifest>{_valid_manifest_json()}</manifest>"
            return "response"

        orch = ArgusOrchestrator(zeus, delphi, runtime, llm_call=_capture_llm)
        orch.context_injector.set_regime_context(vol_regime="VOL_SPIKE")
        response = await orch.chat("research during high vol")

        # At least one system prompt should contain the injected regime
        assert any("VOL_SPIKE" in s for s in captured_systems)

    @pytest.mark.asyncio
    async def test_research_blocked_in_data_only(self, tmp_path):
        """Research is blocked when system is in DATA_ONLY mode."""
        zeus, delphi, runtime = _make_stack(tmp_path)
        await zeus.set_mode(RuntimeMode.DATA_ONLY)
        await runtime.transition_to(RuntimeMode.DATA_ONLY)

        async def _never_called_llm(messages, model):
            raise AssertionError("LLM should not be called in DATA_ONLY mode")

        orch = ArgusOrchestrator(zeus, delphi, runtime, llm_call=_never_called_llm)
        response = await orch.chat("research a new strategy")

        assert "DATA_ONLY" in response
        assert "restricted" in response.lower() or "switch" in response.lower()

    @pytest.mark.asyncio
    async def test_escalation_priority_affects_model_selection(self, tmp_path):
        """Roles with higher escalation_priority use more powerful models."""
        zeus, delphi, runtime = _make_stack(tmp_path)

        models_used: List[str] = []

        async def _model_tracking_llm(messages, model):
            models_used.append(model)
            system = messages[0]["content"] if messages else ""
            if "You are Athena" in system:
                return f"<verdict>{_valid_verdict_json()}</verdict>"
            elif "You are Ares" in system:
                return f"<critique>{_valid_critique_json()}</critique>"
            elif "You are Prometheus" in system:
                return f"<manifest>{_valid_manifest_json()}</manifest>"
            return "response"

        orch = ArgusOrchestrator(zeus, delphi, runtime, llm_call=_model_tracking_llm)
        await orch.chat("research momentum signals")

        # Prometheus and Ares should use at least 32B (escalation_priority=1)
        # Default model is 14B, so stages should escalate to 32B
        assert UPGRADE_MODEL in models_used

    @pytest.mark.asyncio
    async def test_active_case_cleared_after_research(self, tmp_path):
        """active_case is None after research completes."""
        zeus, delphi, runtime = _make_stack(tmp_path)

        async def _simple_llm(messages, model):
            system = messages[0]["content"] if messages else ""
            if "You are Athena" in system:
                return f"<verdict>{_valid_verdict_json()}</verdict>"
            elif "You are Ares" in system:
                return f"<critique>{_valid_critique_json()}</critique>"
            elif "You are Prometheus" in system:
                return f"<manifest>{_valid_manifest_json()}</manifest>"
            return "response"

        orch = ArgusOrchestrator(zeus, delphi, runtime, llm_call=_simple_llm)
        await orch.chat("research strategy X")
        assert orch.active_case is None
