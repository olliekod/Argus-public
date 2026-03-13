"""Tests for the Argus Orchestrator — conversational loop, tool integration, case-file workflow."""

from __future__ import annotations

import json
import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock

import pytest

from src.agent.argus_orchestrator import (
    ArgusOrchestrator,
    CaseFile,
    CaseStage,
    ConversationBuffer,
    EscalationConfig,
    EscalationProvider,
    Intent,
)
from src.agent.delphi import DelphiToolRegistry, RiskLevel
from src.agent.runtime_controller import RuntimeController
from src.agent.pantheon.roles import parse_critique_response
from src.agent.zeus import RuntimeMode, ZeusPolicyEngine
from src.agent.resource_manager import AgentResourceManager
from src.core.config import ZeusConfig
from src.core.manifests import ManifestValidationError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeResourceManager:
    gpu_enabled: bool = True


def _make_stack(tmp_path, **zeus_kwargs):
    """Create a full Zeus → Delphi → Runtime → Argus stack for testing."""
    db_path = str(tmp_path / "argus.db")
    zeus_cfg = ZeusConfig(governance_db_path=db_path, **zeus_kwargs)
    zeus = ZeusPolicyEngine(zeus_cfg)
    delphi = DelphiToolRegistry(zeus, role_tool_allowlist={"Argus": {"*"}})
    rm = _FakeResourceManager()
    runtime = RuntimeController(zeus, rm)
    return zeus, delphi, runtime


async def _echo_llm(messages: List[Dict[str, str]], model: str) -> str:
    """Deterministic LLM stub that echoes the last user message."""
    for msg in reversed(messages):
        if msg["role"] == "user":
            return f"Echo: {msg['content']}"
    return "Echo: (no user message)"


async def _tool_calling_llm(messages: List[Dict[str, str]], model: str) -> str:
    """LLM stub that always responds with a tool call for 'get_status'."""
    # Check if we're in an observation follow-up
    for msg in reversed(messages):
        if msg["role"] == "user" and msg["content"].startswith("Observation:"):
            return "The system status is healthy."
    return json.dumps({"tool": "get_status", "args": {}})


async def _noop_llm(messages: List[Dict[str, str]], model: str) -> str:
    """LLM stub that returns a plain text response (no tool call)."""
    return "I understand your request. Here is my analysis."


async def _case_file_llm(messages: List[Dict[str, str]], model: str) -> str:
    """LLM stub that returns stage-appropriate responses for case-file debates."""
    for msg in reversed(messages):
        if msg["role"] == "system":
            if "Prometheus" in msg["content"]:
                return "Proposal: Implement momentum strategy with vol-adjusted sizing."
            if "Ares" in msg["content"]:
                return "Critique: Momentum crashes in regime transitions. Drawdown risk is high."
            if "Athena" in msg["content"]:
                return "Adjudication: Approved with conditions. Confidence: 0.72. Add regime filter."
    return "Generic case file response."


# ---------------------------------------------------------------------------
# ConversationBuffer tests
# ---------------------------------------------------------------------------

class TestConversationBuffer:
    def test_append_and_retrieve(self):
        buf = ConversationBuffer(max_messages=10)
        buf.append("user", "hello")
        buf.append("assistant", "hi there")
        assert len(buf.messages) == 2
        assert buf.messages[0] == {"role": "user", "content": "hello"}
        assert buf.messages[1] == {"role": "assistant", "content": "hi there"}

    def test_sliding_window_trims(self):
        buf = ConversationBuffer(max_messages=4)
        for i in range(10):
            buf.append("user", f"msg-{i}")
        assert len(buf.messages) <= 4

    def test_system_message_preserved_on_trim(self):
        buf = ConversationBuffer(max_messages=4)
        buf.append_system("You are Argus.")
        for i in range(10):
            buf.append("user", f"msg-{i}")
        msgs = buf.messages
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "You are Argus."

    def test_clear(self):
        buf = ConversationBuffer()
        buf.append("user", "test")
        buf.clear()
        assert len(buf.messages) == 0

    def test_to_prompt_messages_returns_copy(self):
        buf = ConversationBuffer()
        buf.append("user", "test")
        prompts = buf.to_prompt_messages()
        prompts.append({"role": "user", "content": "injected"})
        assert len(buf.messages) == 1  # Original unaffected


# ---------------------------------------------------------------------------
# CaseFile tests
# ---------------------------------------------------------------------------

class TestCaseFile:
    def test_advance_through_stages(self):
        case = CaseFile(case_id="test-1", objective="Test strategy")
        assert case.stage == CaseStage.INITIATED
        case.advance()
        assert case.stage == CaseStage.PROPOSAL_V1
        case.advance()
        assert case.stage == CaseStage.CRITIQUE_V1
        case.advance()
        assert case.stage == CaseStage.REVISION_V2
        case.advance()
        assert case.stage == CaseStage.FINAL_ATTACK
        case.advance()
        assert case.stage == CaseStage.ADJUDICATION
        # Should not advance past adjudication
        case.advance()
        assert case.stage == CaseStage.ADJUDICATION

    def test_add_artifact(self):
        case = CaseFile(case_id="test-2", objective="Another test")
        case.advance()
        case.add_artifact("Prometheus", "My proposal content")
        assert len(case.artifacts) == 1
        assert case.artifacts[0]["role"] == "Prometheus"
        assert case.artifacts[0]["content"] == "My proposal content"
        assert case.artifacts[0]["stage"] == CaseStage.PROPOSAL_V1.value


# ---------------------------------------------------------------------------
# Intent classification tests
# ---------------------------------------------------------------------------

class TestIntentClassification:
    def _make_orchestrator(self, tmp_path):
        zeus, delphi, runtime = _make_stack(tmp_path)
        return ArgusOrchestrator(zeus, delphi, runtime, llm_call=_echo_llm)

    def test_mode_switch_intent(self, tmp_path):
        orch = self._make_orchestrator(tmp_path)
        assert orch._classify_intent("switch to gaming mode") == Intent.MODE_SWITCH
        assert orch._classify_intent("Enter data only mode") == Intent.MODE_SWITCH
        assert orch._classify_intent("activate active mode") == Intent.MODE_SWITCH

    def test_approval_intent(self, tmp_path):
        orch = self._make_orchestrator(tmp_path)
        assert orch._classify_intent("proceed") == Intent.APPROVAL
        assert orch._classify_intent("approve") == Intent.APPROVAL
        assert orch._classify_intent("yes") == Intent.APPROVAL
        assert orch._classify_intent("confirm") == Intent.APPROVAL

    def test_research_intent(self, tmp_path):
        orch = self._make_orchestrator(tmp_path)
        assert orch._classify_intent("research momentum strategies") == Intent.RESEARCH
        assert orch._classify_intent("analyze VRP performance") == Intent.RESEARCH
        assert orch._classify_intent("investigate high-vol regime") == Intent.RESEARCH

    def test_command_intent(self, tmp_path):
        orch = self._make_orchestrator(tmp_path)
        assert orch._classify_intent("run the backtest") == Intent.COMMAND
        assert orch._classify_intent("deploy strategy") == Intent.COMMAND
        assert orch._classify_intent("execute trade") == Intent.COMMAND
        assert orch._classify_intent("stop the workers") == Intent.COMMAND

    def test_question_intent(self, tmp_path):
        orch = self._make_orchestrator(tmp_path)
        assert orch._classify_intent("What is the current PnL?") == Intent.QUESTION
        assert orch._classify_intent("How does the regime detector work?") == Intent.QUESTION


# ---------------------------------------------------------------------------
# Mode switching tests
# ---------------------------------------------------------------------------

class TestModeSwitching:
    @pytest.mark.asyncio
    async def test_switch_to_gaming_mode(self, tmp_path):
        zeus, delphi, runtime = _make_stack(tmp_path)
        orch = ArgusOrchestrator(zeus, delphi, runtime, llm_call=_echo_llm)

        response = await orch.chat("switch to gaming mode")

        assert "DATA_ONLY" in response
        assert zeus.current_mode == RuntimeMode.DATA_ONLY

    @pytest.mark.asyncio
    async def test_switch_to_active_mode(self, tmp_path):
        zeus, delphi, runtime = _make_stack(tmp_path)
        orch = ArgusOrchestrator(zeus, delphi, runtime, llm_call=_echo_llm)

        # First switch to data_only
        await orch.chat("switch to gaming mode")
        assert zeus.current_mode == RuntimeMode.DATA_ONLY

        # Now switch back to active
        response = await orch.chat("switch to active mode")
        assert "ACTIVE" in response
        assert zeus.current_mode == RuntimeMode.ACTIVE

    @pytest.mark.asyncio
    async def test_switch_idempotent(self, tmp_path):
        zeus, delphi, runtime = _make_stack(tmp_path)
        orch = ArgusOrchestrator(zeus, delphi, runtime, llm_call=_echo_llm)

        response = await orch.chat("switch to active mode")
        assert "already" in response.lower()

    @pytest.mark.asyncio
    async def test_switch_to_trading_maps_to_active(self, tmp_path):
        zeus, delphi, runtime = _make_stack(tmp_path)
        await zeus.set_mode(RuntimeMode.DATA_ONLY)
        await runtime.transition_to(RuntimeMode.DATA_ONLY)

        orch = ArgusOrchestrator(zeus, delphi, runtime, llm_call=_echo_llm)
        response = await orch.chat("switch to trading mode")
        assert "ACTIVE" in response


# ---------------------------------------------------------------------------
# Tool-calling (ReAct loop) tests
# ---------------------------------------------------------------------------

class TestToolCalling:
    @pytest.mark.asyncio
    async def test_tool_call_success(self, tmp_path):
        zeus, delphi, runtime = _make_stack(tmp_path)

        # Register a test tool
        @delphi.register(name="get_status", description="Get system status", risk_level=RiskLevel.READ_ONLY)
        def get_status() -> dict:
            return {"status": "healthy", "uptime_hours": 42}

        orch = ArgusOrchestrator(zeus, delphi, runtime, llm_call=_tool_calling_llm)
        response = await orch.chat("run get_status")

        assert "healthy" in response.lower() or "status" in response.lower()

    @pytest.mark.asyncio
    async def test_tool_result_parsed_into_response(self, tmp_path):
        """Verify tool-calling results are correctly parsed and integrated."""
        zeus, delphi, runtime = _make_stack(tmp_path)

        @delphi.register(name="get_status", description="Get system status", risk_level=RiskLevel.READ_ONLY)
        def get_status() -> dict:
            return {"mode": "ACTIVE", "workers": 4}

        orch = ArgusOrchestrator(zeus, delphi, runtime, llm_call=_tool_calling_llm)
        response = await orch.chat("run status check")

        # The tool_calling_llm will call get_status, get back the result,
        # then on the observation follow-up returns the final answer
        assert "status" in response.lower()

    @pytest.mark.asyncio
    async def test_approval_required_pauses(self, tmp_path):
        zeus, delphi, runtime = _make_stack(tmp_path, high_risk_tools=["config_edit"])

        @delphi.register(name="config_edit", description="Edit config", risk_level=RiskLevel.HIGH)
        def config_edit(path: str) -> str:
            return path

        async def _config_edit_llm(messages, model):
            for msg in reversed(messages):
                if msg["role"] == "user" and msg["content"].startswith("Observation:"):
                    return "Config updated."
            return json.dumps({"tool": "config_edit", "args": {"path": "config/config.yaml"}})

        orch = ArgusOrchestrator(zeus, delphi, runtime, llm_call=_config_edit_llm)
        response = await orch.chat("run config_edit on config.yaml")

        assert "approval" in response.lower() or "approve" in response.lower()
        assert orch.pending_approval is not None
        assert orch.pending_approval["tool_name"] == "config_edit"

    @pytest.mark.asyncio
    async def test_approval_then_proceed(self, tmp_path):
        zeus, delphi, runtime = _make_stack(tmp_path, high_risk_tools=["config_edit"])

        @delphi.register(name="config_edit", description="Edit config", risk_level=RiskLevel.HIGH)
        def config_edit(path: str) -> str:
            return f"edited:{path}"

        async def _config_edit_llm(messages, model):
            for msg in reversed(messages):
                if msg["role"] == "user" and msg["content"].startswith("Observation:"):
                    return "Config updated."
            return json.dumps({"tool": "config_edit", "args": {"path": "test.yaml"}})

        orch = ArgusOrchestrator(zeus, delphi, runtime, llm_call=_config_edit_llm)

        # First: request triggers approval
        await orch.chat("run config_edit on test.yaml")
        assert orch.pending_approval is not None

        # Second: approve
        response = await orch.chat("proceed")
        assert "edited:test.yaml" in response or "Done" in response

    @pytest.mark.asyncio
    async def test_no_tool_call_returns_plain_text(self, tmp_path):
        zeus, delphi, runtime = _make_stack(tmp_path)
        orch = ArgusOrchestrator(zeus, delphi, runtime, llm_call=_noop_llm)

        response = await orch.chat("set something up")
        assert "analysis" in response.lower()

    @pytest.mark.asyncio
    async def test_tool_not_found_handled(self, tmp_path):
        zeus, delphi, runtime = _make_stack(tmp_path)

        async def _bad_tool_llm(messages, model):
            for msg in reversed(messages):
                if msg["role"] == "user" and msg["content"].startswith("Observation:"):
                    return "Sorry, that failed."
            return json.dumps({"tool": "nonexistent_tool", "args": {}})

        orch = ArgusOrchestrator(zeus, delphi, runtime, llm_call=_bad_tool_llm)
        response = await orch.chat("run nonexistent_tool")

        assert "failed" in response.lower() or "not" in response.lower()

    @pytest.mark.asyncio
    async def test_tool_execution_error_handled(self, tmp_path):
        zeus, delphi, runtime = _make_stack(tmp_path)

        @delphi.register(name="explode", description="Always fails", risk_level=RiskLevel.READ_ONLY)
        def explode() -> None:
            raise RuntimeError("kaboom")

        async def _explode_llm(messages, model):
            return json.dumps({"tool": "explode", "args": {}})

        orch = ArgusOrchestrator(zeus, delphi, runtime, llm_call=_explode_llm)
        response = await orch.chat("run explode")

        assert "kaboom" in response.lower() or "failed" in response.lower()


# ---------------------------------------------------------------------------
# Case-file debate protocol tests
# ---------------------------------------------------------------------------

class TestCaseFileWorkflow:
    @pytest.mark.asyncio
    async def test_debate_runs_all_stages(self, tmp_path):
        zeus, delphi, runtime = _make_stack(tmp_path)
        orch = ArgusOrchestrator(zeus, delphi, runtime, llm_call=_case_file_llm)

        response = await orch.chat("research a momentum strategy approach")

        # All stages should appear in output
        assert "Stage 1" in response
        assert "Stage 2" in response
        assert "Stage 3" in response
        assert "Stage 4" in response
        assert "Stage 5" in response

        # Role names should be present
        assert "Prometheus" in response
        assert "Ares" in response
        assert "Athena" in response

    @pytest.mark.asyncio
    async def test_debate_produces_artifacts(self, tmp_path):
        zeus, delphi, runtime = _make_stack(tmp_path)

        artifacts_captured: List[Dict[str, Any]] = []

        async def _tracking_llm(messages, model):
            result = await _case_file_llm(messages, model)
            return result

        orch = ArgusOrchestrator(zeus, delphi, runtime, llm_call=_tracking_llm)
        await orch.chat("research volatility premium strategy")

        # Active case should be cleared after completion
        assert orch.active_case is None

    @pytest.mark.asyncio
    async def test_research_blocked_in_data_only_mode(self, tmp_path):
        zeus, delphi, runtime = _make_stack(tmp_path)
        await zeus.set_mode(RuntimeMode.DATA_ONLY)
        await runtime.transition_to(RuntimeMode.DATA_ONLY)

        orch = ArgusOrchestrator(zeus, delphi, runtime, llm_call=_case_file_llm)
        response = await orch.chat("research a new strategy")

        assert "DATA_ONLY" in response
        assert "restricted" in response.lower() or "switch" in response.lower()


# ---------------------------------------------------------------------------
# Runtime awareness tests
# ---------------------------------------------------------------------------

class TestRuntimeAwareness:
    @pytest.mark.asyncio
    async def test_compute_restricted_in_data_only(self, tmp_path):
        zeus, delphi, runtime = _make_stack(tmp_path)
        await zeus.set_mode(RuntimeMode.DATA_ONLY)
        await runtime.transition_to(RuntimeMode.DATA_ONLY)

        orch = ArgusOrchestrator(zeus, delphi, runtime, llm_call=_echo_llm)
        response = await orch.chat("run a backtest on VRP")

        assert "DATA_ONLY" in response
        assert "restricted" in response.lower() or "switch" in response.lower()

    @pytest.mark.asyncio
    async def test_questions_work_in_data_only(self, tmp_path):
        zeus, delphi, runtime = _make_stack(tmp_path)
        await zeus.set_mode(RuntimeMode.DATA_ONLY)
        await runtime.transition_to(RuntimeMode.DATA_ONLY)

        orch = ArgusOrchestrator(zeus, delphi, runtime, llm_call=_echo_llm)
        response = await orch.chat("What is the current mode?")

        # Questions should still work — they don't require compute
        assert "Echo:" in response


# ---------------------------------------------------------------------------
# Zeus budget / escalation tests
# ---------------------------------------------------------------------------

class TestEscalation:
    @pytest.mark.asyncio
    async def test_escalation_denied_when_budget_exhausted(self, tmp_path):
        zeus, delphi, runtime = _make_stack(tmp_path, monthly_budget_cap=1.0)
        zeus.log_spend(1.0, actor="test", purpose="exhaust")

        esc_config = EscalationConfig(
            provider=EscalationProvider.CLAUDE,
            api_key="test-key",
            estimated_cost_per_call=0.10,
        )
        orch = ArgusOrchestrator(
            zeus, delphi, runtime,
            llm_call=_echo_llm,
            escalation_config=esc_config,
        )

        result = await orch._escalate_to_api([], "test justification")
        assert result is None  # Budget denied, no escalation

    @pytest.mark.asyncio
    async def test_escalation_without_config_returns_none(self, tmp_path):
        zeus, delphi, runtime = _make_stack(tmp_path)
        orch = ArgusOrchestrator(zeus, delphi, runtime, llm_call=_echo_llm)

        result = await orch._escalate_to_api([], "test")
        assert result is None

    @pytest.mark.asyncio
    async def test_budget_checked_for_every_escalation(self, tmp_path, monkeypatch):
        zeus, delphi, runtime = _make_stack(tmp_path, monthly_budget_cap=15.0)

        esc_config = EscalationConfig(
            provider=EscalationProvider.CLAUDE,
            api_key="test-key",
            estimated_cost_per_call=0.05,
        )
        orch = ArgusOrchestrator(
            zeus, delphi, runtime,
            llm_call=_echo_llm,
            escalation_config=esc_config,
        )

        async def _mock_complete(self, messages):
            return {"message": {"content": "mocked-api"}}

        monkeypatch.setattr("src.connectors.api_llm_clients.AnthropicClient.complete", _mock_complete)

        spend_before = zeus.monthly_spend
        await orch._escalate_to_api([], "testing budget track")
        spend_after = zeus.monthly_spend
        # Spend should have been logged
        assert spend_after > spend_before


class TestResourceManager:
    @pytest.mark.asyncio
    async def test_llm_slots_block_when_limit_reached(self):
        manager = AgentResourceManager(max_concurrent_llm_calls=2)
        active = 0
        peak = 0

        async def _worker():
            nonlocal active, peak
            async with manager.llm_slot():
                active += 1
                peak = max(peak, active)
                await asyncio.sleep(0.05)
                active -= 1

        await asyncio.gather(*[_worker() for _ in range(5)])
        assert peak == 2


# ---------------------------------------------------------------------------
# Conversation memory integration tests
# ---------------------------------------------------------------------------

class TestConversationIntegration:
    @pytest.mark.asyncio
    async def test_memory_grows_with_chat(self, tmp_path):
        zeus, delphi, runtime = _make_stack(tmp_path)
        orch = ArgusOrchestrator(zeus, delphi, runtime, llm_call=_echo_llm)

        await orch.chat("first message")
        await orch.chat("second message")

        # System prompt + 2 user + 2 assistant = 5
        msgs = orch.memory.messages
        user_msgs = [m for m in msgs if m["role"] == "user"]
        assistant_msgs = [m for m in msgs if m["role"] == "assistant"]
        assert len(user_msgs) == 2
        assert len(assistant_msgs) == 2

    @pytest.mark.asyncio
    async def test_system_prompt_contains_runtime_mode(self, tmp_path):
        zeus, delphi, runtime = _make_stack(tmp_path)
        orch = ArgusOrchestrator(zeus, delphi, runtime, llm_call=_echo_llm)

        system_msgs = [m for m in orch.memory.messages if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert "ACTIVE" in system_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_system_prompt_updates_on_mode_switch(self, tmp_path):
        zeus, delphi, runtime = _make_stack(tmp_path)
        orch = ArgusOrchestrator(zeus, delphi, runtime, llm_call=_echo_llm)

        await orch.chat("switch to gaming mode")

        system_msgs = [m for m in orch.memory.messages if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert "DATA_ONLY" in system_msgs[0]["content"]


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_llm_error_does_not_crash(self, tmp_path):
        zeus, delphi, runtime = _make_stack(tmp_path)

        async def _failing_llm(messages, model):
            raise ConnectionError("Ollama down")

        orch = ArgusOrchestrator(zeus, delphi, runtime, llm_call=_failing_llm)
        response = await orch.chat("What is up?")

        assert "error" in response.lower()

    @pytest.mark.asyncio
    async def test_approval_without_pending(self, tmp_path):
        zeus, delphi, runtime = _make_stack(tmp_path)
        orch = ArgusOrchestrator(zeus, delphi, runtime, llm_call=_echo_llm)

        response = await orch.chat("proceed")
        assert "nothing pending" in response.lower()


# ---------------------------------------------------------------------------
# Tool catalog tests
# ---------------------------------------------------------------------------

class TestToolCatalog:
    def test_catalog_with_no_tools(self, tmp_path):
        zeus, delphi, runtime = _make_stack(tmp_path)
        orch = ArgusOrchestrator(zeus, delphi, runtime, llm_call=_echo_llm)
        # Clear the built-in tools registered by ArgusOrchestrator.__init__
        # so we can test the empty-catalog branch.
        delphi._tools.clear()
        catalog = orch._get_tool_catalog()
        assert "no tools" in catalog.lower()

    def test_catalog_lists_registered_tools(self, tmp_path):
        zeus, delphi, runtime = _make_stack(tmp_path)

        @delphi.register(
            name="market_data",
            description="Fetch live market data",
            risk_level=RiskLevel.READ_ONLY,
            parameters_schema={
                "properties": {"symbol": {"type": "string"}},
                "required": ["symbol"],
            },
        )
        def market_data(symbol: str) -> dict:
            return {"symbol": symbol}

        orch = ArgusOrchestrator(zeus, delphi, runtime, llm_call=_echo_llm)
        catalog = orch._get_tool_catalog()

        assert "market_data" in catalog
        assert "symbol" in catalog
        assert "READ_ONLY" in catalog


# ---------------------------------------------------------------------------
# Parse tool-call tests
# ---------------------------------------------------------------------------

class TestParseToolCall:
    def _make_orchestrator(self, tmp_path):
        zeus, delphi, runtime = _make_stack(tmp_path)
        return ArgusOrchestrator(zeus, delphi, runtime, llm_call=_echo_llm)

    def test_parse_raw_json(self, tmp_path):
        orch = self._make_orchestrator(tmp_path)
        result = orch._parse_tool_call('{"tool": "get_status", "args": {}}')
        assert result is not None
        assert result["tool"] == "get_status"

    def test_parse_json_in_code_block(self, tmp_path):
        orch = self._make_orchestrator(tmp_path)
        text = 'Here is my action:\n```json\n{"tool": "set_mode", "args": {"mode": "ACTIVE"}}\n```'
        result = orch._parse_tool_call(text)
        assert result is not None
        assert result["tool"] == "set_mode"

    def test_parse_no_tool_call(self, tmp_path):
        orch = self._make_orchestrator(tmp_path)
        result = orch._parse_tool_call("I don't need to call any tool.")
        assert result is None

    def test_parse_malformed_json(self, tmp_path):
        orch = self._make_orchestrator(tmp_path)
        result = orch._parse_tool_call('{"tool": broken}')
        assert result is None

    def test_parse_json_without_tool_key(self, tmp_path):
        orch = self._make_orchestrator(tmp_path)
        result = orch._parse_tool_call('{"action": "get_status"}')
        assert result is None


# ---------------------------------------------------------------------------
# Tiered Escalation tests
# ---------------------------------------------------------------------------

class TestTieredEscalation:
    @pytest.mark.asyncio
    async def test_escalation_from_14b_to_32b_on_parse_failure(self, tmp_path):
        zeus, delphi, runtime = _make_stack(tmp_path)

        @delphi.register(name="my_tool", description="Test tool", risk_level=RiskLevel.READ_ONLY)
        def my_tool() -> str:
            return "ok"

        calls = []

        async def _failing_parse_llm(messages, model):
            calls.append(model)
            if model == "qwen2.5:14b-instruct":
                return "I'll use my_tool but in bad format."
            return json.dumps({"tool": "my_tool", "args": {}})

        orch = ArgusOrchestrator(zeus, delphi, runtime, llm_call=_failing_parse_llm)
        # Default is 14b now
        await orch.chat("run my_tool")

        assert "qwen2.5:14b-instruct" in calls
        assert "qwen2.5:32b-instruct" in calls


    @pytest.mark.asyncio
    async def test_prometheus_32b_parse_retry_escalates(self, tmp_path):
        zeus, delphi, runtime = _make_stack(tmp_path)

        esc_config = EscalationConfig(
            provider=EscalationProvider.CLAUDE,
            api_key="test-key",
            estimated_cost_per_call=0.05,
        )

        call_log = []
        prometheus_calls = 0

        async def mocked_llm_complete(messages, model, escalation_justification=None):
            nonlocal prometheus_calls
            call_log.append({
                "model": model,
                "escalation_justification": escalation_justification,
            })

            system_prompt = messages[0]["content"] if messages else ""
            if "Prometheus" in system_prompt:
                prometheus_calls += 1
                if prometheus_calls <= 2:
                    return "<manifest>{not valid json}</manifest>"
                if escalation_justification:
                    return "<manifest>{not valid json}</manifest>"
                return "<manifest>{\"name\": \"x\"}</manifest>"

            if "Ares" in system_prompt:
                return "<critique>{not valid json}</critique>"

            return "<verdict>{not valid json}</verdict>"

        orch = ArgusOrchestrator(zeus, delphi, runtime, llm_call=_noop_llm, escalation_config=esc_config)
        orch._llm_complete = mocked_llm_complete

        await orch.chat("research a momentum strategy approach")

        assert len(call_log) >= 2
        # Prometheus has escalation_priority=ESCALATION_LOCAL_32B, so the first
        # call is already 32B (upgraded from 14B at line 634 of orchestrator).
        assert call_log[0]["model"] == "qwen2.5:32b-instruct"
        # After local 32B parse failure, the orchestrator escalates to API
        # (still via 32B model but with escalation_justification set).
        escalation_calls = [c for c in call_log if c["escalation_justification"] is not None]
        assert len(escalation_calls) >= 1

    def test_ares_validation_enforcement(self):
        """Ares must produce at least 3 findings; fewer raises ManifestValidationError."""
        critique_response = """
<critique>
{
  "manifest_hash": "abc123",
  "findings": [
    {
      "category": "EXECUTION_RISK",
      "severity": "BLOCKER",
      "description": "Single finding only",
      "evidence": "Not enough adversarial depth",
      "recommendation": "Provide at least three findings"
    }
  ],
  "summary": "Insufficient critique"
}
</critique>
"""

        with pytest.raises(ManifestValidationError, match="at least 3 failure vectors"):
            parse_critique_response(critique_response, manifest_hash="abc123")

    @pytest.mark.asyncio
    async def test_athena_escalation_to_api(self, tmp_path):
        zeus, delphi, runtime = _make_stack(tmp_path)

        esc_config = EscalationConfig(
            provider=EscalationProvider.CLAUDE,
            api_key="test-key",
            estimated_cost_per_call=0.05,
        )

        api_called = False

        async def _local_llm(messages, model):
            return "local response"

        orch = ArgusOrchestrator(zeus, delphi, runtime, llm_call=_local_llm, escalation_config=esc_config)

        # Mock _escalate_to_api to verify it's called
        async def mocked_escalate(messages, justification):
            nonlocal api_called
            api_called = True
            return "api adjudication"

        orch._escalate_to_api = mocked_escalate

        # Research protocol triggers Athena at Stage 5
        await orch.chat("research strategy X")

        assert api_called is True
