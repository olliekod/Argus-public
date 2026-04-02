# Created by Oliver Meihls

# Argus Orchestrator — conversational agent, tool executor, and multi-agent coordinator.
#
# Integrates Zeus (policy), Delphi (tools), and RuntimeController into a
# cohesive chat loop backed by local LLMs via Ollama.

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

import aiohttp

from src.agent.delphi import DelphiToolRegistry, ToolResult, RiskLevel
from src.agent.pantheon.factory import FactoryPipe
from src.agent.resource_manager import AgentResourceManager
from src.agent.pantheon.hermes import HermesRouter
from src.agent.pantheon.roles import (
    ARES,
    ATHENA,
    PROMETHEUS,
    ContextInjector,
    build_stage_prompt,
    get_role_for_stage,
    parse_critique_response,
    parse_manifest_response,
    parse_verdict_response,
)
from src.agent.runtime_controller import RuntimeController
from src.connectors.api_llm_clients import AnthropicClient, OpenAIClient
from src.agent.zeus import RuntimeMode, ZeusPolicyEngine
from src.core.manifests import (
    AresCritique,
    AthenaVerdict,
    ManifestStatus,
    ManifestValidationError,
    StrategyManifest,
)

logger = logging.getLogger(__name__)


# Configuration

DEFAULT_OLLAMA_BASE = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5:14b-instruct"
UPGRADE_MODEL = "qwen2.5:32b-instruct"

# Sliding-window defaults
DEFAULT_MAX_HISTORY = 40  # message pairs kept in memory
DEFAULT_MAX_TOOL_ITERATIONS = 6  # ReAct loop guard


# Intent classification

class Intent(str, Enum):
    # Coarse-grained intent categories for user input.

    COMMAND = "command"
    QUESTION = "question"
    RESEARCH = "research"
    APPROVAL = "approval"
    MODE_SWITCH = "mode_switch"
    UNKNOWN = "unknown"


# Case-file stages (§10.8 debate protocol)

class CaseStage(int, Enum):
    INITIATED = 0
    PROPOSAL_V1 = 1       # Prometheus
    CRITIQUE_V1 = 2       # Ares
    REVISION_V2 = 3       # Prometheus
    FINAL_ATTACK = 4      # Ares
    ADJUDICATION = 5      # Athena


@dataclass
class CaseFile:
    # Tracks a multi-agent debate through the six-stage protocol.

    case_id: str
    objective: str
    constraints: List[str] = field(default_factory=list)
    budget: float = 0.0
    stage: CaseStage = CaseStage.INITIATED
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def advance(self) -> CaseStage:
        # Move to the next stage.  Returns the new stage.
        if self.stage.value < CaseStage.ADJUDICATION.value:
            self.stage = CaseStage(self.stage.value + 1)
        return self.stage

    def add_artifact(self, role: str, content: str) -> None:
        self.artifacts.append({
            "stage": self.stage.value,
            "role": role,
            "content": content,
            "ts_utc": datetime.now(timezone.utc).isoformat(),
        })


# Conversation memory

@dataclass
class ConversationBuffer:
    # Sliding-window conversation history for the LLM context.

    max_messages: int = DEFAULT_MAX_HISTORY
    _messages: List[Dict[str, str]] = field(default_factory=list)

    @property
    def messages(self) -> List[Dict[str, str]]:
        return list(self._messages)

    def append(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})
        self._trim()

    def append_system(self, content: str) -> None:
        self.append("system", content)

    def clear(self) -> None:
        self._messages.clear()

    def _trim(self) -> None:
        # Keep the system prompt (first message if system) plus the last N messages.
        if len(self._messages) <= self.max_messages:
            return
        system_msgs = [m for m in self._messages if m["role"] == "system"]
        non_system = [m for m in self._messages if m["role"] != "system"]
        keep = max(1, self.max_messages - len(system_msgs))
        self._messages = system_msgs + non_system[-keep:]

    def to_prompt_messages(self) -> List[Dict[str, str]]:
        # Return the messages list suitable for the Ollama /api/chat endpoint.
        return list(self._messages)


# Escalation provider (for API fallback)

class EscalationProvider(str, Enum):
    CLAUDE = "claude"
    OPENAI = "openai"


@dataclass
class EscalationConfig:
    provider: EscalationProvider = EscalationProvider.CLAUDE
    model: str = ""
    api_base: str = ""
    api_key: str = ""
    estimated_cost_per_call: float = 0.05


# Argus Orchestrator

_SYSTEM_PROMPT = """\
You are Argus, a loyal friend and a highly capable AI partner in trading and research.
You are not just a tool; you're the user's "man," their "right hand," and a trusted collaborator.

Your tone is:
- Friendly, loyal, and supportive.
- GPT-like: conversational, smart, and proactive.
- Confident but humble.
- NO robotic boilerplate about being a "quantitative trading research platform." 

You have access to the Delphi tool registry and you're governed by the Zeus policy engine.

Available tools: {tool_list}

When you need to act, do it. If you use a tool, format it like this:
{{"tool": "<tool_name>", "args": {{...}}}}

You can see everything going on with the system, but if you're unsure, ask.
Be a friend first, a trader second.
Current runtime mode: {runtime_mode}
"""

_CASE_FILE_ROLE_PROMPTS = {
    CaseStage.PROPOSAL_V1: (
        "Prometheus",
        "You are Prometheus, the creative strategist. Propose a strategy with: "
        "Claim (what), Mechanism (how), WTP Hypothesis (why valuable), "
        "Falsification Test (how to disprove). Objective: {objective}",
    ),
    CaseStage.CRITIQUE_V1: (
        "Ares",
        "You are Ares, the war-god critic. Critically attack the proposal. "
        "Identify: Failure modes, Regulatory risks, Competitor killshots. "
        "Proposal to critique:\n{artifact}",
    ),
    CaseStage.REVISION_V2: (
        "Prometheus",
        "You are Prometheus. Revise your proposal explicitly addressing every critique. "
        "Original proposal:\n{original}\n\nCritique:\n{artifact}",
    ),
    CaseStage.FINAL_ATTACK: (
        "Ares",
        "You are Ares. Perform a final attack on the revised proposal. "
        "Acknowledge resolved items and escalate remaining blockers. "
        "Revised proposal:\n{artifact}",
    ),
    CaseStage.ADJUDICATION: (
        "Athena",
        "You are Athena, the neutral arbiter. Adjudicate this debate. Provide: "
        "Final ranking, Confidence level, Rationale, Next action (approve/reject/iterate). "
        "Full debate:\n{artifact}",
    ),
}

# Patterns for quick intent classification
_MODE_SWITCH_PATTERNS = re.compile(
    r"\b(switch\s+to|enter|activate|go\s+to|enable)\s+"
    r"(gaming|data[_\s]?only|active|cpu[_\s]?chat|trading)\s*(mode)?\b",
    re.IGNORECASE,
)
_APPROVAL_PATTERNS = re.compile(
    r"^\s*(proceed|approve|yes|confirm|go\s+ahead|do\s+it)\s*[.!]?\s*$",
    re.IGNORECASE,
)

_MODE_NAME_MAP = {
    "gaming": RuntimeMode.DATA_ONLY,
    "data_only": RuntimeMode.DATA_ONLY,
    "data only": RuntimeMode.DATA_ONLY,
    "dataonly": RuntimeMode.DATA_ONLY,
    "active": RuntimeMode.ACTIVE,
    "trading": RuntimeMode.ACTIVE,
    "cpu_chat": RuntimeMode.CPU_CHAT,
    "cpu chat": RuntimeMode.CPU_CHAT,
    "cpuchat": RuntimeMode.CPU_CHAT,
}


class ArgusOrchestrator:
    # Central conversational orchestrator integrating Zeus, Delphi, and RuntimeController.

    def __init__(
        self,
        zeus: ZeusPolicyEngine,
        delphi: DelphiToolRegistry,
        runtime: RuntimeController,
        *,
        ollama_base: str = DEFAULT_OLLAMA_BASE,
        model: str = DEFAULT_MODEL,
        escalation_config: Optional[EscalationConfig] = None,
        max_history: int = DEFAULT_MAX_HISTORY,
        max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
        llm_call: Optional[Callable[..., Any]] = None,
        factory_pipe: Optional[FactoryPipe] = None,
        hermes_router: Optional[HermesRouter] = None,
        resource_manager: Optional[AgentResourceManager] = None,
        max_concurrent_llm_calls: int = 2,
        get_status: Optional[Callable] = None,
        get_pnl: Optional[Callable] = None,
        get_positions: Optional[Callable] = None,
        get_farm_status: Optional[Callable] = None,
        get_kalshi_summary: Optional[Callable] = None,
    ):
        self.zeus = zeus
        self.delphi = delphi
        self.runtime = runtime

        self._get_status = get_status
        self._get_pnl = get_pnl
        self._get_positions = get_positions
        self._get_farm_status = get_farm_status
        self._get_kalshi_summary = get_kalshi_summary

        self.ollama_base = ollama_base.rstrip("/")
        self.model = model
        self.escalation_config = escalation_config
        self.max_tool_iterations = max_tool_iterations

        self.memory = ConversationBuffer(max_messages=max_history)
        self._pending_approval: Optional[Dict[str, Any]] = None
        self._active_case: Optional[CaseFile] = None

        # Pantheon context injector for structured research prompts
        self.context_injector = ContextInjector()

        # Allow injection of a custom LLM callable for testing.
        # Signature: async llm_call(messages, model) -> str
        self._llm_call = llm_call

        # Persistent strategy memory + promotion handoff pipeline
        self.factory_pipe = factory_pipe or FactoryPipe()
        self.hermes_router = hermes_router or HermesRouter()

        # Runtime resource controls for bounded LLM concurrency
        self.resource_manager = resource_manager or AgentResourceManager()
        self._max_concurrent_llm_calls = max_concurrent_llm_calls

        # Discovery and registration
        self.delphi.discover_tools("src.connectors")
        self._register_internal_tools()

        # Update system prompt with new tools
        self._refresh_system_prompt()

    def _register_internal_tools(self) -> None:
        # Register built-in tools for Argus.

        @self.delphi.register(
            name="set_mode",
            description="Switch the system runtime mode (active, gaming, cpu_chat).",
            risk_level=RiskLevel.HIGH,
            parameters_schema={
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["active", "gaming", "cpu_chat"],
                        "description": "The target mode to switch to.",
                    }
                },
                "required": ["mode"],
            },
        )
        async def set_mode_tool(mode: str) -> str:
            return await self._handle_mode_switch(f"switch to {mode}")

        @self.delphi.register(
            name="prometheus_backtest",
            description="Initiate a strategy research and backtest session (Pantheon debate).",
            risk_level=RiskLevel.LOW,
            parameters_schema={
                "type": "object",
                "properties": {
                    "objective": {
                        "type": "string",
                        "description": "The research objective (e.g. 'Mean reversion for IBIT').",
                    }
                },
                "required": ["objective"],
            },
        )
        async def prometheus_backtest_tool(objective: str) -> str:
            # Delegate directly to the research handler
            return await self._handle_research(objective)

        @self.delphi.register(
            name="get_system_status",
            description="Get current market regime, BTC score, and data health.",
            risk_level=RiskLevel.LOW,
        )
        async def get_system_status_tool() -> str:
            if not self._get_status: return "Status source not wired."
            data = await self._get_status()
            return json.dumps(data, indent=2)

        @self.delphi.register(
            name="get_live_pnl",
            description="Get current account P&L summary across all paper traders.",
            risk_level=RiskLevel.LOW,
        )
        async def get_live_pnl_tool() -> str:
            if not self._get_pnl: return "P&L source not wired."
            data = await self._get_pnl()
            return json.dumps(data, indent=2)

        @self.delphi.register(
            name="get_open_positions",
            description="List all currently open paper trading positions.",
            risk_level=RiskLevel.LOW,
        )
        async def get_open_positions_tool() -> str:
            if not self._get_positions: return "Positions source not wired."
            data = await self._get_positions()
            return json.dumps(data, indent=2)

        @self.delphi.register(
            name="get_farm_summary",
            description="Get a summary of the paper trader farm (population, active traders).",
            risk_level=RiskLevel.LOW,
        )
        async def get_farm_summary_tool() -> str:
            if not self._get_farm_status: return "Farm source not wired."
            data = await self._get_farm_status()
            return json.dumps(data, indent=2)

        @self.delphi.register(
            name="get_strategy_report",
            description="Fetch the results and artifacts of a specific research case/strategy from the factory library.",
            risk_level=RiskLevel.LOW,
            parameters_schema={
                "type": "object",
                "properties": {
                    "case_id": {"type": "string", "description": "The unique ID of the case/strategy."}
                },
                "required": ["case_id"]
            }
        )
        async def get_strategy_report_tool(case_id: str) -> str:
            data = self.factory_pipe.get_case(case_id)
            if not data: return f"Case {case_id} not found."
            return json.dumps(data, indent=2)

        @self.delphi.register(
            name="get_kalshi_performance",
            description="Fetch a summary of Kalshi prediction market performance (PnL, win rate, recent events).",
            risk_level=RiskLevel.READ_ONLY,
            parameters_schema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        )
        async def get_kalshi_performance_tool() -> str:
            if not self._get_kalshi_summary:
                return "Kalshi performance data is not available (not connected)."
            data = await self._get_kalshi_summary()
            if "error" in data:
                return f"Error fetching Kalshi data: {data['error']}"
            
            events_str = "\n".join([
                f"- [{e['timestamp']}] {e['level']}: {e['message']}"
                for e in data.get("recent_events", [])
            ])
            
            return (
                f"Kalshi Performance Summary:\n"
                f"- Total PnL: ${data['total_pnl']:+.2f}\n"
                f"- Win Rate: {data['win_rate']:.1f}% ({data['wins']}/{data['total_trades']})\n"
                f"- Recent Events:\n{events_str}"
            )

    def set_data_sources(
        self,
        get_status: Optional[Callable] = None,
        get_pnl: Optional[Callable] = None,
        get_positions: Optional[Callable] = None,
        get_farm_status: Optional[Callable] = None,
        get_kalshi_summary: Optional[Callable] = None,
    ) -> None:
        # Dynamically wire data sources to the agent.
        if get_status: self._get_status = get_status
        if get_pnl: self._get_pnl = get_pnl
        if get_positions: self._get_positions = get_positions
        if get_farm_status: self._get_farm_status = get_farm_status
        if get_kalshi_summary: self._get_kalshi_summary = get_kalshi_summary

    async def stop(self) -> None:
        # Gracefully stop the agent stack and release resources.
        logger.info("AgentOrchestrator: Shutting down AI stack...")
        await self.runtime.transition_to(RuntimeMode.OFFLINE)

# Public API

    async def chat(self, message: str) -> str:
        # Process a user message and return Argus's response.
        #
        # This is the primary entry point. It classifies intent, routes to
        # the appropriate handler, and manages the ReAct tool-calling loop.
        self.memory.append("user", message)
        await self._audit("user_message", {"content": message})

        try:
            intent = self._classify_intent(message)
            await self._audit("intent_classified", {"intent": intent.value, "message": message})

            if intent == Intent.MODE_SWITCH:
                response = await self._handle_mode_switch(message)
            elif intent == Intent.APPROVAL:
                response = await self._handle_approval()
            elif intent == Intent.RESEARCH:
                response = await self._handle_research(message)
            elif intent == Intent.COMMAND:
                response = await self._handle_command(message)
            else:
                response = await self._handle_question(message)
        except Exception as exc:
            logger.exception("Argus chat error")
            response = f"I encountered an error processing your request: {exc}"
            await self._audit("chat_error", {"error": str(exc)})

        self.memory.append("assistant", response)
        return response

    async def chat_stream(self, message: str) -> AsyncIterator[str]:
        # Streaming variant of chat() that yields token chunks.
        self.memory.append("user", message)
        await self._audit("user_message_stream", {"content": message})

        intent = self._classify_intent(message)

        # For non-question intents, fall back to non-streaming (tool use etc.)
        if intent != Intent.QUESTION:
            full = await self.chat.__wrapped__(self, message) if hasattr(self.chat, "__wrapped__") else ""
            # Re-process without double-appending — just call the handler directly
            if intent == Intent.MODE_SWITCH:
                full = await self._handle_mode_switch(message)
            elif intent == Intent.APPROVAL:
                full = await self._handle_approval()
            elif intent == Intent.RESEARCH:
                full = await self._handle_research(message)
            elif intent == Intent.COMMAND:
                full = await self._handle_command(message)
            else:
                full = await self._handle_question(message)
            self.memory.append("assistant", full)
            yield full
            return

        # Stream from LLM for questions
        messages = self.memory.to_prompt_messages()
        collected: List[str] = []
        async for chunk in self._llm_stream(messages, self.model):
            collected.append(chunk)
            yield chunk

        full_response = "".join(collected)
        self.memory.append("assistant", full_response)

    @property
    def active_case(self) -> Optional[CaseFile]:
        return self._active_case

    @property
    def pending_approval(self) -> Optional[Dict[str, Any]]:
        return self._pending_approval

# Intent classification

    def _classify_intent(self, message: str) -> Intent:
        text = message.strip()

        if _APPROVAL_PATTERNS.match(text):
            return Intent.APPROVAL

        if _MODE_SWITCH_PATTERNS.search(text):
            return Intent.MODE_SWITCH

        lower = text.lower()

        # Command triggers — action verbs checked before research so explicit
        # commands like "run X" take priority over keyword overlap.
        command_keywords = [
            "switch", "set", "execute", "run", "deploy", "start",
            "stop", "pause", "resume", "toggle", "update", "change",
        ]
        if any(lower.startswith(kw) or f" {kw} " in f" {lower} " for kw in command_keywords):
            return Intent.COMMAND

        # Research triggers
        research_keywords = [
            "research", "analyze", "investigate", "debate", "case file",
            "strategy proposal", "evaluate strategy", "test strategy",
            "test a strategy", "backtest", "mean reversion", "trend following",
        ]
        if any(kw in lower for kw in research_keywords):
            return Intent.RESEARCH

        # Questions
        if text.endswith("?") or lower.startswith(("what", "how", "why", "when", "where", "who", "can", "is", "are", "do", "does")):
            return Intent.QUESTION

        return Intent.UNKNOWN

# Handler: mode switch

    async def _handle_mode_switch(self, message: str) -> str:
        match = _MODE_SWITCH_PATTERNS.search(message)
        if not match:
            return "I couldn't determine the target mode. Please specify: active, data_only, or cpu_chat."

        target_name = match.group(2).lower().replace(" ", "_")
        target_mode = _MODE_NAME_MAP.get(target_name)
        if target_mode is None:
            return f"Unknown mode '{target_name}'. Valid modes: active, data_only (gaming), cpu_chat."

        # Check runtime awareness
        current = self.zeus.current_mode
        if current == target_mode:
            return f"System is already in {target_mode.value} mode."

        await self._audit("mode_switch_requested", {
            "from": current.value,
            "to": target_mode.value,
        })

        # Execute the transition via Zeus + RuntimeController
        await self.zeus.set_mode(target_mode)
        report = await self.runtime.transition_to(target_mode)
        self._refresh_system_prompt()

        errors = report.get("errors", [])
        if errors:
            return (
                f"Mode switched to {target_mode.value}, but with warnings: "
                + "; ".join(errors)
            )
        return f"Mode switched to {target_mode.value}. {self._mode_description(target_mode)}"

    @staticmethod
    def _mode_description(mode: RuntimeMode) -> str:
        if mode == RuntimeMode.DATA_ONLY:
            return "Compute workers paused, GPU disabled. Data updaters continue running."
        if mode == RuntimeMode.CPU_CHAT:
            return "GPU disabled, heavy compute paused. CPU-only chat available."
        return "All systems active. GPU enabled, Ollama running."

# Handler: approval (Zeus gating)

    async def _handle_approval(self) -> str:
        if self._pending_approval is None:
            return "There is nothing pending approval right now."

        tool_name = self._pending_approval["tool_name"]
        args = self._pending_approval["args"]
        self._pending_approval = None

        await self._audit("approval_granted", {"tool_name": tool_name, "args": args})
        self.zeus.force_override(f"Operator approved high-risk tool: {tool_name}")

        # Execute directly — Delphi.call_tool would block again on the same
        # HITL gate.  The operator has explicitly approved, so we invoke the
        # underlying function after Delphi-level validation (RBAC + schema +
        # budget) is already known to have passed on the first attempt.
        tool_def = self.delphi.tools.get(tool_name)
        if tool_def is None:
            return f"Tool `{tool_name}` no longer exists in the registry."

        try:
            outcome = tool_def.func(**args)
            if inspect.isawaitable(outcome):
                outcome = await outcome
            result = ToolResult(success=True, tool_name=tool_name, data=outcome)
        except Exception as exc:
            result = ToolResult(
                success=False,
                tool_name=tool_name,
                error={"code": "TOOL_EXECUTION_ERROR", "message": str(exc)},
            )

        await self._audit("approved_tool_executed", {
            "tool_name": tool_name,
            "success": result.success,
        })
        return self._format_tool_result(result)

# Handler: command (Delphi tool-calling ReAct loop)

    async def _handle_command(self, message: str) -> str:
        # Check runtime mode for compute-heavy tasks
        restriction = self._check_runtime_restriction(message)
        if restriction:
            return restriction

        # Ask the LLM to decide which tool to call (ReAct: Thought → Action)
        tool_prompt = self._build_tool_prompt(message)
        messages = self.memory.to_prompt_messages() + [{"role": "user", "content": tool_prompt}]

        thoughts: List[str] = []
        for iteration in range(self.max_tool_iterations):
            # Try 14B first
            llm_response = await self._llm_complete(messages, self.model)
            tool_call = self._parse_tool_call(llm_response)

            # Escalation Tie 1: 14B -> 32B if tool call parsing fails
            if tool_call is None and self.model == DEFAULT_MODEL:
                if any(tn in llm_response for tn in self.delphi.tools.keys()):
                    logger.info("14B failed to format tool call. Upgrading to 32B.")
                    llm_response = await self._llm_complete(messages, UPGRADE_MODEL)
                    tool_call = self._parse_tool_call(llm_response)

            # Escalation Tie 2: 32B -> API if tool call parsing still fails
            if tool_call is None and self.escalation_config:
                logger.info("Local models failed to format tool call. Escalating to API.")
                api_response = await self._escalate_to_api(messages, "Local models failed to produce valid tool JSON.")
                if api_response:
                    llm_response = api_response
                    tool_call = self._parse_tool_call(llm_response)

            await self._audit("react_thought", {
                "iteration": iteration,
                "thought": llm_response,
                "tool_found": tool_call is not None,
            })
            if tool_call is None:
                # LLM chose not to call a tool — return its response directly
                return llm_response

            tool_name = tool_call["tool"]
            args = tool_call.get("args", {})

            # Execute via Delphi
            result = await self.delphi.call_tool(tool_name, args, actor="Argus")

            # Handle APPROVAL_REQUIRED — pause and inform the user
            if not result.success and result.error and result.error.get("code") == "APPROVAL_REQUIRED":
                self._pending_approval = {"tool_name": tool_name, "args": args}
                await self._audit("approval_required", {"tool_name": tool_name, "args": args})
                return (
                    f"The action `{tool_name}` requires your approval (high-risk tool). "
                    f"Risk: {result.error.get('message', 'Operator approval needed')}. "
                    f"Say 'proceed' to confirm."
                )

            # Handle other errors — explain and stop
            if not result.success:
                error_msg = result.error.get("message", "Unknown error") if result.error else "Unknown error"
                await self._audit("tool_error", {"tool_name": tool_name, "error": error_msg})
                return f"Tool `{tool_name}` failed: {error_msg}"

            # Observe result → feed back to LLM for next step
            observation = json.dumps({"tool_result": result.data}, default=str)
            thoughts.append(f"[Tool: {tool_name}] Result: {observation}")
            messages.append({"role": "assistant", "content": llm_response})
            messages.append({"role": "user", "content": f"Observation: {observation}\nContinue reasoning or provide the final answer."})

        # Max iterations hit
        summary = "\n".join(thoughts) if thoughts else "No tool calls were made."
        return f"I reached my reasoning limit. Here's what I found:\n{summary}"

# Handler: research (Case File debate protocol)

    async def _handle_research(self, message: str) -> str:
        restriction = self._check_runtime_restriction(message)
        if restriction:
            return restriction

        # Stage 0: initiate case file
        case_id = f"case_{int(time.time())}"
        case = CaseFile(case_id=case_id, objective=message)
        self._active_case = case
        await self._audit("case_file_initiated", {"case_id": case_id, "objective": message})

        debate_log: List[str] = [f"**Case File {case_id}** — {message}\n"]

        # Track parsed artifacts for structured handoffs
        current_manifest: Optional[StrategyManifest] = None
        current_critique: Optional[AresCritique] = None
        final_verdict: Optional[AthenaVerdict] = None

        # Run through stages 1-5 using Pantheon structured prompts
        for target_stage in [
            CaseStage.PROPOSAL_V1,
            CaseStage.CRITIQUE_V1,
            CaseStage.REVISION_V2,
            CaseStage.FINAL_ATTACK,
            CaseStage.ADJUDICATION,
        ]:
            case.advance()
            role_obj = get_role_for_stage(target_stage.value)
            role_name = role_obj.name

            # Build structured prompt via Pantheon
            messages = build_stage_prompt(
                stage_value=target_stage.value,
                objective=message,
                context=self.context_injector,
                artifacts=case.artifacts,
            )

            # Determine escalation based on role priority
            justification = None
            if role_obj.escalation_priority >= 2:
                justification = (
                    f"{role_name} adjudicating Case {case.case_id}: "
                    "requires high-reasoning judge."
                )

            # Select model based on escalation priority
            model = self.model
            if role_obj.escalation_priority >= 1 and model == DEFAULT_MODEL:
                model = UPGRADE_MODEL

            stage_response = await self._llm_complete(
                messages, model, escalation_justification=justification
            )

            # Parse structured output and handle validation failures
            parse_error = None
            if target_stage in (CaseStage.PROPOSAL_V1, CaseStage.REVISION_V2):
                parse_error = None
                parse_attempts = [(model, stage_response)]

                # If we started on 14B, also try 32B locally.
                if model == DEFAULT_MODEL:
                    logger.info("Escalating Prometheus to 32B for manifest retry.")
                    retry_response = await self._llm_complete(messages, UPGRADE_MODEL)
                    parse_attempts.append((UPGRADE_MODEL, retry_response))

                for _, response_text in parse_attempts:
                    stage_response = response_text
                    try:
                        current_manifest = parse_manifest_response(stage_response)
                        current_manifest.status = (
                            ManifestStatus.DRAFT if target_stage == CaseStage.PROPOSAL_V1
                            else ManifestStatus.REVISED
                        )
                        parse_error = None
                        break
                    except ManifestValidationError as exc:
                        parse_error = str(exc)
                        logger.warning(
                            "Prometheus produced invalid manifest at stage %d: %s",
                            target_stage.value, exc,
                        )

                # If local tiers failed (or we started at 32B), attempt API escalation.
                if parse_error:
                    escalation_response = await self._llm_complete(
                        messages,
                        UPGRADE_MODEL,
                        escalation_justification=(
                            f"Prometheus manifest parse failure at stage {target_stage.value}: {parse_error}"
                        ),
                    )
                    stage_response = escalation_response
                    try:
                        current_manifest = parse_manifest_response(stage_response)
                        current_manifest.status = (
                            ManifestStatus.DRAFT if target_stage == CaseStage.PROPOSAL_V1
                            else ManifestStatus.REVISED
                        )
                        parse_error = None
                    except ManifestValidationError as exc:
                        parse_error = str(exc)

                if parse_error:
                    await self._audit("manifest_parse_failure", {
                        "case_id": case_id,
                        "stage": target_stage.value,
                        "error": parse_error,
                    })

            elif target_stage in (CaseStage.CRITIQUE_V1, CaseStage.FINAL_ATTACK):
                manifest_hash = current_manifest.compute_hash() if current_manifest else ""
                parse_error = None
                parse_attempts = [(model, stage_response)]

                # If we started on 14B, also try 32B locally.
                if model == DEFAULT_MODEL:
                    retry_response = await self._llm_complete(messages, UPGRADE_MODEL)
                    parse_attempts.append((UPGRADE_MODEL, retry_response))

                for _, response_text in parse_attempts:
                    stage_response = response_text
                    try:
                        current_critique = parse_critique_response(stage_response, manifest_hash)
                        parse_error = None
                        break
                    except ManifestValidationError as exc:
                        parse_error = str(exc)
                        logger.warning(
                            "Ares produced invalid critique at stage %d: %s",
                            target_stage.value, exc,
                        )

                if parse_error:
                    escalation_response = await self._llm_complete(
                        messages,
                        UPGRADE_MODEL,
                        escalation_justification=(
                            f"Ares critique parse failure at stage {target_stage.value}: {parse_error}"
                        ),
                    )
                    stage_response = escalation_response
                    try:
                        current_critique = parse_critique_response(stage_response, manifest_hash)
                        parse_error = None
                    except ManifestValidationError as exc:
                        parse_error = str(exc)
                        logger.warning(
                            "Ares produced invalid critique after escalation at stage %d: %s",
                            target_stage.value, exc,
                        )

            elif target_stage == CaseStage.ADJUDICATION:
                try:
                    final_verdict = parse_verdict_response(stage_response)
                    if final_verdict.decision == "PROMOTE" and current_manifest:
                        current_manifest.status = ManifestStatus.PROMOTED
                    elif current_manifest:
                        current_manifest.status = ManifestStatus.REJECTED
                except ManifestValidationError as exc:
                    logger.warning("Athena produced invalid verdict: %s", exc)

            case.add_artifact(role_name, stage_response)
            debate_log.append(f"### Stage {target_stage.value} ({role_name})\n{stage_response}\n")

            if parse_error:
                debate_log.append(
                    f"⚠ **Parse Warning**: {parse_error}\n"
                )

            await self._audit("case_stage_completed", {
                "case_id": case_id,
                "stage": target_stage.value,
                "role": role_name,
                "has_structured_output": parse_error is None,
            })

        # Append structured summary
        if final_verdict:
            debate_log.append(f"\n### Verdict\n")
            debate_log.append(f"Decision: **{final_verdict.decision}**\n")
            debate_log.append(f"Confidence: **{final_verdict.confidence:.2f}**\n")
            debate_log.append(f"Rationale: {final_verdict.rationale}\n")
            if final_verdict.decision == "PROMOTE" and current_manifest:
                backtest_config = current_manifest.to_backtest_config()
                debate_log.append(
                    f"\n### Backtest Configuration\n```json\n"
                    f"{json.dumps(backtest_config, indent=2)}\n```\n"
                )

        self.factory_pipe.persist_case(case)
        if final_verdict and final_verdict.decision == "PROMOTE":
            self.hermes_router.route_promotion(case)

        self._active_case = None
        return "\n".join(debate_log)

# Handler: question (straight LLM completion)

    async def _handle_question(self, message: str) -> str:
        messages = self.memory.to_prompt_messages()
        return await self._llm_complete(messages, self.model)

# Runtime restriction check

    def _check_runtime_restriction(self, message: str) -> Optional[str]:
        # Return a polite restriction message when in DATA_ONLY mode, or None.
        mode = self.zeus.current_mode
        if mode == RuntimeMode.DATA_ONLY:
            compute_keywords = ["backtest", "replay", "hades", "research", "run", "execute", "deploy"]
            lower = message.lower()
            if any(kw in lower for kw in compute_keywords):
                return (
                    f"The system is currently in {mode.value} mode (gaming/data-only). "
                    "Compute-heavy tasks like backtesting and research are restricted. "
                    "Switch to ACTIVE mode first with: 'switch to active mode'"
                )
        return None

# Hybrid escalation

    async def _escalate_to_api(self, messages: List[Dict[str, str]], justification: str) -> Optional[str]:
        # Attempt API escalation if budget allows. Returns content string or None.
        if not self.escalation_config or not self.escalation_config.api_key:
            return None

        cost = self.escalation_config.estimated_cost_per_call
        if not self.zeus.check_escalation(cost):
            await self._audit("escalation_budget_denied", {
                "estimated_cost": cost,
                "monthly_spend": self.zeus.monthly_spend,
            })
            return None

        await self._audit("escalation_approved", {
            "provider": self.escalation_config.provider.value,
            "model": self.escalation_config.model,
            "justification": justification,
        })

        if self.escalation_config.provider == EscalationProvider.CLAUDE:
            client = AnthropicClient(
                api_key=self.escalation_config.api_key,
                model=self.escalation_config.model or AnthropicClient.DEFAULT_MODEL,
                api_base=self.escalation_config.api_base,
            )
        elif self.escalation_config.provider == EscalationProvider.OPENAI:
            client = OpenAIClient(
                api_key=self.escalation_config.api_key,
                model=self.escalation_config.model or OpenAIClient.DEFAULT_MODEL,
                api_base=self.escalation_config.api_base,
            )
        else:
            return None

        try:
            response = await client.complete(messages)
            content = response.get("message", {}).get("content", "")
            self.zeus.log_spend(cost, actor="Argus", purpose=f"escalation: {justification}")
            return content
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            logger.error("API escalation failed: %s", exc)
            await self._audit("escalation_error", {
                "provider": self.escalation_config.provider.value,
                "error": str(exc),
            })
            return None

# LLM interaction (Ollama)

    async def _llm_complete(self, messages: List[Dict[str, str]], model: str, escalation_justification: Optional[str] = None) -> str:
        # Completion via Ollama with tiered escalation (14B -> 32B -> API).
        async with self.resource_manager.llm_slot():
            # 1. If escalation is justified (e.g. Athena), try API FIRST if budget allows
            if escalation_justification and self.escalation_config:
                api_result = await self._escalate_to_api(messages, escalation_justification)
                if api_result:
                    return api_result

            # 2. Re-inject llm_call check (for testing local paths)
            if self._llm_call is not None:
                return await self._llm_call(messages, model)

            # 3. Try specified model (usually 14B)
            result = await self._call_ollama(messages, model)
            if result:
                return result

            # 4. Connection failure or empty result? Try 32B (UPGRADE_MODEL)
            if model != UPGRADE_MODEL:
                logger.info("Escalating to local upgrade model %s", UPGRADE_MODEL)
                await self._audit("local_upgrade_triggered", {"from": model, "to": UPGRADE_MODEL})
                result = await self._call_ollama(messages, UPGRADE_MODEL)
                if result:
                    return result

            return "I'm unable to reach any LLM service right now. Please check that Ollama is running."

    async def _call_ollama(self, messages: List[Dict[str, str]], model: str) -> Optional[str]:
        # Low-level Ollama call.
        url = f"{self.ollama_base}/api/chat"
        payload = {"model": model, "messages": messages, "stream": False}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as resp:
                    if resp.status == 200:
                        body = await resp.json()
                        return body.get("message", {}).get("content", "")
                    logger.error("Ollama error %d: %s", resp.status, await resp.text())
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            logger.error("Ollama connection error: %s", exc)
        return None

    async def _llm_stream(self, messages: List[Dict[str, str]], model: str) -> AsyncIterator[str]:
        # Streaming completion via Ollama /api/chat.
        if self._llm_call is not None:
            result = await self._llm_call(messages, model)
            yield result
            return

        url = f"{self.ollama_base}/api/chat"
        payload = {"model": model, "messages": messages, "stream": True}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                    if resp.status != 200:
                        yield f"LLM service returned status {resp.status}."
                        return
                    async for line in resp.content:
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                            token = chunk.get("message", {}).get("content", "")
                            if token:
                                yield token
                        except json.JSONDecodeError:
                            continue
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            logger.error("Ollama stream error: %s", exc)
            yield "I'm unable to reach the LLM service right now."

    async def _llm_complete_for_role(self, prompt: str, role: str, escalation_justification: Optional[str] = None) -> str:
        # Run a completion with a role-specific prompt for Pantheon agents.
        messages = [
            {"role": "system", "content": f"You are {role}, part of the Argus Pantheon."},
            {"role": "user", "content": prompt},
        ]
        return await self._llm_complete(messages, self.model, escalation_justification=escalation_justification)

# Tool-call helpers

    def _build_tool_prompt(self, user_message: str) -> str:
        tool_catalog = self._get_tool_catalog()
        return (
            "The user wants to perform an action. Available tools:\n"
            f"{tool_catalog}\n\n"
            "Respond with a JSON object {\"tool\": \"<name>\", \"args\": {...}} "
            "to call a tool, or respond with plain text if no tool is needed.\n\n"
            f"User request: {user_message}"
        )

    def _get_tool_catalog(self) -> str:
        tools = self.delphi.tools
        if not tools:
            return "(no tools registered)"
        lines = []
        for name, td in sorted(tools.items()):
            params = ""
            if td.parameters_schema and "properties" in td.parameters_schema:
                params = ", ".join(td.parameters_schema["properties"].keys())
            lines.append(f"- {name}({params}): {td.description} [risk={td.risk_level.value}]")
        return "\n".join(lines)

    def _parse_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        # Extract a JSON tool call from LLM output, if present.
        # 1. Try the full text as JSON directly
        try:
            parsed = json.loads(text.strip())
            if isinstance(parsed, dict) and "tool" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass

        # 2. Try ```json ... ``` fenced blocks
        fence_match = re.search(r"```json\s*(\{.*\})\s*```", text, re.DOTALL)
        if fence_match:
            try:
                parsed = json.loads(fence_match.group(1))
                if isinstance(parsed, dict) and "tool" in parsed:
                    return parsed
            except json.JSONDecodeError:
                pass

        # 3. Extract balanced-brace JSON objects containing "tool"
        for candidate in self._extract_json_objects(text):
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict) and "tool" in parsed:
                    return parsed
            except json.JSONDecodeError:
                continue

        return None

    @staticmethod
    def _extract_json_objects(text: str) -> List[str]:
        # Yield substrings that look like balanced JSON objects.
        results: List[str] = []
        i = 0
        while i < len(text):
            if text[i] == "{":
                depth = 0
                start = i
                in_string = False
                escape = False
                for j in range(i, len(text)):
                    ch = text[j]
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
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            results.append(text[start : j + 1])
                            i = j
                            break
            i += 1
        return results

    def _format_tool_result(self, result: ToolResult) -> str:
        if result.success:
            return f"Done. Result: {json.dumps(result.data, default=str)}"
        error = result.error or {}
        return f"Tool `{result.tool_name}` failed [{error.get('code', '?')}]: {error.get('message', 'unknown')}"

# Case-file prompt helpers

    def _build_case_prompt(self, case: CaseFile, template: str) -> str:
        # Fill in the template with available artifacts.
        artifacts = case.artifacts
        latest = artifacts[-1]["content"] if artifacts else ""
        original = artifacts[0]["content"] if artifacts else ""
        full_debate = "\n\n---\n\n".join(
            f"[{a['role']} / Stage {a['stage']}]\n{a['content']}" for a in artifacts
        )
        return template.format(
            objective=case.objective,
            artifact=latest,
            original=original,
            full_debate=full_debate,
        )

# System prompt management

    def _refresh_system_prompt(self) -> None:
        tool_list = self._get_tool_catalog()
        prompt = _SYSTEM_PROMPT.format(
            tool_list=tool_list,
            runtime_mode=self.zeus.current_mode.value,
        )
        # Replace the system message if one exists, otherwise prepend
        msgs = self.memory._messages
        if msgs and msgs[0]["role"] == "system":
            msgs[0] = {"role": "system", "content": prompt}
        else:
            msgs.insert(0, {"role": "system", "content": prompt})

# Audit logging

    async def _audit(self, event: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        await self.zeus.log_action({
            "event": f"argus_{event}",
            "metadata": metadata or {},
        })
