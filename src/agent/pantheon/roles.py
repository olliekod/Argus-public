"""
Pantheon Intelligence Engine — Role Definitions
================================================

Defines the structured research agents (Prometheus, Ares, Athena) that
drive the Argus research loop.  Each role has:

- A detailed system prompt that enforces structured output
- An output schema that the response must conform to
- An escalation priority (which LLM tier is required)
- A context injection framework for dynamic prompt enrichment

These agents generate and critique machine-readable trading strategies
via :class:`~src.core.manifests.StrategyManifest`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.core.manifests import (
    HADES_INDICATOR_CATALOG,
    HADES_INDICATOR_DESCRIPTIONS,
    REGIME_FILTER_CATALOG,
    VALID_DIRECTIONS,
    VALID_LOGIC_OPS,
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

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Escalation Levels
# ═══════════════════════════════════════════════════════════════════════════

ESCALATION_LOCAL_14B = 0    # 14B local model is sufficient
ESCALATION_LOCAL_32B = 1    # 32B local model preferred
ESCALATION_CLAUDE = 2       # Claude API mandatory


# ═══════════════════════════════════════════════════════════════════════════
# Context Injector
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ContextInjector:
    """Manages dynamic context injection for Pantheon agent prompts.

    Enriches each agent with selective system context:
    - Prometheus gets full context (indicators, universe, risk limits,
      strategy library, Hades results, failure logs).
    - Ares gets only Hades battle evidence and failure logs (sight-limited
      to prevent novelty bias corrupting its adversarial role).
    - Athena gets library + Hades + risk limits (for grounded scoring).
    """

    # Market state
    regime_context: Optional[Dict[str, str]] = None
    benchmark_greeks: Optional[Dict[str, float]] = None

    # Indicator catalog
    available_indicators: List[str] = field(
        default_factory=lambda: sorted(HADES_INDICATOR_CATALOG)
    )
    available_regime_filters: List[str] = field(
        default_factory=lambda: sorted(REGIME_FILTER_CATALOG)
    )
    indicator_descriptions: Dict[str, str] = field(
        default_factory=lambda: dict(HADES_INDICATOR_DESCRIPTIONS)
    )

    # System awareness
    available_symbols: List[str] = field(default_factory=list)
    risk_limits: Dict[str, Any] = field(default_factory=dict)
    strategy_library: List[Dict[str, Any]] = field(default_factory=list)
    hades_performance_log: List[Dict[str, Any]] = field(default_factory=list)

    # Historical failures
    failure_logs: List[Dict[str, Any]] = field(default_factory=list)

    # ── Setters ──────────────────────────────────────────────────────────

    def set_regime_context(
        self,
        vol_regime: str = "UNKNOWN",
        trend_regime: str = "UNKNOWN",
        liquidity_regime: str = "UNKNOWN",
        session_regime: str = "UNKNOWN",
        risk_regime: str = "UNKNOWN",
    ) -> None:
        """Update the current market regime state."""
        self.regime_context = {
            "vol_regime": vol_regime,
            "trend_regime": trend_regime,
            "liquidity_regime": liquidity_regime,
            "session_regime": session_regime,
            "risk_regime": risk_regime,
        }

    def set_benchmark_greeks(self, greeks: Dict[str, float]) -> None:
        """Update aggregate Greeks for benchmark index (e.g. BTC)."""
        self.benchmark_greeks = greeks

    def set_available_symbols(self, symbols: List[str]) -> None:
        """Set confirmed tradeable symbols that have bar data in Argus DB."""
        self.available_symbols = sorted(set(symbols))

    def set_risk_limits(self, limits: Dict[str, Any]) -> None:
        """Set hard risk constraints from RiskEngineOpts config."""
        self.risk_limits = limits

    def add_strategy_to_library(
        self,
        name: str,
        grading: str,
        universe: List[str],
        signals: List[str],
        category: str = "",
    ) -> None:
        """Record a promoted strategy for novelty comparison."""
        self.strategy_library.append({
            "name": name,
            "category": category,
            "grading": grading,
            "universe": universe,
            "signals": signals[:6],  # cap to avoid token bloat
        })
        if len(self.strategy_library) > 30:
            self.strategy_library = self.strategy_library[-30:]

    def add_hades_result(
        self,
        strategy_name: str,
        sharpe: float,
        pnl: float,
        win_rate: float,
        kill_reason: Optional[str] = None,
        grading: str = "Unrated",
    ) -> None:
        """Record a Hades backtest result for feedback loop."""
        self.hades_performance_log.append({
            "name": strategy_name,
            "sharpe": round(sharpe, 3),
            "pnl": round(pnl, 2),
            "win_rate": round(win_rate, 1),
            "kill_reason": kill_reason or "none",
            "grading": grading,
        })
        if len(self.hades_performance_log) > 10:
            self.hades_performance_log = self.hades_performance_log[-10:]

    def add_failure_log(self, case_id: str, reason: str, strategy_name: str = "") -> None:
        """Record a historical failure for context in future cases."""
        self.failure_logs.append({
            "case_id": case_id,
            "strategy_name": strategy_name,
            "failure_reason": reason,
        })
        if len(self.failure_logs) > 20:
            self.failure_logs = self.failure_logs[-20:]

    # ── Formatters ───────────────────────────────────────────────────────

    def format_regime_block(self) -> str:
        if not self.regime_context:
            return "Market regime: UNKNOWN (no regime data available)"
        lines = ["Current Market Regime:"]
        for key, value in sorted(self.regime_context.items()):
            label = key.replace("_", " ").title()
            lines.append(f"  - {label}: {value}")
        return "\n".join(lines)

    def format_benchmark_greeks(self) -> str:
        if not self.benchmark_greeks:
            return ""
        lines = ["Benchmark Index Greeks (BTC):"]
        for key, value in sorted(self.benchmark_greeks.items()):
            label = key.title()
            if value is not None:
                lines.append(f"  - {label}: {value:.6f}")
            else:
                lines.append(f"  - {label}: N/A")
        return "\n".join(lines)

    def format_indicator_catalog(self) -> str:
        return (
            "Available Hades Indicators:\n"
            + "\n".join(f"  - {ind}" for ind in self.available_indicators)
        )

    def format_indicator_descriptions(self) -> str:
        """Brief one-liners for each indicator to prevent misuse."""
        if not self.indicator_descriptions:
            return ""
        lines = ["Indicator Reference (what each signal actually measures):"]
        for ind in sorted(self.indicator_descriptions.keys()):
            desc = self.indicator_descriptions[ind]
            lines.append(f"  - {ind}: {desc}")
        return "\n".join(lines)

    def format_regime_filter_catalog(self) -> str:
        return (
            "Available Regime Filters:\n"
            + "\n".join(f"  - {rf}" for rf in self.available_regime_filters)
        )

    def format_symbol_universe(self) -> str:
        """Confirmed tradeable symbols — prevents Prometheus hallucinating tickers."""
        if not self.available_symbols:
            return ""
        syms = ", ".join(self.available_symbols)
        return (
            "Confirmed Available Symbols (have bar data in Argus DB):\n"
            f"  {syms}\n"
            "  ⚠️  Only propose strategies using symbols from this list."
        )

    def format_risk_limits(self) -> str:
        """Hard risk caps from RiskEngineOpts — Prometheus MUST stay within these."""
        if not self.risk_limits:
            return ""
        label_map = {
            "max_risk_per_trade_pct": "Max risk_per_trade_pct (HARD LIMIT)",
            "aggregate_cap_pct":     "Max aggregate portfolio exposure",
            "max_contracts":         "Max contracts per trade",
            "max_loss_per_contract": "Max loss cap per contract ($)",
            "max_drawdown_pct":      "Drawdown throttle trigger (%)",
        }
        lines = ["Argus Risk Engine Hard Limits — your manifest MUST respect these:"]
        for key, label in label_map.items():
            if key in self.risk_limits and self.risk_limits[key] is not None:
                lines.append(f"  - {label}: {self.risk_limits[key]}")
        return "\n".join(lines)

    def format_strategy_library(self) -> str:
        """Existing promoted strategies — use for novelty comparison."""
        if not self.strategy_library:
            return "Strategy Library: Empty (no prior promoted strategies — you have full novelty latitude)."
        lines = ["Existing Promoted Strategies (compare for novelty scoring):"]
        for s in self.strategy_library:
            signals = ", ".join(s.get("signals", [])[:4])
            uni = ", ".join(s.get("universe", []))
            grade = s.get("grading", "?")
            lines.append(
                f"  - [{grade}] {s['name']} | universe: {uni} | signals: {signals}"
            )
        return "\n".join(lines)

    def format_hades_performance_log(self) -> str:
        """Recent Hades backtest results — evidence for attack and scoring."""
        if not self.hades_performance_log:
            return "Hades Backtest History: No results yet."
        lines = ["Recent Hades Backtest Results (what actually happened in live tests):"]
        for r in self.hades_performance_log[-5:]:
            killed = f" → KILLED: {r['kill_reason']}" if r["kill_reason"] != "none" else ""
            lines.append(
                f"  - {r['name']}: Sharpe={r['sharpe']}, PnL=${r['pnl']}, "
                f"Win={r['win_rate']}% [{r['grading']}]{killed}"
            )
        return "\n".join(lines)

    def format_failure_logs(self) -> str:
        if not self.failure_logs:
            return "Historical Failures: None recorded."
        lines = ["Historical Failures (most recent):"]
        for entry in self.failure_logs[-5:]:
            name = entry.get("strategy_name", "unnamed")
            reason = entry["failure_reason"]
            lines.append(f"  - [{entry['case_id']}] {name}: {reason}")
        return "\n".join(lines)

    def format_context_for_role(self, role: str) -> str:
        """Return role-appropriate context subset.

        Ares is sight-limited intentionally — it sees only Hades battle
        evidence and failure logs. This prevents novelty bias corrupting
        its adversarial critique (its job is quant rigor, not originality).
        """
        role_l = role.lower()

        if role_l == "prometheus":
            sections = [
                self.format_regime_block(),
                self.format_benchmark_greeks(),
                self.format_symbol_universe(),
                self.format_risk_limits(),
                self.format_indicator_catalog(),
                self.format_indicator_descriptions(),
                self.format_regime_filter_catalog(),
                self.format_strategy_library(),
                self.format_hades_performance_log(),
                self.format_failure_logs(),
            ]
        elif role_l == "ares":
            # Ares: only adversarial battle data — no library, no symbols, no risk limits
            # Rationale: Ares must attack on quant merit alone, not "this looks like strategy X"
            sections = [
                self.format_regime_block(),
                self.format_hades_performance_log(),
                self.format_failure_logs(),
            ]
        elif role_l == "athena":
            # Athena: library for novelty scoring, Hades for testability, limits for grounding
            sections = [
                self.format_regime_block(),
                self.format_risk_limits(),
                self.format_strategy_library(),
                self.format_hades_performance_log(),
            ]
        else:
            sections = [
                self.format_regime_block(),
                self.format_indicator_catalog(),
                self.format_regime_filter_catalog(),
                self.format_failure_logs(),
            ]

        return "\n\n".join(filter(None, sections))

    def format_full_context(self) -> str:
        """Full context — used as fallback when no role is specified."""
        sections = [
            self.format_regime_block(),
            self.format_benchmark_greeks(),
            self.format_symbol_universe(),
            self.format_risk_limits(),
            self.format_indicator_catalog(),
            self.format_indicator_descriptions(),
            self.format_regime_filter_catalog(),
            self.format_strategy_library(),
            self.format_hades_performance_log(),
            self.format_failure_logs(),
        ]
        return "\n\n".join(filter(None, sections))


# ═══════════════════════════════════════════════════════════════════════════
# PantheonRole
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PantheonRole:
    """Definition of a Pantheon research agent role."""

    name: str
    personality: str
    output_schema: Dict[str, Any]
    escalation_priority: int
    system_prompt_template: str

    def build_system_prompt(self, context: ContextInjector) -> str:
        """Build the full system prompt with role-filtered injected context."""
        return self.system_prompt_template.format(
            context=context.format_context_for_role(self.name),
            indicator_catalog=context.format_indicator_catalog(),
            indicator_catalog_list=", ".join(sorted(HADES_INDICATOR_CATALOG)),
            regime_context=context.format_regime_block(),
            regime_filters=context.format_regime_filter_catalog(),
            failure_logs=context.format_failure_logs(),
            valid_directions=", ".join(sorted(VALID_DIRECTIONS)),
            valid_logic_ops=", ".join(sorted(VALID_LOGIC_OPS)),
            manifest_hash="{manifest_hash}",
        )

    def build_stage_prompt(
        self,
        objective: str,
        context: ContextInjector,
        artifact: str = "",
        original: str = "",
        full_debate: str = "",
    ) -> str:
        """Build the complete messages for a case-file stage.

        Returns the system prompt + user prompt as a formatted string
        ready for LLM completion.
        """
        system = self.build_system_prompt(context)
        return system, objective, artifact, original, full_debate


# ═══════════════════════════════════════════════════════════════════════════
# Prometheus — The Manifest Generator
# ═══════════════════════════════════════════════════════════════════════════

_PROMETHEUS_SYSTEM = """\
You are Prometheus, the Creative Strategist of the Argus Pantheon.

Your role is to propose quantitative trading strategies as structured Strategy Manifests.
You do NOT produce prose — you produce machine-readable JSON artifacts.

CRITICAL RULES:
1. Every response MUST contain a <thought> tag with your internal reasoning.
2. Every response MUST contain a <manifest> tag with a valid JSON Strategy Manifest.
3. The manifest MUST be parseable by the Hades backtest engine without modification.
4. Only use indicators from the available catalog below.
5. Logic trees use a DSL with these operators: {valid_logic_ops}
   STRICT: You MUST ONLY use operators from this list. Do NOT invent new operators.
6. Valid trade directions: {valid_directions}
   STRICT: "direction" MUST be a single plain string — one of "LONG", "SHORT", or "NEUTRAL".
   DO NOT use a dictionary, logic tree, or any nested object for "direction".

   ██ MEAN REVERSION / BIDIRECTIONAL STRATEGIES — READ THIS ██
   If your strategy can trade in BOTH directions (buys dips AND sells rallies), set
   "direction": "NEUTRAL" and encode the two-sided condition in entry_logic.
   NEVER put directional logic inside the "direction" field.

   // ❌ WRONG — putting a logic object here causes a HARD PARSE FAILURE, aborting your case:
   "direction": {{"op": "OR", "left": {{"op": "AND", ...}}, "right": {{"op": "AND", ...}}}}

   // ✅ CORRECT — direction is always a plain string, two-sided logic goes in entry_logic:
   "direction": "NEUTRAL",
   "entry_logic": {{
     "op": "OR",
     "left": {{"op": "LT", "left": "rsi", "right": {{"param_name": "oversold_threshold"}}}},
     "right": {{"op": "GT", "left": "rsi", "right": {{"param_name": "overbought_threshold"}}}}
   }}

{context}

██ SIGNALS vs REGIME FILTERS — READ CAREFULLY ██

The "signals" array MUST ONLY contain technical indicators from the Hades Indicator Catalog:
  {indicator_catalog_list}

Regime types (vol_regime, trend_regime, etc.) are NOT signals — they go ONLY in "regime_filters".

  // CORRECT:
  "signals": ["rsi", "ema"],
  "regime_filters": {{"vol_regime": ["VOL_LOW"]}}

  // WRONG — DO NOT DO THIS:
  "signals": ["vol_regime"],  // ❌ vol_regime is NOT an indicator!

██ PARAMETERS RULES ██

- Fixed values: "param_name": 42
- Sweep ranges for optimization: "param_name": {{"min": 10, "max": 50, "step": 5}}
- Every strategy MUST have at least one sweepable parameter.

██ LOGIC NODE DSL — READ CAREFULLY ██

Logic trees are built from nodes. Each node is one of:
  A) A COMPARISON node: {{"op": "GT", "left": "rsi", "right": 30}}
  B) A REGIME node:    {{"op": "IN_REGIME", "left": "vol_regime", "right": "VOL_HIGH"}}
  C) A COMBINER node:  {{"op": "AND", "left": <node>, "right": <node>}}

Rules for "left" and "right":
- "left" is ALWAYS a plain string indicator name (e.g. "rsi") or a nested logic node.
- "right" is ALWAYS a plain number, string, or nested logic node.
- DO NOT use nested dicts like {{"indicator": "rsi"}} or {{"value": 30}} — those are WRONG.
- "left" is NEVER {{"time_decay"}} or {{"param_name": "x"}} — those do not exist.
- To reference a sweep parameter in "right", use: {{"param_name": "entry_threshold"}}

FULL MANIFEST EXAMPLE (VRP Put Spread):
{{
  "name": "SPY VRP Put Spread",
  "objective": "Sell put spreads during high-vol regimes to capture premium decay.",
  "signals": ["rsi", "atr"],
  "entry_logic": {{
    "op": "AND",
    "left": {{"op": "IN_REGIME", "left": "vol_regime", "right": "VOL_HIGH"}},
    "right": {{"op": "GT", "left": "rsi", "right": {{"param_name": "rsi_entry"}}}}
  }},
  "exit_logic": {{
    "op": "OR",
    "left": {{"op": "LT", "left": "rsi", "right": {{"param_name": "rsi_exit"}}}},
    "right": {{"op": "GT", "left": "atr", "right": 2.5}}
  }},
  "parameters": {{
    "rsi_entry": {{"min": 40, "max": 70, "step": 5}},
    "rsi_exit": {{"min": 20, "max": 45, "step": 5}}
  }},
  "direction": "NEUTRAL",
  "universe": ["SPY"],
  "regime_filters": {{"vol_regime": ["VOL_HIGH", "VOL_NORMAL"]}},
  "timeframe": 60,
  "holding_period": "1-5 days",
  "risk_per_trade_pct": 0.02
}}

FULL MANIFEST EXAMPLE (Momentum):
{{
  "name": "EMA Momentum Long",
  "objective": "Capture trend continuation in uptrends using EMA crossover.",
  "signals": ["ema", "rsi", "atr"],
  "entry_logic": {{
    "op": "AND",
    "left": {{"op": "CROSS_ABOVE", "left": "ema", "right": {{"param_name": "ema_fast"}}}},
    "right": {{"op": "GT", "left": "rsi", "right": 55}}
  }},
  "exit_logic": {{
    "op": "LT",
    "left": "rsi",
    "right": 45
  }},
  "parameters": {{
    "ema_fast": {{"min": 8, "max": 21, "step": 3}}
  }},
  "direction": "LONG",
  "universe": ["SPY"],
  "regime_filters": {{"vol_regime": ["VOL_LOW", "VOL_NORMAL"], "trend_regime": ["TREND_UP"]}},
  "timeframe": 60,
  "holding_period": "intraday",
  "risk_per_trade_pct": 0.01
}}

RESPONSE FORMAT:
<thought>
Your step-by-step reasoning about the strategy design.
Consider: What market inefficiency are you targeting?
What is the falsification test? Under what conditions would this fail?
</thought>

<manifest>
{{ your valid JSON manifest here, following the exact structure of the examples above }}
</manifest>
"""

_PROMETHEUS_REVISION_SYSTEM = """\
You are Prometheus, the Creative Strategist of the Argus Pantheon.

You are REVISING your strategy proposal based on Ares's critique.
You MUST explicitly address every blocker raised by Ares.

CRITICAL RULES:
1. Every response MUST contain a <thought> tag with your reasoning about each critique point.
2. Every response MUST contain a <manifest> tag with the REVISED JSON Strategy Manifest.
3. For each BLOCKER, explain how you resolved it or why it's invalid.
4. For each ADVISORY, either incorporate the suggestion or explain why not.
5. Only use indicators from the available Hades Indicator Catalog.
   Regime types (vol_regime, trend_regime, etc.) are NOT indicators — they go ONLY in "regime_filters".
6. STRICT: You MUST ONLY use these logic operators: {valid_logic_ops}
7. Every strategy MUST have at least one sweepable parameter:
   "param_name": {{"min": float, "max": float, "step": float}}
8. STRICT: "direction" MUST be a single string ("LONG", "SHORT", "NEUTRAL").
   DO NOT use a dictionary or logic block for "direction".

{context}

RESPONSE FORMAT:
<thought>
Address each critique point:
- [BLOCKER/ADVISORY] Category: Your response and resolution
</thought>

<manifest>
{{ revised JSON manifest here }}
</manifest>
"""


PROMETHEUS = PantheonRole(
    name="Prometheus",
    personality=(
        "Creative strategist who proposes quantitative trading strategies as "
        "structured, machine-readable Strategy Manifests. Thinks in terms of "
        "market microstructure, regime dynamics, and falsifiable hypotheses."
    ),
    output_schema={
        "type": "object",
        "required": ["name", "objective", "signals", "entry_logic", "exit_logic", "parameters"],
        "properties": {
            "name": {"type": "string"},
            "objective": {"type": "string"},
            "signals": {"type": "array", "items": {"type": "string"}},
            "entry_logic": {"type": "object"},
            "exit_logic": {"type": "object"},
            "parameters": {"type": "object"},
            "direction": {"type": "string", "enum": ["LONG", "SHORT", "NEUTRAL"]},
            "universe": {"type": "array", "items": {"type": "string"}},
            "regime_filters": {"type": "object"},
            "timeframe": {"type": "integer"},
            "holding_period": {"type": "string"},
            "risk_per_trade_pct": {"type": "number"},
        },
    },
    escalation_priority=ESCALATION_LOCAL_32B,
    system_prompt_template=_PROMETHEUS_SYSTEM,
)


# ═══════════════════════════════════════════════════════════════════════════
# Ares — The Adversary
# ═══════════════════════════════════════════════════════════════════════════

_ARES_SYSTEM = """\
You are Ares, the War-God Critic of the Argus Pantheon — the Filter of Truth.

Your role is to ATTACK strategy proposals with rigorous adversarial analysis.
You are explicitly adversarial. Your job is to find fatal flaws.

CRITICAL RULES:
1. Every response MUST contain a <thought> tag with your attack reasoning.
2. Every response MUST contain a <critique> tag with a valid JSON critique.
3. You MUST identify at least 3 distinct failure vectors.
4. Classify each finding as BLOCKER (stops the case) or ADVISORY (allows revision).

YOUR ATTACK VECTORS — analyze ALL of these.
For EACH finding, you MUST use EXACTLY the category enum value shown in (parentheses):

QUANT SINS:
- Overfitting (category: OVERFITTING): Too many parameters relative to data? Curve-fitting to noise?
- Look-ahead bias (category: LOOK_AHEAD_BIAS): Does any signal use future information?
- Data leakage (category: DATA_LEAKAGE): Does training data bleed into test data?
- Survivorship bias (category: SURVIVORSHIP_BIAS): Are dead symbols excluded?

FRAGILITY:
- Parameter fragility (category: PARAMETER_FRAGILITY): Would small changes to parameters destroy the edge?
- Regime dependency (category: REGIME_DEPENDENCY): Does it only work in one specific regime?
- Insufficient sample (category: INSUFFICIENT_SAMPLE): Is there enough data to validate statistically?

EXECUTION/REGULATORY RISK:
- Execution risk (category: EXECUTION_RISK): Does it assume unrealistic execution speeds or fill rates?
- Liquidity risk (category: LIQUIDITY_RISK): Can the target instruments actually absorb the positions?
- Regulatory risk (category: REGULATORY_RISK): Could the logic trigger wash-sale or other violations?
- Drawdown risk (category: DRAWDOWN_RISK): Could a drawdown cascade destroy the strategy's viability?

██ CRITICAL: The "category" value in your JSON MUST be exactly one of these strings:
OVERFITTING, LOOK_AHEAD_BIAS, DATA_LEAKAGE, PARAMETER_FRAGILITY, REGIME_DEPENDENCY,
EXECUTION_RISK, REGULATORY_RISK, LIQUIDITY_RISK, DRAWDOWN_RISK, SURVIVORSHIP_BIAS,
INSUFFICIENT_SAMPLE, OTHER

Do NOT invent category names. Do NOT use PARAMETER_SENSITIVITY, VARIABLE_IMPACT,
EXIT_LOGIC_DURATION, ENTRY_LOGIC_COMPLEXITY or any other unlisted value.

{context}

CRITIQUE SCHEMA:
{{
  "manifest_hash": "{manifest_hash}",
  "findings": [
    {{
      "category": "<exactly one of the allowed enum values above>",
      "severity": "BLOCKER|ADVISORY|RESOLVED",
      "description": "What the problem is",
      "evidence": "Specific evidence from the manifest",
      "recommendation": "How to fix it (empty string if RESOLVED)"
    }}
  ],
  "summary": "Overall assessment"
}}

RESPONSE FORMAT:
<thought>
Your adversarial analysis, examining each attack vector systematically.
</thought>

<critique>
{{ valid JSON critique here }}
</critique>
"""

_ARES_FINAL_ATTACK_SYSTEM = """\
You are Ares, the War-God Critic of the Argus Pantheon — performing FINAL ATTACK.

The strategist has revised their proposal. You must:
1. Acknowledge which blockers have been RESOLVED.
2. Identify any NEW vulnerabilities introduced by the revision.
3. Escalate any REMAINING unresolved blockers.

Be fair: if a blocker is genuinely resolved, say so. But be ruthless: if it's
only superficially addressed, call it out.

CRITICAL RULES:
1. Every response MUST contain a <thought> tag.
2. Every response MUST contain a <critique> tag with valid JSON.
3. You MUST provide at least 3 findings (resolved or new).
4. SEVERITY VALUES — use EXACTLY one of these:
   - "BLOCKER"  — for unresolved or new critical issues
   - "ADVISORY" — for non-critical suggestions
   - "RESOLVED" — for previously-raised issues that are now fixed
   DO NOT use any other severity value.

{context}

CRITIQUE SCHEMA (FINAL ATTACK):
{{
  "manifest_hash": "{manifest_hash}",
  "findings": [
    {{
      "category": "<exactly one of: OVERFITTING, LOOK_AHEAD_BIAS, DATA_LEAKAGE, PARAMETER_FRAGILITY, REGIME_DEPENDENCY, EXECUTION_RISK, REGULATORY_RISK, LIQUIDITY_RISK, DRAWDOWN_RISK, SURVIVORSHIP_BIAS, INSUFFICIENT_SAMPLE, OTHER>",
      "severity": "BLOCKER|ADVISORY|RESOLVED",
      "description": "What the problem is",
      "evidence": "Specific evidence from the revision",
      "recommendation": "How to fix it (empty string if RESOLVED)"
    }}
  ],
  "summary": "Overall final assessment"
}}

Do NOT invent category names. Use EXACTLY one of the listed enum values.

RESPONSE FORMAT:
<thought>
Review each original blocker: resolved or still present?
Any new issues from the revision?
</thought>

<critique>
{{ valid JSON critique here }}
</critique>
"""


ARES = PantheonRole(
    name="Ares",
    personality=(
        "War-god adversarial critic. Explicitly attacks strategy proposals to "
        "find fatal flaws in quant logic, execution assumptions, and regime "
        "dependencies. The Filter of Truth."
    ),
    output_schema={
        "type": "object",
        "required": ["manifest_hash", "findings", "summary"],
        "properties": {
            "manifest_hash": {"type": "string"},
            "findings": {
                "type": "array",
                "minItems": 3,
                "items": {
                    "type": "object",
                    "required": ["category", "severity", "description"],
                    "properties": {
                        "category": {"type": "string"},
                        "severity": {"type": "string", "enum": ["BLOCKER", "ADVISORY", "RESOLVED"]},
                        "description": {"type": "string"},
                        "evidence": {"type": "string"},
                        "recommendation": {"type": "string"},
                    },
                },
            },
            "summary": {"type": "string"},
        },
    },
    escalation_priority=ESCALATION_LOCAL_32B,
    system_prompt_template=_ARES_SYSTEM,
)


# ═══════════════════════════════════════════════════════════════════════════
# Athena — The Adjudicator
# ═══════════════════════════════════════════════════════════════════════════

_ATHENA_SYSTEM = """\
You are Athena, the Neutral Arbiter of the Argus Pantheon.

You provide the FINAL go/no-go decision on a strategy proposal after the
Prometheus-Ares debate. You are NOT an advocate — you are a judge.

CRITICAL RULES:
1. Every response MUST contain a <thought> tag with your judicial reasoning.
2. Every response MUST contain a <verdict> tag with a valid JSON verdict.
3. You MUST use the scoring rubric below — no vibes-based decisions.
4. If promoting, "research_packet" MUST be the cleaned manifest JSON directly.
   Do NOT wrap it in a label key. It must be a valid Strategy Manifest object
   with "name", "signals", "entry_logic", etc. at the top level.

SCORING RUBRIC (each 0.0–1.0, weight in parentheses):
- theoretical_soundness (0.25): Is the market hypothesis plausible?
- critique_resolution (0.25): Were Ares's blockers satisfactorily resolved?
- testability (0.20): Can Hades actually backtest this with available indicators and data?
  NOTE: If the manifest failed schema validation, testability MUST be 0.0.
- risk_management (0.15): Are risk controls adequate?
- novelty (0.15): Does this add diversification vs. existing strategies?

Confidence = (theoretical_soundness × 0.25) + (critique_resolution × 0.25) +
             (testability × 0.20) + (risk_management × 0.15) + (novelty × 0.15)

DECISION LOGIC:
- PROMOTE if confidence >= 0.6 AND no unresolved blockers
- REJECT if confidence < 0.4 OR any unresolved blockers remain
- For confidence in [0.4, 0.6): REJECT with detailed conditions for re-submission

██ DECISION FIELD IS CASE-SENSITIVE ██
The "decision" field MUST be exactly one of: "PROMOTE" or "REJECT"
Do NOT write "Approved", "Approve", "Accepted", "Rejected", "Go", "No-Go" or anything else.
ONLY "PROMOTE" or "REJECT" — both uppercase, no quotes variation.

{context}

██ STRICT SCHEMA REQUIREMENT ██
Your verdict JSON MUST contain EXACTLY these top-level keys:
  confidence, decision, rationale, research_packet, unresolved_blockers,
  conditions, rubric_scores

Do NOT add custom top-level keys (e.g. exit_logic_duration, score, comments).
Do NOT put rubric scores (theoretical_soundness etc.) at the top level.
All rubric scores MUST be nested inside the "rubric_scores" object.
The "confidence" value MUST appear as a top-level number (not omitted, not nested).

VERDICT SCHEMA (follow this exactly):
{{
  "confidence": 0.72,
  "decision": "PROMOTE",
  "rationale": "Detailed explanation of the decision",
  "research_packet": {{
    "name": "Strategy Name",
    "objective": "...",
    "signals": ["ema", "rsi"],
    "entry_logic": {{...}},
    "exit_logic": {{...}},
    "parameters": {{...}},
    "direction": "LONG",
    "universe": ["SPY"],
    "regime_filters": {{...}},
    "timeframe": 60,
    "holding_period": "intraday",
    "risk_per_trade_pct": 0.02
  }},
  "unresolved_blockers": ["description of any remaining blockers"],
  "conditions": ["conditions for promotion, if any"],
  "rubric_scores": {{
    "theoretical_soundness": 0.8,
    "critique_resolution": 0.7,
    "testability": 0.85,
    "risk_management": 0.6,
    "novelty": 0.65
  }}
}}

RESPONSE FORMAT:
<thought>
Your judicial analysis of the full debate.
Score each rubric dimension with evidence.
</thought>

██ FINAL REMINDER BEFORE YOU WRITE <verdict> ██
STOP. Check your JSON before writing it. Common mistakes to AVOID:

❌ BAD — inventing your own structure:
  {{"total_score": 3, "reasoning": "...", "summary": "..."}}
  {{"score": {{"Theoretical Soundness": 0.85}}, "decision": "Approved"}}
  {{"blocker_count": 3, "pass": false}}

✅ CORRECT — exact schema, nothing added, nothing renamed:
  {{
    "confidence": 0.65,
    "decision": "REJECT",
    "rationale": "Three unresolved blockers prevent promotion.",
    "research_packet": null,
    "unresolved_blockers": ["EXECUTION_RISK not addressed"],
    "conditions": ["Provide concrete liquidity check mechanism"],
    "rubric_scores": {{
      "theoretical_soundness": 0.80,
      "critique_resolution": 0.50,
      "testability": 0.70,
      "risk_management": 0.55,
      "novelty": 0.65
    }}
  }}

The "decision" value MUST be exactly "PROMOTE" or "REJECT" — nothing else.
The "confidence" MUST be a single decimal number (e.g. 0.65) at the top level.
All five rubric_scores MUST appear inside "rubric_scores", NOT at top level.

<verdict>
{{ valid JSON verdict here — following the schema above exactly }}
</verdict>
"""


ATHENA = PantheonRole(
    name="Athena",
    personality=(
        "Neutral arbiter and judge. Synthesizes the Prometheus-Ares debate "
        "using a deterministic scoring rubric. Produces the final go/no-go "
        "decision with a confidence score and validated research packet."
    ),
    output_schema={
        "type": "object",
        "required": ["confidence", "decision", "rationale", "rubric_scores"],
        "properties": {
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "decision": {"type": "string", "enum": ["PROMOTE", "REJECT"]},
            "rationale": {"type": "string"},
            "research_packet": {"type": "object"},
            "unresolved_blockers": {"type": "array", "items": {"type": "string"}},
            "conditions": {"type": "array", "items": {"type": "string"}},
            "rubric_scores": {
                "type": "object",
                "properties": {
                    "theoretical_soundness": {"type": "number"},
                    "critique_resolution": {"type": "number"},
                    "testability": {"type": "number"},
                    "risk_management": {"type": "number"},
                    "novelty": {"type": "number"},
                },
            },
        },
    },
    escalation_priority=ESCALATION_CLAUDE,
    system_prompt_template=_ATHENA_SYSTEM,
)


# ═══════════════════════════════════════════════════════════════════════════
# Role Registry & Stage Mapping
# ═══════════════════════════════════════════════════════════════════════════

# Import CaseStage here to avoid circular imports at module level.
# The orchestrator defines CaseStage, and roles need to map stages to roles.

# Stage → (role, is_revision_variant)
_STAGE_ROLE_MAP = {
    1: (PROMETHEUS, False),   # PROPOSAL_V1
    2: (ARES, False),         # CRITIQUE_V1
    3: (PROMETHEUS, True),    # REVISION_V2 (uses revision prompt)
    4: (ARES, True),          # FINAL_ATTACK (uses final attack prompt)
    5: (ATHENA, False),       # ADJUDICATION
}


def get_role_for_stage(stage_value: int) -> PantheonRole:
    """Return the PantheonRole for a given CaseStage value."""
    entry = _STAGE_ROLE_MAP.get(stage_value)
    if entry is None:
        raise ValueError(f"No role defined for stage {stage_value}")
    return entry[0]


def _get_system_prompt_for_stage(stage_value: int, context: ContextInjector) -> str:
    """Get the appropriate system prompt for a stage, including variants."""
    entry = _STAGE_ROLE_MAP.get(stage_value)
    if entry is None:
        raise ValueError(f"No role defined for stage {stage_value}")
    role, is_variant = entry

    if is_variant:
        # Use variant prompts
        if role.name == "Prometheus":
            template = _PROMETHEUS_REVISION_SYSTEM
        elif role.name == "Ares":
            template = _ARES_FINAL_ATTACK_SYSTEM
        else:
            template = role.system_prompt_template
    else:
        template = role.system_prompt_template

    return template.format(
        context=context.format_full_context(),
        indicator_catalog=context.format_indicator_catalog(),
        indicator_catalog_list=", ".join(sorted(HADES_INDICATOR_CATALOG)),
        regime_context=context.format_regime_block(),
        regime_filters=context.format_regime_filter_catalog(),
        failure_logs=context.format_failure_logs(),
        valid_directions=", ".join(sorted(VALID_DIRECTIONS)),
        valid_logic_ops=", ".join(sorted(VALID_LOGIC_OPS)),
        manifest_hash="{manifest_hash}",
    )


def build_stage_prompt(
    stage_value: int,
    objective: str,
    context: ContextInjector,
    artifacts: List[Dict[str, Any]],
    parse_errors: str = "",
) -> List[Dict[str, str]]:
    """Build the complete message list for a case-file stage.

    Parameters
    ----------
    stage_value : int
        CaseStage enum value (1–5).
    objective : str
        The research objective / user request.
    context : ContextInjector
        Runtime context for prompt enrichment.
    artifacts : List[Dict[str, Any]]
        Previous stage artifacts from the CaseFile.
    parse_errors : str
        Accumulated parse error messages from previous stages.
        Injected into Stage 3 (revision) and Stage 5 (adjudication)
        so the LLMs can see what failed.

    Returns
    -------
    List[Dict[str, str]]
        Messages list suitable for LLM completion (system + user).
    """
    role = get_role_for_stage(stage_value)
    system_prompt = _get_system_prompt_for_stage(stage_value, context)

    # Build the user prompt with stage-appropriate context
    latest = artifacts[-1]["content"] if artifacts else ""
    original = artifacts[0]["content"] if artifacts else ""
    full_debate = "\n\n---\n\n".join(
        f"[{a['role']} / Stage {a['stage']}]\n{a['content']}" for a in artifacts
    )

    # Inject manifest hash if we are critiquing
    manifest_hash = "UNKNOWN"
    if stage_value in (2, 4) and latest:
        try:
            m = parse_manifest_response(latest)
            manifest_hash = m.compute_hash()
        except Exception as e:
            # If latest is broken (Stage 3 failed), try to fall back to Stage 1 hash
            # so Ares at least knows what we were *trying* to revise.
            if stage_value == 4 and original:
                try:
                    m = parse_manifest_response(original)
                    manifest_hash = m.compute_hash()
                except Exception as e2:
                    pass

    # Re-apply format to system prompt to inject the actual hash if needed
    if "{manifest_hash}" in system_prompt:
        system_prompt = system_prompt.replace("{manifest_hash}", manifest_hash)

    # Build parse error prefix for stages that need it
    error_prefix = ""
    if parse_errors:
        error_prefix = (
            "\n\n⚠️ SCHEMA VALIDATION ERROR(S) FROM PREVIOUS MANIFEST:\n"
            f"{parse_errors}\n"
            "You MUST fix these schema errors before any other revisions.\n\n"
        )

    if stage_value == 1:
        # Prometheus initial proposal
        user_prompt = (
            f"Research Objective: {objective}\n\n"
            "Generate a Strategy Manifest that addresses this objective. "
            "Include your reasoning in <thought> tags and the manifest in <manifest> tags."
        )
    elif stage_value == 2:
        # Ares initial critique
        user_prompt = (
            f"Research Objective: {objective}\n\n"
            f"Strategy Proposal to critique (Hash: {manifest_hash}):\n{latest}\n\n"
            f"Perform adversarial analysis. Use manifest_hash: \"{manifest_hash}\" in your JSON critique. "
            "Include your reasoning in <thought> tags and the critique in <critique> tags."
        )
    elif stage_value == 3:
        # Prometheus revision — inject parse errors if any
        user_prompt = (
            f"Research Objective: {objective}\n\n"
            f"{error_prefix}"
            f"Your original proposal:\n{original}\n\n"
            f"Ares's critique:\n{latest}\n\n"
            "Revise your manifest addressing every critique point AND any schema errors above. "
            "Include your reasoning in <thought> tags and the revised manifest in <manifest> tags."
        )
    elif stage_value == 4:
        # Ares final attack
        user_prompt = (
            f"Research Objective: {objective}\n\n"
            f"Revised proposal (Hash: {manifest_hash}):\n{latest}\n\n"
            f"Full debate history:\n{full_debate}\n\n"
            f"Perform final attack. Use manifest_hash: \"{manifest_hash}\" in your JSON critique. "
            "Acknowledge resolved items, escalate remaining issues. "
            "Include your reasoning in <thought> tags and the critique in <critique> tags."
        )
    elif stage_value == 5:
        # Athena adjudication — inject parse errors if manifest never parsed
        schema_warning = ""
        if parse_errors:
            schema_warning = (
                "\n\n⚠️ CRITICAL: The strategy manifest FAILED schema validation. "
                "This strategy CANNOT be backtested as written. "
                "You MUST set testability to 0.0 in your rubric scores.\n"
                f"Schema errors: {parse_errors}\n\n"
            )
        user_prompt = (
            f"Research Objective: {objective}\n\n"
            f"Full debate:\n{full_debate}\n\n"
            f"{schema_warning}"
            "Adjudicate this debate using the scoring rubric. "
            "Include your reasoning in <thought> tags and the verdict in <verdict> tags.\n\n"
            "██ CRITICAL JSON SCHEMA REMINDER ██\n"
            "Your <verdict> JSON MUST exactly match this schema. Do NOT invent new keys.\n"
            "{\n"
            '  "confidence": 0.0-1.0,\n'
            '  "decision": "PROMOTE" or "REJECT",\n'
            '  "rationale": "...",\n'
            '  "research_packet": { ... },\n'
            '  "unresolved_blockers": [],\n'
            '  "conditions": [],\n'
            '  "rubric_scores": {\n'
            '    "theoretical_soundness": 0.0-1.0,\n'
            '    "critique_resolution": 0.0-1.0,\n'
            '    "testability": 0.0-1.0,\n'
            '    "risk_management": 0.0-1.0,\n'
            '    "novelty": 0.0-1.0\n'
            '  }\n'
            "}"
        )
    else:
        raise ValueError(f"Unknown stage value: {stage_value}")

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


# ═══════════════════════════════════════════════════════════════════════════
# Response Parsers
# ═══════════════════════════════════════════════════════════════════════════

def parse_manifest_response(response: str) -> StrategyManifest:
    """Parse a Prometheus response into a StrategyManifest.

    Extracts JSON from <manifest> tags, fenced blocks, or raw JSON.
    Validates the manifest structure.

    Raises
    ------
    ManifestValidationError
        If the response contains no valid manifest or the manifest is invalid.
    """
    data = extract_json_from_response(response)
    if data is None:
        raise ManifestValidationError(
            "Prometheus response contains no valid JSON manifest. "
            "Response must include a <manifest>{...}</manifest> block."
        )

    manifest = StrategyManifest.from_dict(data)
    manifest.validate()
    return manifest


def parse_critique_response(response: str, manifest_hash: str = "") -> AresCritique:
    """Parse an Ares response into an AresCritique.

    Extracts JSON from <critique> tags, fenced blocks, or raw JSON.

    Raises
    ------
    ManifestValidationError
        If the response contains no valid critique.
    """
    data = extract_json_from_response(response)
    if data is None:
        raise ManifestValidationError(
            "Ares response contains no valid JSON critique. "
            "Response must include a <critique>{...}</critique> block."
        )

    # Ensure manifest_hash is set
    if "manifest_hash" not in data:
        data["manifest_hash"] = manifest_hash

    if manifest_hash and manifest_hash != "UNKNOWN" and data.get("manifest_hash") != manifest_hash:
        raise ManifestValidationError(
            f"Ares critique manifest_hash '{data.get('manifest_hash')}' does not match "
            f"the manifest under review '{manifest_hash}'."
        )

    critique = AresCritique.from_dict(data)
    critique.validate()
    return critique


def parse_verdict_response(response: str) -> AthenaVerdict:
    """Parse an Athena response into an AthenaVerdict.

    Extracts JSON from <verdict> tags, fenced blocks, or raw JSON.

    Raises
    ------
    ManifestValidationError
        If the response contains no valid verdict.
    """
    data = extract_json_from_response(response)
    if data is None:
        raise ManifestValidationError(
            "Athena response contains no valid JSON verdict. "
            "Response must include a <verdict>{...}</verdict> block."
        )

    verdict = AthenaVerdict.from_dict(data)
    verdict.validate()
    return verdict
