#!/usr/bin/env python3
"""
Autonomous Research Engine
==========================

A proactive service that watches a research backlog and drives each
objective through the full Pantheon 5-stage debate protocol:

    Prometheus (Proposal) -> Ares (Critique) -> Prometheus (Revision)
    -> Ares (Final Attack) -> Athena (Adjudication)

Upon completion:
- The :class:`~src.agent.pantheon.factory.FactoryPipe` persists the
  case and evidence to the strategy library.
- If Athena grades the strategy "Gold" or "Silver",
  :class:`~src.agent.pantheon.hermes.HermesRouter` automatically
  routes it to the Hades backtest queue.

Queue Source
------------
The engine reads research objectives from either:
1. ``config/research_backlog.yaml`` — a YAML file with a list of objectives.
2. The ``research_queue`` table in ``data/argus.db`` (if it exists).

Usage
-----
::

    # Process the entire backlog, one objective at a time
    python scripts/research_engine.py

    # Process a single objective from the CLI
    python scripts/research_engine.py --objective "BTC/USDT Breakout seasonality"

    # Dry-run: show what would be processed without running debates
    python scripts/research_engine.py --dry-run

    # Process one objective and exit
    python scripts/research_engine.py --once

Clean Shutdown
--------------
The engine installs a SIGINT / SIGTERM handler.  When interrupted:
1. The current debate stage is allowed to finish (if one is in progress).
2. The partial case file is persisted with whatever stages completed.
3. The remaining backlog is left untouched for the next run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import platform
import signal
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on path
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import yaml

from src.agent.argus_orchestrator import (
    ArgusOrchestrator,
    CaseFile,
    CaseStage,
    EscalationConfig,
)
from src.agent.delphi import DelphiToolRegistry
from src.agent.pantheon.factory import EvidenceGrader, FactoryPipe
from src.agent.pantheon.hermes import HermesRouter
from src.agent.pantheon.roles import (
    build_stage_prompt,
    get_role_for_stage,
    parse_critique_response,
    parse_manifest_response,
    parse_verdict_response,
    ContextInjector,
)
from src.agent.resource_manager import AgentResourceManager
from src.agent.runtime_controller import RuntimeController
from src.agent.zeus import ZeusPolicyEngine
from src.connectors.news_sentiment_tools import configure_sentiment_client
from src.core.config import load_config, ZeusConfig
from src.core.manifests import (
    AresCritique,
    AthenaVerdict,
    ManifestStatus,
    ManifestValidationError,
    StrategyManifest,
)

logger = logging.getLogger("argus.research_engine")

# ═══════════════════════════════════════════════════════════════════════════
# Console Dialogue & Stats Helpers
# ═══════════════════════════════════════════════════════════════════════════

_ROLE_COLORS = {
    "Prometheus": "\033[94m",  # Blue
    "Ares":       "\033[91m",  # Red
    "Athena":     "\033[93m",  # Yellow
}
_RESET = "\033[0m"
_BOLD  = "\033[1m"
_DIM   = "\033[2m"
_GREEN = "\033[92m"
_CYAN  = "\033[96m"
_BOX   = "-"


def _print_dialogue_block(role_name: str, stage_name: str, response: str) -> None:
    """Pretty-print a single LLM dialogue block to stdout.

    Uses ANSI colours so the terminal output is easy to scan.
    Colour codes degrade gracefully if the terminal doesn't support them.
    """
    color = _ROLE_COLORS.get(role_name, "")
    width = 80
    header = f" {role_name} [{stage_name}] "
    pad_total = width - len(header)
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    border = _BOX * width

    print(f"\n{color}{_BOLD}{border}{_RESET}")
    print(f"{color}{_BOLD}{_BOX * pad_left}{header}{_BOX * pad_right}{_RESET}")
    print(f"{color}{_BOLD}{border}{_RESET}")
    # Print the response with slight indent
    for line in response.splitlines():
        print(f"  {line}")
    print(f"{color}{_DIM}{border}{_RESET}\n")


def _print_hades_summary(results: list) -> None:
    """Print a formatted table of Hades backtest experiment results to stdout."""
    if not results:
        print(f"\n{_DIM}[Hades] No experiment results returned.{_RESET}\n")
        return

    width = 80
    border = "-" * width
    print(f"\n{_CYAN}{_BOLD}{border}{_RESET}")
    print(f"{_CYAN}{_BOLD}{'  HADES BACKTEST RESULTS':^{width}}{_RESET}")
    print(f"{_CYAN}{_BOLD}{border}{_RESET}")

    # Header row
    print(
        f"  {'Strategy':<32}  {'PnL':>10}  {'Return%':>8}  "
        f"{'Sharpe':>7}  {'Win%':>6}  {'Trades':>6}"
    )
    print(f"  {'-'*32}  {'-'*10}  {'-'*8}  {'-'*7}  {'-'*6}  {'-'*6}")

    for r in results:
        strat_id = str(r.get("strategy_id", r.get("strategy_class", "unknown")))[:32]
        portfolio = r.get("portfolio", r.get("portfolio_summary", {}))
        pnl      = portfolio.get("total_realized_pnl", 0.0)
        ret_pct  = portfolio.get("total_return_pct", 0.0)
        sharpe   = portfolio.get("sharpe_annualized_proxy", 0.0)
        win_rate = portfolio.get("win_rate", 0.0)
        trades   = portfolio.get("total_trades", 0)

        pnl_color = _GREEN if pnl >= 0 else "\033[91m"
        print(
            f"  {strat_id:<32}  "
            f"{pnl_color}${pnl:>9.2f}{_RESET}  "
            f"{ret_pct:>7.2f}%  "
            f"{sharpe:>7.2f}  "
            f"{win_rate:>5.1f}%  "
            f"{trades:>6}"
        )

    print(f"{_CYAN}{_DIM}{border}{_RESET}\n")


# ═══════════════════════════════════════════════════════════════════════════

_BACKLOG_PATH = Path("config/research_backlog.yaml")
_DB_PATH = Path("data/argus.db")


def _load_yaml_backlog() -> List[Dict[str, Any]]:
    """Load research objectives from YAML backlog."""
    if not _BACKLOG_PATH.exists():
        logger.info("No research backlog found at %s", _BACKLOG_PATH)
        return []

    with open(_BACKLOG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    objectives = data.get("objectives", [])
    items: List[Dict[str, Any]] = []
    for obj in objectives:
        if isinstance(obj, str):
            items.append({"objective": obj, "priority": "normal", "status": "pending"})
        elif isinstance(obj, dict):
            if obj.get("status", "pending") == "pending":
                items.append({
                    "objective": str(obj.get("objective", "")),
                    "priority": str(obj.get("priority", "normal")),
                    "status": "pending",
                    "constraints": obj.get("constraints", []),
                })
    return items


def _load_db_queue() -> List[Dict[str, Any]]:
    """Load pending research objectives from SQLite queue table."""
    if not _DB_PATH.exists():
        return []

    try:
        conn = sqlite3.connect(_DB_PATH)
        conn.row_factory = sqlite3.Row

        # Check if table exists
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='research_queue'"
        )
        if cur.fetchone() is None:
            conn.close()
            return []

        rows = conn.execute(
            "SELECT * FROM research_queue WHERE status = 'pending' ORDER BY priority DESC, id ASC"
        ).fetchall()
        conn.close()

        return [
            {
                "id": row["id"],
                "objective": row["objective"],
                "priority": row.get("priority", "normal"),
                "status": row["status"],
            }
            for row in rows
        ]
    except Exception as exc:
        logger.warning("Failed to read research_queue from DB: %s", exc)
        return []


def _mark_db_objective_done(obj_id: int, case_id: str, grading: str) -> None:
    """Mark a DB queue objective as completed."""
    if not _DB_PATH.exists():
        return
    try:
        conn = sqlite3.connect(_DB_PATH)
        conn.execute(
            "UPDATE research_queue SET status = 'completed', case_id = ?, grading = ? WHERE id = ?",
            (case_id, grading, obj_id),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.warning("Failed to update research_queue: %s", exc)


def _mark_yaml_objective_done(objective: str) -> None:
    """Mark a YAML backlog objective as completed by updating its status."""
    if not _BACKLOG_PATH.exists():
        return
    try:
        with open(_BACKLOG_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        objectives = data.get("objectives", [])
        for i, obj in enumerate(objectives):
            if isinstance(obj, str) and obj == objective:
                objectives[i] = {"objective": obj, "status": "completed"}
            elif isinstance(obj, dict) and obj.get("objective") == objective:
                obj["status"] = "completed"

        data["objectives"] = objectives
        with open(_BACKLOG_PATH, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    except Exception as exc:
        logger.warning("Failed to update YAML backlog: %s", exc)


def load_research_queue() -> List[Dict[str, Any]]:
    """Load pending research objectives from all sources."""
    items: List[Dict[str, Any]] = []
    items.extend(_load_yaml_backlog())
    items.extend(_load_db_queue())

    # Sort by priority (high > normal > low)
    priority_order = {"high": 0, "normal": 1, "low": 2}
    items.sort(key=lambda x: priority_order.get(x.get("priority", "normal"), 1))
    return items


# ═══════════════════════════════════════════════════════════════════════════
# Research Engine
# ═══════════════════════════════════════════════════════════════════════════

class ResearchEngine:
    """Autonomous research service that drives the Pantheon debate protocol.

    Parameters
    ----------
    orchestrator : ArgusOrchestrator
        The fully-wired orchestrator with Zeus, Delphi, and LLM access.
    """

    def __init__(
        self,
        orchestrator: ArgusOrchestrator,
        trigger_backtest: bool = False,
    ) -> None:
        self.orchestrator = orchestrator
        self.trigger_backtest = trigger_backtest
        self.factory_pipe = orchestrator.factory_pipe
        self.hermes_router = orchestrator.hermes_router
        self.grader = EvidenceGrader()

        # Bootstrap context injector with system awareness
        from src.agent.pantheon.context_loader import bootstrap_context_injector
        from src.core.config import load_config
        self._config = load_config()
        self.orchestrator.context_injector = bootstrap_context_injector(config=self._config)

        # Shutdown coordination
        self._shutdown_requested = False
        self._current_case: Optional[CaseFile] = None


    def request_shutdown(self) -> None:
        """Signal the engine to stop after the current debate stage."""
        self._shutdown_requested = True
        logger.info("Shutdown requested — will finish current stage and exit.")

    async def process_objective(self, objective: str) -> Dict[str, Any]:
        """Run a single research objective through the full Pantheon debate.

        Returns
        -------
        dict
            Summary of the research outcome including case_id, grading,
            decision, and confidence.
        """
        case_id = f"research_{int(time.time())}_{hash(objective) % 10000:04d}"
        logger.info("Starting research case %s: %s", case_id, objective[:80])

        # Delegate to the orchestrator's research handler
        result_text = await self.orchestrator._handle_research(objective)

        # Extract the case from factory pipe
        case_data = self.factory_pipe.get_case(case_id)

        # Parse outcome from result
        summary = self._parse_outcome(result_text, case_id, objective)
        return summary

    async def process_objective_standalone(self, objective: str) -> Dict[str, Any]:
        """Run the debate protocol independently without the orchestrator chat loop.

        This directly drives the 5-stage debate using the Pantheon roles,
        which gives us finer control over shutdown coordination and error
        handling.

        Returns
        -------
        dict
            Summary of the research outcome.
        """
        case_id = f"research_{int(time.time())}_{abs(hash(objective)) % 10000:04d}"
        case = CaseFile(case_id=case_id, objective=objective)
        self._current_case = case

        logger.info("=== Research Case %s ===", case_id)
        logger.info("Objective: %s", objective[:120])

        context = self.orchestrator.context_injector
        current_manifest: Optional[StrategyManifest] = None
        current_critique: Optional[AresCritique] = None
        final_verdict: Optional[AthenaVerdict] = None
        accumulated_parse_errors: List[str] = []
        parse_error_count = 0

        stages = [
            CaseStage.PROPOSAL_V1,
            CaseStage.CRITIQUE_V1,
            CaseStage.REVISION_V2,
            CaseStage.FINAL_ATTACK,
            CaseStage.ADJUDICATION,
        ]

        for target_stage in stages:
            if self._shutdown_requested:
                logger.info(
                    "Shutdown requested — saving partial case at stage %d",
                    case.stage.value,
                )
                break

            case.advance()
            role_obj = get_role_for_stage(target_stage.value)
            role_name = role_obj.name

            logger.info(
                "  Stage %d/%d: %s (%s)",
                target_stage.value, 5, role_name, target_stage.name,
            )

            # Build structured prompt (inject parse errors at stages 3 and 5)
            parse_errors_str = "\n".join(accumulated_parse_errors)
            messages = build_stage_prompt(
                stage_value=target_stage.value,
                objective=objective,
                context=context,
                artifacts=case.artifacts,
                parse_errors=parse_errors_str if target_stage.value in (3, 5) else "",
            )

            # Determine escalation
            justification = None
            if role_obj.escalation_priority >= 2:
                justification = (
                    f"{role_name} adjudicating Case {case.case_id}: "
                    "requires high-reasoning judge."
                )

            model = self.orchestrator.model
            from src.agent.argus_orchestrator import DEFAULT_MODEL, UPGRADE_MODEL
            if role_obj.escalation_priority >= 1 and model == DEFAULT_MODEL:
                model = UPGRADE_MODEL

            # LLM completion
            stage_response = await self.orchestrator._llm_complete(
                messages, model, escalation_justification=justification,
            )

            # Parse structured output
            parse_error = None
            if target_stage in (CaseStage.PROPOSAL_V1, CaseStage.REVISION_V2):
                try:
                    current_manifest = parse_manifest_response(stage_response)
                    current_manifest.status = (
                        ManifestStatus.DRAFT
                        if target_stage == CaseStage.PROPOSAL_V1
                        else ManifestStatus.REVISED
                    )
                except Exception as exc:
                    parse_error = str(exc)
                    logger.warning("Prometheus parse error: %s", exc)

                    # Retry with upgrade model
                    if model != UPGRADE_MODEL:
                        retry_response = await self.orchestrator._llm_complete(
                            messages, UPGRADE_MODEL,
                        )
                        try:
                            current_manifest = parse_manifest_response(retry_response)
                            current_manifest.status = (
                                ManifestStatus.DRAFT
                                if target_stage == CaseStage.PROPOSAL_V1
                                else ManifestStatus.REVISED
                            )
                            stage_response = retry_response
                            parse_error = None
                        except ManifestValidationError as exc2:
                            parse_error = str(exc2)

                    # If still broken after retry, log and abort remaining stages
                    if parse_error:
                        accumulated_parse_errors.append(parse_error)
                        parse_error_count += 1
                        if target_stage == CaseStage.PROPOSAL_V1:
                            logger.error(
                                "  FATAL: Proposal manifest failed to parse after retry — "
                                "aborting debate."
                            )
                            case.add_artifact(role_name, stage_response)
                            _print_dialogue_block(role_name, target_stage.name, stage_response)
                            break

            elif target_stage in (CaseStage.CRITIQUE_V1, CaseStage.FINAL_ATTACK):
                manifest_hash = current_manifest.compute_hash() if current_manifest else ""
                try:
                    current_critique = parse_critique_response(
                        stage_response, manifest_hash,
                    )
                except Exception as exc:
                    parse_error = str(exc)
                    logger.warning("Ares parse error: %s", exc)

                    if model != UPGRADE_MODEL:
                        retry_response = await self.orchestrator._llm_complete(
                            messages, UPGRADE_MODEL,
                        )
                        try:
                            current_critique = parse_critique_response(
                                retry_response, manifest_hash,
                            )
                            stage_response = retry_response
                            parse_error = None
                        except ManifestValidationError as exc2:
                            parse_error = str(exc2)

                    if parse_error:
                        accumulated_parse_errors.append(parse_error)
                        parse_error_count += 1

            elif target_stage == CaseStage.ADJUDICATION:
                try:
                    final_verdict = parse_verdict_response(stage_response)
                    if final_verdict.decision == "PROMOTE" and current_manifest:
                        current_manifest.status = ManifestStatus.PROMOTED
                    elif current_manifest:
                        current_manifest.status = ManifestStatus.REJECTED
                except Exception as exc:
                    parse_error = str(exc)
                    logger.warning("Athena parse error: %s", exc)

            case.add_artifact(role_name, stage_response)

            if parse_error:
                logger.warning("  Parse warning at stage %d: %s", target_stage.value, parse_error[:80])

            # ── Print LLM Dialogue ───────────────────────────────────────
            _print_dialogue_block(role_name, target_stage.name, stage_response)

        # Persist results
        self.factory_pipe.persist_case(case)

        # Determine grading
        athena_confidence = final_verdict.confidence if final_verdict else 0.0
        raw_blockers = (
            len(current_critique.blockers) if current_critique else 999
        )
        resolved_count = (
            len(current_critique.resolved) if current_critique else 0
        )
        # Net blockers = raw minus any that were explicitly resolved
        final_blockers = max(0, raw_blockers - resolved_count)
        grading = self.grader.grade(athena_confidence, final_blockers)

        logger.info(
            "  Result: decision=%s confidence=%.2f grading=%s",
            final_verdict.decision if final_verdict else "INCOMPLETE",
            athena_confidence,
            grading,
        )

        # Auto-promote Gold/Silver to Hades backtest queue
        if (
            final_verdict
            and final_verdict.decision == "PROMOTE"
            and grading in ("Gold", "Silver")
        ):
            queue_path = self.hermes_router.route_promotion(case)
            if queue_path:
                logger.info("  Promoted to Hades queue: %s", queue_path)

                # Instantly add to context library so subsequent cases see it
                from src.agent.pantheon.context_loader import _infer_category
                self.orchestrator.context_injector.add_strategy_to_library(
                    name=current_manifest.name,
                    category=_infer_category({
                        "name": current_manifest.name,
                        "signals": current_manifest.signals,
                        "direction": current_manifest.direction,
                    }),
                    grading=grading,
                    universe=current_manifest.universe,
                    signals=current_manifest.signals,
                )

                if self.trigger_backtest:
                    # Run backtest in a separate thread to prevent asyncio loop conflicts
                    loop = asyncio.get_running_loop()
                    universe_to_test = current_manifest.universe if current_manifest and current_manifest.universe else None
                    backtest_results = await loop.run_in_executor(
                        None, self._run_hades_cycle, case_id, universe_to_test
                    )
                    if backtest_results:
                        _print_hades_summary(backtest_results)
                        
                        # Add results to context injector context
                        for br in backtest_results:
                            portfolio = br.get("portfolio", br.get("portfolio_summary", {}))
                            strat_id = str(br.get("strategy_id", br.get("strategy_class", "unknown")))
                            self.orchestrator.context_injector.add_hades_result(
                                strategy_name=strat_id,
                                sharpe=float(portfolio.get("sharpe_annualized_proxy", 0.0)),
                                pnl=float(portfolio.get("total_realized_pnl", 0.0)),
                                win_rate=float(portfolio.get("win_rate", 0.0)),
                                kill_reason=br.get("kill_reason"),
                                grading=grading,
                            )

        self._current_case = None

        return {
            "case_id": case_id,
            "objective": objective,
            "decision": final_verdict.decision if final_verdict else "INCOMPLETE",
            "confidence": athena_confidence,
            "grading": grading,
            "stages_completed": case.stage.value,
            "manifest_name": current_manifest.name if current_manifest else None,
            "parse_error_count": parse_error_count,
            "promoted": grading in ("Gold", "Silver") and final_verdict is not None and final_verdict.decision == "PROMOTE",
        }

    def _run_hades_cycle(self, case_id: str, universe: Optional[List[str]] = None) -> Optional[List[Dict[str, Any]]]:
        """Trigger the Hades research loop for a specific case.

        Returns a list of experiment result summaries, or None on failure.
        """
        logger.info("  [Singularity] Triggering Hades backtest for case_id=%s...", case_id)
        try:
            # Inline import to avoid circular dependency
            from scripts.strategy_research_loop import run_cycle
            from src.analysis.research_loop_config import load_research_loop_config

            config_path = "config/research_loop.yaml"
            config = load_research_loop_config(config_path)

            if universe:
                config.pack.symbols = universe
                config.pack.mode = "single"
                logger.info("  [Singularity] Overriding backtest universe with manifesto symbols: %s", universe)

            # Execute one cycle — returns a list of experiment summaries
            results = run_cycle(config, case_id=case_id)
            logger.info("  [Singularity] Hades backtest complete for case_id=%s", case_id)
            return results  # List[Dict] or None
        except Exception as exc:
            logger.error("  [Singularity] Hades backtest failed: %s", exc)
            return None

    def _parse_outcome(
        self, result_text: str, case_id: str, objective: str
    ) -> Dict[str, Any]:
        """Parse the orchestrator's debate output into a summary dict."""
        # Extract key metrics from the debate log
        decision = "UNKNOWN"
        confidence = 0.0
        grading = "Unrated"

        if "**PROMOTE**" in result_text:
            decision = "PROMOTE"
        elif "**REJECT**" in result_text:
            decision = "REJECT"

        # Try to extract confidence
        import re
        conf_match = re.search(r"Confidence:\s*\*\*(\d+\.\d+)\*\*", result_text)
        if conf_match:
            confidence = float(conf_match.group(1))

        grading = self.grader.grade(confidence, 0 if decision == "PROMOTE" else 999)

        return {
            "case_id": case_id,
            "objective": objective,
            "decision": decision,
            "confidence": confidence,
            "grading": grading,
        }

    async def run_backlog(
        self,
        once: bool = False,
        dry_run: bool = False,
    ) -> List[Dict[str, Any]]:
        """Process the research backlog.

        Parameters
        ----------
        once : bool
            If True, process only the first pending objective and exit.
        dry_run : bool
            If True, print objectives but don't run debates.

        Returns
        -------
        list
            List of outcome summaries.
        """
        queue = load_research_queue()

        if not queue:
            logger.info("Research backlog is empty. Nothing to process.")
            return []

        logger.info("Research backlog: %d pending objectives", len(queue))
        if dry_run:
            for i, item in enumerate(queue, 1):
                print(
                    f"  [{i}] [{item.get('priority', 'normal')}] "
                    f"{item['objective'][:100]}"
                )
            return []

        results: List[Dict[str, Any]] = []
        for item in queue:
            if self._shutdown_requested:
                logger.info("Shutdown requested — stopping backlog processing.")
                break

            objective = item["objective"]
            obj_id = item.get("id")

            outcome = await self.process_objective_standalone(objective)
            results.append(outcome)

            # Mark as completed
            if obj_id is not None:
                _mark_db_objective_done(
                    obj_id, outcome["case_id"], outcome["grading"]
                )
            else:
                _mark_yaml_objective_done(objective)

            if once:
                break

        return results


# ═══════════════════════════════════════════════════════════════════════════
# Startup & Signal Handling
# ═══════════════════════════════════════════════════════════════════════════

def _build_orchestrator(config: Dict[str, Any]) -> ArgusOrchestrator:
    """Wire up an ArgusOrchestrator with all dependencies."""
    # 1. Initialize ZeusConfig from dict
    z_cfg_dict = config.get("zeus", {})
    zeus_cfg = ZeusConfig(
        monthly_budget_cap=float(z_cfg_dict.get("monthly_budget_cap", 15.0)),
        high_risk_tools=list(z_cfg_dict.get("high_risk_tools", [])),
        ollama_service_name=str(z_cfg_dict.get("ollama_service_name", "ollama")),
        governance_db_path=str(z_cfg_dict.get("governance_db_path", "data/argus.db")),
    )
    
    zeus = ZeusPolicyEngine(config=zeus_cfg)
    delphi = DelphiToolRegistry(zeus=zeus, role_tool_allowlist={"Argus": {"*"}})

    # 2. Initialize Resource Manager
    agent_cfg = config.get("agent", {})
    max_concurrent = int(agent_cfg.get("max_concurrent_llm_calls", 2))
    resource_manager = AgentResourceManager(max_concurrent_llm_calls=max_concurrent)

    # 3. Initialize Runtime Controller with resource manager
    runtime = RuntimeController(zeus=zeus, resource_manager=resource_manager)

    # Configure sentiment tools for discovery
    configure_sentiment_client(config)

    # Delphi Tool Registry (ArgusOrchestrator handles discovery on init)

    # Escalation config from yaml
    escalation = None
    esc_cfg = agent_cfg.get("escalation", {})
    if esc_cfg.get("api_key"):
        from src.agent.argus_orchestrator import EscalationConfig, EscalationProvider
        escalation = EscalationConfig(
            provider=EscalationProvider(esc_cfg.get("provider", "claude")),
            model=esc_cfg.get("model", ""),
            api_key=esc_cfg["api_key"],
            api_base=esc_cfg.get("api_base", ""),
            estimated_cost_per_call=float(esc_cfg.get("cost_per_call", 0.05)),
        )

    orchestrator = ArgusOrchestrator(
        zeus=zeus,
        delphi=delphi,
        runtime=runtime,
        resource_manager=resource_manager,
        ollama_base=agent_cfg.get("ollama_base", "http://localhost:11434"),
        model=agent_cfg.get("model", "qwen2.5:14b-instruct"),
        escalation_config=escalation,
    )
    return orchestrator


async def _main(args: argparse.Namespace) -> None:
    """Async entry point."""
    config = load_config()

    orchestrator = _build_orchestrator(config)
    engine = ResearchEngine(orchestrator, trigger_backtest=getattr(args, "backtest", False))

    # Install signal handlers for clean shutdown (not supported on Windows asyncio loop)
    if platform.system() != "Windows":
        loop = asyncio.get_running_loop()

        def _signal_handler() -> None:
            engine.request_shutdown()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _signal_handler)

    if args.objective:
        # Single objective from CLI
        outcome = await engine.process_objective_standalone(args.objective)
        print(json.dumps(outcome, indent=2))
    else:
        # Process backlog
        results = await engine.run_backlog(
            once=args.once,
            dry_run=args.dry_run,
        )
        if results:
            print(f"\n{'='*60}")
            print(f"Research Engine: {len(results)} objectives processed")
            print(f"{'='*60}")
            for r in results:
                status = "PROMOTED" if r.get("promoted") else r["decision"]
                print(
                    f"  [{r['grading']:>8}] {status:>10} "
                    f"(conf={r['confidence']:.2f}) {r['objective'][:60]}"
                )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Argus Autonomous Research Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="",
        help="Single research objective to process (bypasses backlog).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process only the first pending backlog item, then exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show pending objectives without running debates.",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Trigger the Hades research loop immediately upon promotion.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)-7s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
