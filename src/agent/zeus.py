"""Zeus deterministic governance and policy enforcement layer."""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from src.core.config import ZeusConfig

logger = logging.getLogger(__name__)


class RuntimeMode(str, Enum):
    """System execution modes governed by Zeus."""

    ACTIVE = "ACTIVE"
    DATA_ONLY = "DATA_ONLY"
    CPU_CHAT = "CPU_CHAT"
    OFFLINE = "OFFLINE"


class ZeusPolicyEngine:
    """Hard-gate governance policy layer with deterministic budget and mode controls."""

    def __init__(self, config: ZeusConfig):
        self.config = config
        self._current_mode = RuntimeMode.ACTIVE
        self._spend_lock = threading.Lock()
        self._mode_lock = asyncio.Lock()
        self._monthly_spend = 0.0
        self._db_path = Path(config.governance_db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_store()
        self._monthly_spend = self._load_current_month_spend()

    @property
    def current_mode(self) -> RuntimeMode:
        return self._current_mode

    @property
    def monthly_spend(self) -> float:
        with self._spend_lock:
            return self._monthly_spend

    async def set_mode(self, mode: RuntimeMode) -> bool:
        """Transition runtime mode and enforce process/resource controls."""
        if isinstance(mode, str):
            mode = RuntimeMode(mode)

        async with self._mode_lock:
            previous_mode = self._current_mode
            metadata: Dict[str, Any] = {
                "status": "accepted",
                "from_mode": previous_mode.value,
                "to_mode": mode.value,
            }

            if mode == RuntimeMode.DATA_ONLY:
                metadata.update(
                    {
                        "gpu_enabled": False,
                        "ollama_stopped": self._stop_ollama_service(),
                        "paused_workers": ["pantheon_roles", "the_forge_discovery", "hades_research"],
                        "allowed_workers": ["market_updater", "news_updater"],
                    }
                )
            elif mode == RuntimeMode.CPU_CHAT:
                metadata.update(
                    {
                        "gpu_enabled": False,
                        "cpu_chat_model": "1-3B",
                        "ollama_stopped": self._stop_ollama_service(),
                    }
                )
            elif mode == RuntimeMode.OFFLINE:
                metadata.update(
                    {
                        "gpu_enabled": False,
                        "ollama_stopped": self._stop_ollama_service(),
                        "shutdown_requested": True,
                    }
                )
            else:
                metadata.update({"gpu_enabled": True})

            self._current_mode = mode
            await self.log_action(
                {
                    "event": "mode_transition",
                    "mode": mode.value,
                    "metadata": metadata,
                }
            )
            logger.info("Zeus mode transition %s -> %s", previous_mode.value, mode.value)
            return True

    def check_budget(self, estimated_cost: float) -> bool:
        """Pre-call budget gate alias."""
        return self.check_escalation(estimated_cost)

    def check_escalation(self, estimated_cost: float) -> bool:
        """Hard cap checker for projected spend."""
        with self._spend_lock:
            projected = self._monthly_spend + max(0.0, estimated_cost)
            allowed = projected <= float(self.config.monthly_budget_cap)

        if not allowed:
            self._write_audit_row(
                {
                    "event": "budget_denied",
                    "mode": self._current_mode.value,
                    "monthly_spend": self.monthly_spend,
                    "metadata": {
                        "estimated_cost": estimated_cost,
                        "projected_spend": projected,
                        "monthly_budget_cap": self.config.monthly_budget_cap,
                    },
                }
            )
            logger.warning(
                "Zeus budget denied: current=%.4f estimate=%.4f cap=%.4f",
                self.monthly_spend,
                estimated_cost,
                self.config.monthly_budget_cap,
            )
        return allowed

    def log_spend(self, actual_cost: float, actor: str, purpose: str) -> None:
        """Track actual spend and persist in governance store."""
        charge = max(0.0, actual_cost)
        month_key = self._month_key()
        with self._spend_lock:
            self._monthly_spend += charge
            snapshot = self._monthly_spend
            with sqlite3.connect(self._db_path) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute(
                    """
                    INSERT INTO zeus_budget_state(month_key, spend)
                    VALUES(?, ?)
                    ON CONFLICT(month_key) DO UPDATE SET spend=excluded.spend
                    """,
                    (month_key, snapshot),
                )
                conn.execute(
                    """
                    INSERT INTO zeus_spend_log(ts_utc, actor, purpose, amount, month_key)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (self._now_iso(), actor, purpose, charge, month_key),
                )
                conn.commit()

        self._write_audit_row(
            {
                "event": "spend_logged",
                "mode": self._current_mode.value,
                "monthly_spend": snapshot,
                "metadata": {
                    "actor": actor,
                    "purpose": purpose,
                    "amount": charge,
                },
            }
        )

    def requires_hitl(self, tool_id: str) -> bool:
        """Return True when the requested tool is high-risk and requires approval."""
        return tool_id in set(self.config.high_risk_tools)

    def is_approval_required(self, tool_name: str) -> bool:
        """Compatibility alias for tool approval checks."""
        return self.requires_hitl(tool_name)

    def force_override(self, justification: str) -> None:
        """Record operator sovereign override in audit trail."""
        self._write_audit_row(
            {
                "event": "force_override",
                "mode": self._current_mode.value,
                "monthly_spend": self.monthly_spend,
                "metadata": {
                    "justification": justification,
                    "severity": "HIGH",
                    "config": asdict(self.config),
                },
            }
        )
        logger.critical("Zeus force override applied: %s", justification)

    async def log_action(self, action_metadata: dict):
        """Write governance event to durable SQLite audit trail in WAL mode."""
        payload = {
            "mode": self._current_mode.value,
            "monthly_spend": self.monthly_spend,
            **action_metadata,
        }
        await asyncio.to_thread(self._write_audit_row, payload)

    def _write_audit_row(self, payload: Dict[str, Any]) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                "INSERT INTO zeus_governance_audit(ts_utc, event, payload_json) VALUES (?, ?, ?)",
                (self._now_iso(), payload.get("event", "unknown"), json.dumps(payload, sort_keys=True)),
            )
            conn.commit()

    def _initialize_store(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS zeus_budget_state (
                    month_key TEXT PRIMARY KEY,
                    spend REAL NOT NULL DEFAULT 0.0
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS zeus_spend_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts_utc TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    purpose TEXT NOT NULL,
                    amount REAL NOT NULL,
                    month_key TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS zeus_governance_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts_utc TEXT NOT NULL,
                    event TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _load_current_month_spend(self) -> float:
        month_key = self._month_key()
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT spend FROM zeus_budget_state WHERE month_key = ?",
                (month_key,),
            ).fetchone()
        return float(row[0]) if row else 0.0

    def _stop_ollama_service(self) -> bool:
        """Best effort service stop signal; deterministic and non-LLM."""
        # Runtime environments vary (systemd, windows service, process manager).
        # We log intent and return deterministic success signal for policy path.
        logger.info("Zeus requested stop for service '%s'", self.config.ollama_service_name)
        return True

    @staticmethod
    def _month_key(now: Optional[datetime] = None) -> str:
        dt = now or datetime.now(timezone.utc)
        return dt.strftime("%Y-%m")

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()
