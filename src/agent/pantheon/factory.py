# Created by Oliver Meihls

# Persistent strategy/evidence store for Pantheon research cases.

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.agent.pantheon.roles import (
    parse_critique_response,
    parse_manifest_response,
    parse_verdict_response,
)


@dataclass
class EvidenceGrader:
    # Grades strategy quality from Athena confidence + Ares blockers.

    def grade(self, athena_confidence: float, final_blockers: int) -> str:
        if athena_confidence > 0.8 and final_blockers == 0:
            return "Gold"
        if 0.6 <= athena_confidence <= 0.8 and final_blockers < 2:
            return "Silver"
        if 0.4 <= athena_confidence <= 0.6:
            return "Bronze"
        return "Unrated"


class FactoryPipe:
    # Persists completed case files into an on-disk sqlite strategy library.

    def __init__(self, db_path: str = "data/factory.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.grader = EvidenceGrader()
        self._write_lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_schema(self) -> None:
        with self._write_lock:
            with self._connect() as conn:
                conn.executescript(
                    """
                CREATE TABLE IF NOT EXISTS strategies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id TEXT NOT NULL UNIQUE,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    final_manifest TEXT,
                    athena_confidence REAL,
                    grading TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS evidence (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_id INTEGER NOT NULL,
                    stage INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    manifest_hash TEXT,
                    FOREIGN KEY(strategy_id) REFERENCES strategies(id) ON DELETE CASCADE
                );
                """
                )

    def persist_case(self, case: Any) -> int:
        # Persist a completed case object and all stage artifacts.
        artifacts: List[Dict[str, Any]] = list(getattr(case, "artifacts", []))
        case_id = str(getattr(case, "case_id", ""))
        objective = str(getattr(case, "objective", "Unnamed Strategy"))

        final_manifest: Optional[Dict[str, Any]] = None
        athena_confidence = 0.0
        status = "UNKNOWN"
        final_blockers = 999
        latest_manifest_hash = ""

        for artifact in artifacts:
            content = artifact.get("content", "")
            role = str(artifact.get("role", "")).lower()

            if role == "prometheus":
                try:
                    parsed_manifest = parse_manifest_response(content)
                    latest_manifest_hash = parsed_manifest.compute_hash()
                    final_manifest = parsed_manifest.to_dict()
                except Exception:
                    pass

            if role == "ares":
                try:
                    critique = parse_critique_response(content, manifest_hash="")
                    final_blockers = len(critique.blockers)
                    if critique.manifest_hash:
                        latest_manifest_hash = critique.manifest_hash
                except Exception:
                    pass

            if role == "athena":
                try:
                    verdict = parse_verdict_response(content)
                    athena_confidence = verdict.confidence
                    status = verdict.decision
                    if verdict.research_packet is not None:
                        final_manifest = verdict.research_packet
                except Exception:
                    pass

        strategy_name = final_manifest.get("name", objective[:120]) if final_manifest else objective[:120]
        grading = self.grader.grade(athena_confidence, final_blockers)

        with self._write_lock:
            with self._connect() as conn:
                cur = conn.execute(
                    """
                    INSERT OR REPLACE INTO strategies
                        (case_id, name, status, final_manifest, athena_confidence, grading)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        case_id,
                        strategy_name,
                        status,
                        json.dumps(final_manifest) if final_manifest is not None else None,
                        athena_confidence,
                        grading,
                    ),
                )

                strategy_id = cur.lastrowid
                if strategy_id == 0:
                    row = conn.execute(
                        "SELECT id FROM strategies WHERE case_id = ?", (case_id,)
                    ).fetchone()
                    strategy_id = int(row["id"])

                conn.execute("DELETE FROM evidence WHERE strategy_id = ?", (strategy_id,))
                for artifact in artifacts:
                    role = str(artifact.get("role", ""))
                    stage = int(artifact.get("stage", -1))
                    content = artifact.get("content", "")
                    manifest_hash = self._derive_manifest_hash(content, role, latest_manifest_hash)
                    conn.execute(
                        """
                        INSERT INTO evidence(strategy_id, stage, role, content, manifest_hash)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (strategy_id, stage, role, content, manifest_hash),
                    )

        return strategy_id

    def get_case(self, case_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            strategy = conn.execute(
                "SELECT * FROM strategies WHERE case_id = ?", (case_id,)
            ).fetchone()
            if strategy is None:
                return None
            evidence = conn.execute(
                """
                SELECT stage, role, content, manifest_hash
                FROM evidence
                WHERE strategy_id = ?
                ORDER BY stage ASC, id ASC
                """,
                (strategy["id"],),
            ).fetchall()

        return {
            "strategy": dict(strategy),
            "evidence": [dict(e) for e in evidence],
        }

    def get_promoted_strategies(
        self, gradings: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        # Return all strategies that were promoted by Athena.
        target_gradings = gradings or ["Gold", "Silver"]
        with self._connect() as conn:
            placeholders = ",".join(["?"] * len(target_gradings))
            query = f"SELECT * FROM strategies WHERE status = 'PROMOTE' AND grading IN ({placeholders})"
            rows = conn.execute(query, target_gradings).fetchall()
            return [dict(r) for r in rows]

    def _derive_manifest_hash(self, content: str, role: str, fallback_hash: str) -> str:
        try:
            if role.lower() == "prometheus":
                return parse_manifest_response(content).compute_hash()
            if role.lower() == "ares":
                return parse_critique_response(content, manifest_hash="").manifest_hash
        except Exception:
            pass

        return fallback_hash
