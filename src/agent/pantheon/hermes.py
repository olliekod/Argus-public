"""Hermes Router: hand promoted Pantheon strategies to the Hades queue."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.agent.pantheon.roles import parse_manifest_response, parse_verdict_response
from src.core.manifests import StrategyManifest


@dataclass
class HadesBacktestConfig:
    """Queue payload consumed by the autonomous experiment runner."""

    case_id: str
    strategy_name: str
    generated_utc: str
    backtest_config: Dict[str, Any]


class HermesRouter:
    """Routes PROMOTE outcomes from Athena to a filesystem queue."""

    def __init__(self, queue_dir: str = "data/backtest_queue") -> None:
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)

    def route_promotion(self, case: Any) -> Optional[Path]:
        artifacts = list(getattr(case, "artifacts", []))
        case_id = str(getattr(case, "case_id", ""))

        verdict = None
        fallback_manifest = None
        for artifact in artifacts:
            role = str(artifact.get("role", "")).lower()
            content = artifact.get("content", "")
            if role == "prometheus":
                try:
                    fallback_manifest = parse_manifest_response(content)
                except Exception:
                    pass
            if role == "athena":
                try:
                    verdict = parse_verdict_response(content)
                except Exception:
                    pass

        if verdict is None or verdict.decision != "PROMOTE":
            return None

        manifest = self._manifest_from_verdict(verdict, fallback_manifest)
        if manifest is None:
            return None

        payload = HadesBacktestConfig(
            case_id=case_id,
            strategy_name=manifest.name,
            generated_utc=datetime.now(timezone.utc).isoformat(),
            backtest_config=manifest.to_backtest_config(),
        )

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        out_path = self.queue_dir / f"{case_id}_{ts}.json"
        out_path.write_text(json.dumps(asdict(payload), indent=2), encoding="utf-8")
        return out_path

    @staticmethod
    def _manifest_from_verdict(verdict: Any, fallback: Optional[StrategyManifest]) -> Optional[StrategyManifest]:
        packet = getattr(verdict, "research_packet", None)
        if isinstance(packet, dict):
            try:
                return StrategyManifest.from_dict(packet)
            except Exception:
                return fallback
        return fallback
