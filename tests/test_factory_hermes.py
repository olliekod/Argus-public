from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest

from src.agent.argus_orchestrator import ArgusOrchestrator, CaseFile
from src.agent.delphi import DelphiToolRegistry
from src.agent.pantheon.factory import FactoryPipe
from src.agent.pantheon.roles import parse_verdict_response
from src.agent.runtime_controller import RuntimeController
from src.agent.zeus import ZeusPolicyEngine
from src.core.config import ZeusConfig


class _FakeResourceManager:
    gpu_enabled: bool = True


def _make_stack(tmp_path):
    zeus_cfg = ZeusConfig(governance_db_path=str(tmp_path / "argus.db"))
    zeus = ZeusPolicyEngine(zeus_cfg)
    delphi = DelphiToolRegistry(zeus, role_tool_allowlist={"Argus": {"*"}})
    runtime = RuntimeController(zeus, _FakeResourceManager())
    return zeus, delphi, runtime


def _manifest_json(name: str = "Pantheon Alpha") -> str:
    return json.dumps(
        {
            "name": name,
            "objective": "Capture momentum continuation.",
            "signals": ["rsi", "macd"],
            "entry_logic": {"op": "GT", "left": "rsi", "right": 55},
            "exit_logic": {"op": "LT", "left": "rsi", "right": 45},
            "parameters": {"lookback": {"min": 5, "max": 20, "step": 5}},
            "direction": "LONG",
            "universe": ["SPY"],
            "regime_filters": {"vol_regime": ["VOL_NORMAL"]},
            "timeframe": 60,
            "holding_period": "intraday",
            "risk_per_trade_pct": 0.01,
        }
    )


def _critique_json(blockers: int = 1) -> str:
    findings = []
    for i in range(max(blockers, 3)):
        findings.append(
            {
                "category": "OTHER",
                "severity": "BLOCKER" if i < blockers else "ADVISORY",
                "description": f"finding-{i}",
                "evidence": "test",
                "recommendation": "fix",
            }
        )
    return json.dumps({"manifest_hash": "", "findings": findings, "summary": "summary"})


def _verdict_json(confidence: float = 0.9, decision: str = "PROMOTE") -> str:
    return json.dumps(
        {
            "confidence": confidence,
            "decision": decision,
            "rationale": "Strong and testable strategy.",
            "research_packet": json.loads(_manifest_json()),
            "unresolved_blockers": [],
            "conditions": [],
            "rubric_scores": {
                "theoretical_soundness": 0.9,
                "critique_resolution": 0.9,
            },
        }
    )


def _build_case(case_id: str = "case_test", confidence: float = 0.9, blockers_final: int = 0) -> CaseFile:
    case = CaseFile(case_id=case_id, objective="research momentum strategy")
    case.artifacts = [
        {"stage": 1, "role": "Prometheus", "content": f"<manifest>{_manifest_json('V1')}</manifest>"},
        {"stage": 2, "role": "Ares", "content": f"<critique>{_critique_json(blockers=2)}</critique>"},
        {"stage": 3, "role": "Prometheus", "content": f"<manifest>{_manifest_json('V2')}</manifest>"},
        {"stage": 4, "role": "Ares", "content": f"<critique>{_critique_json(blockers=blockers_final)}</critique>"},
        {"stage": 5, "role": "Athena", "content": f"<verdict>{_verdict_json(confidence=confidence)}</verdict>"},
    ]
    return case


def test_factory_persistence(tmp_path):
    factory = FactoryPipe(db_path=str(tmp_path / "factory.db"))
    case = _build_case(case_id="case_123", confidence=0.72, blockers_final=1)

    factory.persist_case(case)
    stored = factory.get_case("case_123")

    assert stored is not None
    assert stored["strategy"]["case_id"] == "case_123"
    assert len(stored["evidence"]) == 5
    assert [e["stage"] for e in stored["evidence"]] == [1, 2, 3, 4, 5]


@pytest.mark.asyncio
async def test_hermes_handoff(tmp_path):
    zeus, delphi, runtime = _make_stack(tmp_path)
    factory_mock = Mock()
    hermes_mock = Mock()

    async def _promote_llm(messages, model):
        system = messages[0]["content"]
        if "Athena" in system:
            return f"<verdict>{_verdict_json(confidence=0.88, decision='PROMOTE')}</verdict>"
        if "Ares" in system:
            return f"<critique>{_critique_json(blockers=0)}</critique>"
        return f"<manifest>{_manifest_json()}</manifest>"

    orch = ArgusOrchestrator(
        zeus,
        delphi,
        runtime,
        llm_call=_promote_llm,
        factory_pipe=factory_mock,
        hermes_router=hermes_mock,
    )

    await orch.chat("research a strategy proposal")

    hermes_mock.route_promotion.assert_called_once()
    passed_case = hermes_mock.route_promotion.call_args[0][0]
    athena_artifact = [a for a in passed_case.artifacts if a["role"] == "Athena"][-1]
    verdict = parse_verdict_response(athena_artifact["content"])
    assert verdict.decision == "PROMOTE"
    assert verdict.research_packet is not None
    assert "entry_logic" in verdict.research_packet


def test_evidence_grading(tmp_path):
    factory = FactoryPipe(db_path=str(tmp_path / "factory.db"))
    case = _build_case(case_id="case_gold", confidence=0.91, blockers_final=0)

    factory.persist_case(case)
    stored = factory.get_case("case_gold")

    assert stored is not None
    assert stored["strategy"]["grading"] == "Gold"


def test_factory_concurrent_writes_are_consistent(tmp_path):
    factory = FactoryPipe(db_path=str(tmp_path / "factory.db"))

    def _persist(i: int) -> int:
        case = _build_case(case_id=f"case_{i}", confidence=0.65, blockers_final=i % 2)
        return factory.persist_case(case)

    with ThreadPoolExecutor(max_workers=6) as pool:
        ids = list(pool.map(_persist, range(12)))

    assert len(ids) == 12
    for i in range(12):
        stored = factory.get_case(f"case_{i}")
        assert stored is not None
        assert len(stored["evidence"]) == 5
