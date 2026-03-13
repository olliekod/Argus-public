from __future__ import annotations

import sqlite3

import pytest

from src.agent.zeus import RuntimeMode, ZeusPolicyEngine
from src.core.config import ZeusConfig


def test_budget_cap_blocks_1501(tmp_path):
    db_path = tmp_path / "argus.db"
    engine = ZeusPolicyEngine(
        ZeusConfig(monthly_budget_cap=15.0, governance_db_path=str(db_path))
    )

    engine.log_spend(15.0, actor="tester", purpose="seed")
    assert engine.check_escalation(0.01) is False


def test_requires_hitl_for_high_risk_tool(tmp_path):
    db_path = tmp_path / "argus.db"
    engine = ZeusPolicyEngine(
        ZeusConfig(
            governance_db_path=str(db_path),
            high_risk_tools=["live_trading_toggle", "budget_cap_increase"],
        )
    )

    assert engine.requires_hitl("live_trading_toggle") is True
    assert engine.is_approval_required("safe_read_only_tool") is False


@pytest.mark.asyncio
async def test_set_mode_data_only_pauses_gpu_workers(tmp_path):
    db_path = tmp_path / "argus.db"
    engine = ZeusPolicyEngine(ZeusConfig(governance_db_path=str(db_path)))

    result = await engine.set_mode(RuntimeMode.DATA_ONLY)

    assert result is True
    assert engine.current_mode == RuntimeMode.DATA_ONLY

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT payload_json FROM zeus_governance_audit WHERE event='mode_transition' ORDER BY id DESC LIMIT 1"
        ).fetchone()
    assert row is not None
    payload = row[0]
    assert '"to_mode": "DATA_ONLY"' in payload
    assert '"paused_workers": ["pantheon_roles", "the_forge_discovery", "hades_research"]' in payload
