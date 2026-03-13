from __future__ import annotations

import sys

import pytest

from src.agent.delphi import DelphiToolRegistry, RiskLevel, tool
from src.agent.zeus import ZeusPolicyEngine
from src.core.config import ZeusConfig


@pytest.mark.asyncio
async def test_validation_error_for_bad_types(tmp_path):
    engine = ZeusPolicyEngine(ZeusConfig(governance_db_path=str(tmp_path / "argus.db")))
    registry = DelphiToolRegistry(engine, role_tool_allowlist={"Apollo": {"echo"}})

    @registry.register(name="echo", description="Echo int", risk_level=RiskLevel.READ_ONLY)
    def echo(value: int) -> int:
        return value

    result = await registry.call_tool("echo", {"value": "x"}, actor="Apollo")

    assert result.success is False
    assert result.error is not None
    assert result.error["code"] == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_rbac_blocks_unallowlisted_actor(tmp_path):
    engine = ZeusPolicyEngine(ZeusConfig(governance_db_path=str(tmp_path / "argus.db")))
    registry = DelphiToolRegistry(engine, role_tool_allowlist={"Poseidon": {"market_scrape"}})

    @registry.register(name="trade_execute", description="Execute trade", risk_level=RiskLevel.HIGH)
    def trade_execute(symbol: str) -> dict:
        return {"symbol": symbol}

    result = await registry.call_tool("trade_execute", {"symbol": "SPY"}, actor="Poseidon")

    assert result.success is False
    assert result.error is not None
    assert result.error["code"] == "RBAC_DENIED"


@pytest.mark.asyncio
async def test_high_risk_tool_blocked_when_approval_required(tmp_path):
    engine = ZeusPolicyEngine(
        ZeusConfig(
            governance_db_path=str(tmp_path / "argus.db"),
            high_risk_tools=["config_edit"],
        )
    )
    registry = DelphiToolRegistry(engine, role_tool_allowlist={"Hermes": {"config_edit"}})

    @registry.register(name="config_edit", description="Edit config", risk_level=RiskLevel.HIGH)
    def config_edit(path: str) -> str:
        return path

    result = await registry.call_tool("config_edit", {"path": "config/config.yaml"}, actor="Hermes")

    assert result.success is False
    assert result.error is not None
    assert result.error["code"] == "APPROVAL_REQUIRED"


@pytest.mark.asyncio
async def test_discover_tools_loads_decorated_functions(tmp_path):
    pkg_root = tmp_path / "demo_pkg"
    pkg_root.mkdir()
    (pkg_root / "__init__.py").write_text("", encoding="utf-8")
    (pkg_root / "sample.py").write_text(
        "from src.agent.delphi import RiskLevel, tool\n\n"
        "@tool(name='catalog_read', description='Read catalog', risk_level=RiskLevel.READ_ONLY)\n"
        "def catalog_read(symbol: str) -> str:\n"
        "    return symbol\n",
        encoding="utf-8",
    )

    sys.path.insert(0, str(tmp_path))
    try:
        engine = ZeusPolicyEngine(ZeusConfig(governance_db_path=str(tmp_path / "argus.db")))
        registry = DelphiToolRegistry(engine, role_tool_allowlist={"Athena": {"catalog_read"}})
        registry.discover_tools("demo_pkg")

        result = await registry.call_tool("catalog_read", {"symbol": "AAPL"}, actor="Athena")
        assert result.success is True
        assert result.data == "AAPL"
    finally:
        sys.path.remove(str(tmp_path))
