import pytest

from src.orchestrator import ArgusOrchestrator


def _base_config():
    return {
        "system": {"database_path": "data/test_public_opts.db", "log_level": "INFO"},
        "market_hours": {},
        "dashboard": {"enabled": False},
        "soak": {},
        "symbols": {"monitored": []},
        "exchanges": {"alpaca": {"enabled": False}, "yahoo": {"enabled": False}},
        "tastytrade": {"snapshot_sampling": {"enabled": False}},
        "public_options": {"enabled": True},
        "public": {},
        "thresholds": {},
        "secrets": {"public": {}},
    }


@pytest.mark.asyncio
async def test_public_options_enabled_requires_api_secret(monkeypatch):
    cfg = _base_config()

    monkeypatch.setattr("src.orchestrator.load_all_config", lambda *_: cfg)
    monkeypatch.setattr("src.orchestrator.validate_secrets", lambda *_: [])
    monkeypatch.setattr("src.orchestrator.setup_logger", lambda *a, **k: None)

    orch = ArgusOrchestrator(config_dir="config")
    with pytest.raises(ValueError, match="public_options.enabled=true requires secrets.public.api_secret"):
        await orch._setup_connectors()


@pytest.mark.asyncio
async def test_public_options_enabled_requires_account_id(monkeypatch):
    cfg = _base_config()
    cfg["secrets"]["public"] = {"api_secret": "a-real-token"}
    cfg["public"] = {}

    monkeypatch.setattr("src.orchestrator.load_all_config", lambda *_: cfg)
    monkeypatch.setattr("src.orchestrator.validate_secrets", lambda *_: [])
    monkeypatch.setattr("src.orchestrator.setup_logger", lambda *a, **k: None)

    orch = ArgusOrchestrator(config_dir="config")
    with pytest.raises(ValueError, match="public.account_id"):
        await orch._setup_connectors()
