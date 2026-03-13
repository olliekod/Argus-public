from __future__ import annotations

import pytest

from src.agent.runtime_controller import RuntimeController
from src.agent.zeus import RuntimeMode


class FakeZeus:
    def __init__(self):
        self.current_mode = RuntimeMode.ACTIVE
        self.events = []

    async def log_action(self, metadata):
        self.events.append(metadata)


class FakeResourceManager:
    def __init__(self):
        self.gpu_enabled = True


@pytest.mark.asyncio
async def test_transition_to_data_only_pauses_workers_and_disables_gpu(monkeypatch):
    zeus = FakeZeus()
    rm = FakeResourceManager()
    controller = RuntimeController(zeus=zeus, resource_manager=rm)

    async def fake_stop() -> bool:
        return True

    monkeypatch.setattr(controller, "_stop_ollama", fake_stop)

    report = await controller.transition_to(RuntimeMode.DATA_ONLY)

    assert report["to_mode"] == RuntimeMode.DATA_ONLY.value
    assert rm.gpu_enabled is False
    assert report["services"]["ollama_stopped"] is True
    assert report["workers"]["status"]["pantheon_roles"] == "paused"
    assert report["workers"]["status"]["the_forge"] == "paused"
    assert report["workers"]["status"]["hades_research"] == "paused"
    assert report["workers"]["status"]["data_updaters"] == "running"
    assert zeus.events[-1]["event"] == "runtime_transition_complete"


@pytest.mark.asyncio
async def test_transition_is_idempotent(monkeypatch):
    zeus = FakeZeus()
    rm = FakeResourceManager()
    controller = RuntimeController(zeus=zeus, resource_manager=rm)

    async def fake_stop() -> bool:
        return True

    monkeypatch.setattr(controller, "_stop_ollama", fake_stop)

    first = await controller.transition_to(RuntimeMode.DATA_ONLY)
    second = await controller.transition_to(RuntimeMode.DATA_ONLY)

    assert first["idempotent"] is False
    assert second["idempotent"] is True
    assert second["workers"]["data_updaters"] == "running"
    assert zeus.events[-1]["status"] == "noop"


@pytest.mark.asyncio
async def test_failed_ollama_stop_logs_critical(monkeypatch):
    zeus = FakeZeus()
    rm = FakeResourceManager()
    controller = RuntimeController(zeus=zeus, resource_manager=rm)

    async def fake_stop() -> bool:
        return False

    monkeypatch.setattr(controller, "_stop_ollama", fake_stop)

    report = await controller.transition_to(RuntimeMode.DATA_ONLY)

    assert report["services"]["ollama_stopped"] is False
    assert rm.gpu_enabled is False
    assert any(event["event"] == "runtime_transition_error" for event in zeus.events)
    assert zeus.events[-1]["event"] == "runtime_transition_complete"


@pytest.mark.asyncio
async def test_transition_to_active_resumes_workers_and_enables_gpu(monkeypatch):
    zeus = FakeZeus()
    rm = FakeResourceManager()
    controller = RuntimeController(zeus=zeus, resource_manager=rm)

    async def fake_stop() -> bool:
        return True

    async def fake_start() -> bool:
        return True

    monkeypatch.setattr(controller, "_stop_ollama", fake_stop)
    monkeypatch.setattr(controller, "_start_ollama", fake_start)

    await controller.transition_to(RuntimeMode.DATA_ONLY)
    report = await controller.transition_to(RuntimeMode.ACTIVE)

    assert report["services"]["ollama_started"] is True
    assert rm.gpu_enabled is True
    assert report["workers"]["status"]["pantheon_roles"] == "running"
    assert report["workers"]["status"]["the_forge"] == "running"
    assert report["workers"]["status"]["hades_research"] == "running"
