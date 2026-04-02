# Created by Oliver Meihls

# Runtime mode transition controller for deterministic resource orchestration.

from __future__ import annotations

import asyncio
import logging
import os
import platform
import signal
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from src.agent.zeus import RuntimeMode, ZeusPolicyEngine

logger = logging.getLogger(__name__)


class ResourceManager(Protocol):
    # Minimal ResourceManager protocol required by RuntimeController.

    gpu_enabled: bool


@dataclass(frozen=True)
class WorkerScope:
    # Named worker scope and whether it can be paused.

    name: str
    pausable: bool = True


class RuntimeController:
    # Authority for physically applying Zeus-approved runtime transitions.

    _DATA_UPDATERS = WorkerScope("data_updaters", pausable=False)
    _PANTHEON = WorkerScope("pantheon_roles")
    _FORGE = WorkerScope("the_forge")
    _HADES = WorkerScope("hades_research")

    def __init__(self, zeus: ZeusPolicyEngine, resource_manager: ResourceManager):
        self.zeus = zeus
        self.rm = resource_manager
        self._transition_lock = asyncio.Lock()
        self._current_mode: RuntimeMode = getattr(zeus, "current_mode", RuntimeMode.ACTIVE)
        self._ollama_pid: Optional[int] = None
        self._worker_states: Dict[str, str] = {
            self._PANTHEON.name: "running",
            self._FORGE.name: "running",
            self._HADES.name: "running",
            self._DATA_UPDATERS.name: "running",
        }

    async def transition_to(self, mode: RuntimeMode) -> Dict[str, Any]:
        # Transition into a runtime mode and return a structured execution report.
        if isinstance(mode, str):
            mode = RuntimeMode(mode)

        async with self._transition_lock:
            report: Dict[str, Any] = {
                "from_mode": self._current_mode.value,
                "to_mode": mode.value,
                "idempotent": mode == self._current_mode,
                "workers": {},
                "services": {},
                "resource_manager": {},
                "errors": [],
            }

            if report["idempotent"]:
                report["workers"] = self.worker_status()
                report["resource_manager"] = {"gpu_enabled": bool(getattr(self.rm, "gpu_enabled", True))}
                await self.zeus.log_action(
                    {
                        "event": "runtime_transition_complete",
                        "status": "noop",
                        "metadata": report,
                    }
                )
                return report

            if mode == RuntimeMode.DATA_ONLY:
                report["workers"]["pause"] = self.pause_workers("all")
                report["services"]["ollama_stopped"] = await self._stop_ollama()
                if not report["services"]["ollama_stopped"]:
                    error = "Failed to stop Ollama while entering DATA_ONLY."
                    report["errors"].append(error)
                    await self._report_critical(error)
                self._set_gpu_enabled(False)
            elif mode == RuntimeMode.OFFLINE:
                report["workers"]["pause"] = self.pause_workers("all")
                report["services"]["ollama_stopped"] = await self._stop_ollama()
                self._set_gpu_enabled(False)
                logger.info("RuntimeController: Transitioned to OFFLINE (Ollama stopped)")
            elif mode == RuntimeMode.CPU_CHAT:
                report["workers"]["pause"] = self.pause_workers("the_forge,hades_research")
                report["services"]["ollama_stopped"] = await self._stop_ollama()
                if not report["services"]["ollama_stopped"]:
                    error = "Failed to stop Ollama while entering CPU_CHAT."
                    report["errors"].append(error)
                    await self._report_critical(error)
                self._set_gpu_enabled(False)
            elif mode == RuntimeMode.ACTIVE:
                report["workers"]["resume"] = self.resume_workers("all")
                report["services"]["ollama_started"] = await self._start_ollama()
                if not report["services"]["ollama_started"]:
                    error = "Failed to start Ollama while entering ACTIVE."
                    report["errors"].append(error)
                    await self._report_critical(error)
                self._set_gpu_enabled(True)

            self._current_mode = mode
            report["resource_manager"] = {"gpu_enabled": bool(getattr(self.rm, "gpu_enabled", True))}
            report["workers"]["status"] = self.worker_status()

            await self.zeus.log_action(
                {
                    "event": "runtime_transition_complete",
                    "status": "success" if not report["errors"] else "degraded",
                    "metadata": report,
                }
            )
            return report

    def pause_workers(self, scope: str) -> Dict[str, List[str]]:
        # Signal workers to gracefully pause by scope while preserving data updaters.
        targets = self._resolve_scopes(scope)
        changed: List[str] = []
        already: List[str] = []
        skipped: List[str] = []

        for name in targets:
            if name == self._DATA_UPDATERS.name:
                skipped.append(name)
                continue
            if self._worker_states.get(name) == "paused":
                already.append(name)
                continue
            self._worker_states[name] = "paused"
            changed.append(name)

        return {"paused": changed, "already_paused": already, "skipped": skipped}

    def resume_workers(self, scope: str) -> Dict[str, List[str]]:
        # Resume worker loops by scope.
        targets = self._resolve_scopes(scope)
        changed: List[str] = []
        already: List[str] = []

        for name in targets:
            if self._worker_states.get(name) == "running":
                already.append(name)
                continue
            self._worker_states[name] = "running"
            changed.append(name)

        return {"resumed": changed, "already_running": already}

    def worker_status(self) -> Dict[str, str]:
        # Get current worker state by scope.
        return dict(self._worker_states)

    def _set_gpu_enabled(self, enabled: bool) -> None:
        setattr(self.rm, "gpu_enabled", enabled)

    def _resolve_scopes(self, scope: str) -> List[str]:
        normalized = (scope or "").strip().lower()
        if normalized in {"all", "*"}:
            return list(self._worker_states.keys())

        raw_tokens = [token.strip() for token in normalized.split(",") if token.strip()]
        tokens: List[str] = []
        alias_map = {
            "pantheon": self._PANTHEON.name,
            "pantheon_roles": self._PANTHEON.name,
            "forge": self._FORGE.name,
            "the_forge": self._FORGE.name,
            "the_forge_discovery": self._FORGE.name,
            "hades": self._HADES.name,
            "hades_research": self._HADES.name,
            "data": self._DATA_UPDATERS.name,
            "data_updaters": self._DATA_UPDATERS.name,
        }
        for token in raw_tokens:
            mapped = alias_map.get(token, token)
            if mapped in self._worker_states:
                tokens.append(mapped)
        return tokens

    async def _stop_ollama(self) -> bool:
        # Best-effort stop for Ollama, prioritizing service manager then process kill.

        def _runner() -> bool:
            system = platform.system().lower()
            if system == "windows":
                return self._try_windows_service_stop() or self._try_windows_process_kill()
            return self._try_unix_stop()

        return await asyncio.to_thread(_runner)

    async def _start_ollama(self) -> bool:
        # Best-effort start for Ollama service/process.

        def _runner() -> bool:
            system = platform.system().lower()
            if system == "windows":
                return self._try_windows_service_start() or self._try_windows_process_start()
            return self._try_unix_start()

        return await asyncio.to_thread(_runner)

    async def _report_critical(self, message: str) -> None:
        logger.critical(message)
        await self.zeus.log_action(
            {
                "event": "runtime_transition_error",
                "severity": "CRITICAL",
                "metadata": {"message": message},
            }
        )

    @staticmethod
    def _run_command(command: List[str]) -> bool:
        try:
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
            return completed.returncode == 0
        except OSError:
            return False

    def _try_windows_service_stop(self) -> bool:
        return self._run_command(["sc", "stop", "Ollama"])

    def _try_windows_service_start(self) -> bool:
        return self._run_command(["sc", "start", "Ollama"])

    def _try_windows_process_kill(self) -> bool:
        # /F = force, /T = tree (kills children like llama-server), /IM = image name
        success = self._run_command(["taskkill", "/F", "/T", "/IM", "ollama.exe"])
        # Also explicitly target llama-server.exe if it exists, as it's the actual VRAM consumer
        self._run_command(["taskkill", "/F", "/T", "/IM", "llama-server.exe"])
        return success

    def _try_windows_process_start(self) -> bool:
        try:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except OSError:
            return False

    def _try_unix_stop(self) -> bool:
        if self._run_command(["systemctl", "--user", "stop", "ollama"]):
            return True
        if self._run_command(["systemctl", "stop", "ollama"]):
            return True

        try:
            output = subprocess.check_output(["pgrep", "-x", "ollama"], text=True)
        except (OSError, subprocess.CalledProcessError):
            return False

        success = True
        for pid in [line.strip() for line in output.splitlines() if line.strip().isdigit()]:
            try:
                pid_int = int(pid)
                os.kill(pid_int, signal.SIGTERM)
                # Wait up to 5 seconds for graceful exit
                import time as _time
                for _ in range(50):
                    _time.sleep(0.1)
                    try:
                        os.kill(pid_int, 0)  # Check if still alive
                    except ProcessLookupError:
                        break  # Process exited
                else:
                    # Still alive after 5s — escalate to SIGKILL
                    try:
                        os.kill(pid_int, signal.SIGKILL)
                        logger.warning("Ollama PID %d did not exit after SIGTERM; sent SIGKILL", pid_int)
                    except ProcessLookupError:
                        pass  # Died between check and kill
            except OSError:
                success = False
        return success

    def _try_unix_start(self) -> bool:
        if self._run_command(["systemctl", "--user", "start", "ollama"]):
            return True
        if self._run_command(["systemctl", "start", "ollama"]):
            return True
        try:
            proc = subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._ollama_pid = proc.pid
            logger.info("Started Ollama serve (PID %d)", proc.pid)
            return True
        except OSError:
            return False
