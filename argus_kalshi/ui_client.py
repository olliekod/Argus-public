"""
Standalone UI client for Argus Kalshi when visualizer_process == "separate".

Connect to the trading process IPC server and run the terminal UI from
received snapshots so the trading event loop is not blocked by rendering.

    python -m argus_kalshi.ui_client --connect 127.0.0.1:9999
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import traceback
from typing import Optional

# StreamReader limit for IPC; 7488 bots + 863 markets can exceed 4 MB per snapshot line.
IPC_READ_LIMIT = 32 * 1024 * 1024  # 32 MB


def _crash_log_path() -> str:
    """Use cwd so the log is always in project/logs/ when run as subprocess from runner."""
    return os.path.join(os.getcwd(), "logs", "argus_ui_crash.log")


def _log_crash(msg: str, exc: Optional[Exception] = None) -> None:
    """Write crash info to a file so it survives the terminal closing."""
    path = _crash_log_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n--- {msg} ---\n")
            if exc:
                traceback.print_exception(type(exc), exc, getattr(exc, "__traceback__", None), file=f)
    except Exception:
        pass
    print(msg, file=sys.stderr)
    if exc:
        traceback.print_exception(type(exc), exc, getattr(exc, "__traceback__", None), file=sys.stderr)


def _checkpoint(msg: str) -> None:
    """Append a checkpoint line to the crash log (no exception). Use to see how far we got before a non-Python crash."""
    path = _crash_log_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"  checkpoint: {msg}\n")
    except Exception:
        pass


async def _connect_with_retry(host: str, port: int, max_attempts: int = 10) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Connect to IPC server with retries so UI can start before backend is fully ready."""
    for attempt in range(max_attempts):
        try:
            return await asyncio.open_connection(host, port, limit=IPC_READ_LIMIT)
        except (ConnectionRefusedError, OSError):
            if attempt == max_attempts - 1:
                raise
            await asyncio.sleep(0.5 * (attempt + 1))


async def run_ui_client(host: str, port: int) -> None:
    from .terminal_ui import TerminalVisualizer
    from .ipc import ipc_client_recv_line_async

    def _task_done_cb(task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            _log_crash(f"Async task failed: {task.get_name()!r}", exc)

    snapshot_queue: asyncio.Queue = asyncio.Queue(maxsize=2)

    async def reader_task(reader: asyncio.StreamReader) -> None:
        try:
            while True:
                snapshot = await ipc_client_recv_line_async(reader)
                if snapshot is None:
                    break
                try:
                    snapshot_queue.put_nowait(snapshot)
                except asyncio.QueueFull:
                    snapshot_queue.get_nowait()
                    snapshot_queue.put_nowait(snapshot)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            _log_crash(f"UI reader error: {e}", e)

    reader, writer = await _connect_with_retry(host, port)
    _checkpoint("IPC connected")
    recv_task = asyncio.create_task(reader_task(reader))
    vision = TerminalVisualizer(
        bus=None,
        metadata={},
        dry_run=True,
        primary_bot_id=None,
        leaderboard_only=False,
        unique_config_count=None,
        initial_kalshi_rtt_ms=None,
        remote_snapshot_queue=snapshot_queue,
        checkpoint_cb=_checkpoint,
    )
    await vision.start()
    _checkpoint("vision.start() done")
    if vision._task:
        vision._task.add_done_callback(_task_done_cb)
    try:
        await recv_task
    except asyncio.CancelledError:
        pass
    finally:
        await vision.stop()
        writer.close()
        await writer.wait_closed()


def main(connect: Optional[str] = None) -> None:
    # Log startup so we know the process ran and where the log file is
    try:
        path = _crash_log_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n--- UI process started ---\n")
    except Exception:
        pass

    def _excepthook(etyp, val, tb):
        _log_crash(f"Uncaught {etyp.__name__}: {val}", val if isinstance(val, Exception) else None)
        if tb:
            traceback.print_exception(etyp, val, tb, file=sys.stderr)
        sys.__excepthook__(etyp, val, tb)
    sys.excepthook = _excepthook

    parser = argparse.ArgumentParser(
        prog="argus_kalshi.ui_client",
        description="Connect to Kalshi trading process IPC and run terminal UI",
    )
    parser.add_argument(
        "--connect",
        default=connect or "127.0.0.1:9999",
        help="Host:port of the trading process IPC server (default: 127.0.0.1:9999)",
    )
    parser.add_argument("--ui-only", action="store_true", help=argparse.SUPPRESS)  # accepted when invoked via python -m argus_kalshi --ui-only
    args = parser.parse_args()
    parts = (args.connect or "127.0.0.1:9999").rsplit(":", 1)
    host = parts[0] if len(parts) == 2 else "127.0.0.1"
    port = int(parts[1]) if len(parts) == 2 else 9999
    try:
        asyncio.run(run_ui_client(host, port))
    except KeyboardInterrupt:
        pass
    except ConnectionRefusedError:
        print(f"Cannot connect to {host}:{port} — is the trading process running with visualizer_process=separate?", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        _log_crash(f"UI client error: {e}", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
