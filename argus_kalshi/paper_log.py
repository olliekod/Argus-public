from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict

from .logging_utils import ComponentLogger

log = ComponentLogger("paper_log")

_PAPER_LOG_PATH = "logs/paper_trades.jsonl"
_PAPER_LOG_LOCK = threading.Lock()


def append_paper_log_sync(record: Dict[str, Any]) -> None:
    """Thread-safe JSONL append used by execution + settlement paths."""
    try:
        os.makedirs("logs", exist_ok=True)
        line = json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n"
        with _PAPER_LOG_LOCK:
            with open(_PAPER_LOG_PATH, "a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()
    except Exception as exc:
        log.warning(f"paper_log write failed: {exc}")
