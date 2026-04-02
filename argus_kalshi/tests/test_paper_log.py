# Created by Oliver Meihls

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

from argus_kalshi import paper_log


def test_paper_log_threadsafe_writes_jsonl(monkeypatch):
    tmp_dir = Path("tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / f"paper_trades_{uuid4().hex}.jsonl"
    monkeypatch.setattr(paper_log, "_PAPER_LOG_PATH", str(path))

    def _write(i: int) -> None:
        paper_log.append_paper_log_sync({"type": "paper_fill", "timestamp": float(i), "i": i})

    with ThreadPoolExecutor(max_workers=16) as ex:
        for i in range(500):
            ex.submit(_write, i)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 500
    parsed = [json.loads(line) for line in lines]
    assert len(parsed) == 500
    assert sorted(int(r["i"]) for r in parsed) == list(range(500))
    path.unlink(missing_ok=True)
