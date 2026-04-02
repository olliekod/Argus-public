# Created by Oliver Meihls

# One-command post-run pipeline:
# 1) Build runpack
# 2) Build cross-run rollup
# 3) Run readiness gate
#
# Exit codes:
# 0 -> readiness PASS
# 1 -> readiness FAIL
# 2 -> pipeline/config/runtime error

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def _run(cmd: List[str]) -> int:
    print("[pipeline] >", " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def _latest_rollup_json(rollup_dir: Path) -> Path | None:
    files = sorted(rollup_dir.glob("rollup_summary_*.json"))
    return files[-1] if files else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Run runpack + rollup + readiness gate in one command")
    ap.add_argument("--log", default="logs/paper_trades.jsonl", help="Path to paper_trades.jsonl")
    ap.add_argument("--config", default="config/kalshi_family_adaptive.yaml", help="Config used for runpack hash")
    ap.add_argument("--hours", type=float, default=8.0, help="Target window hours (default: 8)")
    ap.add_argument("--out-root", default="logs/analysis", help="Runpack root dir (default: logs/analysis)")
    ap.add_argument("--rollup-out", default="logs/analysis/rollup", help="Rollup output dir")
    ap.add_argument("--min-settlements", type=int, default=50, help="Min settlements for rollup/gate")
    ap.add_argument("--gate-config", default="", help="Optional readiness thresholds JSON")
    ap.add_argument("--json-out", default="", help="Optional readiness JSON report path")
    args = ap.parse_args()

    py = sys.executable
    out_root = Path(args.out_root)
    rollup_out = Path(args.rollup_out)

    cmd_runpack = [
        py,
        "scripts/kalshi_run_pack.py",
        "--log",
        args.log,
        "--hours",
        str(args.hours),
        "--out-root",
        str(out_root),
        "--config",
        args.config,
        "--validate",
    ]
    rc = _run(cmd_runpack)
    if rc != 0:
        print("[pipeline] runpack failed")
        sys.exit(2)

    cmd_rollup = [
        py,
        "scripts/kalshi_runpack_rollup.py",
        "--dir",
        str(out_root),
        "--out",
        str(rollup_out),
        "--hours",
        str(args.hours),
        "--min-settlements",
        str(args.min_settlements),
    ]
    rc = _run(cmd_rollup)
    if rc != 0:
        print("[pipeline] rollup failed")
        sys.exit(2)

    latest_rollup = _latest_rollup_json(rollup_out)
    if latest_rollup is None:
        print("[pipeline] no rollup_summary_*.json found")
        sys.exit(2)

    cmd_gate = [
        py,
        "scripts/kalshi_readiness_gate.py",
        "--dir",
        str(out_root),
        "--rollup",
        str(latest_rollup),
        "--hours",
        str(args.hours),
        "--min-settlements",
        str(args.min_settlements),
    ]
    if args.gate_config:
        cmd_gate.extend(["--config", args.gate_config])
    if args.json_out:
        cmd_gate.extend(["--json-out", args.json_out])

    rc = _run(cmd_gate)
    # Preserve readiness gate contract: 0 pass / 1 fail
    if rc in (0, 1):
        sys.exit(rc)

    print("[pipeline] readiness gate execution error")
    sys.exit(2)


if __name__ == "__main__":
    main()

