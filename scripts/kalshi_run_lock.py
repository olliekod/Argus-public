"""
Kalshi clean-run lock helper.

Purpose:
- Capture a baseline workspace/settings snapshot at run start.
- Verify later whether a run remained "clean" or became "contaminated".

Design notes:
- Existing pre-run dirty files are allowed and recorded as baseline.
- A run is marked contaminated only if state changes relative to baseline.
- Volatile runtime paths are ignored by default (logs/, data/, tmp/, caches).

Usage:
    python scripts/kalshi_run_lock.py start --settings config/kalshi_family_adaptive.yaml
    python scripts/kalshi_run_lock.py check --run-id <run_id>
    python scripts/kalshi_run_lock.py check-latest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


LOCK_ROOT = Path("logs/run_locks")
IGNORE_PREFIXES = (
    "logs/",
    "data/",
    "tmp/",
    ".pytest_cache/",
    "__pycache__/",
)
IGNORE_SUFFIXES = (".pyc", ".pyo", ".md")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _run_git(args: List[str]) -> Tuple[int, str, str]:
    proc = subprocess.run(
        ["git", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc.returncode, proc.stdout, proc.stderr


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/").strip()


def _is_ignored(path: str) -> bool:
    p = _normalize_path(path)
    if p.startswith(IGNORE_PREFIXES):
        return True
    return p.endswith(IGNORE_SUFFIXES)


def _sha256_file(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class DirtyRow:
    x: str
    y: str
    path: str


def _parse_porcelain_v1(raw: str) -> List[DirtyRow]:
    rows: List[DirtyRow] = []
    for line in raw.splitlines():
        if not line:
            continue
        if len(line) < 4:
            continue
        x, y = line[0], line[1]
        body = line[3:]
        if " -> " in body:
            body = body.split(" -> ", 1)[1]
        body = _normalize_path(body)
        rows.append(DirtyRow(x=x, y=y, path=body))
    return rows


def _snapshot_workspace() -> Dict[str, object]:
    rc, out, err = _run_git(["rev-parse", "HEAD"])
    head = out.strip() if rc == 0 else None
    git_ok = rc == 0
    git_error = err.strip() if rc != 0 else ""

    dirty_rows: List[DirtyRow] = []
    if git_ok:
        rc2, out2, err2 = _run_git(["status", "--porcelain=1"])
        if rc2 != 0:
            git_ok = False
            git_error = err2.strip() or "git status failed"
        else:
            dirty_rows = _parse_porcelain_v1(out2)

    file_map: Dict[str, Dict[str, Optional[str]]] = {}
    for row in dirty_rows:
        path = row.path
        if _is_ignored(path):
            continue
        p = Path(path)
        file_map[path] = {
            "xy": f"{row.x}{row.y}",
            "sha256": _sha256_file(p),
            "exists": p.exists(),
        }

    return {
        "git_ok": git_ok,
        "git_error": git_error,
        "head": head,
        "dirty_files": file_map,
    }


def _snapshot_settings(settings_path: Path) -> Dict[str, object]:
    return {
        "path": str(settings_path).replace("\\", "/"),
        "exists": settings_path.exists(),
        "sha256": _sha256_file(settings_path),
    }


def _load_lock(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_lock(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _compare_lock(lock: Dict[str, object]) -> Dict[str, object]:
    baseline_ws = lock["baseline"]["workspace"]  # type: ignore[index]
    baseline_settings = lock["baseline"]["settings"]  # type: ignore[index]
    settings_path = Path(str(baseline_settings["path"]))  # type: ignore[index]

    current_ws = _snapshot_workspace()
    current_settings = _snapshot_settings(settings_path)

    reasons: List[str] = []
    details: Dict[str, object] = {
        "new_or_changed_files": [],
        "removed_files": [],
        "head_changed": False,
        "settings_changed": False,
    }

    b_files: Dict[str, Dict[str, object]] = baseline_ws["dirty_files"]  # type: ignore[index]
    c_files: Dict[str, Dict[str, object]] = current_ws["dirty_files"]  # type: ignore[index]

    b_paths = set(b_files.keys())
    c_paths = set(c_files.keys())

    for path in sorted(c_paths - b_paths):
        details["new_or_changed_files"].append({"path": path, "baseline": None, "current": c_files[path]})  # type: ignore[index]
    for path in sorted(b_paths - c_paths):
        details["removed_files"].append({"path": path, "baseline": b_files[path], "current": None})  # type: ignore[index]
    for path in sorted(b_paths & c_paths):
        b = b_files[path]
        c = c_files[path]
        if b.get("xy") != c.get("xy") or b.get("sha256") != c.get("sha256"):
            details["new_or_changed_files"].append({"path": path, "baseline": b, "current": c})  # type: ignore[index]

    if details["new_or_changed_files"] or details["removed_files"]:  # type: ignore[index]
        reasons.append("workspace_changed")

    b_head = baseline_ws.get("head")
    c_head = current_ws.get("head")
    if b_head != c_head:
        details["head_changed"] = True
        reasons.append("git_head_changed")

    if baseline_settings.get("sha256") != current_settings.get("sha256"):
        details["settings_changed"] = True
        reasons.append("settings_changed")

    status = "contaminated" if reasons else "clean"
    return {
        "status": status,
        "reasons": reasons,
        "details": details,
        "current": {
            "workspace": current_ws,
            "settings": current_settings,
            "checked_at_utc": _utc_now(),
        },
    }


def _latest_lock_file() -> Optional[Path]:
    if not LOCK_ROOT.exists():
        return None
    candidates = sorted(LOCK_ROOT.glob("*.json"))
    return candidates[-1] if candidates else None


def _print_human(report: Dict[str, object], run_id: str) -> None:
    print(f"run_id={run_id}")
    print(f"status={report['status']}")
    reasons = report.get("reasons") or []
    if reasons:
        print(f"reasons={','.join(str(x) for x in reasons)}")
    details = report.get("details") or {}
    changed = details.get("new_or_changed_files") or []
    removed = details.get("removed_files") or []
    print(f"changed_file_count={len(changed)} removed_file_count={len(removed)}")
    if details.get("settings_changed"):
        print("settings_changed=true")
    if details.get("head_changed"):
        print("head_changed=true")


def cmd_start(args: argparse.Namespace) -> int:
    settings_path = Path(args.settings)
    run_id = args.run_id or _utc_now()
    lock_path = LOCK_ROOT / f"{run_id}.json"
    if lock_path.exists():
        print(f"error: lock already exists: {lock_path}", file=sys.stderr)
        return 2

    payload = {
        "run_id": run_id,
        "created_at_utc": _utc_now(),
        "note": args.note or "",
        "baseline": {
            "settings": _snapshot_settings(settings_path),
            "workspace": _snapshot_workspace(),
        },
    }
    _write_lock(lock_path, payload)
    print(f"created={lock_path}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    if args.run_id:
        lock_path = LOCK_ROOT / f"{args.run_id}.json"
    else:
        lock_path = _latest_lock_file()
        if lock_path is None:
            print("error: no lock files found under logs/run_locks", file=sys.stderr)
            return 2
    if lock_path is None or not lock_path.exists():
        print(f"error: lock not found: {lock_path}", file=sys.stderr)
        return 2

    lock = _load_lock(lock_path)
    report = _compare_lock(lock)
    run_id = str(lock.get("run_id") or lock_path.stem)
    _print_human(report, run_id)

    if args.write_report:
        out = lock_path.with_name(f"{run_id}.check.json")
        _write_lock(out, report)
        print(f"report={out}")

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Run integrity lock/check helper for Kalshi experiments")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_start = sub.add_parser("start", help="Capture baseline lock for a run")
    ap_start.add_argument("--settings", default="config/kalshi_family_adaptive.yaml", help="Settings file path")
    ap_start.add_argument("--run-id", help="Optional run id (default UTC timestamp)")
    ap_start.add_argument("--note", help="Optional note")
    ap_start.set_defaults(func=cmd_start)

    ap_check = sub.add_parser("check", help="Check a run lock for contamination")
    ap_check.add_argument("--run-id", help="Run id to check; default latest")
    ap_check.add_argument("--write-report", action="store_true", help="Write JSON report beside lock file")
    ap_check.set_defaults(func=cmd_check)

    ap_latest = sub.add_parser("check-latest", help="Check latest run lock for contamination")
    ap_latest.add_argument("--write-report", action="store_true", help="Write JSON report beside lock file")
    ap_latest.set_defaults(func=cmd_check)

    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
