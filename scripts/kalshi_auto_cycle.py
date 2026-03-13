#!/usr/bin/env python3
"""
kalshi_auto_cycle.py — Automated Kalshi farm run loop.

Each cycle:
  1. Start farm process
  2. Wait for --duration-hours
  3. Send SIGTERM to farm; wait for clean exit
  4. Archive logs/paper_trades.jsonl → logs/training_data/run_TIMESTAMP.jsonl
  5. Run kalshi_run_pack.py → logs/analysis/
  6. Run kalshi_apply_promotion.py → update context policy
  7. Run reset_kalshi_paper.py (preserves training archive)
  8. Run kalshi_auto_promote.py on schedule → update promoted_bot_id for UI
  9. Repeat until --cycles reached or KeyboardInterrupt

Usage:
  python scripts/kalshi_auto_cycle.py
  python scripts/kalshi_auto_cycle.py --cycles 6 --duration-hours 8
  python scripts/kalshi_auto_cycle.py --dry-run --cycles 1 --duration-hours 0.01
  python scripts/kalshi_auto_cycle.py --skip-farm --cycles 1 --duration-hours 0
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def _ts() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")


def log(msg: str) -> None:
    print(f"[auto_cycle {datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _resolve_settings_path(settings: str) -> Path:
    path = Path(settings)
    if path.is_absolute():
        return path
    return ROOT / path


def _apply_farm_cycle_offset(text: str, cycle_offset: int) -> str:
    newline = "\r\n" if "\r\n" in text else "\n"
    line_re = re.compile(
        r"(?m)^(?P<indent>\s*)farm_cycle_offset:\s*(?:\"[^\"]*\"|'[^']*'|[^\n#]*)(?P<comment>\s*(?:#.*)?)$"
    )
    match = line_re.search(text)
    if match:
        indent = match.group("indent")
        comment = match.group("comment") or ""
        replacement = f"{indent}farm_cycle_offset: {int(cycle_offset)}{comment}"
        return text[:match.start()] + replacement + text[match.end():]

    farm_match = re.search(r"(?m)^(?P<indent>\s*)farm:\s*(?:#.*)?$", text)
    if not farm_match:
        raise ValueError("Settings YAML is missing an argus_kalshi.farm block")
    child_indent = farm_match.group("indent") + "  "
    insert_line = f"{child_indent}farm_cycle_offset: {int(cycle_offset)}"
    return text[:farm_match.end()] + newline + insert_line + text[farm_match.end():]


@contextmanager
def temporary_farm_cycle_offset(settings: str, cycle_offset: int, dry_run: bool):
    settings_path = _resolve_settings_path(settings)
    if dry_run:
        log(
            f"[DRY RUN] Would set argus_kalshi.farm.farm_cycle_offset={cycle_offset} "
            f"in {settings_path}"
        )
        yield
        return

    original_text = settings_path.read_text(encoding="utf-8")
    updated_text = _apply_farm_cycle_offset(original_text, cycle_offset)
    settings_path.write_text(updated_text, encoding="utf-8")
    log(f"Set farm_cycle_offset={cycle_offset} in {settings_path}")
    try:
        yield
    finally:
        settings_path.write_text(original_text, encoding="utf-8")
        log(f"Restored {settings_path}")


def archive_paper_trades(dry_run: bool) -> Path | None:
    """Copy paper_trades.jsonl to logs/training_data/run_TIMESTAMP.jsonl."""
    src = ROOT / "logs" / "paper_trades.jsonl"
    if not src.exists() or src.stat().st_size == 0:
        log("paper_trades.jsonl empty or missing — skipping archive")
        return None
    dest_dir = ROOT / "logs" / "training_data"
    dest = dest_dir / f"run_{_ts()}.jsonl"
    if dry_run:
        log(f"[DRY RUN] Would archive {src.name} -> training_data/{dest.name}")
        return dest
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    log(f"Archived {src.stat().st_size // 1024}KB → {dest.name}")
    return dest


def run_script(script: str, *args: str, dry_run: bool = False) -> int:
    """Run a script in the scripts/ directory."""
    cmd = [PYTHON, str(ROOT / "scripts" / script), *args]
    label = " ".join(Path(c).name if c.endswith(".py") else c for c in cmd)
    if dry_run:
        log(f"[DRY RUN] Would run: {label}")
        return 0
    log(f"Running: {label}")
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + existing if existing else "")
    result = subprocess.run(cmd, cwd=ROOT, env=env)
    return result.returncode


def start_farm(settings: str) -> subprocess.Popen:
    cmd = [PYTHON, "-m", "argus_kalshi", "--settings", settings]
    log(f"Starting farm: {' '.join(cmd)}")
    return subprocess.Popen(cmd, cwd=ROOT)


def stop_farm(proc: subprocess.Popen, timeout: float = 30.0) -> None:
    if proc.poll() is not None:
        log(f"Farm already exited (code {proc.returncode})")
        return
    log("Sending shutdown signal to farm...")
    if os.name == "nt":
        proc.terminate()
    else:
        proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=timeout)
        log(f"Farm exited cleanly (code {proc.returncode})")
    except subprocess.TimeoutExpired:
        log("Farm did not exit within timeout — killing")
        proc.kill()
        proc.wait()


def run_backtest_eval(dry_run: bool, skip_cross_val: bool = False) -> None:
    """Cross-validate winner zone stability across runs (>=2 tapes required).

    Note: grid_eval (param sweep against tape) is intentionally NOT run here.
    The farm's 7,488 bots already constitute a live grid search — each bot is
    one param combo. kalshi_bot_performance.json (written by apply_promotion)
    captures per-param PnL directly. Running grid_eval against the aggregate
    tape would mix signals from all bots and produce misleading results.
    Cross-validation is still useful: it checks whether the winner zone is
    stable across runs or just overfit to one market regime.
    """
    tape_dir = ROOT / "logs" / "decision_tape"
    training_dir = ROOT / "logs" / "training_data"
    settlement_glob = str(training_dir / "*.jsonl")

    tape_files = sorted(tape_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime) if tape_dir.exists() else []
    if skip_cross_val:
        return
    if len(tape_files) < 2:
        log(f"Only {len(tape_files)} tape file(s) — skipping cross-validation (need >=2 runs)")
        return
    log(f"Cross-validating across {len(tape_files)} tape file(s)...")
    run_script(
        "kalshi_cross_validate.py",
        "--tape-dir", str(tape_dir),
        "--settlement", settlement_glob,
        dry_run=dry_run,
    )


def run_cycle(
    cycle_num: int,
    total_cycles: int,
    settings: str,
    duration_hours: float,
    dry_run: bool,
    skip_farm: bool,
    skip_backtest: bool = False,
    promote_every: int = 3,
) -> None:
    log(f"=== Cycle {cycle_num}/{total_cycles} start ===")

    farm_proc: subprocess.Popen | None = None
    cycle_ctx = temporary_farm_cycle_offset(settings, cycle_num, dry_run=dry_run) if (dry_run or not skip_farm) else nullcontext()
    with cycle_ctx:
        if not skip_farm and not dry_run:
            farm_proc = start_farm(settings)
        elif dry_run:
            log(f"[DRY RUN] Would start farm: python -m argus_kalshi --settings {settings}")

        duration_s = duration_hours * 3600
        if duration_s > 0:
            log(f"Waiting {duration_hours:.2f}h ({duration_s:.0f}s)...")
        try:
            if farm_proc is not None:
                farm_proc.wait(timeout=duration_s)
                log("Farm exited early — continuing with cycle teardown")
            elif duration_s > 0:
                time.sleep(duration_s)
        except subprocess.TimeoutExpired:
            pass  # normal — farm ran for full duration
        except KeyboardInterrupt:
            log("Interrupted — stopping farm")
            if farm_proc is not None:
                stop_farm(farm_proc)
            raise

        if farm_proc is not None:
            stop_farm(farm_proc)

    # Archive training data BEFORE reset
    archive_paper_trades(dry_run=dry_run)

    # Run analysis pack (uses --out-root; run_pack creates its own timestamped subdir)
    run_script("kalshi_run_pack.py", "--out-root", str(ROOT / "logs" / "analysis"), dry_run=dry_run)

    # Apply promotions → update context policy
    run_script("kalshi_apply_promotion.py", "--merge", dry_run=dry_run)

    # Backtest eval (grid param search + cross-validation)
    if not skip_backtest:
        run_backtest_eval(dry_run=dry_run)

    # Reset paper state (preserves training archive, uses --no-backup since we archived above)
    run_script("reset_kalshi_paper.py", "--no-backup", "--keep-policy", dry_run=dry_run)

    should_run_promoter = (
        cycle_num == total_cycles
        or (promote_every > 0 and cycle_num % promote_every == 0)
    )
    if should_run_promoter:
        run_script(
            "kalshi_auto_promote.py",
            "--settings", settings,
            "--lifetime", str(ROOT / "config" / "kalshi_lifetime_performance.json"),
            "--tape-dir", str(ROOT / "logs" / "training_data"),
            dry_run=dry_run,
        )
    else:
        log(
            f"Skipping promoted-bot evaluation this cycle "
            f"(runs every {promote_every} cycles; final cycle always runs)"
        )

    log(f"=== Cycle {cycle_num}/{total_cycles} complete ===\n")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Automated Kalshi farm run loop — runs N cycles of (farm + archive + promote + reset)."
    )
    ap.add_argument(
        "--cycles", type=int, default=6,
        help="Number of cycles to run"
    )
    ap.add_argument(
        "--duration-hours", type=float, default=8.0,
        help="Duration of each farm run in hours"
    )
    ap.add_argument(
        "--settings", default="config/kalshi_family_adaptive.yaml",
        help="Farm settings YAML (default: config/kalshi_family_adaptive.yaml)"
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Print what would happen without starting the farm or modifying files"
    )
    ap.add_argument(
        "--skip-farm", action="store_true",
        help="Skip starting/stopping the farm process (for testing teardown logic)"
    )
    ap.add_argument(
        "--skip-backtest", action="store_true",
        help="Skip grid eval and cross-validation (useful when tape is disabled or for quick cycles)"
    )
    ap.add_argument(
        "--promote-every", type=int, default=3,
        help="Run kalshi_auto_promote.py every N cycles and always on the final cycle"
    )
    args = ap.parse_args()

    log(f"Starting auto-cycle: {args.cycles} cycle(s) × {args.duration_hours}h each")
    if args.dry_run:
        log("DRY RUN mode — no farm started, no files modified")

    for i in range(1, args.cycles + 1):
        try:
            run_cycle(
                cycle_num=i,
                total_cycles=args.cycles,
                settings=args.settings,
                duration_hours=args.duration_hours,
                dry_run=args.dry_run,
                skip_farm=args.skip_farm,
                skip_backtest=args.skip_backtest,
                promote_every=args.promote_every,
            )
        except KeyboardInterrupt:
            log(f"Auto-cycle interrupted at cycle {i}/{args.cycles}")
            return 1

    log(f"All {args.cycles} cycle(s) complete.")
    log("Next step: python scripts/kalshi_aggregate_training.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
