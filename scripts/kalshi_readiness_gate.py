"""
Kalshi runpack readiness gate — strict PASS/FAIL from JSON artifacts only.

Reads metrics_summary.json (and optionally rollup JSON) from one or more runpack
directories. Applies hard threshold checks; prints a structured PASS/FAIL report
with exact failed checks and remediation hints.

Usage:
    python scripts/kalshi_readiness_gate.py logs/analysis/runpack_20260308_1600_abc12345
    python scripts/kalshi_readiness_gate.py --rollup logs/analysis/rollup/rollup_summary_*.json
    python scripts/kalshi_readiness_gate.py --dir logs/analysis --hours 8 --min-settlements 200

Exit codes:
    0 — PASS
    1 — FAIL
    2 — Usage/config error
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
#  Thresholds (all configurable via --config JSON)
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS: Dict[str, Any] = {
    # -------------------------------------------------------------------------
    # Sample size
    # Derivation: statistical power for a binary outcome.
    # To distinguish a 55% win rate from 50% at 95% confidence requires
    # n ≥ (1.96 / 0.05)² × 0.25 ≈ 384. We use 200 as a practical minimum
    # (gives ±7% CI), enough to separate a real edge from noise.
    # *** PENDING CALIBRATION: tighten once we have 10+ runs of real data. ***
    "min_settlements": 200,

    # -------------------------------------------------------------------------
    # PnL quality
    # min_avg_pnl_usd: must be strictly positive — pending calibration for a
    # meaningful floor (need empirical distribution from real runs).
    "min_avg_pnl_usd": 0.0,

    # min_win_rate_pct: breakeven derivation.
    # Kalshi fee ≈ 2% of notional = ~2 cents at avg entry 50c.
    # Breakeven WR = (entry_cents + fee_cents) / 100 = (50 + 2) / 100 = 52%.
    # We gate at 52% so a strategy must beat its own cost of capital.
    # Source: config effective_edge_fee_pct=0.02, avg entry ~50c.
    "min_win_rate_pct": 52.0,

    # min_robust_score: must be positive (penalised avg_pnl > 0).
    # *** PENDING CALIBRATION: set a meaningful floor from real run distribution. ***
    "min_robust_score": 0.0,

    # -------------------------------------------------------------------------
    # Risk
    # max_drawdown_usd: derived from risk budget.
    # Daily drawdown limit = 5% × $5,000 = $250 (from config).
    # 8h window = 1/3 of a day → window budget = $250 × (8/24) = $83.
    # Gate threshold = 80% of budget (warn before hitting hard stop) → $66.
    # Rounded to $65. Scale with --config if bankroll or window_hours differs.
    # Source: config bankroll_usd=5000, daily_drawdown_limit=0.05.
    "max_drawdown_usd": 65.0,

    # max_top_concentration_share / max_market_side_hhi:
    # *** PENDING CALIBRATION: needs empirical run data. ***
    # Placeholder bounds are conservative guesses only. With ~20-30 active
    # Kalshi markets, uniform spread ≈ 4-5% per market; >30% is concerning.
    "max_top_concentration_share": 0.40,
    "max_market_side_hhi": 0.20,

    # -------------------------------------------------------------------------
    # Context / sleeve health
    # *** ALL PENDING CALIBRATION: entirely system-specific, no first-principles
    # derivation available. Placeholders only. ***
    "min_context_coverage_pct": 30.0,
    "max_edge_retention_poor_keys": 5,
    "min_promotable_contexts": 1,

    # -------------------------------------------------------------------------
    # Rollup consistency (only checked when --rollup is provided)
    # *** PENDING CALIBRATION ***
    "max_rollup_rank": 3,
    "min_rollup_runs": 2,
}


# ---------------------------------------------------------------------------
#  Data structures
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    remediation: str = ""


@dataclass
class GateReport:
    runpack: str
    window_hours: float
    checks: List[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failed_checks(self) -> List[CheckResult]:
        return [c for c in self.checks if not c.passed]


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[gate] ERROR reading {path}: {e}", file=sys.stderr)
        return None


def _find_window(metrics: Dict[str, Any], target_hours: float) -> Optional[Dict[str, Any]]:
    windows = metrics.get("windows", [])
    if not windows:
        return None
    # Find closest window within 50% of target
    candidates = [w for w in windows if abs(float(w.get("hours", 0)) - target_hours) <= target_hours * 0.5]
    if not candidates:
        return None
    return min(candidates, key=lambda w: abs(float(w.get("hours", 0)) - target_hours))


def _robust_score(w: Dict[str, Any]) -> float:
    avg = float(w.get("avg_pnl_usd", 0.0))
    total_pnl = float(w.get("total_pnl_usd", 0.0))
    dd = float(w.get("max_drawdown_usd", 0.0))
    conc = float(w.get("top_concentration_share", 1.0))
    dd_penalty = min(1.0, dd / max(0.01, abs(total_pnl) if total_pnl != 0 else 0.01))
    conc_penalty = max(0.0, conc - 0.30)
    return avg * (1.0 - 0.5 * dd_penalty) * (1.0 - conc_penalty)


# ---------------------------------------------------------------------------
#  Gate checks
# ---------------------------------------------------------------------------

def _check_sample_size(w: Dict[str, Any], t: Dict[str, Any]) -> CheckResult:
    n = int(w.get("settlements", 0))
    mn = int(t["min_settlements"])
    return CheckResult(
        name="sample_size",
        passed=n >= mn,
        detail=f"settlements={n} (min {mn})",
        remediation=f"Run longer or lower --min-settlements to {max(10, mn // 2)}.",
    )


def _check_avg_pnl(w: Dict[str, Any], t: Dict[str, Any]) -> CheckResult:
    v = float(w.get("avg_pnl_usd", 0.0))
    mn = float(t["min_avg_pnl_usd"])
    return CheckResult(
        name="avg_pnl_positive",
        passed=v > mn,
        detail=f"avg_pnl=${v:+.4f} (must be > {mn:+.4f})",
        remediation="Edge is negative. Check entry thresholds, fee config, and strategy params.",
    )


def _check_win_rate(w: Dict[str, Any], t: Dict[str, Any]) -> CheckResult:
    v = float(w.get("win_rate_pct", 0.0))
    mn = float(t["min_win_rate_pct"])
    return CheckResult(
        name="win_rate",
        passed=v >= mn,
        detail=f"win_rate={v:.1f}% (min {mn:.1f}% — breakeven at ~50c entry + 2c fee)",
        remediation="Win rate below breakeven. Review edge filter, entry price distribution, or fee config.",
    )


def _check_robust_score(w: Dict[str, Any], t: Dict[str, Any]) -> CheckResult:
    score = _robust_score(w)
    mn = float(t["min_robust_score"])
    return CheckResult(
        name="robust_score",
        passed=score > mn,
        detail=f"robust_score={score:.4f} (must be > {mn:.4f})",
        remediation="Risk-adjusted score is non-positive. Reduce concentration or improve raw edge.",
    )


def _check_drawdown(w: Dict[str, Any], t: Dict[str, Any]) -> CheckResult:
    v = float(w.get("max_drawdown_usd", 0.0))
    mx = float(t["max_drawdown_usd"])
    return CheckResult(
        name="max_drawdown",
        passed=v <= mx,
        detail=f"max_drawdown=${v:.2f} (max ${mx:.2f} = 80% of 8h risk budget)",
        remediation=(
            "Drawdown exceeds 8h risk budget. Lower position size, tighten stop config, "
            "or reduce simultaneous exposure. Use --config to override threshold if window_hours differs."
        ),
    )


def _check_concentration(w: Dict[str, Any], t: Dict[str, Any]) -> CheckResult:
    v = float(w.get("top_concentration_share", 1.0))
    mx = float(t["max_top_concentration_share"])
    return CheckResult(
        name="top_concentration",
        passed=v <= mx,
        detail=f"top_concentration={v:.1%} (max {mx:.1%})",
        remediation="A single market-side dominates fills. Increase market diversity or cap per-market quantity.",
    )


def _check_hhi(w: Dict[str, Any], t: Dict[str, Any]) -> CheckResult:
    v = float(w.get("market_side_hhi", 0.0))
    mx = float(t["max_market_side_hhi"])
    # HHI of 0 means data was not computed — skip check gracefully
    if v == 0.0:
        return CheckResult(name="market_side_hhi", passed=True, detail="hhi=0.0 (not computed — skipped)")
    return CheckResult(
        name="market_side_hhi",
        passed=v <= mx,
        detail=f"market_side_hhi={v:.4f} (max {mx:.4f})",
        remediation="Fill concentration is high (few dominant market-sides). Diversify traded markets/families.",
    )


def _check_context_coverage(w: Dict[str, Any], t: Dict[str, Any]) -> CheckResult:
    v = float(w.get("context_coverage_pct", 0.0))
    mn = float(t["min_context_coverage_pct"])
    if v == 0.0:
        # Metric not present in older runpacks — skip
        return CheckResult(name="context_coverage", passed=True, detail="context_coverage=0.0 (not computed — skipped)")
    return CheckResult(
        name="context_coverage",
        passed=v >= mn,
        detail=f"context_coverage={v:.1f}% (min {mn:.1f}%)",
        remediation="Too few settlements have decision context attached. Check context_policy config.",
    )


def _check_edge_retention(w: Dict[str, Any], t: Dict[str, Any]) -> CheckResult:
    v = int(w.get("edge_retention_poor_keys", 0))
    mx = int(t["max_edge_retention_poor_keys"])
    return CheckResult(
        name="edge_retention_poor_keys",
        passed=v <= mx,
        detail=f"edge_retention_poor_keys={v} (max {mx})",
        remediation="Too many context keys showing poor edge retention. Review context_policy promotion thresholds.",
    )


def _check_promotable_contexts(w: Dict[str, Any], t: Dict[str, Any]) -> CheckResult:
    v = int(w.get("promotable_contexts", 0))
    mn = int(t["min_promotable_contexts"])
    return CheckResult(
        name="promotable_contexts",
        passed=v >= mn,
        detail=f"promotable_contexts={v} (min {mn})",
        remediation="No contexts qualify for promotion. Check edge_retention and context_coverage thresholds.",
    )


# ---------------------------------------------------------------------------
#  Rollup consistency check
# ---------------------------------------------------------------------------

def _check_rollup_rank(run_id: str, rollup: Dict[str, Any], t: Dict[str, Any]) -> Optional[CheckResult]:
    runs = rollup.get("runs", [])
    n_runs = len(runs)
    min_runs = int(t["min_rollup_runs"])
    if n_runs < min_runs:
        return CheckResult(
            name="rollup_rank",
            passed=True,
            detail=f"rollup only has {n_runs} run(s) (< {min_runs}) — rank check skipped",
        )
    # Find rank of this run_id (1-based, sorted by robust_score desc)
    rank = next((i + 1 for i, r in enumerate(runs) if r.get("run_id") == run_id), None)
    if rank is None:
        return CheckResult(
            name="rollup_rank",
            passed=False,
            detail=f"run_id '{run_id}' not found in rollup",
            remediation="Run kalshi_runpack_rollup.py to include this runpack, then re-run gate.",
        )
    max_rank = int(t["max_rollup_rank"])
    return CheckResult(
        name="rollup_rank",
        passed=rank <= max_rank,
        detail=f"rollup_rank={rank} (must be ≤ {max_rank} out of {n_runs} runs)",
        remediation=f"This run ranks #{rank} by robust_score. Promote a higher-ranked run instead.",
    )


# ---------------------------------------------------------------------------
#  Core gate logic
# ---------------------------------------------------------------------------

def run_gate(
    runpack_dir: Path,
    thresholds: Dict[str, Any],
    target_hours: float,
    rollup: Optional[Dict[str, Any]] = None,
) -> GateReport:
    ms_path = runpack_dir / "metrics_summary.json"
    metrics = _load_json(ms_path)
    if not metrics:
        # Try manifest fallback
        mf_path = runpack_dir / "manifest.json"
        metrics = _load_json(mf_path) if mf_path.exists() else None
    if not metrics:
        report = GateReport(runpack=runpack_dir.name, window_hours=target_hours)
        report.checks.append(CheckResult(
            name="metrics_loaded",
            passed=False,
            detail=f"Cannot read metrics_summary.json or manifest.json from {runpack_dir}",
            remediation="Re-run kalshi_run_pack.py --validate to regenerate metrics.",
        ))
        return report

    window = _find_window(metrics, target_hours)
    if not window:
        report = GateReport(runpack=runpack_dir.name, window_hours=target_hours)
        report.checks.append(CheckResult(
            name="window_found",
            passed=False,
            detail=f"No window within 50% of {target_hours}h found in metrics",
            remediation=f"Runpack windows: {[w.get('hours') for w in metrics.get('windows', [])]}",
        ))
        return report

    actual_hours = float(window.get("hours", target_hours))
    report = GateReport(runpack=runpack_dir.name, window_hours=actual_hours)
    t = thresholds

    report.checks.extend([
        _check_sample_size(window, t),
        _check_avg_pnl(window, t),
        _check_win_rate(window, t),
        _check_robust_score(window, t),
        _check_drawdown(window, t),
        _check_concentration(window, t),
        _check_hhi(window, t),
        _check_context_coverage(window, t),
        _check_edge_retention(window, t),
        _check_promotable_contexts(window, t),
    ])

    if rollup is not None:
        run_id = metrics.get("run_id", runpack_dir.name)
        rc = _check_rollup_rank(run_id, rollup, t)
        if rc:
            report.checks.append(rc)

    return report


# ---------------------------------------------------------------------------
#  Reporting
# ---------------------------------------------------------------------------

_PASS = "\033[32mPASS\033[0m"
_FAIL = "\033[31mFAIL\033[0m"
_WARN = "\033[33mWARN\033[0m"
_BOLD = "\033[1m"
_RST  = "\033[0m"


def _render_report(report: GateReport) -> str:
    lines = []
    verdict = _PASS if report.passed else _FAIL
    lines.append(f"\n{_BOLD}{'='*60}{_RST}")
    lines.append(f"{_BOLD}Runpack:{_RST} {report.runpack}")
    lines.append(f"{_BOLD}Window: {_RST} {report.window_hours:.1f}h")
    lines.append(f"{_BOLD}Verdict:{_RST} {verdict}")
    lines.append(f"{_BOLD}{'='*60}{_RST}")

    for c in report.checks:
        icon = "  [PASS]" if c.passed else "  [FAIL]"
        color = "\033[32m" if c.passed else "\033[31m"
        lines.append(f"{color}{icon}{_RST}  {c.name}: {c.detail}")
        if not c.passed and c.remediation:
            lines.append(f"          {_WARN}Remediation:{_RST} {c.remediation}")

    if not report.passed:
        lines.append(f"\n{_BOLD}Failed checks ({len(report.failed_checks)}):{_RST}")
        for c in report.failed_checks:
            lines.append(f"  - {c.name}")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Kalshi runpack readiness gate — strict PASS/FAIL from JSON artifacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("runpacks", nargs="*", help="Runpack directories to gate")
    ap.add_argument("--dir", default="logs/analysis",
                    help="Auto-discover runpack_* subdirs here (default: logs/analysis)")
    ap.add_argument("--rollup", metavar="ROLLUP_JSON",
                    help="Path to rollup_summary_*.json for cross-run rank check")
    ap.add_argument("--hours", type=float, default=8.0,
                    help="Target window to evaluate (default: 8h)")
    ap.add_argument("--min-settlements", type=int, default=None,
                    help="Override min_settlements threshold")
    ap.add_argument("--config", metavar="THRESHOLDS_JSON",
                    help="JSON file with threshold overrides")
    ap.add_argument("--json-out", metavar="PATH",
                    help="Write structured JSON report to this path")
    ap.add_argument("--no-color", action="store_true",
                    help="Disable ANSI color codes in output")
    args = ap.parse_args()

    # Load thresholds
    thresholds = dict(DEFAULT_THRESHOLDS)
    if args.config:
        cfg = _load_json(Path(args.config))
        if cfg:
            thresholds.update(cfg)
        else:
            sys.exit(2)
    if args.min_settlements is not None:
        thresholds["min_settlements"] = args.min_settlements

    # Collect runpack dirs
    dirs: List[Path] = []
    if args.runpacks:
        for p in args.runpacks:
            d = Path(p)
            if d.is_dir():
                dirs.append(d)
            else:
                print(f"[gate] skip {p}: not a directory", file=sys.stderr)
    else:
        base = Path(args.dir)
        if base.is_dir():
            dirs = sorted(d for d in base.iterdir() if d.is_dir() and d.name.startswith("runpack_"))

    if not dirs:
        print(f"[gate] No runpack directories found (scanned: {args.dir})", file=sys.stderr)
        sys.exit(2)

    # Load rollup if provided
    rollup: Optional[Dict[str, Any]] = None
    if args.rollup:
        rollup = _load_json(Path(args.rollup))
        if not rollup:
            print(f"[gate] WARNING: could not load rollup JSON from {args.rollup}", file=sys.stderr)

    # Run gate on each runpack
    reports: List[GateReport] = []
    for d in dirs:
        report = run_gate(d, thresholds, args.hours, rollup)
        reports.append(report)
        if not args.no_color:
            print(_render_report(report))
        else:
            # Strip ANSI
            import re
            text = _render_report(report)
            print(re.sub(r"\033\[[0-9;]*m", "", text))

    # Summary
    total = len(reports)
    passed = sum(1 for r in reports if r.passed)
    failed = total - passed

    print(f"\n{'='*60}")
    print(f"GATE SUMMARY: {passed}/{total} PASS  |  {failed}/{total} FAIL")
    print(f"{'='*60}\n")

    # JSON output
    if args.json_out:
        out = {
            "target_hours": args.hours,
            "thresholds": thresholds,
            "total": total,
            "passed": passed,
            "failed": failed,
            "overall_pass": failed == 0,
            "reports": [
                {
                    "runpack": r.runpack,
                    "window_hours": r.window_hours,
                    "passed": r.passed,
                    "checks": [
                        {
                            "name": c.name,
                            "passed": c.passed,
                            "detail": c.detail,
                            "remediation": c.remediation,
                        }
                        for c in r.checks
                    ],
                }
                for r in reports
            ],
        }
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"[gate] JSON report written to {out_path}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
