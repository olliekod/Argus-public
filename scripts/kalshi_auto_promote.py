#!/usr/bin/env python3
"""
kalshi_auto_promote.py - Promote the single best Kalshi paper bot for UI review.

Reads cumulative lifetime performance, derives per-bot lifetime risk diagnostics
from archived training runs, applies strict promotion gates, optionally runs an
offline stress replay, and writes the promoted bot metadata used by the terminal
UI. This never changes execution settings.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

from argus_kalshi.offline_sim import load_tapes, params_from_kalshi_config, simulate
from argus_kalshi.settlement_index import SettlementIndex
from argus_kalshi.simulation import calculate_robustness_score
from scripts.kalshi_readiness_gate import CheckResult


DEFAULT_THRESHOLDS: Dict[str, Any] = {
    "min_lifetime_fills": 2000,
    "min_cycles_active": 5,
    "min_robustness_score": 48.0,
    "min_win_rate": 0.52,
    "min_avg_pnl": 0.0,
    "min_positive_cycle_fraction": 0.60,
    "min_cycle_fills_to_count": 50,
    "max_top_concentration_share": 0.35,
    "promotion_upgrade_margin": 5.0,
}

DEFAULT_SETTINGS_PATH = Path("config/kalshi_family_adaptive.yaml")
DEFAULT_LIFETIME_PATH = Path("config/kalshi_lifetime_performance.json")
DEFAULT_PROMOTED_JSON_PATH = Path("config/kalshi_promoted_bot.json")
DEFAULT_HISTORY_PATH = Path("logs/promotion_history.jsonl")
DEFAULT_TAPE_DIR = Path("logs/training_data")
MIN_BACKTEST_RECORDS = 500
TOP_FAILED_TO_LOG = 10
TOP_HISTORY_CANDIDATES = 5


@dataclass
class BotDerivedStats:
    pnl: float = 0.0
    wins: int = 0
    losses: int = 0
    trade_count: int = 0
    max_drawdown: float = 0.0
    peak_pnl: float = 0.0
    qty_e_contracts: float = 0.0
    qty_s_contracts: float = 0.0
    qty_a_contracts: float = 0.0
    top_concentration_share: float = 0.0
    cycle_consistency_rate: float = 0.0
    qualifying_cycles: int = 0
    positive_cycles: int = 0
    cycle_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    market_side_qty: Dict[str, float] = field(default_factory=dict)
    trade_pnls: List[float] = field(default_factory=list)

    def robustness_input(self) -> Dict[str, Any]:
        return {
            "pnl": self.pnl,
            "wins": self.wins,
            "losses": self.losses,
            "trade_count": self.trade_count,
            "max_drawdown": self.max_drawdown,
            "qty_e_contracts": self.qty_e_contracts,
            "qty_s_contracts": self.qty_s_contracts,
            "qty_a_contracts": self.qty_a_contracts,
            "tail_loss_10pct": _tail_loss(self.trade_pnls),
        }

    def finalize(self, min_cycle_fills_to_count: int) -> None:
        total_qty = sum(self.market_side_qty.values())
        if total_qty > 0:
            self.top_concentration_share = max(self.market_side_qty.values()) / total_qty
        qualifying = 0
        positive = 0
        for cyc in self.cycle_stats.values():
            fills = int(cyc.get("fills", 0))
            if fills < min_cycle_fills_to_count:
                continue
            qualifying += 1
            if float(cyc.get("avg_pnl", 0.0)) > 0.0:
                positive += 1
        self.qualifying_cycles = qualifying
        self.positive_cycles = positive
        self.cycle_consistency_rate = (positive / qualifying) if qualifying > 0 else 0.0


@dataclass
class BotEvaluation:
    bot_id: str
    params: Dict[str, Any]
    lifetime_stats: Dict[str, Any]
    derived: BotDerivedStats
    robustness_score: float
    checks: List[CheckResult] = field(default_factory=list)
    gate_passed: bool = False
    failed_gate: Optional[str] = None
    passed_gate_count: int = 0
    failure_gap: float = 0.0
    backtest_stress_pnl: Optional[float] = None
    backtest_status: str = "not_run"
    backtest_records: int = 0

    @property
    def cycle_consistency_rate(self) -> float:
        return self.derived.cycle_consistency_rate


def log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[auto_promote {now}] {msg}", flush=True)


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log(f"ERROR reading {path}: {exc}")
        return None


def _tail_loss(pnls: List[float], percentile: float = 0.10) -> float:
    if not pnls:
        return 0.0
    ordered = sorted(pnls)
    count = max(1, int(math.ceil(len(ordered) * percentile)))
    return sum(ordered[:count]) / count


def _cycle_id_from_run_file(path: Path) -> str:
    return path.stem.removeprefix("run_")


def _settlement_dedup_key(rec: Dict[str, Any]) -> Optional[Tuple[str, str, str]]:
    bot_id = str(rec.get("bot_id") or "").strip()
    ticker = str(rec.get("market_ticker") or rec.get("ticker") or "").strip()
    timestamp = rec.get("timestamp")
    if not bot_id or not ticker or timestamp is None:
        return None
    return bot_id, ticker, str(timestamp)


def _source_bucket(source: str) -> str:
    src = (source or "").strip()
    if src == "mispricing_scalp":
        return "scalp"
    if src == "pair_arb":
        return "arb"
    return "expiry"


def _read_current_promoted_bot_id(settings_path: Path) -> Optional[str]:
    if not settings_path.exists():
        return None
    try:
        text = settings_path.read_text(encoding="utf-8")
    except Exception:
        return None
    match = re.search(r"(?m)^promoted_bot_id:\s*(?:\"([^\"]*)\"|'([^']*)'|([^\n#]+))", text)
    if not match:
        return None
    value = next((g for g in match.groups() if g is not None), "")
    value = value.strip()
    return value or None


def _load_current_promoted_score(promoted_json_path: Path) -> Optional[float]:
    payload = _load_json(promoted_json_path)
    if not payload:
        return None
    try:
        return float(payload.get("robustness_score"))
    except (TypeError, ValueError):
        return None


def _load_settings_base_config(settings_path: Path) -> Dict[str, Any]:
    if not settings_path.exists():
        return {}
    try:
        payload = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    farm_base = (
        payload.get("argus_kalshi", {})
        .get("farm", {})
        .get("base", {})
    )
    return dict(farm_base) if isinstance(farm_base, dict) else {}


def _update_yaml_promoted_bot(settings_path: Path, bot_id: str) -> None:
    text = settings_path.read_text(encoding="utf-8")
    replacement = f'promoted_bot_id: "{bot_id}"'
    updated, count = re.subn(
        r"(?m)^promoted_bot_id:\s*(?:\"[^\"]*\"|'[^']*'|[^\n#]*)",
        replacement,
        text,
        count=1,
    )
    if count == 0:
        updated = replacement + "\n" + text
    settings_path.write_text(updated, encoding="utf-8")


def _check_min_lifetime_fills(stats: Dict[str, Any], thresholds: Dict[str, Any]) -> Tuple[CheckResult, float]:
    fills = int(stats.get("fills", 0) or 0)
    minimum = int(thresholds["min_lifetime_fills"])
    return (
        CheckResult(
            name="min_lifetime_fills",
            passed=fills >= minimum,
            detail=f"fills={fills} (min {minimum})",
            remediation="Accumulate more lifetime settlements before promotion.",
        ),
        max(0.0, minimum - fills),
    )


def _check_min_cycles_active(stats: Dict[str, Any], thresholds: Dict[str, Any]) -> Tuple[CheckResult, float]:
    cycles = int(stats.get("cycles_active", 0) or 0)
    minimum = int(thresholds["min_cycles_active"])
    return (
        CheckResult(
            name="min_cycles_active",
            passed=cycles >= minimum,
            detail=f"cycles_active={cycles} (min {minimum})",
            remediation="Require performance across more independent cycle archives.",
        ),
        max(0.0, minimum - cycles),
    )


def _check_min_robustness_score(score: float, thresholds: Dict[str, Any]) -> Tuple[CheckResult, float]:
    minimum = float(thresholds["min_robustness_score"])
    return (
        CheckResult(
            name="min_robustness_score",
            passed=score >= minimum,
            detail=f"robustness_score={score:.2f} (min {minimum:.2f})",
            remediation="Risk-adjusted lifetime score is below the promotion floor.",
        ),
        max(0.0, minimum - score),
    )


def _check_min_win_rate(stats: Dict[str, Any], thresholds: Dict[str, Any]) -> Tuple[CheckResult, float]:
    win_rate = float(stats.get("win_rate", 0.0) or 0.0)
    minimum = float(thresholds["min_win_rate"])
    return (
        CheckResult(
            name="min_win_rate",
            passed=win_rate >= minimum,
            detail=f"win_rate={win_rate:.2%} (min {minimum:.2%})",
            remediation="Win rate is below the 52% breakeven-derived floor.",
        ),
        max(0.0, minimum - win_rate),
    )


def _check_min_avg_pnl(stats: Dict[str, Any], thresholds: Dict[str, Any]) -> Tuple[CheckResult, float]:
    avg_pnl = float(stats.get("avg_pnl", 0.0) or 0.0)
    minimum = float(thresholds["min_avg_pnl"])
    return (
        CheckResult(
            name="min_avg_pnl",
            passed=avg_pnl > minimum,
            detail=f"avg_pnl=${avg_pnl:+.4f} (must be > {minimum:+.4f})",
            remediation="Lifetime expectancy must remain strictly positive.",
        ),
        max(0.0, minimum - avg_pnl) if avg_pnl <= minimum else 0.0,
    )


def _check_positive_cycle_fraction(derived: BotDerivedStats, thresholds: Dict[str, Any]) -> Tuple[CheckResult, float]:
    minimum = float(thresholds["min_positive_cycle_fraction"])
    minimum_cycle_fills = int(thresholds["min_cycle_fills_to_count"])
    rate = derived.cycle_consistency_rate
    return (
        CheckResult(
            name="min_positive_cycle_fraction",
            passed=rate >= minimum,
            detail=(
                f"positive_cycle_fraction={rate:.2%} "
                f"({derived.positive_cycles}/{derived.qualifying_cycles} qualifying cycles, "
                f"min_cycle_fills={minimum_cycle_fills})"
            ),
            remediation="Too few qualifying cycles are independently profitable.",
        ),
        max(0.0, minimum - rate),
    )


def _check_top_concentration_share(derived: BotDerivedStats, thresholds: Dict[str, Any]) -> Tuple[CheckResult, float]:
    maximum = float(thresholds["max_top_concentration_share"])
    share = derived.top_concentration_share
    return (
        CheckResult(
            name="max_top_concentration_share",
            passed=share <= maximum,
            detail=f"top_concentration_share={share:.2%} (max {maximum:.2%})",
            remediation="One market-side dominates too much of the promoted bot's exposure.",
        ),
        max(0.0, share - maximum),
    )


def _check_upgrade_margin(
    candidate_score: float,
    current_score: Optional[float],
    current_bot_id: Optional[str],
    candidate_bot_id: str,
    thresholds: Dict[str, Any],
) -> CheckResult:
    margin = float(thresholds["promotion_upgrade_margin"])
    if not current_bot_id or current_score is None:
        return CheckResult(
            name="promotion_upgrade_margin",
            passed=True,
            detail="no current promoted bot score available - margin check skipped",
        )
    if current_bot_id == candidate_bot_id:
        return CheckResult(
            name="promotion_upgrade_margin",
            passed=True,
            detail="candidate already matches current promoted bot - replacement margin not required",
        )
    delta = candidate_score - current_score
    return CheckResult(
        name="promotion_upgrade_margin",
        passed=delta >= margin,
        detail=f"score_delta={delta:.2f} vs current {current_bot_id} (min +{margin:.2f})",
        remediation="Keep the current promoted bot until a materially better replacement emerges.",
    )


def _evaluate_bot(
    bot_id: str,
    lifetime_stats: Dict[str, Any],
    derived: BotDerivedStats,
    thresholds: Dict[str, Any],
) -> BotEvaluation:
    params = dict(lifetime_stats.get("params") or {})
    score = calculate_robustness_score(derived.robustness_input())
    evaluation = BotEvaluation(
        bot_id=bot_id,
        params=params,
        lifetime_stats=dict(lifetime_stats),
        derived=derived,
        robustness_score=score,
    )
    checks_with_gap = [
        _check_min_lifetime_fills(lifetime_stats, thresholds),
        _check_min_cycles_active(lifetime_stats, thresholds),
        _check_min_robustness_score(score, thresholds),
        _check_min_win_rate(lifetime_stats, thresholds),
        _check_min_avg_pnl(lifetime_stats, thresholds),
        _check_positive_cycle_fraction(derived, thresholds),
        _check_top_concentration_share(derived, thresholds),
    ]
    for idx, (check, gap) in enumerate(checks_with_gap, start=1):
        evaluation.checks.append(check)
        if not check.passed:
            evaluation.failed_gate = check.name
            evaluation.failure_gap = gap
            evaluation.passed_gate_count = idx - 1
            return evaluation
    evaluation.gate_passed = True
    evaluation.passed_gate_count = len(checks_with_gap)
    return evaluation


def _iter_settlement_rows(run_file: Path) -> Iterable[Dict[str, Any]]:
    seen_in_file: set[Tuple[str, str, str]] = set()
    try:
        with run_file.open("r", encoding="utf-8", errors="ignore") as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("type") != "settlement":
                    continue
                dedup_key = _settlement_dedup_key(rec)
                if dedup_key is None or dedup_key in seen_in_file:
                    continue
                seen_in_file.add(dedup_key)
                yield rec
    except Exception as exc:
        log(f"WARNING unable to stream {run_file.name}: {exc}")


def build_derived_stats(
    training_dir: Path,
    thresholds: Dict[str, Any],
) -> Dict[str, BotDerivedStats]:
    derived: Dict[str, BotDerivedStats] = {}
    for run_file in sorted(training_dir.glob("run_*.jsonl")):
        cycle_id = _cycle_id_from_run_file(run_file)
        for rec in _iter_settlement_rows(run_file):
            bot_id = str(rec.get("bot_id") or "").strip()
            if not bot_id:
                continue
            pnl = float(rec.get("pnl_usd") or 0.0)
            won = bool(rec.get("won"))
            qty = float(rec.get("quantity_contracts") or 1.0)
            ticker = str(rec.get("market_ticker") or rec.get("ticker") or "").strip()
            side = str(rec.get("side") or "").strip().lower()
            source = str(rec.get("source") or "")
            stats = derived.setdefault(bot_id, BotDerivedStats())
            stats.pnl += pnl
            stats.trade_count += 1
            if won:
                stats.wins += 1
            else:
                stats.losses += 1
            stats.trade_pnls.append(pnl)
            stats.peak_pnl = max(stats.peak_pnl, stats.pnl)
            stats.max_drawdown = max(stats.max_drawdown, stats.peak_pnl - stats.pnl)
            cycle_bucket = stats.cycle_stats.setdefault(cycle_id, {"fills": 0.0, "pnl": 0.0, "avg_pnl": 0.0})
            cycle_bucket["fills"] += 1.0
            cycle_bucket["pnl"] += pnl
            cycle_bucket["avg_pnl"] = cycle_bucket["pnl"] / max(1.0, cycle_bucket["fills"])
            if ticker and side:
                market_side = f"{ticker}|{side}"
                stats.market_side_qty[market_side] = stats.market_side_qty.get(market_side, 0.0) + qty
            bucket = _source_bucket(source)
            if bucket == "scalp":
                stats.qty_s_contracts += qty
            elif bucket == "arb":
                stats.qty_a_contracts += qty
            else:
                stats.qty_e_contracts += qty
    for stats in derived.values():
        stats.finalize(int(thresholds["min_cycle_fills_to_count"]))
    return derived


def _history_candidate_row(evaluation: BotEvaluation) -> Dict[str, Any]:
    return {
        "bot_id": evaluation.bot_id,
        "robustness_score": evaluation.robustness_score,
        "gate_passed": evaluation.gate_passed,
        "failed_gate": evaluation.failed_gate,
        "cycle_consistency_rate": evaluation.cycle_consistency_rate,
        "top_concentration_share": evaluation.derived.top_concentration_share,
        "backtest_stress_pnl": evaluation.backtest_stress_pnl,
        "gate_results": [asdict(check) for check in evaluation.checks],
    }


def _log_failed_candidates(failed: List[BotEvaluation]) -> None:
    if not failed:
        return
    ranked = sorted(
        failed,
        key=lambda ev: (-ev.passed_gate_count, ev.failure_gap, -ev.robustness_score, ev.bot_id),
    )
    log(f"Closest failed candidates (top {min(TOP_FAILED_TO_LOG, len(ranked))}):")
    for ev in ranked[:TOP_FAILED_TO_LOG]:
        failed_check = ev.checks[-1] if ev.checks else None
        detail = failed_check.detail if failed_check else "unknown"
        log(
            f"  {ev.bot_id}: failed {ev.failed_gate} after {ev.passed_gate_count} gate(s) "
            f"| score={ev.robustness_score:.2f} | {detail}"
        )


def _stress_backtest(
    params: Dict[str, Any],
    tape_dir: Path,
    settings_base: Dict[str, Any],
) -> Tuple[Optional[float], str, int]:
    tape_paths = [str(path) for path in sorted(tape_dir.glob("*.jsonl"))]
    if not tape_paths:
        return None, "no_tape_files", 0
    tape_records = load_tapes(tape_paths, include_rejections=False)
    record_count = len(tape_records)
    if record_count < MIN_BACKTEST_RECORDS:
        return None, "insufficient_data", record_count
    settlement_index = SettlementIndex.from_jsonl_files(tape_paths)
    sim_cfg = dict(settings_base)
    sim_cfg.update(params)
    sim_params = params_from_kalshi_config(sim_cfg)
    result = simulate(tape_records, settlement_index, sim_params, include_rejections=False, scenario="stress")
    return float(result.total_pnl_usd), "ran", record_count


def _write_promoted_metadata(
    promoted_json_path: Path,
    evaluation: BotEvaluation,
    upgrade_check: CheckResult,
) -> None:
    promoted_json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "bot_id": evaluation.bot_id,
        "params": evaluation.params,
        "promotion_timestamp": datetime.now(timezone.utc).isoformat(),
        "lifetime_stats": evaluation.lifetime_stats,
        "robustness_score": evaluation.robustness_score,
        "backtest_stress_pnl": evaluation.backtest_stress_pnl,
        "cycle_consistency_rate": evaluation.cycle_consistency_rate,
        "gate_results": [asdict(check) for check in [*evaluation.checks, upgrade_check]],
    }
    promoted_json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _append_history(
    history_path: Path,
    action: str,
    top_candidates: List[BotEvaluation],
    promoted_bot_id: Optional[str],
    reason: str,
) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "top_candidates": [_history_candidate_row(ev) for ev in top_candidates[:TOP_HISTORY_CANDIDATES]],
        "promoted_bot_id": promoted_bot_id or "",
        "reason": reason,
    }
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def run_promotion(
    lifetime_path: Path,
    tape_dir: Path,
    settings_path: Path,
    promoted_json_path: Path,
    history_path: Path,
    thresholds: Dict[str, Any],
) -> int:
    lifetime_doc = _load_json(lifetime_path)
    if not lifetime_doc:
        log(f"ERROR: lifetime performance file not readable: {lifetime_path}")
        return 2
    bots = lifetime_doc.get("bots")
    if not isinstance(bots, dict) or not bots:
        _append_history(history_path, "no_eligible_bot", [], _read_current_promoted_bot_id(settings_path), "lifetime_doc_empty")
        log("No bots found in lifetime performance document.")
        return 0

    derived_by_bot = build_derived_stats(tape_dir, thresholds)
    evaluations: List[BotEvaluation] = []
    for bot_id, stats in bots.items():
        if not isinstance(stats, dict):
            continue
        derived = derived_by_bot.get(bot_id, BotDerivedStats())
        evaluations.append(_evaluate_bot(bot_id, stats, derived, thresholds))
    evaluations.sort(key=lambda ev: (ev.robustness_score, float(ev.lifetime_stats.get("total_pnl", 0.0))), reverse=True)

    top_candidates = evaluations[:TOP_HISTORY_CANDIDATES]
    failed_candidates = [ev for ev in evaluations if not ev.gate_passed]
    passed_candidates = [ev for ev in evaluations if ev.gate_passed]
    _log_failed_candidates(failed_candidates)

    current_promoted_bot_id = _read_current_promoted_bot_id(settings_path)
    stored_promoted_score = _load_current_promoted_score(promoted_json_path) if current_promoted_bot_id else None
    current_promoted_score = None
    if current_promoted_bot_id:
        current_eval = next((ev for ev in evaluations if ev.bot_id == current_promoted_bot_id), None)
        if current_eval is not None:
            current_promoted_score = current_eval.robustness_score
        if current_promoted_score is None:
            current_promoted_score = stored_promoted_score

    if not passed_candidates:
        reason = "no bot passed lifetime promotion gates"
        _append_history(history_path, "no_eligible_bot", top_candidates, current_promoted_bot_id, reason)
        log(reason)
        return 0

    settings_base = _load_settings_base_config(settings_path)
    final_candidate: Optional[BotEvaluation] = None
    backtest_reason = ""
    for candidate in passed_candidates:
        stress_pnl, status, record_count = _stress_backtest(candidate.params, tape_dir, settings_base)
        candidate.backtest_stress_pnl = stress_pnl
        candidate.backtest_status = status
        candidate.backtest_records = record_count
        if status == "insufficient_data":
            log(
                f"{candidate.bot_id}: offline stress replay skipped "
                f"(usable tape records {record_count} < {MIN_BACKTEST_RECORDS})"
            )
            final_candidate = candidate
            break
        if status == "no_tape_files":
            log(f"{candidate.bot_id}: no tape files found in {tape_dir}, skipping stress replay")
            final_candidate = candidate
            break
        if stress_pnl is None:
            log(f"{candidate.bot_id}: offline stress replay unavailable ({status})")
            continue
        log(
            f"{candidate.bot_id}: offline stress replay on {record_count} records "
            f"-> stress_pnl=${stress_pnl:+.2f}"
        )
        if stress_pnl > 0.0:
            final_candidate = candidate
            break
        backtest_reason = f"{candidate.bot_id} failed stress replay with pnl={stress_pnl:+.2f}"
    if final_candidate is None:
        reason = backtest_reason or "all gate-passing bots failed offline stress replay"
        _append_history(history_path, "no_eligible_bot", top_candidates, current_promoted_bot_id, reason)
        log(reason)
        return 0

    upgrade_check = _check_upgrade_margin(
        final_candidate.robustness_score,
        current_promoted_score,
        current_promoted_bot_id,
        final_candidate.bot_id,
        thresholds,
    )
    if not upgrade_check.passed:
        reason = upgrade_check.detail
        _append_history(
            history_path,
            "no_change",
            top_candidates,
            current_promoted_bot_id or final_candidate.bot_id,
            reason,
        )
        log(f"No promotion change: {reason}")
        return 0

    if current_promoted_bot_id == final_candidate.bot_id:
        prior_score = stored_promoted_score if stored_promoted_score is not None else final_candidate.robustness_score
        meaningful_delta = abs(final_candidate.robustness_score - prior_score)
        margin = float(thresholds["promotion_upgrade_margin"])
        if meaningful_delta < margin:
            reason = (
                f"{final_candidate.bot_id} remains promoted; score delta {meaningful_delta:.2f} "
                f"is below meaningful-change threshold {margin:.2f}"
            )
            _append_history(history_path, "no_change", top_candidates, current_promoted_bot_id, reason)
            log(reason)
            return 0

    _update_yaml_promoted_bot(settings_path, final_candidate.bot_id)
    _write_promoted_metadata(promoted_json_path, final_candidate, upgrade_check)
    reason = (
        f"promoted {final_candidate.bot_id} with robustness_score={final_candidate.robustness_score:.2f}"
        if current_promoted_bot_id != final_candidate.bot_id
        else f"refreshed promotion metadata for {final_candidate.bot_id}"
    )
    _append_history(history_path, "promoted", top_candidates, final_candidate.bot_id, reason)
    log(reason)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate lifetime Kalshi paper bots and promote the best bot for terminal review.",
    )
    parser.add_argument("--lifetime", default=str(DEFAULT_LIFETIME_PATH), help="Path to kalshi_lifetime_performance.json")
    parser.add_argument("--tape-dir", default=str(DEFAULT_TAPE_DIR), help="Directory containing archived run_*.jsonl files")
    parser.add_argument("--settings", default=str(DEFAULT_SETTINGS_PATH), help="Farm settings YAML containing promoted_bot_id")
    parser.add_argument("--promoted-json", default=str(DEFAULT_PROMOTED_JSON_PATH), help="Output metadata JSON for promoted bot")
    parser.add_argument("--history-log", default=str(DEFAULT_HISTORY_PATH), help="Promotion history JSONL path")
    parser.add_argument("--config", help="JSON file with threshold overrides")
    args = parser.parse_args()

    thresholds = dict(DEFAULT_THRESHOLDS)
    if args.config:
        override = _load_json(Path(args.config))
        if override is None:
            return 2
        thresholds.update(override)

    return run_promotion(
        lifetime_path=Path(args.lifetime),
        tape_dir=Path(args.tape_dir),
        settings_path=Path(args.settings),
        promoted_json_path=Path(args.promoted_json),
        history_path=Path(args.history_log),
        thresholds=thresholds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
