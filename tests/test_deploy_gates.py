"""
Tests for Deploy Gates Integration
=====================================

Verifies:
- DSR kill threshold integration in StrategyEvaluator
- Slippage sensitivity kill reason
- Reality Check kill reason
- cost_multiplier in ExecutionConfig
- New config schema fields
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from src.analysis.execution_model import ExecutionConfig, ExecutionModel, Quote
from src.analysis.strategy_evaluator import StrategyEvaluator


# ── Helpers ──────────────────────────────────────────────────────────

def _make_experiment(
    strategy_id: str = "TEST",
    run_id: str = "abc",
    strategy_class: str = "TestStrategy",
    total_pnl: float = 500.0,
    sharpe: float = 1.5,
    max_drawdown_pct: float = 5.0,
    fill_rate: float = 0.95,
    total_trades: int = 50,
    bars_replayed: int = 1000,
    dsr: float = 0.0,
    slippage_sensitivity: Dict[str, Any] = None,
    reality_check: Dict[str, Any] = None,
) -> Dict[str, Any]:
    manifest = {
        "run_id": run_id,
        "strategy_class": strategy_class,
        "strategy_params": {},
        "execution_config": "DEFAULT",
        "replay_packs": [],
        "data_sources": {},
        "environment": {"git_commit": "test", "timestamp": "20240101"},
    }
    if slippage_sensitivity is not None:
        manifest["slippage_sensitivity"] = slippage_sensitivity
    if reality_check is not None:
        manifest["reality_check"] = reality_check

    return {
        "manifest": manifest,
        "result": {
            "strategy_id": strategy_id,
            "bars_replayed": bars_replayed,
            "portfolio": {
                "total_realized_pnl": total_pnl,
                "starting_cash": 10000.0,
                "total_return_pct": total_pnl / 100.0,
                "sharpe_annualized_proxy": sharpe,
                "max_drawdown": total_pnl * 0.1,
                "max_drawdown_pct": max_drawdown_pct,
                "expectancy": total_pnl / max(total_trades, 1),
                "profit_factor": 2.0,
                "win_rate": 60.0,
                "total_trades": total_trades,
                "winners": int(total_trades * 0.6),
                "losers": int(total_trades * 0.4),
                "regime_breakdown": {},
            },
            "execution": {
                "fills": int(total_trades * fill_rate),
                "rejects": int(total_trades * (1 - fill_rate)),
                "fill_rate": fill_rate,
                "total_commission": 32.50,
                "total_slippage": 18.75,
            },
        },
    }


class TestCostMultiplier:
    def test_default_multiplier_is_1(self):
        cfg = ExecutionConfig()
        assert cfg.cost_multiplier == 1.0

    def test_custom_multiplier(self):
        cfg = ExecutionConfig(cost_multiplier=1.5)
        assert cfg.cost_multiplier == 1.5

    def test_multiplier_scales_slippage(self):
        """cost_multiplier=2.0 should double slippage."""
        quote = Quote(bid=1.00, ask=1.10, bid_size=10, ask_size=10)

        model_1x = ExecutionModel(ExecutionConfig(
            slippage_per_contract=0.02,
            cost_multiplier=1.0,
        ))
        model_2x = ExecutionModel(ExecutionConfig(
            slippage_per_contract=0.02,
            cost_multiplier=2.0,
        ))

        fill_1x = model_1x.attempt_fill(quote, "SELL", 1, 1000)
        fill_2x = model_2x.attempt_fill(quote, "SELL", 1, 1000)

        assert fill_1x.filled and fill_2x.filled
        # 1x: bid - 0.02 = 0.98
        # 2x: bid - 0.04 = 0.96
        assert abs(fill_1x.fill_price - 0.98) < 1e-4
        assert abs(fill_2x.fill_price - 0.96) < 1e-4

    def test_multiplier_scales_commission(self):
        quote = Quote(bid=1.00, ask=1.10, bid_size=10, ask_size=10)

        model_1x = ExecutionModel(ExecutionConfig(
            commission_per_contract=0.65,
            cost_multiplier=1.0,
        ))
        model_1_5x = ExecutionModel(ExecutionConfig(
            commission_per_contract=0.65,
            cost_multiplier=1.5,
        ))

        fill_1x = model_1x.attempt_fill(quote, "SELL", 2, 1000)
        fill_1_5x = model_1_5x.attempt_fill(quote, "SELL", 2, 1000)

        # 1x: 0.65 * 1.0 * 2 = 1.30
        # 1.5x: 0.65 * 1.5 * 2 = 1.95
        assert abs(fill_1x.commission - 1.30) < 1e-4
        assert abs(fill_1_5x.commission - 1.95) < 1e-4


class TestDSRKillThreshold:
    def test_dsr_kill_reason_added(self):
        """Strategy with low DSR should get dsr_below_threshold kill."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use low Sharpe so DSR stays below 0.99 (with n_trials=2, threshold is low
            # but observed Sharpe must be modest for DSR < 0.99)
            for i, (rid, sr) in enumerate([("run1", 0.08), ("run2", 0.05)]):
                exp = _make_experiment(
                    strategy_id=f"S{i}",
                    run_id=rid,
                    sharpe=sr,
                    bars_replayed=500,
                )
                with open(os.path.join(tmpdir, f"exp_{i}.json"), "w") as f:
                    json.dump(exp, f)

            evaluator = StrategyEvaluator(
                input_dir=tmpdir,
                kill_thresholds={
                    "dsr_min": 0.99,  # Very high threshold - should kill
                    "robustness_penalty": 2.0,
                    "walk_forward_penalty": 2.0,
                    "regime_dependency_penalty": 2.0,
                    "composite_score_min": -999.0,
                },
            )
            evaluator.load_experiments()
            rankings = evaluator.evaluate()

            # At least one should have DSR kill (low Sharpe -> DSR < 0.99)
            dsr_kills = [
                r for r in rankings
                for kr in r.get("kill_reasons", [])
                if kr["reason"] == "dsr_below_threshold"
            ]
            assert len(dsr_kills) > 0


class TestSlippageSensitivityKill:
    def test_slippage_kill_from_manifest(self):
        """Strategy with slippage_sensitivity.killed=True in manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = _make_experiment(
                run_id="slip1",
                slippage_sensitivity={
                    "killed": True,
                    "sharpe_at_150pct": -0.3,
                },
            )
            with open(os.path.join(tmpdir, "exp.json"), "w") as f:
                json.dump(exp, f)

            evaluator = StrategyEvaluator(
                input_dir=tmpdir,
                kill_thresholds={
                    "dsr_min": 0.0,  # Disable DSR kill
                    "robustness_penalty": 2.0,
                    "walk_forward_penalty": 2.0,
                    "regime_dependency_penalty": 2.0,
                    "composite_score_min": -999.0,
                },
            )
            evaluator.load_experiments()
            rankings = evaluator.evaluate()

            kills = [
                kr for r in rankings
                for kr in r.get("kill_reasons", [])
                if kr["reason"] == "slippage_sensitivity"
            ]
            assert len(kills) == 1
            assert kills[0]["value"] == -0.3


class TestRealityCheckKill:
    def test_reality_check_kill_from_manifest(self):
        """Strategy with reality_check p_value >= threshold."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = _make_experiment(
                run_id="rc1",
                reality_check={
                    "p_value": 0.30,
                    "best_strategy": "rc1",
                },
            )
            with open(os.path.join(tmpdir, "exp.json"), "w") as f:
                json.dump(exp, f)

            evaluator = StrategyEvaluator(
                input_dir=tmpdir,
                kill_thresholds={
                    "dsr_min": 0.0,
                    "reality_check_p_max": 0.05,
                    "robustness_penalty": 2.0,
                    "walk_forward_penalty": 2.0,
                    "regime_dependency_penalty": 2.0,
                    "composite_score_min": -999.0,
                },
            )
            evaluator.load_experiments()
            rankings = evaluator.evaluate()

            kills = [
                kr for r in rankings
                for kr in r.get("kill_reasons", [])
                if kr["reason"] == "reality_check_failed"
            ]
            assert len(kills) == 1
            assert kills[0]["value"] == 0.30


class TestConfigSchema:
    def test_deploy_gates_config(self):
        from src.analysis.research_loop_config import DeployGatesOpts
        opts = DeployGatesOpts()
        assert opts.dsr_min == 0.95
        assert opts.slippage_sweep is True
        assert opts.reality_check_p_max == 0.05

    def test_allocation_config(self):
        from src.analysis.research_loop_config import AllocationOpts
        opts = AllocationOpts()
        assert opts.kelly_fraction == 0.25
        assert opts.per_play_cap == 0.07
        assert opts.vol_target_annual == 0.10
