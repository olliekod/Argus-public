"""
Tests for the research → allocation loop integration.

Verifies:
- run_allocation() produces allocations.json when config is set
- run_allocation() is skipped when allocation config or output path is null
- Forecasts are built correctly from evaluator rankings
- Allocations include required fields (strategy_id, instrument, weight, etc.)
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Ensure repo root on sys.path
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.analysis.research_loop_config import (
    AllocationOpts,
    EvaluationOpts,
    ExperimentOpts,
    LoopOpts,
    OutcomesConfig,
    PackConfig,
    ResearchLoopConfig,
    StrategySpec,
)
from scripts.strategy_research_loop import run_allocation, run_cycle


def _make_rankings() -> List[Dict[str, Any]]:
    """Create sample evaluator rankings for testing."""
    return [
        {
            "rank": 1,
            "strategy_id": "VRP_v1",
            "run_id": "run_001",
            "strategy_class": "VRPCreditSpreadStrategy",
            "strategy_params": {"underlying": "SPY"},
            "composite_score": 0.45,
            "regime_sensitivity_score": 0.7,
            "dsr": 0.98,
            "killed": False,
            "kill_reasons": [],
            "metrics": {
                "total_pnl": 500.0,
                "total_return_pct": 5.0,
                "sharpe": 1.2,
                "max_drawdown": 200.0,
                "max_drawdown_pct": 2.0,
                "expectancy": 50.0,
                "profit_factor": 2.0,
                "win_rate": 60.0,
                "total_trades": 20,
                "fill_rate": 0.95,
                "fills": 19,
                "rejects": 1,
                "dsr": 0.98,
            },
        },
        {
            "rank": 2,
            "strategy_id": "VRP_v2",
            "run_id": "run_002",
            "strategy_class": "VRPCreditSpreadStrategy",
            "strategy_params": {"underlying": "QQQ"},
            "composite_score": 0.30,
            "regime_sensitivity_score": 0.5,
            "dsr": 0.80,
            "killed": False,
            "kill_reasons": [],
            "metrics": {
                "total_pnl": 300.0,
                "total_return_pct": 3.0,
                "sharpe": 0.8,
                "max_drawdown": 400.0,
                "max_drawdown_pct": 4.0,
                "expectancy": 30.0,
                "profit_factor": 1.5,
                "win_rate": 55.0,
                "total_trades": 15,
                "fill_rate": 0.90,
                "fills": 14,
                "rejects": 1,
                "dsr": 0.80,
            },
        },
        {
            "rank": 3,
            "strategy_id": "KILLED_strat",
            "run_id": "run_003",
            "strategy_class": "VRPCreditSpreadStrategy",
            "strategy_params": {},
            "composite_score": 0.05,
            "regime_sensitivity_score": 0.3,
            "dsr": 0.50,
            "killed": True,
            "kill_reasons": [{"reason": "mc_ruin_prob", "value": 0.5, "threshold": 0.3}],
            "metrics": {
                "total_pnl": -100.0,
                "total_return_pct": -1.0,
                "sharpe": -0.2,
                "max_drawdown": 800.0,
                "max_drawdown_pct": 8.0,
                "expectancy": -10.0,
                "profit_factor": 0.5,
                "win_rate": 40.0,
                "total_trades": 10,
                "fill_rate": 0.85,
                "fills": 9,
                "rejects": 1,
                "dsr": 0.50,
            },
        },
    ]


def _make_config(
    alloc_output_path=None,
    allocation=None,
    equity=10_000.0,
    min_dsr=0.0,
    min_composite_score=0.0,
):
    """Build a minimal ResearchLoopConfig for testing allocation."""
    return ResearchLoopConfig(
        pack=PackConfig(
            mode="single",
            symbols=["SPY"],
            start_date="2026-01-01",
            end_date="2026-01-31",
            bars_provider=None,
            options_snapshot_provider=None,
            packs_output_dir="/tmp/packs",
            db_path="/tmp/argus_test.db",
        ),
        outcomes=OutcomesConfig(ensure_before_pack=False, bar_duration=60),
        strategies=[StrategySpec(strategy_class="VRPCreditSpreadStrategy", params={}, sweep=None)],
        experiment=ExperimentOpts(
            output_dir="/tmp/experiments",
            starting_cash=10000.0,
            regime_stress=False,
            mc_bootstrap=False,
            mc_paths=100,
            mc_method="bootstrap",
            mc_block_size=None,
            mc_seed=42,
            mc_ruin_level=0.2,
            mc_kill_thresholds=None,
        ),
        evaluation=EvaluationOpts(
            input_dir="/tmp/experiments",
            kill_thresholds=None,
            rankings_output_dir="/tmp/logs",
            killed_output_path=None,
            candidate_set_output_path=None,
            allocation=allocation,
            allocations_output_path=alloc_output_path,
            equity=equity,
            min_dsr=min_dsr,
            min_composite_score=min_composite_score,
        ),
        loop=LoopOpts(interval_hours=24, require_recent_bars_hours=None),
    )


class TestRunAllocation:
    def test_skipped_when_no_allocation_config(self):
        """Allocation step is skipped when allocation config is None."""
        config = _make_config(alloc_output_path="/tmp/allocs.json", allocation=None)
        result = run_allocation(config, _make_rankings())
        assert result is None

    def test_skipped_when_no_output_path(self):
        """Allocation step is skipped when output path is None."""
        alloc = AllocationOpts(kelly_fraction=0.25, per_play_cap=0.07)
        config = _make_config(alloc_output_path=None, allocation=alloc)
        result = run_allocation(config, _make_rankings())
        assert result is None

    def test_produces_allocations_json(self, tmp_path):
        """Full allocation run produces a valid JSON file."""
        output = str(tmp_path / "allocations.json")
        alloc = AllocationOpts(kelly_fraction=0.25, per_play_cap=0.07, vol_target_annual=0.10)
        config = _make_config(
            alloc_output_path=output,
            allocation=alloc,
            equity=10_000.0,
        )
        result = run_allocation(config, _make_rankings())
        assert result == output
        assert Path(output).exists()

        with open(output) as f:
            data = json.load(f)

        assert "generated_at" in data
        assert "equity" in data
        assert data["equity"] == 10_000.0
        assert "config" in data
        assert "allocations" in data
        assert isinstance(data["allocations"], list)

    def test_behavior_unchanged_without_max_loss_config(self, tmp_path):
        """Without max_loss_per_contract, allocations still run and contracts are null."""
        output = str(tmp_path / "allocations.json")
        alloc = AllocationOpts(kelly_fraction=0.25, per_play_cap=0.07)
        config = _make_config(alloc_output_path=output, allocation=alloc)
        run_allocation(config, _make_rankings())

        with open(output) as f:
            data = json.load(f)

        assert data["config"].get("max_loss_per_contract") is None
        assert len(data["allocations"]) > 0

    def test_allocation_fields(self, tmp_path):
        """Each allocation entry has required fields."""
        output = str(tmp_path / "allocations.json")
        alloc = AllocationOpts(kelly_fraction=0.25, per_play_cap=0.07)
        config = _make_config(alloc_output_path=output, allocation=alloc)
        run_allocation(config, _make_rankings())

        with open(output) as f:
            data = json.load(f)

        required_fields = {"strategy_id", "instrument", "weight", "dollar_risk", "contracts"}
        for entry in data["allocations"]:
            assert required_fields.issubset(entry.keys()), f"Missing fields in {entry}"

    def test_killed_strategies_excluded(self, tmp_path):
        """Killed strategies should not appear in allocations."""
        output = str(tmp_path / "allocations.json")
        alloc = AllocationOpts(kelly_fraction=0.25, per_play_cap=0.07)
        config = _make_config(alloc_output_path=output, allocation=alloc)
        run_allocation(config, _make_rankings())

        with open(output) as f:
            data = json.load(f)

        strategy_ids = {a["strategy_id"] for a in data["allocations"]}
        assert "KILLED_strat" not in strategy_ids

    def test_min_dsr_filter(self, tmp_path):
        """Strategies below min_dsr are filtered out."""
        output = str(tmp_path / "allocations.json")
        alloc = AllocationOpts(kelly_fraction=0.25, per_play_cap=0.07)
        config = _make_config(
            alloc_output_path=output,
            allocation=alloc,
            min_dsr=0.95,
        )
        run_allocation(config, _make_rankings())

        with open(output) as f:
            data = json.load(f)

        strategy_ids = {a["strategy_id"] for a in data["allocations"]}
        # VRP_v2 has dsr=0.80 < 0.95, should be filtered out
        assert "VRP_v2" not in strategy_ids
        assert "VRP_v1" in strategy_ids

    def test_min_composite_score_filter(self, tmp_path):
        """Strategies below min_composite_score are filtered out."""
        output = str(tmp_path / "allocations.json")
        alloc = AllocationOpts(kelly_fraction=0.25, per_play_cap=0.07)
        config = _make_config(
            alloc_output_path=output,
            allocation=alloc,
            min_composite_score=0.35,
        )
        run_allocation(config, _make_rankings())

        with open(output) as f:
            data = json.load(f)

        strategy_ids = {a["strategy_id"] for a in data["allocations"]}
        # VRP_v2 has composite_score=0.30 < 0.35
        assert "VRP_v2" not in strategy_ids
        assert "VRP_v1" in strategy_ids

    def test_no_candidates_returns_none(self, tmp_path):
        """If all candidates are filtered, returns None."""
        output = str(tmp_path / "allocations.json")
        alloc = AllocationOpts(kelly_fraction=0.25, per_play_cap=0.07)
        config = _make_config(
            alloc_output_path=output,
            allocation=alloc,
            min_dsr=0.99,  # Filters out everything
            min_composite_score=0.99,
        )
        result = run_allocation(config, _make_rankings())
        assert result is None

    def test_max_loss_per_contract_from_allocation_config_reduces_contracts(self, tmp_path):
        """Higher configured max loss should reduce contracts for a strategy."""
        output = str(tmp_path / "allocations.json")
        alloc = AllocationOpts(
            kelly_fraction=0.25,
            per_play_cap=0.07,
            max_loss_per_contract={"VRP_v1": 1000.0, "VRP_v2": 100.0},
        )
        config = _make_config(alloc_output_path=output, allocation=alloc)
        run_allocation(config, _make_rankings())

        with open(output) as f:
            data = json.load(f)

        contracts = {a["strategy_id"]: a["contracts"] for a in data["allocations"]}
        assert contracts["VRP_v1"] < contracts["VRP_v2"]

    def test_max_loss_per_contract_strategy_params_override_config(self, tmp_path):
        """strategy_params max_loss_per_contract should override allocation defaults."""
        output = str(tmp_path / "allocations.json")
        rankings = _make_rankings()
        rankings[0]["strategy_params"]["max_loss_per_contract"] = 1200.0

        alloc = AllocationOpts(
            kelly_fraction=0.25,
            per_play_cap=0.07,
            max_loss_per_contract=100.0,
        )
        config = _make_config(alloc_output_path=output, allocation=alloc)
        run_allocation(config, rankings)

        with open(output) as f:
            data = json.load(f)

        contracts = {a["strategy_id"]: a["contracts"] for a in data["allocations"]}
        assert contracts["VRP_v1"] < contracts["VRP_v2"]

    def test_instrument_from_params(self, tmp_path):
        """Instrument should be derived from strategy_params."""
        output = str(tmp_path / "allocations.json")
        alloc = AllocationOpts(kelly_fraction=0.25, per_play_cap=0.07)
        config = _make_config(alloc_output_path=output, allocation=alloc)
        run_allocation(config, _make_rankings())

        with open(output) as f:
            data = json.load(f)

        instruments = {a["strategy_id"]: a["instrument"] for a in data["allocations"]}
        assert instruments.get("VRP_v1") == "SPY"
        assert instruments.get("VRP_v2") == "QQQ"


def test_run_cycle_writes_allocations_json_end_to_end(tmp_path, monkeypatch):
    """run_cycle reaches Step 5 and writes allocations.json with expected schema."""
    rankings_path = tmp_path / "rankings.json"
    allocations_path = tmp_path / "logs" / "allocations.json"

    alloc = AllocationOpts(kelly_fraction=0.25, per_play_cap=0.07, vol_target_annual=0.10)
    config = _make_config(
        alloc_output_path=str(allocations_path),
        allocation=alloc,
        equity=25_000.0,
    )

    def _fake_outcomes(_config):
        return None

    def _fake_build_packs(_config):
        return [str(tmp_path / "pack.json")]

    def _fake_run_experiments(_config, _pack_paths):
        return None

    def _fake_evaluate(_config):
        rankings_payload = {
            "generated_at": "2026-02-13T00:00:00Z",
            "rankings": _make_rankings(),
        }
        rankings_path.write_text(json.dumps(rankings_payload), encoding="utf-8")
        return str(rankings_path)

    monkeypatch.setattr("scripts.strategy_research_loop.run_outcomes_backfill", _fake_outcomes)
    monkeypatch.setattr("scripts.strategy_research_loop.build_packs", _fake_build_packs)
    monkeypatch.setattr("scripts.strategy_research_loop.run_experiments", _fake_run_experiments)
    monkeypatch.setattr("scripts.strategy_research_loop.evaluate_and_persist", _fake_evaluate)

    run_cycle(config)

    assert allocations_path.exists()
    data = json.loads(allocations_path.read_text(encoding="utf-8"))
    assert "generated_at" in data
    assert "equity" in data
    assert "config" in data
    assert "allocations" in data
    assert isinstance(data["allocations"], list)

    required_fields = {"strategy_id", "instrument", "weight", "dollar_risk", "contracts"}
    for row in data["allocations"]:
        assert required_fields.issubset(row.keys())
