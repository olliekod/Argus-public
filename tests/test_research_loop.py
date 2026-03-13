"""
Tests for the Strategy Research Loop.

Covers:
- Config loader with valid YAML
- Config loader with last_n_days
- Config validation errors (invalid mode, missing symbols, etc.)
- Full integration cycle (tiny pack, minimal strategy)
- Dry-run mode
"""

import json
import pytest
from pathlib import Path

import yaml

from src.analysis.research_loop_config import (
    ConfigValidationError,
    ResearchLoopConfig,
    load_research_loop_config,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Config loader tests
# ═══════════════════════════════════════════════════════════════════════════


def _write_yaml(path: Path, data: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f)
    return str(path)


def _minimal_config(**overrides) -> dict:
    """Return a minimal valid config dict."""
    cfg = {
        "pack": {
            "mode": "single",
            "symbols": ["SPY"],
            "start_date": "2026-02-01",
            "end_date": "2026-02-05",
            "packs_output_dir": "data/packs",
            "db_path": "data/argus.db",
        },
        "outcomes": {
            "ensure_before_pack": False,
            "bar_duration": 60,
        },
        "strategies": [
            {"strategy_class": "VRPCreditSpreadStrategy", "params": {}},
        ],
        "experiment": {
            "output_dir": "logs/experiments",
            "starting_cash": 10000.0,
            "regime_stress": False,
            "mc_bootstrap": False,
            "mc_paths": 100,
            "mc_method": "bootstrap",
            "mc_seed": 42,
            "mc_ruin_level": 0.2,
        },
        "evaluation": {
            "rankings_output_dir": "logs",
        },
        "loop": {
            "interval_hours": 24,
        },
    }
    cfg.update(overrides)
    return cfg


class TestConfigLoader:
    def test_valid_config(self, tmp_path):
        cfg_path = _write_yaml(tmp_path / "config.yaml", _minimal_config())
        config = load_research_loop_config(cfg_path, project_root=tmp_path)

        assert config.pack.mode == "single"
        assert config.pack.symbols == ["SPY"]
        assert config.pack.start_date == "2026-02-01"
        assert config.pack.end_date == "2026-02-05"
        assert len(config.strategies) == 1
        assert config.strategies[0].strategy_class == "VRPCreditSpreadStrategy"
        assert config.experiment.starting_cash == 10000.0
        assert config.outcomes.ensure_before_pack is False

    def test_last_n_days_overrides_dates(self, tmp_path):
        raw = _minimal_config()
        raw["pack"]["last_n_days"] = 3
        cfg_path = _write_yaml(tmp_path / "config.yaml", raw)

        config = load_research_loop_config(cfg_path, project_root=tmp_path)

        # Dates should be computed from today
        from datetime import datetime, timedelta, timezone
        today = datetime.now(timezone.utc).date()
        expected_end = today.strftime("%Y-%m-%d")
        expected_start = (today - timedelta(days=3)).strftime("%Y-%m-%d")
        assert config.pack.end_date == expected_end
        assert config.pack.start_date == expected_start

    def test_universe_mode_no_symbols_required(self, tmp_path):
        raw = _minimal_config()
        raw["pack"]["mode"] = "universe"
        raw["pack"]["symbols"] = []
        cfg_path = _write_yaml(tmp_path / "config.yaml", raw)

        config = load_research_loop_config(cfg_path, project_root=tmp_path)
        assert config.pack.mode == "universe"

    def test_invalid_mode(self, tmp_path):
        raw = _minimal_config()
        raw["pack"]["mode"] = "invalid"
        cfg_path = _write_yaml(tmp_path / "config.yaml", raw)

        with pytest.raises(ConfigValidationError, match="pack.mode"):
            load_research_loop_config(cfg_path, project_root=tmp_path)

    def test_single_mode_no_symbols(self, tmp_path):
        raw = _minimal_config()
        raw["pack"]["symbols"] = []
        cfg_path = _write_yaml(tmp_path / "config.yaml", raw)

        with pytest.raises(ConfigValidationError, match="pack.symbols"):
            load_research_loop_config(cfg_path, project_root=tmp_path)

    def test_no_dates_no_last_n_days(self, tmp_path):
        raw = _minimal_config()
        del raw["pack"]["start_date"]
        del raw["pack"]["end_date"]
        cfg_path = _write_yaml(tmp_path / "config.yaml", raw)

        with pytest.raises(ConfigValidationError, match="start_date"):
            load_research_loop_config(cfg_path, project_root=tmp_path)

    def test_no_strategies(self, tmp_path):
        raw = _minimal_config()
        raw["strategies"] = []
        cfg_path = _write_yaml(tmp_path / "config.yaml", raw)

        with pytest.raises(ConfigValidationError, match="strategies"):
            load_research_loop_config(cfg_path, project_root=tmp_path)

    def test_strategy_missing_class(self, tmp_path):
        raw = _minimal_config()
        raw["strategies"] = [{"params": {}}]
        cfg_path = _write_yaml(tmp_path / "config.yaml", raw)

        with pytest.raises(ConfigValidationError, match="strategy_class"):
            load_research_loop_config(cfg_path, project_root=tmp_path)

    def test_invalid_date_format(self, tmp_path):
        raw = _minimal_config()
        raw["pack"]["start_date"] = "02/01/2026"
        cfg_path = _write_yaml(tmp_path / "config.yaml", raw)

        with pytest.raises(ConfigValidationError, match="YYYY-MM-DD"):
            load_research_loop_config(cfg_path, project_root=tmp_path)

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_research_loop_config(
                str(tmp_path / "nonexistent.yaml"),
                project_root=tmp_path,
            )

    def test_paths_resolved_relative_to_project_root(self, tmp_path):
        raw = _minimal_config()
        raw["pack"]["packs_output_dir"] = "my_packs"
        raw["experiment"]["output_dir"] = "my_logs/exp"
        cfg_path = _write_yaml(tmp_path / "config.yaml", raw)

        config = load_research_loop_config(cfg_path, project_root=tmp_path)
        assert config.pack.packs_output_dir == str(tmp_path / "my_packs")
        assert config.experiment.output_dir == str(tmp_path / "my_logs" / "exp")

    def test_evaluation_input_defaults_to_experiment_output(self, tmp_path):
        raw = _minimal_config()
        cfg_path = _write_yaml(tmp_path / "config.yaml", raw)

        config = load_research_loop_config(cfg_path, project_root=tmp_path)
        assert config.evaluation.input_dir == config.experiment.output_dir

    def test_sweep_path_resolved(self, tmp_path):
        raw = _minimal_config()
        raw["strategies"][0]["sweep"] = "config/vrp_sweep.yaml"
        cfg_path = _write_yaml(tmp_path / "config.yaml", raw)

        config = load_research_loop_config(cfg_path, project_root=tmp_path)
        assert config.strategies[0].sweep == str(tmp_path / "config" / "vrp_sweep.yaml")

    def test_allocation_max_loss_per_contract_float(self, tmp_path):
        raw = _minimal_config()
        raw["evaluation"]["allocation"] = {
            "kelly_fraction": 0.25,
            "per_play_cap": 0.07,
            "max_loss_per_contract": 250.0,
        }
        cfg_path = _write_yaml(tmp_path / "config.yaml", raw)

        config = load_research_loop_config(cfg_path, project_root=tmp_path)
        assert config.evaluation.allocation is not None
        assert config.evaluation.allocation.max_loss_per_contract == 250.0

    def test_allocation_max_loss_per_contract_dict(self, tmp_path):
        raw = _minimal_config()
        raw["evaluation"]["allocation"] = {
            "kelly_fraction": 0.25,
            "per_play_cap": 0.07,
            "max_loss_per_contract": {"VRP_v1": 300.0, "VRP_v2": 450.0},
        }
        cfg_path = _write_yaml(tmp_path / "config.yaml", raw)

        config = load_research_loop_config(cfg_path, project_root=tmp_path)
        assert config.evaluation.allocation is not None
        assert config.evaluation.allocation.max_loss_per_contract == {"VRP_v1": 300.0, "VRP_v2": 450.0}

    def test_multiple_strategies(self, tmp_path):
        raw = _minimal_config()
        raw["strategies"] = [
            {"strategy_class": "StratA", "params": {"a": 1}},
            {"strategy_class": "StratB", "params": {"b": 2}, "sweep": "s.yaml"},
        ]
        cfg_path = _write_yaml(tmp_path / "config.yaml", raw)

        config = load_research_loop_config(cfg_path, project_root=tmp_path)
        assert len(config.strategies) == 2
        assert config.strategies[0].strategy_class == "StratA"
        assert config.strategies[1].strategy_class == "StratB"
        assert config.strategies[1].params == {"b": 2}


# ═══════════════════════════════════════════════════════════════════════════
#  Integration test: full cycle with tiny pack and mock strategy
# ═══════════════════════════════════════════════════════════════════════════


class TestIntegrationCycle:
    """Integration test that runs one full cycle with a mock pack."""

    def _create_mock_pack(self, pack_dir: Path, symbol: str = "SPY") -> str:
        """Create a tiny mock replay pack JSON."""
        pack = {
            "metadata": {
                "symbol": symbol,
                "provider": "test",
                "bars_provider": "test",
                "options_snapshot_provider": "test",
                "bar_duration": 60,
                "start_date": "2026-02-01",
                "end_date": "2026-02-01",
                "packed_at": "2026-02-01T00:00:00+00:00",
                "bar_count": 5,
                "outcome_count": 0,
                "regime_count": 0,
                "snapshot_count": 0,
            },
            "bars": [
                {
                    "timestamp_ms": 1738368000000 + i * 60000,
                    "open": 100.0 + i,
                    "high": 101.0 + i,
                    "low": 99.0 + i,
                    "close": 100.5 + i,
                    "volume": 1000,
                    "symbol": symbol,
                }
                for i in range(5)
            ],
            "outcomes": [],
            "regimes": [],
            "snapshots": [],
        }
        pack_path = pack_dir / f"{symbol}_2026-02-01_2026-02-01.json"
        pack_dir.mkdir(parents=True, exist_ok=True)
        with open(pack_path, "w") as f:
            json.dump(pack, f)
        return str(pack_path)

    def test_run_experiments_and_evaluate(self, tmp_path):
        """Run experiments on a mock pack, then evaluate — check output files."""
        from src.analysis.experiment_runner import ExperimentRunner, ExperimentConfig
        from src.analysis.replay_harness import ReplayStrategy, TradeIntent
        from src.analysis.strategy_evaluator import StrategyEvaluator

        # Create mock pack
        pack_dir = tmp_path / "packs"
        pack_path = self._create_mock_pack(pack_dir)

        # Use a simple mock strategy that produces no trades
        class TestStrategy(ReplayStrategy):
            def __init__(self, params):
                self._params = params
                self._bars = 0

            @property
            def strategy_id(self) -> str:
                return "TEST_STRATEGY_V1"

            def on_bar(self, bar, sim_ts_ms, session, visible_outcomes, **kw):
                self._bars += 1

            def generate_intents(self, sim_ts_ms):
                return []

            def finalize(self):
                return {"bars": self._bars, "params": self._params}

        # Run experiment
        exp_dir = tmp_path / "experiments"
        runner = ExperimentRunner(output_dir=str(exp_dir))
        config = ExperimentConfig(
            strategy_class=TestStrategy,
            strategy_params={"test": True},
            replay_pack_paths=[pack_path],
            starting_cash=10000.0,
            output_dir=str(exp_dir),
        )
        result = runner.run(config)
        assert result.bars_replayed == 5

        # Check experiment output exists
        exp_files = list(exp_dir.glob("*.json"))
        assert len(exp_files) >= 1

        # Evaluate
        rankings_dir = tmp_path / "rankings"
        evaluator = StrategyEvaluator(
            input_dir=str(exp_dir),
            output_dir=str(rankings_dir),
        )
        count = evaluator.load_experiments()
        assert count >= 1

        evaluator.evaluate()
        rankings_path = evaluator.save_rankings()
        assert Path(rankings_path).exists()

        # Verify rankings structure
        with open(rankings_path) as f:
            rankings_data = json.load(f)
        assert "rankings" in rankings_data
        assert "killed" in rankings_data
        assert rankings_data["experiment_count"] >= 1

        # Verify candidate set generation
        candidates = [
            rec for rec in evaluator.rankings
            if not rec.get("killed", False)
        ]
        assert isinstance(candidates, list)

    def test_dry_run_no_side_effects(self, tmp_path):
        """Dry run should not create any experiment or ranking files."""
        from scripts.strategy_research_loop import run_cycle
        from src.analysis.research_loop_config import (
            ResearchLoopConfig,
            PackConfig,
            OutcomesConfig,
            StrategySpec,
            ExperimentOpts,
            EvaluationOpts,
            LoopOpts,
        )

        exp_dir = tmp_path / "experiments"
        rankings_dir = tmp_path / "rankings"

        config = ResearchLoopConfig(
            pack=PackConfig(
                mode="single",
                symbols=["SPY"],
                start_date="2026-02-01",
                end_date="2026-02-01",
                bars_provider="test",
                options_snapshot_provider=None,
                packs_output_dir=str(tmp_path / "packs"),
                db_path=str(tmp_path / "argus.db"),
            ),
            outcomes=OutcomesConfig(ensure_before_pack=False, bar_duration=60),
            strategies=[
                StrategySpec(
                    strategy_class="VRPCreditSpreadStrategy",
                    params={},
                    sweep=None,
                ),
            ],
            experiment=ExperimentOpts(
                output_dir=str(exp_dir),
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
                input_dir=str(exp_dir),
                kill_thresholds=None,
                rankings_output_dir=str(rankings_dir),
                killed_output_path=None,
                candidate_set_output_path=None,
            ),
            loop=LoopOpts(interval_hours=24, require_recent_bars_hours=None),
        )

        run_cycle(config, dry_run=True)

        # No files should be created
        assert not exp_dir.exists() or len(list(exp_dir.glob("*.json"))) == 0
        assert not rankings_dir.exists() or len(list(rankings_dir.glob("*.json"))) == 0


# ═══════════════════════════════════════════════════════════════════════════
#  Candidate set output test
# ═══════════════════════════════════════════════════════════════════════════


class TestCandidateSet:
    def test_candidate_set_written(self, tmp_path):
        """When candidate_set_output_path is set, candidates JSON is written."""
        from src.analysis.experiment_runner import ExperimentRunner, ExperimentConfig
        from src.analysis.replay_harness import ReplayStrategy
        from scripts.strategy_research_loop import evaluate_and_persist
        from src.analysis.research_loop_config import EvaluationOpts

        class NoopStrategy(ReplayStrategy):
            def __init__(self, params):
                self._p = params

            @property
            def strategy_id(self):
                return "NOOP_V1"

            def on_bar(self, *a, **kw):
                pass

            def generate_intents(self, sim_ts_ms):
                return []

            def finalize(self):
                return {"params": self._p}

        # Create a tiny pack and run one experiment
        pack_dir = tmp_path / "packs"
        pack_dir.mkdir()
        pack = {
            "metadata": {"symbol": "SPY", "provider": "test",
                         "bars_provider": "test",
                         "options_snapshot_provider": "test",
                         "bar_duration": 60,
                         "start_date": "2026-02-01",
                         "end_date": "2026-02-01",
                         "packed_at": "2026-02-01T00:00:00+00:00",
                         "bar_count": 2, "outcome_count": 0,
                         "regime_count": 0, "snapshot_count": 0},
            "bars": [
                {"timestamp_ms": 1000, "open": 100, "high": 101,
                 "low": 99, "close": 100, "volume": 100, "symbol": "SPY"},
                {"timestamp_ms": 2000, "open": 100, "high": 101,
                 "low": 99, "close": 100, "volume": 100, "symbol": "SPY"},
            ],
            "outcomes": [],
            "regimes": [],
            "snapshots": [],
        }
        pack_path = pack_dir / "test.json"
        with open(pack_path, "w") as f:
            json.dump(pack, f)

        exp_dir = tmp_path / "experiments"
        runner = ExperimentRunner(output_dir=str(exp_dir))
        runner.run(ExperimentConfig(
            strategy_class=NoopStrategy,
            strategy_params={},
            replay_pack_paths=[str(pack_path)],
            output_dir=str(exp_dir),
        ))

        # Now evaluate with candidate set output
        cand_path = tmp_path / "candidates.json"
        from src.analysis.research_loop_config import ResearchLoopConfig, PackConfig, OutcomesConfig, StrategySpec, ExperimentOpts, LoopOpts

        config = ResearchLoopConfig(
            pack=PackConfig("single", ["SPY"], "2026-02-01", "2026-02-01",
                            None, None, str(pack_dir), str(tmp_path / "db")),
            outcomes=OutcomesConfig(False, 60),
            strategies=[StrategySpec("NoopStrategy", {}, None)],
            experiment=ExperimentOpts(str(exp_dir), 10000, False, False,
                                     100, "bootstrap", None, 42, 0.2, None),
            evaluation=EvaluationOpts(
                input_dir=str(exp_dir),
                kill_thresholds=None,
                rankings_output_dir=str(tmp_path / "rankings"),
                killed_output_path=str(tmp_path / "killed.json"),
                candidate_set_output_path=str(cand_path),
            ),
            loop=LoopOpts(24, None),
        )

        evaluate_and_persist(config)

        # Verify candidate set
        assert cand_path.exists()
        with open(cand_path) as f:
            cand_data = json.load(f)
        assert "candidates" in cand_data
        assert "generated_at" in cand_data
        assert "candidate_count" in cand_data
        assert isinstance(cand_data["candidates"], list)

        # Verify killed list
        killed_path = tmp_path / "killed.json"
        assert killed_path.exists()
        with open(killed_path) as f:
            killed_data = json.load(f)
        assert "killed" in killed_data
        assert "killed_count" in killed_data
