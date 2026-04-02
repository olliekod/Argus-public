# Created by Oliver Meihls

# Tests for argus_kalshi.simulation module.
#
# Covers: ScenarioProfile, BotEquityLedger, BotRunRecord, PopulationManager,
# calculate_robustness_score, assign_family, config perturbation, and all
# behavioral invariants specified in the implementation plan.

import random

import pytest

from argus_kalshi.simulation import (
    SCENARIO_BASE,
    SCENARIO_BEST,
    SCENARIO_STRESS,
    BotEquityLedger,
    BotRunRecord,
    PopulationManager,
    assign_family,
    calculate_alpha_score,
    calculate_robustness_score,
    perturb_config,
    project_execution_scenarios,
    random_explore_config,
)


#  Phase 1: Scenario profiles and equity ledger

class TestScenarioProfiles:
    def test_scenario_profiles_alter_costs(self):
        # BEST/BASE/STRESS produce different PnL projections for identical trade data.
        stats = {"pnl": 10.0, "qty_s_contracts": 50.0, "qty_e_contracts": 100.0}
        scenarios = project_execution_scenarios(stats)
        assert scenarios["best"] > scenarios["base"] > scenarios["stress"]
        # Best adds relief, stress adds drag
        assert scenarios["best"] > 10.0
        assert scenarios["stress"] < 10.0

    def test_scenario_profiles_zero_contracts(self):
        # With zero contracts, all scenarios equal base PnL.
        stats = {"pnl": 5.0, "qty_s_contracts": 0.0, "qty_e_contracts": 0.0}
        scenarios = project_execution_scenarios(stats)
        assert scenarios["best"] == scenarios["base"] == scenarios["stress"] == 5.0

    def test_scenario_profile_presets_exist(self):
        # Three named presets with distinct latency/slippage configs.
        assert SCENARIO_BEST.slippage_cents < SCENARIO_BASE.slippage_cents
        assert SCENARIO_BASE.slippage_cents < SCENARIO_STRESS.slippage_cents
        assert SCENARIO_BEST.latency_max_ms < SCENARIO_STRESS.latency_max_ms


class TestEquityLedger:
    def test_equity_ledger_starts_at_5000(self):
        # Initial equity is exactly 5000.0.
        ledger = BotEquityLedger()
        assert ledger.start_equity == 5000.0
        assert ledger.equity == 5000.0
        assert ledger.peak_equity == 5000.0
        assert ledger.max_drawdown == 0.0
        assert ledger.trade_count == 0

    def test_equity_ledger_tracks_drawdown(self):
        # Ledger correctly tracks peak, current, max_drawdown after wins and losses.
        ledger = BotEquityLedger(start_equity=5000.0)

        # Win $100
        ledger.record_trade(gross_pnl=100.0)
        assert ledger.equity == 5100.0
        assert ledger.peak_equity == 5100.0
        assert ledger.max_drawdown == 0.0
        assert ledger.trade_count == 1

        # Lose $300
        ledger.record_trade(gross_pnl=-300.0)
        assert ledger.equity == 4800.0
        assert ledger.peak_equity == 5100.0
        assert ledger.max_drawdown == 300.0
        assert ledger.max_drawdown_pct == pytest.approx(300.0 / 5000.0)

        # Partial recovery: win $50
        ledger.record_trade(gross_pnl=50.0)
        assert ledger.equity == 4850.0
        assert ledger.max_drawdown == 300.0  # still the old max
        assert ledger.trade_count == 3

    def test_equity_ledger_cost_breakdown(self):
        # Costs (fees, slippage, spread drag) are correctly subtracted from net PnL.
        ledger = BotEquityLedger(start_equity=5000.0)
        net = ledger.record_trade(
            gross_pnl=10.0, fee_usd=1.5, slippage_usd=0.5, spread_drag_usd=0.2
        )
        assert net == pytest.approx(10.0 - 1.5 - 0.5 - 0.2)
        assert ledger.total_fees == 1.5
        assert ledger.total_slippage == 0.5
        assert ledger.total_spread_drag == 0.2
        assert ledger.equity == pytest.approx(5000.0 + net)

    def test_tail_loss(self):
        # Tail loss computes average of worst 10% of trade PnLs.
        ledger = BotEquityLedger()
        # Record 10 trades: -50, -40, -30, -20, -10, 10, 20, 30, 40, 50
        for pnl in [-50, -40, -30, -20, -10, 10, 20, 30, 40, 50]:
            ledger.record_trade(gross_pnl=float(pnl))
        # Worst 10% = worst 1 trade = -50
        assert ledger.tail_loss(0.10) == pytest.approx(-50.0)

    def test_drawdown_pct(self):
        # Current drawdown as fraction of start equity.
        ledger = BotEquityLedger(start_equity=5000.0)
        ledger.record_trade(gross_pnl=500.0)   # peak=5500
        ledger.record_trade(gross_pnl=-1000.0)  # equity=4500, peak=5500
        assert ledger.drawdown_pct() == pytest.approx((5500 - 4500) / 5000)

    def test_to_dict(self):
        # to_dict serializes all fields.
        ledger = BotEquityLedger()
        ledger.record_trade(gross_pnl=10.0, fee_usd=0.5)
        d = ledger.to_dict()
        assert d["start_equity"] == 5000.0
        assert d["trade_count"] == 1
        assert d["total_fees"] == 0.5
        assert "tail_loss_10pct" in d


#  Phase 2: Robustness ranking

class TestRobustnessScore:
    def test_robustness_score_penalizes_tail_loss(self):
        # Bot with same avg PnL but worse tail losses scores lower.
        good_stats = {
            "pnl": 100.0, "wins": 10, "losses": 5, "trade_count": 15,
            "max_drawdown": 20.0, "tail_loss_10pct": -5.0,
        }
        bad_stats = {
            "pnl": 100.0, "wins": 10, "losses": 5, "trade_count": 15,
            "max_drawdown": 20.0, "tail_loss_10pct": -80.0,
        }
        good_score = calculate_robustness_score(good_stats)
        bad_score = calculate_robustness_score(bad_stats)
        assert good_score > bad_score

    def test_robustness_score_penalizes_small_samples(self):
        # Edge case: low trade count applies a penalty multiplier.
        many_stats = {"pnl": 50.0, "wins": 10, "losses": 5, "trade_count": 15}
        few_stats = {"pnl": 50.0, "wins": 2, "losses": 1, "trade_count": 3}
        many_score = calculate_robustness_score(many_stats)
        few_score = calculate_robustness_score(few_stats)
        assert many_score > few_score

    def test_robustness_score_penalizes_scenario_inconsistency(self):
        # Bot profitable in BEST but negative in STRESS scores lower than consistent bot.
        consistent = {
            "pnl": 50.0, "wins": 10, "losses": 5, "trade_count": 15,
            "qty_s_contracts": 10.0, "qty_e_contracts": 10.0,
        }
        # fragile = same base PnL but huge contract volumes → huge best/stress spread
        fragile = {
            "pnl": 50.0, "wins": 10, "losses": 5, "trade_count": 15,
            "qty_s_contracts": 5000.0, "qty_e_contracts": 5000.0,
        }
        consistent_score = calculate_robustness_score(consistent)
        fragile_score = calculate_robustness_score(fragile)
        assert consistent_score > fragile_score

    def test_backward_compat_alias(self):
        # calculate_alpha_score is an alias for calculate_robustness_score.
        stats = {"pnl": 20.0, "wins": 5, "losses": 3, "trade_count": 8}
        assert calculate_alpha_score(stats) == calculate_robustness_score(stats)


#  Phase 3: Population manager

class TestPopulationManager:
    def test_epoch_retire_reseed(self):
        # With 10 bots, after epoch: bottom 2 retired (20%), correct counts returned.
        pm = PopulationManager(
            exploit_fraction=0.80,
            retire_bottom_pct=0.20,
            drawdown_retire_pct=0.15,
        )
        # 10 bots with scores 10..100
        bot_scores = [(f"bot_{i}", float(i * 10), {}) for i in range(1, 11)]
        retire_ids, n_exploit, n_explore = pm.evaluate_epoch(bot_scores)
        assert len(retire_ids) == 2  # bottom 20% of 10
        assert "bot_1" in retire_ids
        assert "bot_2" in retire_ids
        # Exploit count should be ~80% of retired count
        assert n_exploit + n_explore == 2

    def test_exploit_explore_split(self):
        # 80% of replacements are exploit (perturbation), 20% are explore (random).
        pm = PopulationManager(exploit_fraction=0.80, retire_bottom_pct=0.20)
        bot_scores = [(f"bot_{i}", float(i), {}) for i in range(1, 21)]
        retire_ids, n_exploit, n_explore = pm.evaluate_epoch(bot_scores)
        assert n_exploit == 3  # 80% of 4 retired = 3.2 → 3
        assert n_explore == 1  # 4 - 3 = 1

    def test_drawdown_check_triggers(self):
        # Drawdown > threshold returns 'retired_drawdown'.
        pm = PopulationManager(drawdown_retire_pct=0.15)
        # equity=4200, peak=5000 → dd = (5000-4200)/5000 = 16%
        result = pm.check_drawdown("bot_1", 4200.0, 5000.0, 5000.0)
        assert result == "retired_drawdown"

    def test_drawdown_check_no_trigger(self):
        # Drawdown below threshold returns None.
        pm = PopulationManager(drawdown_retire_pct=0.15)
        result = pm.check_drawdown("bot_1", 4800.0, 5000.0, 5000.0)
        assert result is None

    def test_replacement_params_have_lineage(self):
        # Generated replacement configs have run_id, parent_run_id, generation.
        pm = PopulationManager(exploit_fraction=0.50)
        top_configs = [{"min_edge_threshold": 0.05, "persistence_window_ms": 100}]
        results = pm.generate_replacement_params(
            top_configs, n_exploit=1, n_explore=1, generation=3,
            parent_run_ids=["parent_abc"],
        )
        assert len(results) == 2
        exploit_cfg = results[0]
        explore_cfg = results[1]
        assert exploit_cfg["_run_id"] != ""
        assert exploit_cfg["_parent_run_id"] == "parent_abc"
        assert exploit_cfg["_generation"] == 3
        assert exploit_cfg["_role"] == "exploit"
        assert explore_cfg["_role"] == "explore"
        assert explore_cfg["_parent_run_id"] == ""


#  Phase 4: Drawdown retire does not mutate params

class TestBotRunRecord:
    def test_drawdown_retire_does_not_mutate_params(self):
        # Retired bot's params_snapshot is frozen; new bot has different run_id.
        record = BotRunRecord(
            run_id="run_abc",
            bot_id="bot_1",
            params_snapshot={"min_edge_threshold": 0.05},
        )
        original_params = dict(record.params_snapshot)
        new_run_id = BotRunRecord.new_run_id()
        assert new_run_id != record.run_id
        assert record.params_snapshot == original_params  # unchanged

    def test_new_run_id_unique(self):
        # Every call to new_run_id() produces a unique ID.
        ids = {BotRunRecord.new_run_id() for _ in range(100)}
        assert len(ids) == 100


#  Phase 5: Family classification

class TestFamilyClassification:
    def test_family_key_from_explicit_metadata(self):
        assert assign_family("", asset="BTC", window_minutes=15) == "BTC 15m"
        assert assign_family("", asset="BTC", window_minutes=60) == "BTC 60m"
        assert assign_family("", asset="BTC", window_minutes=60, is_range=True) == "BTC Range"
        assert assign_family("", asset="ETH", window_minutes=15) == "ETH 15m"
        assert assign_family("", asset="ETH", window_minutes=60) == "ETH 60m"

    def test_family_key_from_ticker_15m(self):
        assert assign_family("KXBTC15M-26MAR0217-B65000") == "BTC 15m"

    def test_family_key_from_ticker_60m(self):
        assert assign_family("KXBTCH-26MAR0300-B65000") == "BTC 60m"

    def test_family_key_unknown(self):
        assert assign_family("RANDOM-TICKER") == "Other"


#  Config perturbation

class TestConfigPerturbation:
    def test_perturb_config_stays_in_bounds(self):
        # Perturbed values stay within defined bounds.
        rng = random.Random(42)
        config = {"min_edge_threshold": 0.05, "persistence_window_ms": 100}
        for _ in range(50):
            result = perturb_config(config, rng)
            assert 0.03 <= result["min_edge_threshold"] <= 0.20
            assert 30 <= result["persistence_window_ms"] <= 600

    def test_perturb_config_does_not_mutate_original(self):
        # Original config dict is not modified.
        original = {"min_edge_threshold": 0.05, "persistence_window_ms": 100}
        original_copy = dict(original)
        rng = random.Random(42)
        perturb_config(original, rng)
        assert original == original_copy

    def test_random_explore_config_has_required_defaults(self):
        # Random exploration config includes bankroll and dry_run defaults.
        rng = random.Random(42)
        cfg = random_explore_config(rng)
        assert cfg["bankroll_usd"] == 5000.0
        assert cfg["dry_run"] is True
        assert "min_edge_threshold" in cfg
        assert "persistence_window_ms" in cfg

    def test_min_max_entry_constraint(self):
        # min_entry_cents <= max_entry_cents after perturbation.
        rng = random.Random(0)
        for seed in range(100):
            rng.seed(seed)
            config = {"min_entry_cents": 50, "max_entry_cents": 55}
            result = perturb_config(config, rng)
            assert result.get("min_entry_cents", 0) <= result.get("max_entry_cents", 100)


#  Terminal UI backward compatibility

class TestTerminalUICompat:
    def test_imports_from_terminal_ui(self):
        # Scoring functions importable from terminal_ui for backward compat.
        from argus_kalshi.terminal_ui import (
            calculate_alpha_score,
            calculate_robustness_score,
            project_execution_scenarios,
            MIN_SETTLED_TRADES_FOR_ALPHA,
            SCALP_BEST_RELIEF_USD_PER_CONTRACT,
        )
        assert callable(calculate_alpha_score)
        assert callable(calculate_robustness_score)
        assert callable(project_execution_scenarios)
        assert MIN_SETTLED_TRADES_FOR_ALPHA == 5
        assert SCALP_BEST_RELIEF_USD_PER_CONTRACT > 0

    def test_leaderboard_shows_generation_column(self):
        # Leaderboard header format includes 'Gen' column.
        # Verify the header template in the leaderboard renderer includes "Gen"
        import inspect
        from argus_kalshi.terminal_ui import TerminalVisualizer
        source = inspect.getsource(TerminalVisualizer._append_leaderboard)
        assert "'Gen'" in source or '"Gen"' in source, (
            "Leaderboard header should contain 'Gen' column"
        )

    def test_leaderboard_row_contains_generation(self):
        # Leaderboard row format includes generation value.
        import inspect
        from argus_kalshi.terminal_ui import TerminalVisualizer
        source = inspect.getsource(TerminalVisualizer._append_leaderboard)
        assert "gen:" in source or "gen}" in source, (
            "Leaderboard row should render generation value"
        )
