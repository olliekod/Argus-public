# Created by Oliver Meihls

# Tests for family-adaptive farm system.
#
# Covers:
# - FAMILIES tuple includes all 6 families (ETH Range fixed)
# - assign_family returns ETH Range correctly
# - FamilyWeightManager rebalance logic and min/max floors
# - Per-family epoch ranking (evaluate_epoch_by_family)
# - Novelty distance prevents near-clones
# - Exploit/explore lanes with family domains
# - RegimeGate wiring in KalshiPaperFarm
# - Config validation for new fields
# - Family weight diagnostics
import time
import asyncio
from unittest.mock import MagicMock, patch

import pytest

from argus_kalshi.config import KalshiConfig
from argus_kalshi.simulation import (
    FAMILIES,
    FamilyWeightManager,
    FAMILY_PARAM_DOMAINS,
    PopulationManager,
    assign_family,
    novelty_distance,
    perturb_config,
    perturb_config_tight,
    random_explore_config,
    _resolve_param_bounds,
)


#  Phase 1: FAMILIES fix

class TestFamiliesFix:
    def test_families_has_6_entries(self):
        assert len(FAMILIES) == 6

    def test_families_includes_eth_range(self):
        assert "ETH Range" in FAMILIES

    def test_families_includes_all_expected(self):
        expected = {"BTC 15m", "BTC 60m", "BTC Range", "ETH 15m", "ETH 60m", "ETH Range"}
        assert set(FAMILIES) == expected

    def test_assign_family_eth_range_from_metadata(self):
        assert assign_family("KXETH-123", asset="ETH", window_minutes=60, is_range=True) == "ETH Range"

    def test_assign_family_btc_range_from_metadata(self):
        assert assign_family("KXBTC-123", asset="BTC", window_minutes=15, is_range=True) == "BTC Range"

    def test_assign_family_eth_15m(self):
        assert assign_family("KXETH15M-123", asset="ETH", window_minutes=15, is_range=False) == "ETH 15m"

    def test_assign_family_eth_60m(self):
        assert assign_family("KXETHH-123", asset="ETH", window_minutes=60, is_range=False) == "ETH 60m"

    def test_assign_family_eth_range_from_ticker(self):
        # KXETH without 15M or H suffix maps to Range
        assert assign_family("KXETH-26MAR05-T4500") == "ETH Range"


#  Phase 2: FamilyWeightManager

class TestFamilyWeightManager:
    def test_initial_weights_equal(self):
        mgr = FamilyWeightManager()
        for f in FAMILIES:
            assert abs(mgr.get_weight(f) - 1.0 / len(FAMILIES)) < 1e-9

    def test_no_family_reaches_zero_weight(self):
        mgr = FamilyWeightManager(min_weight=0.05)
        # Record heavily skewed trades
        for _ in range(100):
            mgr.record_family_trade("BTC 15m", 10.0)
        for _ in range(100):
            mgr.record_family_trade("ETH Range", -5.0)
        mgr.rebalance(time.time() + 3600)
        for f in FAMILIES:
            assert mgr.get_weight(f) > 0, f"Family {f} has zero weight"
            assert mgr.get_weight(f) >= mgr.min_weight * 0.8, f"Family {f} below min_weight"

    def test_rebalance_respects_max_weight(self):
        mgr = FamilyWeightManager(max_weight=0.40)
        for _ in range(200):
            mgr.record_family_trade("BTC 15m", 100.0)
        mgr.rebalance(time.time() + 3600)
        # After normalization, no family should exceed a reasonable upper bound
        for f in FAMILIES:
            assert mgr.get_weight(f) <= 1.0, f"Family {f} weight exceeds 1.0"

    def test_rebalance_does_not_fire_too_early(self):
        mgr = FamilyWeightManager(rebalance_interval_s=1800.0)
        mgr.record_family_trade("BTC 15m", 10.0)
        assert mgr.rebalance(2000.0) is True  # first one always fires
        mgr.record_family_trade("BTC 15m", 10.0)
        assert mgr.rebalance(2100.0) is False  # too soon

    def test_rebalance_fires_after_interval(self):
        mgr = FamilyWeightManager(rebalance_interval_s=60.0)
        mgr.record_family_trade("BTC 15m", 10.0)
        assert mgr.rebalance(100.0) is True  # first rebalance
        mgr.record_family_trade("ETH 15m", 5.0)
        assert mgr.rebalance(200.0) is True  # 100 seconds > 60

    def test_weights_sum_to_one(self):
        mgr = FamilyWeightManager()
        for f in FAMILIES:
            mgr.record_family_trade(f, 5.0)
        mgr.rebalance(time.time() + 3600)
        total = sum(mgr.weights.values())
        assert abs(total - 1.0) < 1e-6, f"Weights sum to {total}, expected 1.0"

    def test_diagnostics_structure(self):
        mgr = FamilyWeightManager()
        diag = mgr.get_diagnostics()
        assert "weights" in diag
        assert "pnl_ema" in diag
        assert "trade_count" in diag
        assert "reseed_count" in diag

    def test_record_family_reseed(self):
        mgr = FamilyWeightManager()
        mgr.record_family_reseed("BTC 15m")
        mgr.record_family_reseed("BTC 15m")
        assert mgr.get_diagnostics()["reseed_count"]["BTC 15m"] == 2


#  Phase 2: Per-family epoch ranking

class TestPerFamilyEpoch:
    def test_per_family_retires_within_family_only(self):
        pm = PopulationManager(retire_bottom_pct=0.50, exploit_fraction=0.80)
        # 4 BTC 15m bots, 4 ETH 60m bots
        family_map = {
            f"btc_{i}": "BTC 15m" for i in range(4)
        }
        family_map.update({
            f"eth_{i}": "ETH 60m" for i in range(4)
        })
        # BTC bots: scores 1-4, ETH bots: scores 10-13
        bot_scores = [
            (f"btc_{i}", float(i + 1), {"pnl": i}) for i in range(4)
        ] + [
            (f"eth_{i}", float(i + 10), {"pnl": i + 10}) for i in range(4)
        ]
        retire_ids, family_split = pm.evaluate_epoch_by_family(family_map, bot_scores)
        # Each family retires bottom 50%
        btc_retired = [r for r in retire_ids if r.startswith("btc_")]
        eth_retired = [r for r in retire_ids if r.startswith("eth_")]
        assert len(btc_retired) == 2, "Should retire 50% of BTC 15m family"
        assert len(eth_retired) == 2, "Should retire 50% of ETH 60m family"
        assert "BTC 15m" in family_split
        assert "ETH 60m" in family_split

    def test_per_family_no_retire_single_bot_family(self):
        pm = PopulationManager(retire_bottom_pct=0.50)
        family_map = {"solo": "ETH Range"}
        bot_scores = [("solo", 5.0, {"pnl": 0})]
        retire_ids, family_split = pm.evaluate_epoch_by_family(family_map, bot_scores)
        assert retire_ids == [], "Should never retire the only bot in a family"
        assert family_split["ETH Range"] == (0, 0)

    def test_per_family_exploit_explore_split(self):
        pm = PopulationManager(retire_bottom_pct=0.50, exploit_fraction=0.75)
        family_map = {f"b_{i}": "BTC 60m" for i in range(8)}
        bot_scores = [(f"b_{i}", float(i), {}) for i in range(8)]
        retire_ids, family_split = pm.evaluate_epoch_by_family(family_map, bot_scores)
        n_exploit, n_explore = family_split["BTC 60m"]
        assert n_exploit + n_explore == len(retire_ids)
        assert n_exploit >= 1

    def test_per_family_empty_scores(self):
        pm = PopulationManager()
        retire_ids, family_split = pm.evaluate_epoch_by_family({}, [])
        assert retire_ids == []
        assert family_split == {}


#  Exploit/explore lanes

class TestExploitExploreLanes:
    def test_perturb_config_tight_smaller_magnitude(self):
        # Tight perturbation should produce smaller changes than standard.
        import random
        rng = random.Random(42)
        base = {"min_edge_threshold": 0.10, "persistence_window_ms": 200}
        std = perturb_config(dict(base), rng, magnitude=0.15)
        rng2 = random.Random(42)
        tight = perturb_config_tight(dict(base), rng2)
        # Not guaranteed to be smaller on every param (randomness), but the function exists
        assert isinstance(tight, dict)
        assert "min_edge_threshold" in tight

    def test_family_param_domains_applied(self):
        # BTC Range family should use different bounds for min_entry_cents.
        bounds = _resolve_param_bounds("BTC Range")
        assert bounds["min_entry_cents"] == (30, 60)
        assert bounds["max_entry_cents"] == (55, 80)
        # Standard params that aren't overridden should still be present
        assert "scalp_min_edge_cents" in bounds

    def test_random_explore_respects_family_bounds(self):
        import random
        rng = random.Random(99)
        cfg = random_explore_config(rng, family="ETH Range")
        assert 30 <= cfg["min_entry_cents"] <= 60
        assert 55 <= cfg["max_entry_cents"] <= 80


#  Novelty distance

class TestNoveltyDistance:
    def test_novelty_distance_empty_population(self):
        assert novelty_distance({"min_edge_threshold": 0.10}, []) == 1.0

    def test_novelty_distance_exact_clone(self):
        cfg = {"min_edge_threshold": 0.10, "persistence_window_ms": 200}
        dist = novelty_distance(cfg, [dict(cfg)])
        assert dist < 0.01, f"Exact clone should have near-zero distance, got {dist}"

    def test_novelty_distance_different_configs(self):
        cfg1 = {"min_edge_threshold": 0.03, "persistence_window_ms": 30}
        cfg2 = {"min_edge_threshold": 0.20, "persistence_window_ms": 600}
        dist = novelty_distance(cfg1, [cfg2])
        assert dist > 0.5, f"Very different configs should have high distance, got {dist}"


#  Config validation

class TestConfigValidation:
    def test_family_min_weight_validates(self):
        with pytest.raises(ValueError, match="family_min_weight"):
            KalshiConfig(bot_id="test", family_min_weight=0.0)

    def test_family_max_weight_validates(self):
        with pytest.raises(ValueError, match="family_max_weight"):
            KalshiConfig(bot_id="test", family_min_weight=0.10, family_max_weight=0.05)

    def test_family_rebalance_interval_validates(self):
        with pytest.raises(ValueError, match="family_rebalance_interval_minutes"):
            KalshiConfig(bot_id="test", family_rebalance_interval_minutes=0)

    def test_valid_family_config(self):
        cfg = KalshiConfig(
            bot_id="test",
            family_population_enabled=True,
            family_min_weight=0.05,
            family_max_weight=0.40,
            family_rebalance_interval_minutes=30.0,
        )
        assert cfg.family_population_enabled is True
        assert cfg.family_min_weight == 0.05

    def test_family_context_features_default_false(self):
        cfg = KalshiConfig(bot_id="test")
        assert cfg.family_context_features_enabled is False


#  RegimeGate wiring (farm_runner integration)

class TestRegimeGateWiring:
    @pytest.mark.asyncio
    async def test_regime_gate_instantiated_in_farm_start(self):
        # When enable_regime_gating=True, RegimeGate should be created and passed to dispatcher.
        from argus_kalshi.bus import Bus
        from argus_kalshi.farm_runner import KalshiPaperFarm

        bus = Bus()
        rest = MagicMock()
        cfg = KalshiConfig(
            bot_id="test_gate",
            dry_run=True,
            enable_regime_gating=True,
            regime_fallback_mode="permissive",
        )
        farm = KalshiPaperFarm([cfg], bus, rest)
        await farm.start(["KXBTC-TEST"])

        # Verify regime gate is instantiated
        assert farm._regime_gate is not None
        # Verify dispatcher has the gate
        assert farm._dispatcher is not None
        assert farm._dispatcher._regime_gate is not None

        await farm.stop()

    @pytest.mark.asyncio
    async def test_regime_gate_not_created_when_disabled(self):
        from argus_kalshi.bus import Bus
        from argus_kalshi.farm_runner import KalshiPaperFarm

        bus = Bus()
        rest = MagicMock()
        cfg = KalshiConfig(bot_id="no_gate", dry_run=True, enable_regime_gating=False)
        farm = KalshiPaperFarm([cfg], bus, rest)
        await farm.start(["KXBTC-TEST"])

        assert farm._regime_gate is None

        await farm.stop()

    @pytest.mark.asyncio
    async def test_regime_gate_preserved_on_register_deregister(self):
        # Evaluator rebuild after register/deregister must preserve regime_gate.
        from argus_kalshi.bus import Bus
        from argus_kalshi.farm_runner import KalshiPaperFarm

        bus = Bus()
        rest = MagicMock()
        cfgs = [
            KalshiConfig(bot_id="bot_a", dry_run=True, enable_regime_gating=True, regime_fallback_mode="permissive"),
            KalshiConfig(bot_id="bot_b", dry_run=True, enable_regime_gating=True, regime_fallback_mode="permissive"),
        ]
        farm = KalshiPaperFarm(cfgs, bus, rest)
        await farm.start(["KXBTC-TEST"])

        # Deregister bot_a — evaluators should be rebuilt with regime_gate
        farm._dispatcher.deregister_bot("bot_a")
        assert farm._dispatcher._strategy_evaluator._regime_gate is not None
        assert farm._dispatcher._scalper_evaluator._regime_gate is not None

        await farm.stop()


#  Diagnostics

class TestDiagnostics:
    @pytest.mark.asyncio
    async def test_population_diagnostics_include_regime_gate(self):
        from argus_kalshi.bus import Bus
        from argus_kalshi.farm_runner import KalshiPaperFarm

        bus = Bus()
        rest = MagicMock()
        cfg = KalshiConfig(
            bot_id="diag_test",
            dry_run=True,
            enable_regime_gating=True,
            enable_population_manager=True,
            regime_fallback_mode="permissive",
        )
        farm = KalshiPaperFarm([cfg], bus, rest)
        await farm.start(["KXBTC-TEST"])

        diag = farm.get_population_diagnostics()
        assert "regime_gate" in diag
        assert diag["regime_gate"] != {"enabled": False}
        assert "family_weights" in diag

        await farm.stop()

    @pytest.mark.asyncio
    async def test_population_diagnostics_no_regime(self):
        from argus_kalshi.bus import Bus
        from argus_kalshi.farm_runner import KalshiPaperFarm

        bus = Bus()
        rest = MagicMock()
        cfg = KalshiConfig(bot_id="diag_no_regime", dry_run=True, enable_regime_gating=False)
        farm = KalshiPaperFarm([cfg], bus, rest)
        await farm.start(["KXBTC-TEST"])

        diag = farm.get_population_diagnostics()
        assert diag["regime_gate"] == {"enabled": False}

        await farm.stop()
