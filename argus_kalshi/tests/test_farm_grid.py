from __future__ import annotations

from argus_kalshi.farm_grid import generate_farm_configs


def _is_candidate(cfg: dict) -> bool:
    return (
        0.07 <= float(cfg.get("min_edge_threshold", 0.0)) <= 0.09
        and 120 <= int(cfg.get("persistence_window_ms", 0)) <= 300
        and 40 <= int(cfg.get("min_entry_cents", 0)) <= 45
        and 70 <= int(cfg.get("max_entry_cents", 0)) <= 70
    )


def test_candidate_region_bias_keeps_exploration_floor():
    base = {
        "candidate_region_enabled": True,
        "candidate_region_weight": 0.8,
        "candidate_region_explore_floor": 0.2,
        "candidate_region_min_edge_min": 0.07,
        "candidate_region_min_edge_max": 0.09,
        "candidate_region_persistence_min_ms": 120,
        "candidate_region_persistence_max_ms": 300,
        "candidate_region_entry_min_cents_min": 40,
        "candidate_region_entry_min_cents_max": 45,
        "candidate_region_entry_max_cents_min": 70,
        "candidate_region_entry_max_cents_max": 70,
    }
    bot_ids = [f"bot_{i:04d}" for i in range(400)]
    cfgs = generate_farm_configs(base, bot_ids, seed=7)
    candidate_count = sum(1 for c in cfgs if _is_candidate(c))
    frac = candidate_count / max(1, len(cfgs))
    # We bias toward candidate region, but preserve at least 20% exploration.
    # Upper bound relaxed slightly after grid expansion (more candidate rows → more natural overlap).
    assert 0.70 <= frac <= 0.87

