from __future__ import annotations

from pathlib import Path
import tempfile

from argus_kalshi.farm_grid import MAX_BOT_COUNT, generate_farm_configs

PARAM_KEYS = (
    "min_entry_cents",
    "max_entry_cents",
    "min_edge_threshold",
    "persistence_window_ms",
    "scalp_min_edge_cents",
    "scalp_min_profit_cents",
)

DEFAULT_BASE = {
    "candidate_region_enabled": True,
    "candidate_region_weight": 0.75,
    "candidate_region_explore_floor": 0.25,
    "candidate_region_min_edge_min": 0.09,
    "candidate_region_min_edge_max": 0.11,
    "candidate_region_persistence_min_ms": 90,
    "candidate_region_persistence_max_ms": 360,
    "candidate_region_entry_min_cents_min": 28,
    "candidate_region_entry_min_cents_max": 42,
    "candidate_region_entry_max_cents_min": 70,
    "candidate_region_entry_max_cents_max": 78,
}

WRAP_BASE = {
    "candidate_region_enabled": True,
    "candidate_region_weight": 0.5,
    "candidate_region_explore_floor": 0.5,
    "candidate_region_min_edge_min": 0.08,
    "candidate_region_min_edge_max": 0.08,
    "candidate_region_persistence_min_ms": 120,
    "candidate_region_persistence_max_ms": 120,
    "candidate_region_entry_min_cents_min": 40,
    "candidate_region_entry_min_cents_max": 40,
    "candidate_region_entry_max_cents_min": 70,
    "candidate_region_entry_max_cents_max": 70,
}

WRAP_GRID = [
    {
        "min_entry_cents": 40,
        "max_entry_cents": 70,
        "min_edge_threshold": 0.08,
        "persistence_window_ms": 120,
        "scalp_min_edge_cents": 4,
        "scalp_min_profit_cents": 4,
    },
    {
        "min_entry_cents": 40,
        "max_entry_cents": 70,
        "min_edge_threshold": 0.08,
        "persistence_window_ms": 120,
        "scalp_min_edge_cents": 5,
        "scalp_min_profit_cents": 4,
    },
    {
        "min_entry_cents": 40,
        "max_entry_cents": 70,
        "min_edge_threshold": 0.08,
        "persistence_window_ms": 120,
        "scalp_min_edge_cents": 6,
        "scalp_min_profit_cents": 4,
    },
    {
        "min_entry_cents": 40,
        "max_entry_cents": 70,
        "min_edge_threshold": 0.08,
        "persistence_window_ms": 120,
        "scalp_min_edge_cents": 7,
        "scalp_min_profit_cents": 4,
    },
    {
        "min_entry_cents": 25,
        "max_entry_cents": 65,
        "min_edge_threshold": 0.07,
        "persistence_window_ms": 60,
        "scalp_min_edge_cents": 4,
        "scalp_min_profit_cents": 4,
    },
    {
        "min_entry_cents": 25,
        "max_entry_cents": 70,
        "min_edge_threshold": 0.07,
        "persistence_window_ms": 60,
        "scalp_min_edge_cents": 5,
        "scalp_min_profit_cents": 4,
    },
    {
        "min_entry_cents": 25,
        "max_entry_cents": 75,
        "min_edge_threshold": 0.10,
        "persistence_window_ms": 360,
        "scalp_min_edge_cents": 6,
        "scalp_min_profit_cents": 4,
    },
    {
        "min_entry_cents": 50,
        "max_entry_cents": 78,
        "min_edge_threshold": 0.10,
        "persistence_window_ms": 480,
        "scalp_min_edge_cents": 7,
        "scalp_min_profit_cents": 4,
    },
]


def _param_key(cfg: dict) -> tuple[object, ...]:
    return tuple(cfg[key] for key in PARAM_KEYS)


def _is_candidate(cfg: dict, base: dict) -> bool:
    return (
        float(base["candidate_region_min_edge_min"]) <= float(cfg["min_edge_threshold"]) <= float(base["candidate_region_min_edge_max"])
        and int(base["candidate_region_persistence_min_ms"]) <= int(cfg["persistence_window_ms"]) <= int(base["candidate_region_persistence_max_ms"])
        and int(base["candidate_region_entry_min_cents_min"]) <= int(cfg["min_entry_cents"]) <= int(base["candidate_region_entry_min_cents_max"])
        and int(base["candidate_region_entry_max_cents_min"]) <= int(cfg["max_entry_cents"]) <= int(base["candidate_region_entry_max_cents_max"])
    )


def _candidate_indices(grid: list[dict], base: dict) -> list[int]:
    return [idx for idx, row in enumerate(grid) if _is_candidate(row, base)]


def test_generate_farm_configs_candidate_sampling_has_no_duplicates() -> None:
    bot_ids = [f"bot_{idx:04d}" for idx in range(4000)]
    cfgs = generate_farm_configs(DEFAULT_BASE, bot_ids, seed=7)
    keys = [_param_key(cfg) for cfg in cfgs]

    assert len(keys) == len(set(keys))


def test_generate_farm_configs_candidate_split_matches_fraction() -> None:
    bot_ids = [f"bot_{idx:04d}" for idx in range(1000)]
    cfgs = generate_farm_configs(DEFAULT_BASE, bot_ids, seed=11)
    candidate_fraction = sum(1 for cfg in cfgs if _is_candidate(cfg, DEFAULT_BASE)) / len(cfgs)
    expected = 1.0 - max(
        float(DEFAULT_BASE["candidate_region_explore_floor"]),
        1.0 - float(DEFAULT_BASE["candidate_region_weight"]),
    )

    assert abs(candidate_fraction - expected) <= 0.02


def test_generate_farm_configs_cycle_offset_changes_assignment_order() -> None:
    bot_ids = [f"bot_{idx:04d}" for idx in range(256)]
    cfgs_base = generate_farm_configs(DEFAULT_BASE, bot_ids, seed=23, cycle_offset=0)
    cfgs_offset = generate_farm_configs(DEFAULT_BASE, bot_ids, seed=23, cycle_offset=100)

    assert [_param_key(cfg) for cfg in cfgs_base] != [_param_key(cfg) for cfg in cfgs_offset]


def test_generate_farm_configs_cycle_offset_wraps_after_full_pool_cycle() -> None:
    bot_ids = [f"bot_{idx:04d}" for idx in range(len(WRAP_GRID))]
    candidate_idx = _candidate_indices(WRAP_GRID, WRAP_BASE)
    non_candidate_count = len(WRAP_GRID) - len(candidate_idx)

    assert len(candidate_idx) == non_candidate_count == 4

    cfgs_base = generate_farm_configs(
        WRAP_BASE,
        bot_ids,
        grid_overrides=WRAP_GRID,
        seed=5,
        cycle_offset=0,
        winner_zone_path=None,
    )
    cfgs_wrapped = generate_farm_configs(
        WRAP_BASE,
        bot_ids,
        grid_overrides=WRAP_GRID,
        seed=5,
        cycle_offset=len(candidate_idx),
        winner_zone_path=None,
    )

    assert [_param_key(cfg) for cfg in cfgs_base] == [_param_key(cfg) for cfg in cfgs_wrapped]


def test_max_bot_count_is_10000() -> None:
    assert MAX_BOT_COUNT == 10000


def test_load_farm_configs_passes_explicit_cycle_offset(monkeypatch) -> None:
    from argus_kalshi import farm_runner

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as handle:
        handle.write("Alpha\nBeta\n")
        dwarf_path = Path(handle.name)
    try:
        captured: dict[str, int] = {}

        def fake_generate_farm_configs(
            base: dict,
            bot_ids: list[str],
            grid_overrides: list[dict] | None = None,
            seed: int = 0,
            winner_zone_path: str | None = None,
            cycle_offset: int = 0,
        ) -> list[dict]:
            captured["cycle_offset"] = cycle_offset
            return [{"bot_id": bot_id} for bot_id in bot_ids]

        monkeypatch.setattr(farm_runner, "generate_farm_configs", fake_generate_farm_configs)
        monkeypatch.setattr(farm_runner, "load_config", lambda cfg: cfg)

        raw = {
            "argus_kalshi": {
                "farm": {
                    "base": {"dry_run": True},
                    "dwarf_names_file": str(dwarf_path),
                    "farm_cycle_offset": 17,
                }
            }
        }
        configs = farm_runner.load_farm_configs(raw, settings_path=None)

        assert captured["cycle_offset"] == 17
        assert [cfg["bot_id"] for cfg in configs] == ["Alpha", "Beta"]
    finally:
        dwarf_path.unlink(missing_ok=True)


def test_load_farm_configs_uses_hourly_cycle_offset_fallback(monkeypatch) -> None:
    from argus_kalshi import farm_runner

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as handle:
        handle.write("Alpha\n")
        dwarf_path = Path(handle.name)
    try:
        captured: dict[str, int] = {}

        def fake_generate_farm_configs(
            base: dict,
            bot_ids: list[str],
            grid_overrides: list[dict] | None = None,
            seed: int = 0,
            winner_zone_path: str | None = None,
            cycle_offset: int = 0,
        ) -> list[dict]:
            captured["cycle_offset"] = cycle_offset
            return [{"bot_id": bot_id} for bot_id in bot_ids]

        monkeypatch.setattr(farm_runner, "generate_farm_configs", fake_generate_farm_configs)
        monkeypatch.setattr(farm_runner, "load_config", lambda cfg: cfg)
        monkeypatch.setattr(farm_runner.time, "time", lambda: 7200.0)

        raw = {
            "argus_kalshi": {
                "farm": {
                    "base": {"dry_run": True},
                    "dwarf_names_file": str(dwarf_path),
                    "farm_cycle_offset": 0,
                }
            }
        }
        farm_runner.load_farm_configs(raw, settings_path=None)

        assert captured["cycle_offset"] == 2
    finally:
        dwarf_path.unlink(missing_ok=True)
