# Created by Oliver Meihls

# Deterministic farm config generator — no 21k-line YAML.
#
# Each bot is assigned parameters ONCE from a fixed grid (or seeded ranges).
# Same (grid, seed, bot_count) → same configs every run. No drift unless you
# change the grid/seed or reset the log/DB.
#
# Max bot count is 10000 (see MAX_BOT_COUNT). Grid has 37,120 unique rows.

from __future__ import annotations

import itertools
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

# Cap total farm size so we don't overload the event loop / memory.
MAX_BOT_COUNT = 10000

# Grid dimensions: product gives 37,120 unique param sets, enough for 10,000 bots.
# Order is fixed so index i always gets the same params (no drift).
DEFAULT_ENTRY_PAIRS = [
    # Original sweep — keep full coverage
    (25, 65), (25, 70), (25, 75),
    (30, 65), (30, 70), (30, 75),
    (35, 60), (35, 70), (35, 75),
    (40, 70), (40, 75), (42, 72),
    (45, 70), (45, 75), (45, 78),
    (50, 70), (50, 75), (50, 78),
    # Finer exploration around winners: hold top bots clustered at 30-75¢, scalp at 40-72¢
    (28, 72), (28, 75),
    (30, 72), (32, 70), (32, 75),
    (35, 72), (38, 70), (38, 75),
    (40, 72), (42, 70), (42, 75),
]
DEFAULT_MIN_EDGE_THRESHOLD = [0.07, 0.08, 0.09, 0.10]
DEFAULT_PERSISTENCE_MS = [60, 90, 110, 120, 140, 150, 160, 180, 210, 240, 270, 300, 330, 360, 420, 480]
# Scalp params varied per bot so we can compare strategies.
# Winners: scalp_min_edge=6, scalp_min_profit=3. Added 7 between 6 and 8.
DEFAULT_SCALP_MIN_EDGE_CENTS = [4, 5, 6, 7, 8]
DEFAULT_SCALP_MIN_PROFIT_CENTS = [4, 5, 6, 7]
DEFAULT_SCALP_STOP_LOSS_CENTS = [0]
# Sizing policy: half-Kelly proxy for farm-generated configs.
# Keep this conservative by default; do not use full-Kelly in automated runs.
DEFAULT_SIZING_RISK_FRACTION = 0.005


def _default_grid_product() -> List[Dict[str, Any]]:
    # Product of all dimensions; each row is a unique param set for one bot (>= MAX_BOT_COUNT rows).
    rows: List[Dict[str, Any]] = []
    for (
        (min_e, max_e),
        edge,
        persistence_ms,
        scalp_edge,
        scalp_profit,
    ) in itertools.product(
        DEFAULT_ENTRY_PAIRS,
        DEFAULT_MIN_EDGE_THRESHOLD,
        DEFAULT_PERSISTENCE_MS,
        DEFAULT_SCALP_MIN_EDGE_CENTS,
        DEFAULT_SCALP_MIN_PROFIT_CENTS,
    ):
        # Grid rows set ONLY the parameters being swept.
        # Everything else (bankroll, fees, scalp entry bounds, etc.) lives in
        # the YAML base config and is NOT overridden here.
        rows.append({
            "min_entry_cents": min_e,
            "max_entry_cents": max_e,
            "min_edge_threshold": edge,
            "persistence_window_ms": persistence_ms,
            "scalp_min_edge_cents": scalp_edge,
            "scalp_min_profit_cents": scalp_profit,
        })
    return rows


def load_dwarf_names(path: str) -> List[str]:
    # Load bot IDs from dwarf names file (one name per line).
    p = Path(path)
    if not p.exists():
        return []
    text = p.read_text(encoding="utf-8")
    return [n.strip() for n in text.split() if n.strip()]


def load_winner_zone(path: str) -> Optional[Dict[str, Any]]:
    # Load winner-zone parameter bounds from a kalshi_bot_performance.json file.
    #
    # Returns a base-config overlay dict with candidate_region_* keys set to the
    # winner zone bounds, or None if the file is missing/malformed.
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None

    wz = data.get("winner_zone")
    if not wz:
        return None

    # Extract parameter bounds with tolerances
    def _get(key: str, default_lo: float, default_hi: float) -> tuple:
        entry = wz.get(key)
        if not entry:
            return default_lo, default_hi
        return float(entry["min"]), float(entry["max"])

    entry_min_lo, entry_min_hi = _get("min_entry_cents", 25, 50)
    entry_max_lo, entry_max_hi = _get("max_entry_cents", 60, 78)
    edge_min, edge_max = _get("min_edge_threshold", 0.07, 0.10)
    persist_min, persist_max = _get("persistence_window_ms", 60, 480)

    return {
        "candidate_region_enabled": True,
        "candidate_region_min_edge_min": edge_min,
        "candidate_region_min_edge_max": edge_max,
        "candidate_region_persistence_min_ms": int(persist_min),
        "candidate_region_persistence_max_ms": int(persist_max),
        "candidate_region_entry_min_cents_min": int(entry_min_lo),
        "candidate_region_entry_min_cents_max": int(entry_min_hi),
        "candidate_region_entry_max_cents_min": int(entry_max_lo),
        "candidate_region_entry_max_cents_max": int(entry_max_hi),
        # 70% of bots drawn from winner zone, 30% explore rest of grid
        "candidate_region_weight": 0.70,
        "candidate_region_explore_floor": 0.30,
    }


def generate_farm_configs(
    base: Dict[str, Any],
    bot_ids: List[str],
    grid_overrides: Optional[List[Dict[str, Any]]] = None,
    seed: int = 0,
    winner_zone_path: Optional[str] = None,
    cycle_offset: int = 0,
) -> List[Dict[str, Any]]:
    # Produce one config dict per bot. Params assigned once and deterministic.
    #
    # - base: shared defaults (e.g. from config.yaml argus_kalshi block).
    # - bot_ids: list of bot_id strings (e.g. dwarf names). Length = number of configs.
    # - grid_overrides: if None, use default grid. Else use this list;
    # len(grid_overrides) must be >= len(bot_ids) (we take first len(bot_ids)).
    # - seed: reserved for future range-based sampling; currently unused (grid is fixed).
    # - winner_zone_path: optional path to kalshi_bot_performance.json. If provided and
    # the file exists, the winner zone overrides candidate_region_* keys in base so
    # subsequent bots are biased toward the previous run's top-performer parameters.
    # - cycle_offset: when candidate-region sampling is enabled, rotate each shuffled
    # pool before assignment so repeated runs start from different positions.
    #
    # Returns list of dicts suitable for load_config(); each has bot_id set.
    # Candidate-region sampling uses deterministic pre-split pool sizes and cycles
    # through each shuffled pool without replacement, only repeating after a full
    # pool exhaustion. Same (base, bot_ids, grid_overrides, seed, winner_zone_path,
    # cycle_offset) → same list every time.
    if not bot_ids:
        return []

    # Apply winner zone as candidate region if available
    if winner_zone_path:
        wz_overlay = load_winner_zone(winner_zone_path)
        if wz_overlay:
            base = {**base, **wz_overlay}

    grid = grid_overrides if grid_overrides is not None else _default_grid_product()
    n = min(len(bot_ids), len(grid))
    rng = random.Random(seed)
    all_indices = list(range(len(grid)))
    param_indices: List[int]

    candidate_enabled = bool(base.get("candidate_region_enabled", False))
    if candidate_enabled:
        edge_min = float(base.get("candidate_region_min_edge_min", 0.07))
        edge_max = float(base.get("candidate_region_min_edge_max", 0.09))
        persist_min = int(base.get("candidate_region_persistence_min_ms", 120))
        persist_max = int(base.get("candidate_region_persistence_max_ms", 300))
        entry_min_lo = int(base.get("candidate_region_entry_min_cents_min", 40))
        entry_min_hi = int(base.get("candidate_region_entry_min_cents_max", 45))
        entry_max_lo = int(base.get("candidate_region_entry_max_cents_min", 70))
        entry_max_hi = int(base.get("candidate_region_entry_max_cents_max", 70))
        target_weight = max(0.0, min(1.0, float(base.get("candidate_region_weight", 0.8))))
        explore_floor = max(0.0, min(1.0, float(base.get("candidate_region_explore_floor", 0.2))))

        def _in_candidate(row: Dict[str, Any]) -> bool:
            edge = float(row.get("min_edge_threshold", 0.0))
            persist = int(row.get("persistence_window_ms", 0))
            e_min = int(row.get("min_entry_cents", 0))
            e_max = int(row.get("max_entry_cents", 100))
            return (
                edge_min <= edge <= edge_max
                and persist_min <= persist <= persist_max
                and entry_min_lo <= e_min <= entry_min_hi
                and entry_max_lo <= e_max <= entry_max_hi
            )

        candidate_idx = [i for i in all_indices if _in_candidate(grid[i])]
        candidate_set = set(candidate_idx)
        non_candidate_idx = [i for i in all_indices if i not in candidate_set]
        rng.shuffle(candidate_idx)
        rng.shuffle(non_candidate_idx)
        if cycle_offset > 0:
            offset_c = cycle_offset % len(candidate_idx) if candidate_idx else 0
            offset_e = cycle_offset % len(non_candidate_idx) if non_candidate_idx else 0
            candidate_idx = candidate_idx[offset_c:] + candidate_idx[:offset_c]
            non_candidate_idx = non_candidate_idx[offset_e:] + non_candidate_idx[:offset_e]

        non_candidate_fraction = max(explore_floor, 1.0 - target_weight)
        candidate_fraction = max(0.0, min(1.0, 1.0 - non_candidate_fraction))
        if not candidate_idx:
            candidate_fraction = 0.0
        if not non_candidate_idx:
            candidate_fraction = 1.0
        n_candidate = round(n * candidate_fraction)
        n_explore = n - n_candidate
        cand_iter = itertools.cycle(candidate_idx) if candidate_idx else iter(())
        expl_iter = itertools.cycle(non_candidate_idx) if non_candidate_idx else itertools.cycle(candidate_idx)
        cand_draws = [next(cand_iter) for _ in range(n_candidate)]
        expl_draws = [next(expl_iter) for _ in range(n_explore)]
        param_indices = cand_draws + expl_draws
        rng.shuffle(param_indices)
    else:
        param_indices = all_indices[:n]
        rng.shuffle(param_indices)
    # Shuffle which grid row each config gets, so name order (often alphabetical) is not
    # tied to param order. Same seed → same mapping; different names in top 20.
    # param_indices already prepared above (candidate-biased if enabled).

    configs: List[Dict[str, Any]] = []
    for i in range(n):
        cfg = dict(base)
        cfg.update(grid[param_indices[i]])
        cfg["bot_id"] = bot_ids[i]
        configs.append(cfg)
    return configs
