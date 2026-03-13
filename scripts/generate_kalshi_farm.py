"""
Generate Kalshi farm config. By default writes the FULL 21k-line YAML (legacy).

For a small config that never drifts, use the compact farm format instead:
  python scripts/generate_kalshi_farm.py --compact
Then use: python -m argus_kalshi --settings config/kalshi_farm_compact.yaml
Params are generated at load time from farm_grid.py (same grid, deterministic).
"""
import argparse
import itertools
import yaml
from pathlib import Path

def generate_grid():
    # Define test ranges so we get exactly 468 configs, each with a UNIQUE
    # strategy fingerprint. Previously we varied (min_e, max_e, edge) × scalper
    # params, giving only 52 unique strategy combos → 9 bots shared the same
    # hold-to-expiry decisions → duplicate PnL on the leaderboard.
    #
    # Now we add strategy-affecting dimensions so every bot has a unique
    # (min_entry, max_entry, min_edge, persistence_window_ms) and optionally
    # scalper params. Target: 468 = number of dwarf names.

    # 1. Main strategy entry filters (valid pairs only: min_e < max_e)
    entry_pairs = [
        (0, 40), (0, 60), (0, 80), (0, 100),
        (20, 60), (20, 80), (20, 100),
        (40, 60), (40, 80), (40, 100),
        (60, 80), (60, 100),
    ]  # 12 pairs

    # 2. Minimum edge threshold (strategy). 3 values so 12*3*13 = 468.
    min_edge_threshold = [0.02, 0.03, 0.04]

    # 3. Persistence window ms (strategy: how long edge must hold before firing)
    persistence_window_ms = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]  # 13

    # 4. Scalper params: shared defaults; stop-loss varied per bot to test bands (0-3c, 4-8c).
    scalp_min_edge_cents = 4
    scalp_min_profit_cents = 2
    # Round-robin so each bot gets a different stop-loss; we can compare PnL by band.
    # 0 = no stop; 1-3 = tight; 4,6,8 = wider. Same 468 configs, no extra bots.
    scalp_stop_loss_values = [0, 1, 2, 3, 4, 6, 8]

    # Load dwarf names (expect 468)
    names_path = Path("argus_kalshi/dwarf_names.txt")
    dwarf_names = []
    if names_path.exists():
        text = names_path.read_text(encoding="utf-8")
        dwarf_names = [n.strip() for n in text.split() if n.strip()]

    # Load base config to preserve truth_feeds, series_filter, etc.
    base_cfg = {}
    base_path = Path("config/config.yaml")
    if base_path.exists():
        with base_path.open() as f:
            full_config = yaml.safe_load(f) or {}
            base_cfg = full_config.get("argus_kalshi", {})
            if isinstance(base_cfg, list):
                base_cfg = base_cfg[0] if base_cfg else {}

    configs = []
    for idx, ((min_e, max_e), edge, persistence_ms) in enumerate(itertools.product(
        entry_pairs,
        min_edge_threshold,
        persistence_window_ms,
    )):
        if idx >= len(dwarf_names):
            break
        bot_id = dwarf_names[idx]

        cfg = dict(base_cfg)
        scalp_stop_loss_cents = scalp_stop_loss_values[idx % len(scalp_stop_loss_values)]
        cfg.update({
            "bot_id": bot_id,
            "min_entry_cents": min_e,
            "max_entry_cents": max_e,
            "min_edge_threshold": edge,
            "persistence_window_ms": persistence_ms,
            "scalp_min_edge_cents": scalp_min_edge_cents,
            "scalp_min_profit_cents": scalp_min_profit_cents,
            "scalp_stop_loss_cents": scalp_stop_loss_cents,
            "sizing_risk_fraction": 0.005,
            "bankroll_usd": 5000.0,
            "dry_run": True,
            "ws_trading_enabled": False,
            "scalper_enabled": True,  # Live service: hold-to-expiry + scalp for every bot
            "use_okx_fallback": True,
            "use_luzia_fallback": False,
        })
        configs.append(cfg)

    if len(configs) != 468:
        print(f"Warning: generated {len(configs)} configs (expected 468). Dwarf names: {len(dwarf_names)}")
    print(f"Generated {len(configs)} isolated bot configurations (each with unique strategy params).")
    
    # Write to a test config file
    out_path = Path("config/kalshi_farm.yaml")
    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("w") as f:
        yaml.dump({"argus_kalshi": configs}, f, default_flow_style=False)
    return configs


def write_compact_config() -> None:
    """Write the short farm YAML that uses farm_grid at load time (no 21k lines)."""
    out = Path("config/kalshi_farm_compact.yaml")
    out.parent.mkdir(exist_ok=True)
    content = """# Compact farm — params generated at load from farm_grid.py (deterministic, no drift).
# Usage: python -m argus_kalshi --settings config/kalshi_farm_compact.yaml --secrets config/secrets.yaml

promoted_bot_id: ""

argus_kalshi:
  farm:
    base_path: "config.yaml"
    dwarf_names_file: "argus_kalshi/dwarf_names.txt"
    seed: 0
"""
    out.write_text(content)
    print(f"Wrote {out}. Use --settings {out} to run the farm with generated params.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Kalshi farm config")
    parser.add_argument("--compact", action="store_true", help="Write compact YAML only (no 21k-line file)")
    args = parser.parse_args()
    if args.compact:
        write_compact_config()
    else:
        generate_grid()
