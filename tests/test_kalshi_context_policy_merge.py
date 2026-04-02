# Created by Oliver Meihls

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
import scripts.kalshi_apply_promotion as promo  # noqa: E402


def test_merge_policy_keys_accumulates_and_preserves_prior_key():
    prior_keys = {
        "overlap": {
            "lane": promo.LANE_UNKNOWN,
            "weight": promo.WEIGHT_UNKNOWN,
            "count": 50,
            "total_pnl": 25.0,
            "avg_pnl": 0.5,
            "win_rate": 0.6,
        },
        "prior_only": {
            "lane": promo.LANE_EXPLORE,
            "weight": 0.5,
            "count": 50,
            "total_pnl": -50.0,
            "avg_pnl": -1.0,
            "win_rate": 0.4,
        },
    }
    current_keys = {
        "overlap": {
            "lane": promo.LANE_CORE,
            "weight": 1.3,
            "count": 30,
            "total_pnl": 45.0,
            "avg_pnl": 1.5,
            "win_rate": 0.7,
        },
        "new_only": {
            "lane": promo.LANE_UNKNOWN,
            "weight": promo.WEIGHT_UNKNOWN,
            "count": 20,
            "total_pnl": 4.0,
            "avg_pnl": 0.2,
            "win_rate": 0.5,
        },
    }

    merged = promo._merge_policy_keys(
        current_keys=current_keys,
        prior_keys=prior_keys,
        min_samples=50,
        promote_threshold=0.5,
        demote_threshold=-0.5,
    )

    assert merged["overlap"]["count"] == 80
    assert merged["overlap"]["total_pnl"] == 70.0
    assert merged["overlap"]["avg_pnl"] == 0.875
    assert merged["overlap"]["win_rate"] == 0.6375
    assert merged["overlap"]["lane"] == promo.LANE_CORE

    assert merged["prior_only"] == prior_keys["prior_only"]
    assert merged["new_only"] == current_keys["new_only"]
