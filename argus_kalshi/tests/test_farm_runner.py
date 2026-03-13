import tempfile
from pathlib import Path

import pytest

from argus_kalshi.bus import Bus
from argus_kalshi.config import KalshiConfig
from argus_kalshi.farm_runner import (
    FarmDispatcher,
    KalshiPaperFarm,
    _crowding_key,
    load_farm_configs,
)
from argus_kalshi.models import TradeSignal
from argus_kalshi.shared_state import SharedFarmState


class DummyRest:
    pass


@pytest.mark.asyncio
async def test_farm_runner_starts_multiple_bots_cleanly():
    bus = Bus()
    rest = DummyRest()
    cfgs = [
        KalshiConfig(bot_id="bot_a", dry_run=True),
        KalshiConfig(bot_id="bot_b", dry_run=True),
    ]
    farm = KalshiPaperFarm(cfgs, bus, rest)

    await farm.start(["KXBTC-TEST"])
    assert len(farm._strategies) == 2
    assert len(farm._scalpers) == 2
    assert len(farm._executions) == 2

    await farm.stop()


@pytest.mark.asyncio
async def test_farm_runner_coerces_missing_and_default_bot_ids():
    bus = Bus()
    rest = DummyRest()
    cfgs = [
        KalshiConfig(bot_id="default", dry_run=True),
        KalshiConfig(bot_id="default", dry_run=True),
        KalshiConfig(bot_id="bot_x", dry_run=True),
    ]
    farm = KalshiPaperFarm(cfgs, bus, rest)

    await farm.start(["KXBTC-TEST"])
    ids = [e._cfg.bot_id for e in farm._executions]

    assert ids[0].startswith("farm_")
    assert ids[1].startswith("farm_")
    assert len(set(ids)) == 3
    assert "default" not in ids

    await farm.stop()


def test_load_farm_configs_compact_farm_generates_deterministic_configs():
    """Compact farm block generates configs from grid; same seed → same params per bot."""
    raw = {
        "argus_kalshi": {
            "farm": {
                "base": {"dry_run": True, "bankroll_usd": 1000.0},
                "dwarf_names_file": "argus_kalshi/dwarf_names.txt",
                "seed": 42,
            }
        }
    }
    configs = load_farm_configs(raw, settings_path=None)
    assert len(configs) >= 1
    # All configs have valid params from the grid
    for c in configs:
        assert c.bot_id
        assert 0 <= c.min_entry_cents <= 100
        assert 0 <= c.max_entry_cents <= 100
        assert c.min_entry_cents <= c.max_entry_cents
        assert c.min_edge_threshold >= 0
        assert c.persistence_window_ms >= 0
    # Determinism: same seed → same configs every run
    configs2 = load_farm_configs(raw, settings_path=None)
    assert len(configs2) == len(configs)
    for a, b in zip(configs, configs2):
        assert a.bot_id == b.bot_id
        assert a.min_entry_cents == b.min_entry_cents
        assert a.persistence_window_ms == b.persistence_window_ms


def test_load_farm_configs_expands_bot_ids_by_reuse_when_fewer_names_than_bot_count():
    """
    Problem: File has 3 names but we want 10 bots. Without expansion we get only 3 configs.
    Test: With dwarf_names_file of 3 names and bot_count=10, we get 10 configs and
    every bot_id is non-empty and one of the 3 names (reused).
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Alice\nBob\nCarol\n")
        dwarf_path = f.name
    try:
        raw = {
            "argus_kalshi": {
                "farm": {
                    "base": {"dry_run": True, "bankroll_usd": 1000.0},
                    "dwarf_names_file": dwarf_path,
                    "bot_count": 10,
                    "seed": 7,
                }
            }
        }
        configs = load_farm_configs(raw, settings_path=None)
        valid_names = {"Alice", "Bob", "Carol"}
        assert len(configs) == 10, "Expected 10 configs when bot_count=10 and names reused"
        for c in configs:
            assert c.bot_id, "No bot_id should be null or empty"
            assert c.bot_id in valid_names, f"bot_id {c.bot_id!r} should be one of {valid_names}"
    finally:
        Path(dwarf_path).unlink(missing_ok=True)


def test_load_farm_configs_no_default_or_empty_bot_id_with_compact_farm():
    """With compact farm and non-empty dwarf file, no config has bot_id 'default' or empty."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Dwarf1\nDwarf2\n")
        dwarf_path = f.name
    try:
        raw = {
            "argus_kalshi": {
                "farm": {
                    "base": {"dry_run": True},
                    "dwarf_names_file": dwarf_path,
                    "seed": 0,
                }
            }
        }
        configs = load_farm_configs(raw, settings_path=None)
        assert len(configs) >= 1
        for c in configs:
            assert c.bot_id and c.bot_id.strip(), "bot_id must be non-empty"
            assert c.bot_id != "default", "bot_id must not be 'default' when using dwarf names"
    finally:
        Path(dwarf_path).unlink(missing_ok=True)


def test_dispatcher_prunes_stale_ticker_tracking():
    bus = Bus()
    shared = SharedFarmState()
    disp = FarmDispatcher(bus, shared, strategies=[], scalpers=[])
    disp._last_ob_mono_by_ticker = {"A": 1.0, "B": 2.0}
    disp._last_prob_mono_by_ticker = {"A": 1.0, "C": 3.0}
    disp._queued_tickers = {"A", "B", "C"}
    disp._dispatching_tickers = {"C"}
    disp._redispatch_tickers = {"B"}

    disp._prune_stale_tickers({"A"})

    assert set(disp._last_ob_mono_by_ticker.keys()) == {"A"}
    assert set(disp._last_prob_mono_by_ticker.keys()) == {"A"}
    assert disp._queued_tickers == {"A"}
    assert disp._dispatching_tickers == set()
    assert disp._redispatch_tickers == set()


def test_crowding_key_isolated_by_bot():
    assert _crowding_key("bot_a", "KXBTC-TEST", "yes") != _crowding_key("bot_b", "KXBTC-TEST", "yes")


def test_crowding_key_preserves_same_bot_same_market_side():
    key = _crowding_key("bot_a", "KXBTC-TEST", "yes")
    assert key == _crowding_key("bot_a", "KXBTC-TEST", "yes")


def test_dispatcher_flow_first_defers_hold_entries_when_fast_entry_exists():
    class _Eval:
        def __init__(self, out):
            self._out = out
            self.counts = {}

        def evaluate(self, *args, **kwargs):
            return list(self._out)

        def _count(self, reason: str, n: int = 1):
            self.counts[reason] = self.counts.get(reason, 0) + n

    bus = Bus()
    shared = SharedFarmState()
    cfg = KalshiConfig(bot_id="cfg", sleeve_mode="flow_first")
    disp = FarmDispatcher(bus, shared, strategies=[], scalpers=[], base_cfg0=cfg)
    disp._prepare_ticker_state = lambda _ticker: object()  # type: ignore[assignment]
    hold_sig = TradeSignal(
        market_ticker="KXBTC-TEST",
        side="yes",
        action="buy",
        limit_price_cents=50,
        quantity_contracts=1,
        edge=0.05,
        p_yes=0.55,
        timestamp=0.0,
        bot_id="hold_bot",
    )
    scalp_sig = TradeSignal(
        market_ticker="KXBTC-TEST",
        side="yes",
        action="buy",
        limit_price_cents=50,
        quantity_contracts=1,
        edge=5.0,
        p_yes=0.55,
        timestamp=0.0,
        bot_id="scalp_bot",
        source="mispricing_scalp",
    )
    disp._strategy_evaluator = _Eval([hold_sig])  # type: ignore[assignment]
    disp._scalper_evaluator = _Eval([scalp_sig])  # type: ignore[assignment]
    disp._arb_evaluator = _Eval([])  # type: ignore[assignment]

    out = disp._dispatch_ticker_sync("KXBTC-TEST")
    assert [s.bot_id for s in out] == ["scalp_bot"]
    assert disp._strategy_evaluator.counts.get("hold_deferred_to_flow", 0) == 1  # type: ignore[attr-defined]


def test_dispatcher_parallel_mode_keeps_hold_and_fast_entries():
    class _Eval:
        def __init__(self, out):
            self._out = out

        def evaluate(self, *args, **kwargs):
            return list(self._out)

    bus = Bus()
    shared = SharedFarmState()
    cfg = KalshiConfig(bot_id="cfg", sleeve_mode="parallel")
    disp = FarmDispatcher(bus, shared, strategies=[], scalpers=[], base_cfg0=cfg)
    disp._prepare_ticker_state = lambda _ticker: object()  # type: ignore[assignment]
    hold_sig = TradeSignal(
        market_ticker="KXBTC-TEST",
        side="yes",
        action="buy",
        limit_price_cents=50,
        quantity_contracts=1,
        edge=0.05,
        p_yes=0.55,
        timestamp=0.0,
        bot_id="hold_bot",
    )
    scalp_sig = TradeSignal(
        market_ticker="KXBTC-TEST",
        side="yes",
        action="buy",
        limit_price_cents=50,
        quantity_contracts=1,
        edge=5.0,
        p_yes=0.55,
        timestamp=0.0,
        bot_id="scalp_bot",
        source="mispricing_scalp",
    )
    disp._strategy_evaluator = _Eval([hold_sig])  # type: ignore[assignment]
    disp._scalper_evaluator = _Eval([scalp_sig])  # type: ignore[assignment]
    disp._arb_evaluator = _Eval([])  # type: ignore[assignment]

    out = disp._dispatch_ticker_sync("KXBTC-TEST")
    assert [s.bot_id for s in out] == ["hold_bot", "scalp_bot"]


def test_dispatcher_flow_first_scalp_defers_arb_same_bot_ticker():
    class _Eval:
        def __init__(self, out):
            self._out = out
            self.counts = {}

        def evaluate(self, *args, **kwargs):
            return list(self._out)

        def _count(self, reason: str, n: int = 1):
            self.counts[reason] = self.counts.get(reason, 0) + n

    bus = Bus()
    shared = SharedFarmState()
    cfg = KalshiConfig(bot_id="cfg", sleeve_mode="flow_first")
    disp = FarmDispatcher(bus, shared, strategies=[], scalpers=[], base_cfg0=cfg)
    disp._prepare_ticker_state = lambda _ticker: object()  # type: ignore[assignment]
    arb_yes = TradeSignal(
        market_ticker="KXBTC-TEST",
        side="yes",
        action="buy",
        limit_price_cents=49,
        quantity_contracts=1,
        edge=0.01,
        p_yes=0.5,
        timestamp=0.0,
        bot_id="bot_x",
        source="pair_arb",
    )
    scalp_buy = TradeSignal(
        market_ticker="KXBTC-TEST",
        side="yes",
        action="buy",
        limit_price_cents=50,
        quantity_contracts=1,
        edge=5.0,
        p_yes=0.55,
        timestamp=0.0,
        bot_id="bot_x",
        source="mispricing_scalp",
    )
    disp._strategy_evaluator = _Eval([])  # type: ignore[assignment]
    disp._scalper_evaluator = _Eval([scalp_buy])  # type: ignore[assignment]
    disp._arb_evaluator = _Eval([arb_yes])  # type: ignore[assignment]

    out = disp._dispatch_ticker_sync("KXBTC-TEST")
    assert [(s.bot_id, s.source) for s in out] == [("bot_x", "mispricing_scalp")]
    assert disp._arb_evaluator.counts.get("arb_deferred_to_scalp", 0) == 1  # type: ignore[attr-defined]
