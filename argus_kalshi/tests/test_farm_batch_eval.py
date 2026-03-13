from __future__ import annotations

import time

from argus_kalshi.bus import Bus
from argus_kalshi.config import KalshiConfig
from argus_kalshi.farm_runner import (
    _ParamRegionPenaltyEngine,
    _PreparedTickerState,
    _ScalperBatchEvaluator,
    _StrategyBatchEvaluator,
)
from argus_kalshi.kalshi_strategy import StrategyEngine
from argus_kalshi.mispricing_scalper import MispricingScalper, _ScalpPosition
from argus_kalshi.models import FairProbability, OrderbookState
from argus_kalshi.shared_state import SharedFarmState


def _cfg(bot_id: str, **overrides: object) -> KalshiConfig:
    base = dict(
        bot_id=bot_id,
        kalshi_key_id="test",
        kalshi_private_key_path="/dev/null",
        dry_run=True,
        ws_trading_enabled=False,
        enable_clock_offset_calibration=False,
        truth_feed_stale_timeout_s=30.0,
        effective_edge_fee_pct=0.0,
        persistence_window_ms=0,
        near_expiry_persistence_ms=0,
        min_edge_threshold=0.02,
        near_expiry_min_edge=0.01,
        near_expiry_minutes=5,
        min_entry_cents=0,
        max_entry_cents=100,
        scalp_min_edge_cents=4,
        scalp_min_profit_cents=8,
        scalp_stop_loss_cents=0,
        scalp_max_spread_cents=10,
        scalp_max_entry_cents=90,
        scalp_min_entry_cents=10,
        scalp_max_quantity=40,
        risk_fraction_per_trade=0.01,
        scalp_min_reprice_move_cents=4,
        scalp_reprice_window_s=3.0,
        scalp_entry_cost_buffer_cents=0,
        scalp_exit_grace_s=3.0,
        scalp_exit_edge_threshold_cents=1,
        scalp_directional_score_threshold=0.3,
        scalp_directional_drift_weight=0.30,
        scalp_directional_drift_scale=0.0002,
        scalp_directional_obi_weight=0.15,
        scalp_directional_flow_weight=0.10,
        scalp_directional_depth_weight=0.25,
        scalp_directional_delta_yes_weight=0.10,
        scalp_directional_delta_no_weight=0.10,
    )
    base.update(overrides)
    return KalshiConfig(**base)


def _shared_state() -> SharedFarmState:
    shared = SharedFarmState()
    ticker = "KXBTC-TEST"
    shared.fair_probs[ticker] = FairProbability(market_ticker=ticker, p_yes=0.70, drift=0.0005)
    shared.prev_fair_prob_by_ticker[ticker] = 0.56
    shared.prev_fair_prob_ts_by_ticker[ticker] = time.monotonic() - 1.0
    shared.last_fair_prob_ts_by_ticker[ticker] = time.monotonic()
    shared.orderbooks[ticker] = OrderbookState(
        market_ticker=ticker,
        best_yes_bid_cents=59,
        best_no_bid_cents=39,
        implied_yes_ask_cents=60,
        implied_no_ask_cents=40,
        seq=1,
        valid=True,
        obi=0.5,
        depth_pressure=0.6,
    )
    shared.market_asset[ticker] = "BTC"
    shared.market_is_range[ticker] = False
    shared.market_window_min[ticker] = 15
    shared.market_settlement[ticker] = time.time() + 60
    shared.scalp_eligible[ticker] = True
    shared.scalp_settlement_epoch[ticker] = time.time() + 60
    shared.trade_flow_by_ticker[ticker] = 0.6
    shared.orderbook_delta_flow_yes[ticker] = 0.6
    shared.orderbook_delta_flow_no[ticker] = -0.6
    shared.last_truth_tick_by_asset["BTC"] = time.monotonic()
    shared.truth_stale = False
    return shared


def _prepared(shared: SharedFarmState, ticker: str = "KXBTC-TEST") -> _PreparedTickerState:
    ob = shared.orderbooks[ticker]
    p_yes = shared.fair_probs[ticker].p_yes
    now_wall = time.time()
    now_mono = time.monotonic()
    return _PreparedTickerState(
        ticker=ticker,
        now_wall=now_wall,
        now_mono=now_mono,
        p_yes=p_yes,
        yes_ask_cents=ob.implied_yes_ask_cents,
        no_ask_cents=ob.implied_no_ask_cents,
        yes_bid_cents=ob.best_yes_bid_cents,
        no_bid_cents=ob.best_no_bid_cents,
        yes_bid_depth_centicx=ob.best_yes_depth,
        no_bid_depth_centicx=ob.best_no_depth,
        ev_yes=p_yes - (ob.implied_yes_ask_cents / 100.0),
        ev_no=(1.0 - p_yes) - (ob.implied_no_ask_cents / 100.0),
        asset="BTC",
        last_asset_tick=shared.last_truth_tick_by_asset["BTC"],
        time_to_settle_s=shared.market_settlement[ticker] - now_wall,
        window_minutes=shared.market_window_min[ticker],
        is_range=False,
        scalp_eligible=True,
        prev_p_yes=shared.prev_fair_prob_by_ticker[ticker],
        prev_prob_ts=shared.prev_fair_prob_ts_by_ticker[ticker],
        last_prob_ts=shared.last_fair_prob_ts_by_ticker[ticker],
        momentum_drift=float(getattr(shared.fair_probs[ticker], "drift", 0.0)),
        trade_flow=float(shared.trade_flow_by_ticker.get(ticker, 0.0)),
        obi=float(getattr(ob, "obi", 0.0)),
        depth_pressure=float(getattr(ob, "depth_pressure", 0.0)),
        delta_flow_yes=float(shared.orderbook_delta_flow_yes.get(ticker, 0.0)),
        delta_flow_no=float(shared.orderbook_delta_flow_no.get(ticker, 0.0)),
    )


def test_strategy_batch_evaluator_matches_individual_evaluation():
    shared_a = _shared_state()
    shared_b = _shared_state()
    bus = Bus()
    cfgs = [
        _cfg("a"),
        _cfg("b"),
        _cfg("c", min_edge_threshold=0.03),
        _cfg("d", persistence_window_ms=30),
    ]
    strategies_a = [StrategyEngine(cfg, bus, shared=shared_a) for cfg in cfgs]
    strategies_b = [StrategyEngine(cfg, bus, shared=shared_b) for cfg in cfgs]

    prepared = _prepared(shared_a)
    evaluator = _StrategyBatchEvaluator(strategies_a)
    got = sorted(
        evaluator.evaluate(prepared, truth_stale=False),
        key=lambda s: (s.bot_id, s.action, s.side),
    )

    expected = sorted(
        [
            signal
            for strategy in strategies_b
            if (signal := strategy.evaluate_sync("KXBTC-TEST", truth_stale=False)) is not None
        ],
        key=lambda s: (s.bot_id, s.action, s.side),
    )

    assert [(s.bot_id, s.action, s.side, s.limit_price_cents, s.quantity_contracts) for s in got] == [
        (s.bot_id, s.action, s.side, s.limit_price_cents, s.quantity_contracts) for s in expected
    ]


def test_scalper_batch_evaluator_matches_individual_evaluation():
    shared_a = _shared_state()
    shared_b = _shared_state()
    bus = Bus()
    cfgs = [
        _cfg("s1"),
        _cfg("s2"),
        _cfg("s3", scalp_min_edge_cents=6),
        _cfg("s4", scalp_min_profit_cents=3),
    ]
    scalpers_a = [MispricingScalper(cfg, bus, shared=shared_a) for cfg in cfgs]
    scalpers_b = [MispricingScalper(cfg, bus, shared=shared_b) for cfg in cfgs]

    for scalper in (scalpers_a[0], scalpers_b[0]):
        scalper._open_positions["KXBTC-TEST"] = _ScalpPosition(
            ticker="KXBTC-TEST",
            side="yes",
            entry_price_cents=55,
            quantity_centicx=100,
            opened_at=time.time() - 60,
        )

    prepared = _prepared(shared_a)
    evaluator = _ScalperBatchEvaluator(scalpers_a)
    got = sorted(
        evaluator.evaluate(prepared),
        key=lambda s: (s.bot_id, s.action, s.side),
    )

    expected_signals = []
    for scalper in scalpers_b:
        exit_sig = scalper.exit_sync("KXBTC-TEST")
        if exit_sig is not None:
            expected_signals.append(exit_sig)
        buy_sig = scalper.scalp_sync("KXBTC-TEST")
        if buy_sig is not None:
            expected_signals.append(buy_sig)
    expected = sorted(expected_signals, key=lambda s: (s.bot_id, s.action, s.side))

    assert [(s.bot_id, s.action, s.side, s.limit_price_cents, s.quantity_contracts) for s in got] == [
        (s.bot_id, s.action, s.side, s.limit_price_cents, s.quantity_contracts) for s in expected
    ]


def test_scalper_exit_grace_path_uses_defined_slippage_buffer():
    shared = _shared_state()
    bus = Bus()
    cfg = _cfg(
        "grace",
        scalp_min_profit_cents=5,
        scalp_exit_edge_threshold_cents=20,
        scalp_exit_grace_s=1.0,
    )
    scalper = MispricingScalper(cfg, bus, shared=shared)
    scalper._open_positions["KXBTC-TEST"] = _ScalpPosition(
        ticker="KXBTC-TEST",
        side="yes",
        entry_price_cents=58,
        quantity_centicx=100,
        opened_at=time.time() - 10,
    )
    evaluator = _ScalperBatchEvaluator([scalper])
    prepared = _prepared(shared)

    signals = evaluator.evaluate(prepared)
    assert any(s.action == "sell" and s.source == "mispricing_scalp" for s in signals)


def test_strategy_15m_hold_entries_are_limited_to_last_three_minutes():
    shared = _shared_state()
    bus = Bus()
    cfg = _cfg("late", max_entry_minutes_to_expiry=20)
    strategy = StrategyEngine(cfg, bus, shared=shared)

    shared.market_settlement["KXBTC-TEST"] = time.time() + (5 * 60)
    assert strategy.evaluate_sync("KXBTC-TEST", truth_stale=False) is None

    shared.market_settlement["KXBTC-TEST"] = time.time() + 150
    signal = strategy.evaluate_sync("KXBTC-TEST", truth_stale=False)
    assert signal is not None


def test_strategy_blocks_expensive_no_tail_entries():
    shared = _shared_state()
    bus = Bus()
    shared.fair_probs["KXBTC-TEST"] = FairProbability(market_ticker="KXBTC-TEST", p_yes=0.02)
    shared.orderbooks["KXBTC-TEST"] = OrderbookState(
        market_ticker="KXBTC-TEST",
        best_yes_bid_cents=1,
        best_no_bid_cents=9,
        implied_yes_ask_cents=91,
        implied_no_ask_cents=99,
        seq=1,
        valid=True,
    )
    cfg = _cfg("tail", min_entry_cents=0, max_entry_cents=100, no_avoid_above_cents=80)
    strategy = StrategyEngine(cfg, bus, shared=shared)
    assert strategy.evaluate_sync("KXBTC-TEST", truth_stale=False) is None


def test_strategy_batch_requires_net_edge_after_hold_friction():
    shared = _shared_state()
    bus = Bus()
    # Very small edge: p_yes=61%, ask_yes=60c -> raw edge ~1c.
    shared.fair_probs["KXBTC-TEST"] = FairProbability(market_ticker="KXBTC-TEST", p_yes=0.61)
    shared.orderbooks["KXBTC-TEST"] = OrderbookState(
        market_ticker="KXBTC-TEST",
        best_yes_bid_cents=59,
        best_no_bid_cents=39,
        implied_yes_ask_cents=60,
        implied_no_ask_cents=40,
        seq=1,
        valid=True,
    )
    cfg = _cfg(
        "hold_friction",
        min_edge_threshold=0.005,
        hold_min_net_edge_cents=2,
        hold_entry_cost_buffer_cents=1,
        paper_slippage_cents=1,
    )
    strategy = StrategyEngine(cfg, bus, shared=shared)
    evaluator = _StrategyBatchEvaluator([strategy], shared=shared)
    prepared = _prepared(shared)
    signals = [s for s in evaluator.evaluate(prepared, truth_stale=False) if s.action == "buy"]
    assert not signals


def test_scalper_batch_requires_directional_signal():
    shared = _shared_state()
    bus = Bus()
    cfg = _cfg("fast", scalp_min_profit_cents=8, scalp_entry_cost_buffer_cents=0)
    scalper = MispricingScalper(cfg, bus, shared=shared)
    evaluator = _ScalperBatchEvaluator([scalper])

    prepared = _prepared(shared)
    signals = evaluator.evaluate(prepared)
    assert any(signal.action == "buy" for signal in signals)

    shared.fair_probs["KXBTC-TEST"] = FairProbability(market_ticker="KXBTC-TEST", p_yes=0.50, drift=0.0)
    shared.trade_flow_by_ticker["KXBTC-TEST"] = 0.0
    shared.orderbook_delta_flow_yes["KXBTC-TEST"] = 0.0
    shared.orderbook_delta_flow_no["KXBTC-TEST"] = 0.0
    shared.orderbooks["KXBTC-TEST"] = OrderbookState(
        market_ticker="KXBTC-TEST",
        best_yes_bid_cents=59,
        best_no_bid_cents=39,
        implied_yes_ask_cents=60,
        implied_no_ask_cents=40,
        seq=1,
        valid=True,
        obi=0.0,
        depth_pressure=0.0,
    )
    prepared = _prepared(shared)
    signals = evaluator.evaluate(prepared)
    assert not any(signal.action == "buy" for signal in signals)


def test_scalper_batch_requires_net_profit_after_fees():
    shared = _shared_state()
    bus = Bus()
    shared.fair_probs["KXBTC-TEST"] = FairProbability(market_ticker="KXBTC-TEST", p_yes=0.64)
    shared.prev_fair_prob_by_ticker["KXBTC-TEST"] = 0.58
    shared.prev_fair_prob_ts_by_ticker["KXBTC-TEST"] = time.monotonic() - 1.0
    shared.last_fair_prob_ts_by_ticker["KXBTC-TEST"] = time.monotonic()
    shared.orderbooks["KXBTC-TEST"] = OrderbookState(
        market_ticker="KXBTC-TEST",
        best_yes_bid_cents=59,
        best_no_bid_cents=39,
        implied_yes_ask_cents=60,
        implied_no_ask_cents=40,
        seq=1,
        valid=True,
    )
    cfg = _cfg(
        "fees",
        scalp_min_edge_cents=4,
        scalp_min_profit_cents=2,
        scalp_entry_cost_buffer_cents=0,
    )
    scalper = MispricingScalper(cfg, bus, shared=shared)
    evaluator = _ScalperBatchEvaluator([scalper])

    prepared = _prepared(shared)
    signals = evaluator.evaluate(prepared)
    assert not any(signal.action == "buy" for signal in signals)


def test_scalper_directional_side_follows_score_sign():
    shared = _shared_state()
    bus = Bus()
    cfg = _cfg(
        "dir_side",
        scalp_min_profit_cents=8,
        scalp_entry_cost_buffer_cents=0,
        scalp_cooldown_s=0.0,
    )
    scalper = MispricingScalper(cfg, bus, shared=shared)
    evaluator = _ScalperBatchEvaluator([scalper])
    prepared = _prepared(shared)

    signals = evaluator.evaluate(prepared)
    assert any(s.action == "buy" and s.side == "yes" and s.source == "mispricing_scalp" for s in signals)

    shared.fair_probs["KXBTC-TEST"] = FairProbability(market_ticker="KXBTC-TEST", p_yes=0.50, drift=-0.0006)
    shared.trade_flow_by_ticker["KXBTC-TEST"] = -0.8
    shared.orderbook_delta_flow_yes["KXBTC-TEST"] = -0.4
    shared.orderbook_delta_flow_no["KXBTC-TEST"] = 0.7
    shared.orderbooks["KXBTC-TEST"] = OrderbookState(
        market_ticker="KXBTC-TEST",
        best_yes_bid_cents=59,
        best_no_bid_cents=39,
        implied_yes_ask_cents=60,
        implied_no_ask_cents=40,
        seq=1,
        valid=True,
        obi=-0.6,
        depth_pressure=-0.6,
    )
    prepared = _prepared(shared)
    signals = evaluator.evaluate(prepared)
    assert any(s.action == "buy" and s.side == "no" for s in signals)


def test_side_guard_blocks_hold_entry():
    shared = _shared_state()
    bus = Bus()
    cfg = _cfg("guard_hold")
    strategy = StrategyEngine(cfg, bus, shared=shared)
    evaluator = _StrategyBatchEvaluator([strategy], shared=shared)

    shared.side_guard_block_until["KXBTC-TEST|yes"] = time.time() + 60.0
    prepared = _prepared(shared)
    signals = evaluator.evaluate(prepared, truth_stale=False)
    assert not any(s.action == "buy" and s.side == "yes" for s in signals)


def test_side_guard_blocks_scalp_entry():
    shared = _shared_state()
    bus = Bus()
    cfg = _cfg("guard_scalp", scalp_min_edge_cents=4, scalp_min_profit_cents=1)
    scalper = MispricingScalper(cfg, bus, shared=shared)
    evaluator = _ScalperBatchEvaluator([scalper], shared=shared)

    shared.side_guard_block_until["KXBTC-TEST|yes"] = time.time() + 60.0
    prepared = _prepared(shared)
    signals = evaluator.evaluate(prepared)
    assert not any(s.action == "buy" and s.side == "yes" for s in signals)


def test_strategy_market_side_cap_scales_then_blocks():
    shared = _shared_state()
    bus = Bus()
    cfg1 = _cfg(
        "cap_s1",
        enable_market_side_caps=True,
        market_side_cap_contracts=1,
        market_side_cap_enforcement_mode="scale",
    )
    cfg2 = _cfg(
        "cap_s2",
        enable_market_side_caps=True,
        market_side_cap_contracts=1,
        market_side_cap_enforcement_mode="scale",
    )
    strategies = [StrategyEngine(cfg1, bus, shared=shared), StrategyEngine(cfg2, bus, shared=shared)]
    evaluator = _StrategyBatchEvaluator(strategies, shared=shared)
    prepared = _prepared(shared)
    signals = [s for s in evaluator.evaluate(prepared, truth_stale=False) if s.action == "buy"]
    assert sum(s.quantity_contracts for s in signals) <= 1
    assert len(signals) == 1


def test_scalper_market_side_cap_scales_then_blocks():
    shared = _shared_state()
    bus = Bus()
    cfg1 = _cfg(
        "cap_sc1",
        enable_market_side_caps=True,
        market_side_cap_contracts=1,
        market_side_cap_enforcement_mode="scale",
        scalp_min_edge_cents=4,
        scalp_min_profit_cents=8,
    )
    cfg2 = _cfg(
        "cap_sc2",
        enable_market_side_caps=True,
        market_side_cap_contracts=1,
        market_side_cap_enforcement_mode="scale",
        scalp_min_edge_cents=4,
        scalp_min_profit_cents=8,
    )
    scalpers = [MispricingScalper(cfg1, bus, shared=shared), MispricingScalper(cfg2, bus, shared=shared)]
    evaluator = _ScalperBatchEvaluator(scalpers, shared=shared)
    prepared = _prepared(shared)
    signals = [s for s in evaluator.evaluate(prepared) if s.action == "buy"]
    assert sum(s.quantity_contracts for s in signals) <= 1
    assert len(signals) == 1


def test_param_region_downweights_when_enabled():
    shared = _shared_state()
    bus = Bus()
    shared.orderbooks["KXBTC-TEST"] = OrderbookState(
        market_ticker="KXBTC-TEST",
        best_yes_bid_cents=49,
        best_no_bid_cents=49,
        implied_yes_ask_cents=50,
        implied_no_ask_cents=50,
        seq=1,
        valid=True,
    )
    cfg = _cfg(
        "region_downweight",
        enable_param_region_penalties=True,
        param_region_window_settles=10,
        param_region_min_samples=2,
        param_region_loss_threshold_usd=-1.0,
        param_region_penalty_factor=0.5,
        min_edge_threshold=0.04,
        max_entry_cents=50,
        risk_fraction_per_trade=0.02,
    )
    strategy_a = StrategyEngine(cfg, bus, shared=shared)
    strategy_b = StrategyEngine(cfg, bus, shared=shared)
    engine = _ParamRegionPenaltyEngine(cfg)
    engine.record_settlement(cfg, "BTC 15m", -1.0)
    engine.record_settlement(cfg, "BTC 15m", -1.0)

    baseline_eval = _StrategyBatchEvaluator([strategy_a], shared=shared)
    penalized_eval = _StrategyBatchEvaluator([strategy_b], shared=shared, param_region_engine=engine)
    prepared = _prepared(shared)
    baseline = [s for s in baseline_eval.evaluate(prepared, truth_stale=False) if s.action == "buy"]
    penalized = [s for s in penalized_eval.evaluate(prepared, truth_stale=False) if s.action == "buy"]
    assert baseline and penalized
    assert penalized[0].quantity_contracts < baseline[0].quantity_contracts


def test_param_region_cooldown_blocks_when_enabled():
    shared = _shared_state()
    bus = Bus()
    cfg = _cfg(
        "region_block",
        enable_param_region_penalties=True,
        param_region_mode="cooldown_block",
        param_region_window_settles=10,
        param_region_min_samples=2,
        param_region_loss_threshold_usd=-1.0,
        param_region_block_minutes=5.0,
        min_edge_threshold=0.04,
        max_entry_cents=50,
        risk_fraction_per_trade=0.02,
    )
    strategy = StrategyEngine(cfg, bus, shared=shared)
    engine = _ParamRegionPenaltyEngine(cfg)
    engine.record_settlement(cfg, "BTC 15m", -1.0)
    engine.record_settlement(cfg, "BTC 15m", -1.0)
    evaluator = _StrategyBatchEvaluator([strategy], shared=shared, param_region_engine=engine)
    prepared = _prepared(shared)
    signals = [s for s in evaluator.evaluate(prepared, truth_stale=False) if s.action == "buy"]
    assert not signals


def test_param_region_context_downweights_losing_bucket():
    cfg = _cfg(
        "region_ctx",
        enable_param_region_penalties=True,
        param_region_window_settles=20,
        param_region_min_samples=2,
        param_region_loss_threshold_usd=-1.0,
        param_region_penalty_factor=0.5,
    )
    engine = _ParamRegionPenaltyEngine(cfg)
    ctx = {"eb": "0.10_0.20", "pb": "70_78", "tb": "3_10m"}
    engine.record_settlement(cfg, "BTC 15m", -1.0, side="yes", decision_context=ctx)
    engine.record_settlement(cfg, "BTC 15m", -1.0, side="yes", decision_context=ctx)

    adj = engine.evaluate_candidate(
        cfg,
        family="BTC 15m",
        now_ts=time.time(),
        side="yes",
        decision_context=ctx,
    )
    assert adj.candidate
    assert not adj.blocked
    assert adj.scale_mult < 1.0


def test_param_region_context_upweights_winning_bucket():
    cfg = _cfg(
        "region_ctx_gain",
        enable_param_region_penalties=True,
        param_region_window_settles=20,
        param_region_min_samples=2,
        param_region_context_min_samples=2,
        param_region_gain_factor=1.2,
        param_region_gain_avg_pnl_usd=1.0,
        param_region_loss_threshold_usd=-999.0,
    )
    engine = _ParamRegionPenaltyEngine(cfg)
    ctx = {"eb": "0.10_0.20", "pb": "70_78", "tb": "3_10m"}
    engine.record_settlement(cfg, "BTC 60m", 2.0, side="no", decision_context=ctx)
    engine.record_settlement(cfg, "BTC 60m", 2.0, side="no", decision_context=ctx)

    adj = engine.evaluate_candidate(
        cfg,
        family="BTC 60m",
        now_ts=time.time(),
        side="no",
        decision_context=ctx,
    )
    assert adj.rewarded
    assert not adj.blocked
    assert adj.scale_mult > 1.0


def test_momentum_scalp_exit_source_matches_entry():
    """Exit signal source tag must match the entry path that opened the position."""
    pos = _ScalpPosition(
        ticker="FAKE-T1",
        side="yes",
        entry_price_cents=48,
        quantity_centicx=100,
        opened_at=time.time() - 120,
        entry_source="momentum_scalp",
    )
    assert pos.entry_source == "momentum_scalp"
    # The exit signal source should propagate entry_source, not hardcode "mispricing_scalp"
    source = getattr(pos, "entry_source", "mispricing_scalp")
    assert source == "momentum_scalp"


def test_scalp_position_default_entry_source():
    """Default entry_source is 'mispricing_scalp' when not specified."""
    pos = _ScalpPosition(
        ticker="FAKE-T2",
        side="no",
        entry_price_cents=52,
        quantity_centicx=50,
        opened_at=time.time() - 60,
    )
    assert pos.entry_source == "mispricing_scalp"


# ── Session multiplier tests ────────────────────────────────────────────────

import calendar
import datetime as _dt
import pytest

from argus_kalshi.farm_runner import _session_mult


def _utc_ts(hour: int) -> float:
    """Return a wall-clock timestamp for today at the given UTC hour."""
    now = _dt.datetime.now(_dt.timezone.utc).replace(hour=hour, minute=0, second=0, microsecond=0)
    return float(calendar.timegm(now.timetuple()))


def test_session_mult_us_afternoon_boosts_scalp():
    ts = _utc_ts(17)  # 17:00 UTC = US afternoon
    assert _session_mult(ts, "scalp") == pytest.approx(1.20)


def test_session_mult_us_morning_boosts_hold():
    ts = _utc_ts(13)  # 13:00 UTC = US morning
    assert _session_mult(ts, "hold") == pytest.approx(1.10)
    assert _session_mult(ts, "scalp") == pytest.approx(1.05)


def test_session_mult_overnight_reduces():
    ts = _utc_ts(3)  # 03:00 UTC = overnight
    assert _session_mult(ts, "scalp") == pytest.approx(0.70)
    assert _session_mult(ts, "hold") == pytest.approx(0.70)


def test_session_mult_neutral_hours():
    ts = _utc_ts(10)  # 10:00 UTC = EU session, neutral
    assert _session_mult(ts, "scalp") == pytest.approx(1.00)
