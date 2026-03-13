from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.core.iv_consensus import ContractKey, IVConsensusConfig, IVConsensusEngine
from src.core.option_events import OptionChainSnapshotEvent, OptionQuoteEvent
from src.orchestrator import ArgusOrchestrator


@dataclass
class _DxEvent:
    event_symbol: str
    volatility: float
    receipt_time: int
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None


def _snap(exp_ms: int, recv: int, iv: float | None = None, quote_iv: float | None = None) -> OptionChainSnapshotEvent:
    q = OptionQuoteEvent(
        contract_id="c1",
        symbol="SPY",
        strike=595.0,
        expiration_ms=exp_ms,
        option_type="PUT",
        bid=1.0,
        ask=1.1,
        iv=quote_iv,
        recv_ts_ms=recv,
    )
    return OptionChainSnapshotEvent(
        symbol="SPY",
        expiration_ms=exp_ms,
        underlying_price=595.0,
        puts=(q,),
        calls=(),
        n_strikes=1,
        atm_iv=iv,
        recv_ts_ms=recv,
    )


def test_expiry_isolation_for_atm_keys():
    engine = IVConsensusEngine()
    exp1 = 1_742_515_200_000
    exp2 = 1_743_120_000_000

    engine.observe_public_snapshot(_snap(exp1, recv=1000, iv=0.20))
    engine.observe_public_snapshot(_snap(exp2, recv=1000, iv=0.40))

    r1 = engine.get_atm_consensus("SPY", "PUT", exp1, as_of_ms=2000)
    r2 = engine.get_atm_consensus("SPY", "PUT", exp2, as_of_ms=2000)

    assert r1.consensus_iv == 0.20
    assert r2.consensus_iv == 0.40


def test_freshness_stale_dxlink_not_used():
    cfg = IVConsensusConfig(contract_freshness_ms=1000, policy="prefer_dxlink")
    engine = IVConsensusEngine(cfg)
    key = ContractKey("SPY", 1_742_515_200_000, "PUT", 595.0)

    engine.observe_dxlink_greeks(_DxEvent(".SPY250321P595", 0.33, 1000))
    engine.observe_public_snapshot(_snap(key.expiration_ms, recv=2500, quote_iv=0.22))

    res = engine.get_contract_consensus(key, as_of_ms=2600)
    assert res.iv_source_used == "public"
    assert res.consensus_iv == 0.22


def test_winner_based_switches_to_dxlink_when_discrepancy_high():
    cfg = IVConsensusConfig(policy="winner_based", winner_abs_threshold=0.02, winner_rel_threshold=0.1)
    engine = IVConsensusEngine(cfg)
    key = ContractKey("SPY", 1_742_515_200_000, "PUT", 595.0)

    engine.observe_dxlink_greeks(_DxEvent(".SPY250321P595", 0.35, 1000))
    engine.observe_public_snapshot(_snap(key.expiration_ms, recv=1000, quote_iv=0.20))

    res = engine.get_contract_consensus(key, as_of_ms=1500)
    assert res.iv_source_used == "dxlink"
    assert res.consensus_iv == 0.35


def test_determinism_same_events_same_outputs():
    cfg = IVConsensusConfig(policy="blended")
    e1 = IVConsensusEngine(cfg)
    e2 = IVConsensusEngine(cfg)
    key = ContractKey("SPY", 1_742_515_200_000, "PUT", 595.0)

    events = [
        ("dx", _DxEvent(".SPY250321P595", 0.28, 1000)),
        ("pub", _snap(key.expiration_ms, recv=1100, quote_iv=0.24)),
        ("dx", _DxEvent(".SPY250321P595", 0.27, 1200)),
    ]

    for kind, obj in events:
        if kind == "dx":
            e1.observe_dxlink_greeks(obj)
            e2.observe_dxlink_greeks(obj)
        else:
            e1.observe_public_snapshot(obj)
            e2.observe_public_snapshot(obj)

    assert e1.get_contract_consensus(key, as_of_ms=1500) == e2.get_contract_consensus(key, as_of_ms=1500)


@pytest.mark.asyncio
async def test_orchestrator_wires_dxlink_streamer_callback(monkeypatch, tmp_path):
    captured = {}

    class FakeClient:
        def get_api_quote_token(self):
            return {"dxlink-url": "wss://example", "token": "abc"}

    class FakeOptions:
        def _ensure_client(self):
            return FakeClient()

    class FakeStreamer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def run_forever(self):
            return None

    monkeypatch.setattr("src.orchestrator.TastytradeStreamer", FakeStreamer)

    # Create a minimal secrets file for the orchestrator
    secrets_file = tmp_path / "secrets.yaml"
    secrets_file.write_text(
        "tastytrade:\n  username: test\n  password: test\n"
        "binance:\n  api_key: test\n  api_secret: test\n"
        "okx:\n  api_key: test\n  api_secret: test\n  passphrase: test\n"
        "bybit:\n  api_key: test\n  api_secret: test\n"
        "telegram:\n  bot_token: test\n  chat_id: test\n"
    )
    monkeypatch.setenv("ARGUS_SECRETS", str(secrets_file))

    orch = ArgusOrchestrator(config_dir="config")
    orch.tastytrade_options = FakeOptions()
    orch._dxlink_greeks_symbols = [".SPY250321P595"]

    await orch._start_dxlink_greeks_streamer()

    assert captured["on_event"] == orch._on_dxlink_greeks_event
    assert "Greeks" in captured["event_types"]


def test_discrepancy_rollup_increments_for_divergent_illiquid_symbol():
    cfg = IVConsensusConfig(
        policy="winner_based",
        warn_abs_threshold=0.02,
        warn_rel_threshold=0.10,
        bad_abs_threshold=0.05,
        bad_rel_threshold=0.20,
    )
    engine = IVConsensusEngine(cfg)
    key = ContractKey("SPY", 1_742_515_200_000, "PUT", 595.0)

    # Simulate an illiquid contract where public and dxlink diverge materially.
    engine.observe_dxlink_greeks(_DxEvent(".SPY250321P595", 0.55, 1_000))
    engine.observe_public_snapshot(_snap(key.expiration_ms, recv=1_050, quote_iv=0.40))

    _ = engine.get_contract_consensus(key, as_of_ms=1_200)
    rollup = engine.get_discrepancy_rollup()

    assert rollup["count"] == 1
    assert rollup["warn_count"] == 0
    assert rollup["bad_count"] == 1
    assert rollup["abs_p50"] is not None and rollup["abs_p50"] >= 0.15
    assert rollup["rel_p50"] is not None and rollup["rel_p50"] >= 0.20


def test_consensus_marks_quality_and_selected_source_for_discrepancy():
    cfg = IVConsensusConfig(
        policy="winner_based",
        winner_abs_threshold=0.02,
        winner_rel_threshold=0.10,
        warn_abs_threshold=0.02,
        warn_rel_threshold=0.10,
        bad_abs_threshold=0.05,
        bad_rel_threshold=0.20,
    )
    engine = IVConsensusEngine(cfg)
    key = ContractKey("SPY", 1_742_515_200_000, "PUT", 595.0)

    engine.observe_dxlink_greeks(_DxEvent(".SPY250321P595", 0.55, 1_000))
    engine.observe_public_snapshot(_snap(key.expiration_ms, recv=1_050, quote_iv=0.40))

    res = engine.get_contract_consensus(key, as_of_ms=1_200)

    assert res.iv_source_used == "dxlink"
    assert res.iv_quality == "bad"
