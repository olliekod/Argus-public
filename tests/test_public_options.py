from datetime import datetime, timezone

import pytest

from src.connectors.public_options import PublicOptionsConnector
from src.core.option_events import OptionChainSnapshotEvent, OptionQuoteEvent


class FakeStructureConnector:
    async def get_expirations_in_range(self, symbol, min_dte=7, max_dte=21):
        return [("2025-03-21", 10)]

    async def build_chain_snapshot(self, symbol, expiration):
        exp_ms = int(datetime(2025, 3, 21, tzinfo=timezone.utc).timestamp() * 1000)
        puts = [
            OptionQuoteEvent(
                contract_id="p1",
                symbol=symbol,
                strike=590.0,
                expiration_ms=exp_ms,
                option_type="PUT",
                bid=1.0,
                ask=1.2,
                provider="alpaca",
                timestamp_ms=1,
                source_ts_ms=1,
                recv_ts_ms=1,
                sequence_id=1,
            ),
            OptionQuoteEvent(
                contract_id="p2",
                symbol=symbol,
                strike=600.0,
                expiration_ms=exp_ms,
                option_type="PUT",
                bid=2.0,
                ask=2.2,
                provider="alpaca",
                timestamp_ms=1,
                source_ts_ms=1,
                recv_ts_ms=1,
                sequence_id=2,
            ),
        ]
        calls = [
            OptionQuoteEvent(
                contract_id="c1",
                symbol=symbol,
                strike=600.0,
                expiration_ms=exp_ms,
                option_type="CALL",
                bid=3.0,
                ask=3.2,
                provider="alpaca",
                timestamp_ms=1,
                source_ts_ms=1,
                recv_ts_ms=1,
                sequence_id=3,
            )
        ]
        return OptionChainSnapshotEvent(
            symbol=symbol,
            expiration_ms=exp_ms,
            underlying_price=596.0,
            puts=tuple(puts),
            calls=tuple(calls),
            n_strikes=2,
            provider="alpaca",
            timestamp_ms=1,
            source_ts_ms=1,
            recv_ts_ms=1,
            snapshot_id="alpaca_x",
            sequence_id=10,
        )


class FakePublicClient:
    MAX_GREEKS_SYMBOLS = 250

    def __init__(self):
        self.calls = []

    async def get_option_greeks(self, osi_symbols):
        self.calls.append(list(osi_symbols))
        out = []
        for sym in osi_symbols:
            out.append(
                {
                    "symbol": sym,
                    "greeks": {
                        "impliedVolatility": 0.31 if "P00590000" in sym else 0.22,
                        "delta": -0.4,
                        "gamma": 0.01,
                        "theta": -0.1,
                        "vega": 0.2,
                    },
                }
            )
        return out

    async def close(self):
        return None


@pytest.mark.asyncio
async def test_public_connector_builds_snapshot_and_atm_iv():
    conn = PublicOptionsConnector(FakePublicClient(), FakeStructureConnector())
    snapshots = await conn.build_snapshots_for_symbol("SPY", min_dte=7, max_dte=21)
    assert len(snapshots) == 1
    snap = snapshots[0]
    assert snap.provider == "public"
    assert snap.atm_iv == 0.22
    assert all(q.provider == "public" for q in snap.puts)
    assert all(q.iv is not None for q in snap.puts)


@pytest.mark.asyncio
async def test_public_connector_batches_over_250_symbols():
    class BigStructure(FakeStructureConnector):
        async def build_chain_snapshot(self, symbol, expiration):
            exp_ms = int(datetime(2025, 3, 21, tzinfo=timezone.utc).timestamp() * 1000)
            puts = []
            for idx in range(300):
                puts.append(
                    OptionQuoteEvent(
                        contract_id=f"p{idx}",
                        symbol=symbol,
                        strike=500.0 + idx,
                        expiration_ms=exp_ms,
                        option_type="PUT",
                        bid=1.0,
                        ask=1.1,
                        provider="alpaca",
                        timestamp_ms=1,
                        source_ts_ms=1,
                        recv_ts_ms=1,
                        sequence_id=idx + 1,
                    )
                )
            return OptionChainSnapshotEvent(
                symbol=symbol,
                expiration_ms=exp_ms,
                underlying_price=600.0,
                puts=tuple(puts),
                calls=tuple(),
                n_strikes=300,
                provider="alpaca",
                timestamp_ms=1,
                source_ts_ms=1,
                recv_ts_ms=1,
                snapshot_id="alpaca_big",
                sequence_id=1,
            )

    client = FakePublicClient()
    conn = PublicOptionsConnector(client, BigStructure())
    snaps = await conn.build_snapshots_for_symbol("SPY", min_dte=7, max_dte=21)
    assert len(snaps) == 1
    assert len(client.calls) == 2
    assert len(client.calls[0]) == 250
    assert len(client.calls[1]) == 50


@pytest.mark.asyncio
async def test_public_connector_supports_tastytrade_structure_path(monkeypatch):
    exp_ms = int(datetime(2025, 3, 21, tzinfo=timezone.utc).timestamp() * 1000)

    class FakeTastyStructure:
        def fetch_nested_chain(self, symbol):
            return {"items": [{"symbol": symbol}]}

        def get_expirations_in_range(self, normalized, min_dte=7, max_dte=21):
            return [("2025-03-21", 10)]

        def build_chain_snapshot(self, symbol, expiration, normalized, underlying_price=0.0):
            put = OptionQuoteEvent(
                contract_id="p1",
                symbol=symbol,
                strike=590.0,
                expiration_ms=exp_ms,
                option_type="PUT",
                bid=1.0,
                ask=1.2,
                provider="tastytrade",
                timestamp_ms=1,
                source_ts_ms=1,
                recv_ts_ms=1,
                sequence_id=1,
            )
            return OptionChainSnapshotEvent(
                symbol=symbol,
                expiration_ms=exp_ms,
                underlying_price=590.0,
                puts=(put,),
                calls=tuple(),
                n_strikes=1,
                provider="tastytrade",
                timestamp_ms=1,
                source_ts_ms=1,
                recv_ts_ms=1,
                snapshot_id="tasty_x",
                sequence_id=10,
            )

    monkeypatch.setattr(
        "src.connectors.public_options.normalize_tastytrade_nested_chain",
        lambda raw: [{"expiry": "2025-03-21", "strike": 590.0}],
    )

    conn = PublicOptionsConnector(FakePublicClient(), FakeTastyStructure())
    snapshots = await conn.build_snapshots_for_symbol("SPY", min_dte=7, max_dte=21)
    assert len(snapshots) == 1
    assert snapshots[0].provider == "public"
