"""Public options snapshot connector using Public.com greeks + external chain structure."""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import asyncio

from ..core.options_normalize import normalize_tastytrade_nested_chain

from ..core.option_events import OptionChainSnapshotEvent, OptionQuoteEvent
from .public_client import PublicAPIClient

logger = logging.getLogger(__name__)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _poll_time_ms() -> int:
    return (_now_ms() // 60_000) * 60_000


def _date_to_ms(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _ms_to_osi_date(expiration_ms: int) -> str:
    dt = datetime.fromtimestamp(expiration_ms / 1000, tz=timezone.utc)
    return dt.strftime("%y%m%d")


def _normalize_osi_symbol(osi: str) -> str:
    return "".join((osi or "").upper().split())


def _response_osi_to_canonical(response_symbol: str) -> str:
    """Normalize Public API response symbol to our canonical OSI (8-digit strike).

    We send e.g. SPY250321P00550000. Public may return the same or e.g. SPY250321P550000
    (strike without leading zeros). Convert to canonical so lookup matches.
    """
    s = _normalize_osi_symbol(response_symbol)
    if not s:
        return ""
    # OSI: underlying (letters) + YYMMDD (6 digits) + P|C + strike (digits)
    m = re.match(r"^([A-Z]+)(\d{6})([PC])(\d+)$", s)
    if not m:
        return s
    underlying, yymmdd, right, strike_digits = m.groups()
    try:
        strike_int = int(strike_digits)
    except ValueError:
        return s
    canonical = f"{underlying}{yymmdd}{right}{strike_int:08d}"
    return canonical


def _to_osi_symbol(symbol: str, expiration_ms: int, option_type: str, strike: float) -> str:
    # OCC strike: strike * 1000, zero-padded to width 8
    strike_int = int(round(float(strike) * 1000))
    right = "P" if option_type.upper() == "PUT" else "C"
    return f"{symbol.upper()}{_ms_to_osi_date(expiration_ms)}{right}{strike_int:08d}"


def _compute_contract_id(option_symbol: str) -> str:
    return hashlib.sha256(option_symbol.encode()).hexdigest()[:16]


@dataclass
class PublicOptionsConfig:
    symbols: Optional[List[str]] = None
    poll_interval_seconds: int = 60
    min_dte: int = 7
    max_dte: int = 21
    batch_size: int = 250


class PublicOptionsConnector:
    PROVIDER = "public"

    def __init__(
        self,
        client: PublicAPIClient,
        structure_connector,
        config: Optional[PublicOptionsConfig] = None,
    ) -> None:
        self._client = client
        self._structure = structure_connector
        self._config = config or PublicOptionsConfig()
        self._sequence_id = 0

    def _next_sequence_id(self) -> int:
        self._sequence_id += 1
        return self._sequence_id

    async def build_snapshots_for_symbol(
        self,
        symbol: str,
        *,
        min_dte: int = 7,
        max_dte: int = 21,
        underlying_price: float = 0.0,
    ) -> List[OptionChainSnapshotEvent]:
        """Build Public-based snapshots using Alpaca or Tastytrade structure."""
        snapshots: List[OptionChainSnapshotEvent] = []

        # Async Alpaca structure connector path
        if hasattr(self._structure, "get_expirations_in_range") and hasattr(self._structure, "build_chain_snapshot"):
            try:
                expirations = await self._structure.get_expirations_in_range(
                    symbol,
                    min_dte=min_dte,
                    max_dte=max_dte,
                )
                for exp_date, _ in expirations:
                    structure_snapshot = await self._structure.build_chain_snapshot(symbol, exp_date)
                    if not structure_snapshot:
                        continue
                    snapshot = await self._build_snapshot_from_structure(structure_snapshot)
                    if snapshot:
                        snapshots.append(snapshot)
                return snapshots
            except (TypeError, AttributeError):
                # Alpaca uses (symbol, min_dte, max_dte); Tastytrade uses (normalized, min_dte, max_dte).
                # When structure is Tastytrade we pass symbol and it expects a list -> AttributeError.
                pass

        # Sync Tastytrade structure connector path
        raw_chain = await asyncio.to_thread(self._structure.fetch_nested_chain, symbol)
        if not raw_chain:
            return snapshots
        normalized = normalize_tastytrade_nested_chain(raw_chain)
        if not normalized:
            return snapshots

        expirations = await asyncio.to_thread(
            self._structure.get_expirations_in_range,
            normalized,
            min_dte,
            max_dte,
        )
        for exp_date, _ in expirations:
            structure_snapshot = await asyncio.to_thread(
                self._structure.build_chain_snapshot,
                symbol,
                exp_date,
                normalized,
                underlying_price,
            )
            if not structure_snapshot:
                continue
            snapshot = await self._build_snapshot_from_structure(structure_snapshot)
            if snapshot:
                snapshots.append(snapshot)
        return snapshots

    async def _build_snapshot_from_structure(
        self,
        structure_snapshot: OptionChainSnapshotEvent,
    ) -> Optional[OptionChainSnapshotEvent]:
        recv_ts_ms = _now_ms()
        timestamp_ms = _poll_time_ms()

        all_quotes: Sequence[OptionQuoteEvent] = tuple(structure_snapshot.puts) + tuple(structure_snapshot.calls)
        if not all_quotes:
            return None

        osi_to_quote: Dict[str, OptionQuoteEvent] = {}
        ordered_osi: List[str] = []
        for q in all_quotes:
            osi = _to_osi_symbol(structure_snapshot.symbol, q.expiration_ms, q.option_type, q.strike)
            norm = _normalize_osi_symbol(osi)
            osi_to_quote[norm] = q
            ordered_osi.append(osi)

        greeks_by_osi: Dict[str, Dict] = {}
        sample_response_symbols: List[str] = []
        batch_size = max(1, min(self._config.batch_size, self._client.MAX_GREEKS_SYMBOLS))
        for i in range(0, len(ordered_osi), batch_size):
            batch = ordered_osi[i : i + batch_size]
            rows = await self._client.get_option_greeks(batch)
            for row in rows:
                if not isinstance(row, dict):
                    continue
                greeks_val = row.get("greeks")
                if not isinstance(greeks_val, dict):
                    greeks_val = {}
                raw_sym = str(row.get("symbol", ""))
                if not raw_sym:
                    continue
                if len(sample_response_symbols) < 3:
                    sample_response_symbols.append(raw_sym)
                # Store under canonical key so we match our _to_osi_symbol format
                # (Public may return strike with fewer digits, e.g. 550000 vs 00550000)
                key = _response_osi_to_canonical(raw_sym)
                if key:
                    greeks_by_osi[key] = greeks_val
        if ordered_osi and not greeks_by_osi and sample_response_symbols:
            logger.debug(
                "Public greeks: no keys matched our OSI format; sample request: %s, sample response symbols: %s",
                ordered_osi[:2],
                sample_response_symbols,
            )

        puts: List[OptionQuoteEvent] = []
        calls: List[OptionQuoteEvent] = []

        for osi_norm, base_quote in osi_to_quote.items():
            g = greeks_by_osi.get(osi_norm) or {}
            if not isinstance(g, dict):
                g = {}
            iv = g.get("impliedVolatility") or g.get("iv")
            q = OptionQuoteEvent(
                contract_id=base_quote.contract_id or _compute_contract_id(osi_norm),
                symbol=base_quote.symbol,
                strike=base_quote.strike,
                expiration_ms=base_quote.expiration_ms,
                option_type=base_quote.option_type,
                bid=base_quote.bid,
                ask=base_quote.ask,
                last=base_quote.last,
                mid=base_quote.mid,
                volume=base_quote.volume,
                open_interest=base_quote.open_interest,
                iv=float(iv) if iv is not None else None,
                delta=g.get("delta"),
                gamma=g.get("gamma"),
                theta=g.get("theta"),
                vega=g.get("vega"),
                timestamp_ms=timestamp_ms,
                source_ts_ms=recv_ts_ms,
                recv_ts_ms=recv_ts_ms,
                provider=self.PROVIDER,
                sequence_id=self._next_sequence_id(),
            )
            if q.option_type == "PUT":
                puts.append(q)
            else:
                calls.append(q)

        puts.sort(key=lambda x: x.strike)
        calls.sort(key=lambda x: x.strike)

        atm_iv = None
        if puts and structure_snapshot.underlying_price > 0:
            atm_put = min(puts, key=lambda q: abs(q.strike - structure_snapshot.underlying_price))
            atm_iv = atm_put.iv

        snapshot_id = f"{self.PROVIDER}_{structure_snapshot.symbol}_{structure_snapshot.expiration_ms}_{timestamp_ms}"
        return OptionChainSnapshotEvent(
            symbol=structure_snapshot.symbol,
            expiration_ms=structure_snapshot.expiration_ms,
            underlying_price=structure_snapshot.underlying_price,
            underlying_bid=structure_snapshot.underlying_bid,
            underlying_ask=structure_snapshot.underlying_ask,
            puts=tuple(puts),
            calls=tuple(calls),
            n_strikes=max(len(puts), len(calls)),
            atm_iv=atm_iv,
            timestamp_ms=timestamp_ms,
            source_ts_ms=recv_ts_ms,
            recv_ts_ms=recv_ts_ms,
            provider=self.PROVIDER,
            snapshot_id=snapshot_id,
            sequence_id=self._next_sequence_id(),
        )
    async def close(self) -> None:
        await self._client.close()

