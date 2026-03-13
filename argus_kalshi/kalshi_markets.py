"""
Market discovery and metadata engine for Kalshi multi-asset contracts.

This module discovers target markets either from an explicit ticker list
or by filtering on series/event, then fetches and publishes ``MarketMetadata``
messages on the bus so that the strategy and probability modules have the
strike prices, settlement times, and other contract parameters they need.
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Dict, List, Optional

import aiohttp

from .bus import Bus
from .config import KalshiConfig
from .kalshi_rest import KalshiRestClient
from .logging_utils import ComponentLogger
from .market_selectors import classify_series, filter_supported_markets
from .models import MarketMetadata, SelectedMarkets

log = ComponentLogger("markets")


# Reasonable strike ranges by asset (avoid using time/date segments as strike).
# ETH minimum set to 200 so "60" (from "60 seconds" / "60 minutes" in rules
# text) is rejected as a valid ETH strike — DST-era Kalshi 15m contracts
# have strike_price=30 (the minute marker) and rules text with "60 minutes",
# causing the text-fallback to erroneously accept 60.0 as the ETH strike.
_STRIKE_MIN_MAX = {
    "BTC": (1_000, 500_000),
    "ETH": (200, 100_000),
    "SOL": (1, 50_000),
}
_TIME_TOKEN_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
# Avoid treating year tokens (e.g. 2026) as strikes during text fallback parsing.
_YEAR_TOKEN_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_CURRENCY_NUMBER_RE = re.compile(
    r"\$?\s*([0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)"
)
_INVALID_STRIKE_WARNED: set[str] = set()
_STRIKE_CORRECTED_WARNED: set[str] = set()
_DIRECTIONAL_PHRASES = ("price up", "price down", "go up", "go down", "price higher", "price lower")


def _looks_like_time_fragment(value: str) -> bool:
    """Return True when a numeric ticker segment is really HHMM or HHMMSS."""
    if len(value) == 4:
        hh = int(value[:2])
        mm = int(value[2:])
        return 0 <= hh <= 23 and 0 <= mm <= 59
    if len(value) == 6:
        hh = int(value[:2])
        mm = int(value[2:4])
        ss = int(value[4:])
        return 0 <= hh <= 23 and 0 <= mm <= 59 and 0 <= ss <= 59
    return False


def _parse_strike_from_ticker(ticker: str, asset: Optional[str] = None) -> Optional[float]:
    """Attempt to extract a numeric strike from a market ticker string.

    Kalshi tickers often embed the strike, e.g.
    ``KXBTCD-26FEB21-65000`` → 65000.0.
    They also embed date/time (e.g. 08:30 → 30), which must not be used as strike.
    When asset is provided, only numeric segments within that asset's strike range
    are considered; otherwise the last segment is used (legacy behavior).
    """
    parts = re.findall(r"\d+(?:\.\d+)?", ticker)
    if not parts:
        return None
    is_15m = "15M" in ticker.upper()
    if asset and asset in _STRIKE_MIN_MAX:
        lo, hi = _STRIKE_MIN_MAX[asset]
        candidates: List[float] = []
        for part in parts:
            if is_15m:
                # 15m tickers often include HHMM/HHMMSS rollover fragments
                # and trailing minute markers that are not strikes.
                if _looks_like_time_fragment(part):
                    continue
                if asset != "BTC" and len(part) <= 2:
                    continue
            value = float(part)
            if lo <= value <= hi:
                candidates.append(value)
        if candidates:
            return max(candidates)  # prefer largest (most likely the strike, not a date)
        return None
    return float(parts[-1])


def _parse_strike_from_text(raw: Dict, asset: str) -> Optional[float]:
    """Best-effort strike extraction from human-readable market text.

    Needed for rollover-style 15m contracts where API `strike_price` can be a
    minute marker (e.g. 45) instead of the true strike shown in title/subtitle.
    """
    if asset not in _STRIKE_MIN_MAX:
        return None
    lo, hi = _STRIKE_MIN_MAX[asset]
    text = " ".join(
        str(raw.get(k, ""))
        for k in (
            "title",
            "subtitle",
            "yes_sub_title",
            "rules_primary",
            "rules_secondary",
        )
    )
    if not text.strip():
        return None
    cleaned = _TIME_TOKEN_RE.sub(" ", text)
    cleaned = _YEAR_TOKEN_RE.sub(" ", cleaned)
    candidates: List[float] = []
    for m in _CURRENCY_NUMBER_RE.findall(cleaned):
        try:
            val = float(m.replace(",", ""))
        except ValueError:
            continue
        if lo <= val <= hi:
            candidates.append(val)
    if not candidates:
        return None
    return max(candidates)


class MarketDiscovery:
    """Fetches and tracks metadata for target Kalshi BTC/ETH/SOL markets."""

    def __init__(
        self,
        config: KalshiConfig,
        rest: KalshiRestClient,
        bus: Bus,
    ) -> None:
        self._cfg = config
        self._rest = rest
        self._bus = bus
        self._metadata: Dict[str, MarketMetadata] = {}
        self._running = False
        self._task: Optional[asyncio.Task[None]] = None

    @property
    def metadata(self) -> Dict[str, MarketMetadata]:
        return self._metadata

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        self._running = True
        # Perform one discovery pass synchronously so callers that await start()
        # can immediately consume discovered tickers/metadata.
        await self._discover()
        # Continue periodic refresh in the background.
        self._task = asyncio.create_task(self._refresh_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    # -- discovery -----------------------------------------------------------

    async def _discover(self) -> None:
        """Fetch target markets and publish MarketMetadata messages.

        Uses incremental update: newly found markets are added and stale markets
        are pruned. This avoids the brief EMPTY flash that occurred when metadata
        was cleared before re-population (especially near 15-minute rollovers).
        """
        tickers = self._cfg.target_market_tickers
        old_keys = set(self._metadata.keys())
        api_found: set = set()

        if tickers:
            api_found = await self._fetch_by_tickers(tickers)
        elif self._cfg.series_filter or self._cfg.event_filter:
            api_found = await self._fetch_by_filter()
        else:
            log.warning("No target tickers or filters configured — nothing to discover")

        # Prune markets no longer in the API results (expired / de-listed).
        # Guard: only prune if the fetch returned something; a zero-result cycle
        # (transient API gap, network error) should not wipe the UI.
        if api_found:
            for gone in old_keys - api_found:
                self._metadata.pop(gone, None)

        # Always publish the selected ticker set so components can prune removals.
        await self._bus.publish(
            "kalshi.selected_markets",
            SelectedMarkets(
                tickers=list(self._metadata.keys()),
                timestamp=time.time(),
            ),
        )

    async def _fetch_by_tickers(self, tickers: List[str]) -> set:
        """Fetch individual markets by explicit ticker. Returns set of successfully fetched tickers."""
        tasks = [self._fetch_one(t) for t in tickers]
        await asyncio.gather(*tasks, return_exceptions=True)
        return {t for t in tickers if t in self._metadata}

    async def _fetch_one(self, ticker: str) -> None:
        try:
            resp = await self._rest.get_market(ticker)
            market = resp.get("market", resp)
            meta = self._parse_market(market)
            if meta:
                self._metadata[ticker] = meta
                await self._bus.publish("kalshi.market_metadata", meta)
                log.info(f"Discovered market {ticker}", data={
                    "strike": meta.strike_price,
                    "settlement": meta.settlement_time_iso,
                })
        except Exception as exc:
            log.error(f"Failed to fetch market {ticker}: {exc}")

    async def _fetch_by_filter(self) -> set:
        """Fetch markets using series/event filter with pagination.

        After collecting candidates from the API, applies BTC/ETH/SOL heuristic
        filtering via ``market_selectors`` so only contracts in the correct
        family are selected. Handles comma-separated series filters.

        Returns the set of ticker strings that were successfully parsed and added.
        """
        start_mono = time.monotonic()
        series_filters = [s.strip() for s in self._cfg.series_filter.split(",")] if self._cfg.series_filter else [None]
        event_filters = [e.strip() for e in self._cfg.event_filter.split(",")] if self._cfg.event_filter else [None]

        raw_markets: List[Dict] = []
        for s_filter in series_filters:
            for e_filter in event_filters:
                kwargs: Dict = {"status": "open"}
                if s_filter:
                    kwargs["series_ticker"] = s_filter
                if e_filter:
                    kwargs["event_ticker"] = e_filter

                try:
                    # Use longer timeout (60s) for discovery; API can be slow under load
                    disc_timeout = aiohttp.ClientTimeout(total=60, connect=20)
                    async for market in self._rest.paginate(
                        "GET",
                        "/markets",
                        params=kwargs,
                        items_key="markets",
                        timeout=disc_timeout,
                    ):
                        raw_markets.append(market)
                except Exception as exc:
                    log.error(f"Market filter discovery failed for series={s_filter}, event={e_filter}: {exc}")

        # Apply supported market filtering heuristics.
        title_regex = self._cfg.market_title_regex or None
        filtered = filter_supported_markets(
            raw_markets,
            assets=self._cfg.assets,
            window_minutes=self._cfg.window_minutes,
            include_range_markets=self._cfg.include_range_markets,
            title_regex=title_regex,
        )

        if not filtered and raw_markets:
            log.warning(
                f"No supported open markets currently identified among {len(raw_markets)} candidates. Will retry on next refresh."
            )
            return set()

        found: set = set()
        for raw in filtered:
            meta = self._parse_market(raw)
            if meta:
                self._metadata[meta.market_ticker] = meta
                await self._bus.publish("kalshi.market_metadata", meta)
                found.add(meta.market_ticker)

        elapsed_s = round(time.monotonic() - start_mono, 1)
        # Log at INFO only when count changes; otherwise debug to avoid spam near rollover boundaries.
        _count_now = len(found)
        _log_fn = log.info if _count_now != getattr(self, "_last_logged_count", -1) else log.debug
        _log_fn(
            f"Discovered {_count_now} markets via filter "
            f"(from {len(raw_markets)} candidates)",
            data={"elapsed_s": elapsed_s, "count": _count_now},
        )
        self._last_logged_count = _count_now  # type: ignore[attr-defined]
        return found

    # -- parsing -------------------------------------------------------------

    @staticmethod
    def _parse_market(raw: Dict) -> Optional[MarketMetadata]:
        """Parse a Kalshi market API response into a MarketMetadata message."""
        ticker = raw.get("ticker", "")
        if not ticker:
            return None

        # Strike: try structured field first, fall back to ticker parsing.
        # (Asset is needed for ticker fallback to avoid using time/date as strike;
        # we'll validate after asset is known.)
        strike = raw.get("strike_price")
        if strike is not None:
            strike = float(strike)
        else:
            floor_strike = raw.get("floor_strike")
            cap_strike = raw.get("cap_strike")
            if floor_strike is not None:
                strike = float(floor_strike)
            elif cap_strike is not None:
                strike = float(cap_strike)
            else:
                strike = _parse_strike_from_ticker(ticker) or 0.0

        # Close time is when trading ends; expiration is when it settles.
        # For our UI, "time remaining" should focus on the trading window.
        close_time = raw.get("close_time", "")
        expiration_time = raw.get("expiration_time", "")
        
        # Prefer the earlier of the two if both exist, as that's the "active" limit.
        settlement_time = close_time if close_time else expiration_time
        last_trade_time = raw.get("last_trade_time", close_time)

        # Index name: some BTC contracts reference CF Benchmarks BRTI.
        subtitle = raw.get("subtitle", "") + " " + raw.get("title", "")
        index_name = ""
        if "BRTI" in subtitle.upper() or "CF BENCHMARK" in subtitle.upper():
            index_name = "CF Benchmarks BRTI"

        # API may omit series_ticker in list response; derive from event_ticker (e.g. "KXBTC15M-26FEB25" -> "KXBTC15M").
        series_ticker = raw.get("series_ticker", "")
        if not series_ticker:
            event = (raw.get("event_ticker") or "").strip()
            if event and "-" in event:
                series_ticker = event.split("-")[0]
        series_info = classify_series(series_ticker)
        asset = "BTC"
        window_minutes = 15
        is_range = False
        if series_info:
            asset, window_minutes, is_range = series_info
        else:
            # Fallback: infer asset from ticker/series string when series is unknown.
            upper_ticker = (ticker + " " + series_ticker).upper()
            if "ETH" in upper_ticker:
                asset = "ETH"
            elif "SOL" in upper_ticker:
                asset = "SOL"
            # window_minutes and is_range stay at defaults (15, False) as best guess

        # Detect directional contracts ("BTC price up in next 15 mins?") — may have no dollar strike.
        title_raw = str(raw.get("title", "") or "")
        is_directional = any(p in title_raw.lower() for p in _DIRECTIONAL_PHRASES)

        # Validate strike is in plausible range for asset (API/ticker can return
        # time or wrong units). Correct using floor_strike or asset-aware ticker parse.
        if asset in _STRIKE_MIN_MAX:
            lo, hi = _STRIKE_MIN_MAX[asset]
            if strike < lo or strike > hi:
                corrected = None
                fl = raw.get("floor_strike")
                if fl is not None:
                    fl_val = float(fl)
                    if lo <= fl_val <= hi:
                        corrected = fl_val
                if corrected is None:
                    corrected = _parse_strike_from_ticker(ticker, asset)
                if corrected is None:
                    corrected = _parse_strike_from_text(raw, asset)
                if corrected is not None:
                    # Only warn once per ticker — the correction fires every discovery cycle otherwise.
                    if ticker not in _STRIKE_CORRECTED_WARNED:
                        log.warning(
                            f"Strike {strike} out of range for {asset} [{lo}, {hi}]; "
                            f"using {corrected}",
                            data={"ticker": ticker, "raw_strike": strike,
                                  "title": str(raw.get("title", ""))[:80]},
                        )
                        _STRIKE_CORRECTED_WARNED.add(ticker)
                    strike = corrected
                elif is_directional:
                    # Directional contracts compare two price averages — no fixed dollar strike.
                    # Accept the market with strike=0.0; probability engine will use current price.
                    strike = 0.0
                else:
                    # Do not keep invalid strike metadata for BTC/ETH/SOL markets.
                    # A zero/invalid strike can corrupt fair-prob and UI rows.
                    if ticker not in _INVALID_STRIKE_WARNED:
                        log.warning(
                            f"Strike {strike} out of range for {asset} [{lo}, {hi}]; "
                            f"skipping market until valid strike metadata is available. "
                            f"(title={str(raw.get('title', ''))[:80]!r})",
                            data={"ticker": ticker, "raw_strike": strike,
                                  "floor_strike": raw.get("floor_strike"),
                                  "cap_strike": raw.get("cap_strike")},
                        )
                        _INVALID_STRIKE_WARNED.add(ticker)
                    return None

        # Diagnostic: log raw fields for non-BTC markets to catch strike scale issues.
        if asset != "BTC":
            log.debug(
                f"Non-BTC market parsed: ticker={ticker} asset={asset} "
                f"strike={strike} series={series_ticker} "
                f"raw_strike_price={raw.get('strike_price')} "
                f"floor_strike={raw.get('floor_strike')} "
                f"cap_strike={raw.get('cap_strike')}"
            )

        strike_floor = raw.get("floor_strike")
        strike_cap = raw.get("cap_strike")

        return MarketMetadata(
            market_ticker=ticker,
            strike_price=strike,
            settlement_time_iso=settlement_time,
            last_trade_time_iso=last_trade_time,
            index_name=index_name,
            contract_type=raw.get("contract_type", "binary"),
            series_ticker=series_ticker,
            event_ticker=raw.get("event_ticker", ""),
            status=raw.get("status", ""),
            asset=asset,
            window_minutes=window_minutes,
            is_range=is_range,
            strike_floor=float(strike_floor) if strike_floor is not None else None,
            strike_cap=float(strike_cap) if strike_cap is not None else None,
            is_directional=is_directional,
        )

    # -- refresh loop --------------------------------------------------------

    async def _refresh_loop(self) -> None:
        """Periodically re-fetch metadata to catch lifecycle changes.

        Runs discovery every market_refresh_interval_s. Additionally, triggers
        an immediate discovery when within 10s of a 15min boundary (e.g. :00, :15,
        :30, :45) so 15min markets refresh promptly after expiry.
        """
        interval = self._cfg.market_refresh_interval_s
        _15MIN_S = 900
        while self._running:
            try:
                await self._discover()
                # Near 15min boundary? Sleep less so we catch the rollover.
                now = time.time()
                secs_into_15 = int(now) % _15MIN_S
                if secs_into_15 >= _15MIN_S - 10 or secs_into_15 < 5:
                    # Within 10s of boundary: short sleep to catch new 15min markets
                    await asyncio.sleep(min(interval, 3.0))
                else:
                    await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error(f"Refresh loop error: {exc}")
                await asyncio.sleep(10.0)

    # -- convenience ---------------------------------------------------------

    def get_tickers(self) -> List[str]:
        return list(self._metadata.keys())

    def get_metadata(self, ticker: str) -> Optional[MarketMetadata]:
        return self._metadata.get(ticker)
