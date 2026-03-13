"""
Alpha Vantage Daily Batch Collector
===================================

Performs a single batch pull of daily bars and FX rates at 09:00 AM ET.
This ensures the DB has the latest Asia close and yesterday's Europe/US close
before the NY open.

Budget: config daily_symbols (e.g. 10) + fx_pairs (4) = 14 calls/day (under 25 free tier).
Uses 15s between requests to stay under 5 calls/min; no retries on rate limit.
"""

import asyncio
import logging
import time
from datetime import datetime, time as dt_time, timedelta
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional

from src.connectors.alphavantage_client import AlphaVantageRateLimitError

from src.core.bus import EventBus
from src.core.events import MetricEvent, TOPIC_MARKET_METRICS

logger = logging.getLogger("argus.alphavantage_collector")

# Minimum seconds between API calls (stay under 5/min; 15s = 4/min)
_CALL_INTERVAL_SECONDS = 15.0


def _symbols_from_config(config: Dict[str, Any]) -> List[str]:
    """Build ordered list: daily_symbols (ETFs) then FX pairs as FX:XXXYYY."""
    av_cfg = (config.get("exchanges") or {}).get("alphavantage") or {}
    symbols: List[str] = list(av_cfg.get("daily_symbols") or [])
    for pair in av_cfg.get("fx_pairs") or []:
        # "EUR/USD" or "USD/JPY" -> FX:EURUSD, FX:USDJPY
        parts = pair.replace(" ", "").split("/")
        if len(parts) == 2:
            symbols.append(f"FX:{parts[0]}{parts[1]}")
    return symbols


class AlphaVantageCollector:
    """Daily batch polling of AV daily data into market_bars."""

    def __init__(
        self,
        av_client: Any,
        db: Any,
        bus: EventBus,
        config: Dict[str, Any],
        telegram: Optional[Any] = None,
    ) -> None:
        self._av = av_client
        self._db = db
        self._bus = bus
        self._telegram = telegram
        
        av_cfg = (config.get("exchanges") or {}).get("alphavantage") or {}
        self._enabled = bool(av_cfg.get("enabled", False)) and av_client is not None
        
        # Use config so we never exceed intended budget (e.g. 10 + 4 = 14)
        self._symbols = _symbols_from_config(config)
        if not self._symbols and self._enabled:
            logger.warning("Alpha Vantage enabled but daily_symbols and fx_pairs empty; no batch symbols")
        
        # Target ET time for batch pull
        self._target_hour = 9
        self._target_minute = 0
        
        self._last_run_date: Optional[datetime.date] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None

        if self._enabled:
            logger.info(
                "AlphaVantageCollector initialized (Daily Batch at 09:00 ET) — %d symbols",
                len(self._symbols)
            )

    async def run_forever(self) -> None:
        """Main loop: wait for 9am ET, run batch, sleep."""
        if not self._enabled:
            return

        self._running = True
        logger.info("AlphaVantageCollector loop started (Target: 09:00 ET)")

        while self._running:
            try:
                now_et = datetime.now(ZoneInfo("America/New_York"))
                target_time = dt_time(self._target_hour, self._target_minute)
                
                # If it's a weekday and we haven't run today and it's past 9am
                if self._should_run_now(now_et, target_time):
                    await self._run_batch()
                
                # Calculate sleep until next check or next 9am
                wait_seconds = self._calculate_wait_seconds(now_et, target_time)
                
                try:
                    await asyncio.sleep(wait_seconds)
                except asyncio.CancelledError:
                    break
                    
            except Exception:
                logger.error("AlphaVantageCollector loop error", exc_info=True)
                await asyncio.sleep(60) # Back off on error

    def _should_run_now(self, now_et: datetime, target_time: dt_time) -> bool:
        """True if we haven't run today and it's past target_time."""
        current_date = now_et.date()
        
        # Skip weekends (Saturday, Sunday)
        if current_date.weekday() >= 5:
            return False
            
        # Already ran today?
        if self._last_run_date == current_date:
            return False
            
        # Past target time?
        if now_et.time() >= target_time:
            return True
            
        return False

    def _calculate_wait_seconds(self, now_et: datetime, target_time: dt_time) -> float:
        """Calculate wait until next 9am ET, or a small check interval."""
        current_date = now_et.date()
        
        # If we already ran today, wait until tomorrow 9am
        if self._last_run_date == current_date:
            tomorrow = now_et + timedelta(days=1)
            next_run = datetime.combine(tomorrow.date(), target_time, tzinfo=now_et.tzinfo)
            return max(60, (next_run - now_et).total_seconds())
            
        # If it's before 9am today, wait until 9am today
        if now_et.time() < target_time:
            today_run = datetime.combine(current_date, target_time, tzinfo=now_et.tzinfo)
            return max(60, (today_run - now_et).total_seconds())
            
        # If it's past 9am but we haven't run (unlikely if logic is correct), 
        # short sleep for retry/safety
        return 300

    async def _run_batch(self) -> None:
        """Execute pull for all symbols."""
        n = len(self._symbols)
        logger.info(
            "Starting Alpha Vantage daily batch pull (%d requests, budget 25/day, ~15s apart)",
            n,
        )
        if n > 20:
            logger.warning("Alpha Vantage batch has %d symbols; free tier is 25/day", n)
        total_processed = 0
        total_new = 0
        
        try:
            for symbol in self._symbols:
                if not self._running:
                    break
                    
                try:
                    new_count = await self._collect_symbol(symbol)
                    total_new += new_count
                    total_processed += 1
                    # Stay under 5 calls/min; client also throttles, this is extra safety
                    await asyncio.sleep(_CALL_INTERVAL_SECONDS)
                except Exception as exc:
                    # If it's the daily limit, let it bubble up to the outer try/except
                    if isinstance(exc, AlphaVantageRateLimitError):
                        raise
                    logger.warning("Batch pull failed for %s: %s", symbol, exc)
                    
            self._last_run_date = datetime.now(ZoneInfo("America/New_York")).date()
            summary = f"Batch pull complete: {total_processed}/{len(self._symbols)} instruments updated, {total_new} new bars"
            logger.info(summary)
            
            if self._telegram:
                # Include breakdown for user info
                msg = (
                    f"✅ <b>ALPHA VANTAGE DATA UPDATE</b>\n\n"
                    f"• Symbols: {total_processed}/{len(self._symbols)}\n"
                    f"• New Bars: {total_new}\n"
                    f"• Status: Daily FX/Equity ingestion complete."
                )
                if hasattr(self._telegram, 'send_tiered_message'):
                    await self._telegram.send_tiered_message(msg, priority=2, key="av_batch_complete")
                else:
                    await self._telegram.send_message(msg)

        except AlphaVantageRateLimitError as exc:
            logger.error("Alpha Vantage Daily Batch aborted: Rate limit reached")
            if self._telegram:
                warn_msg = (
                    f"⚠️ <b>ALPHA VANTAGE LIMIT</b>\n\n"
                    f"The daily free tier limit (25 calls) has been exhausted.\n"
                    f"Processed {total_processed}/{len(self._symbols)} symbols.\n"
                    f"Resuming tomorrow @ 09:00 ET."
                )
                if hasattr(self._telegram, 'send_tiered_message'):
                    await self._telegram.send_tiered_message(warn_msg, priority=1, key="av_rate_limit")
                else:
                    await self._telegram.send_message(warn_msg)
            # Still mark as run today to avoid infinite retry loops if the limit is permanent for the day
            self._last_run_date = datetime.now(ZoneInfo("America/New_York")).date()

    async def _collect_symbol(self, symbol: str) -> int:
        """Fetch and save a single symbol/pair. Returns number of new bars."""
        logger.info("Polling Alpha Vantage for %s...", symbol)
        
        bars: List[Dict[str, Any]] = []
        if symbol.startswith("FX:"):
            # Format: FX:USDJPY -> ("USD", "JPY")
            pair = symbol[3:]
            from_cur, to_cur = pair[:3], pair[3:]
            bars = await self._av.fetch_fx_daily(from_cur, to_cur)
        else:
            bars = await self._av.fetch_daily_bars(symbol)

        if not bars:
            logger.debug("No bars returned for %s", symbol)
            return 0

        # Map to market_bars INSERT schema
        db_rows = []
        for b in bars:
            ts_sec = b["timestamp_ms"] / 1000.0
            from datetime import datetime as dt_factory, timezone as tz_factory
            dt = dt_factory.fromtimestamp(ts_sec, tz=tz_factory.utc)
            ts_str = dt.strftime("%Y-%m-%d %H:%M:%S")

            db_rows.append((
                ts_str,
                symbol,
                "alphavantage",
                b["open"],
                b["high"],
                b["low"],
                b["close"],
                b.get("volume", 0.0),
                0, 0, None, None, 0, "DAILY_BAR", 86400
            ))

        new_count = await self._db.upsert_bars_backfill(db_rows)
        logger.debug("Saved %d new bars for %s (total processed: %d)", new_count, symbol, len(db_rows))

        # Publish metric event to the bus
        self._bus.publish(
            TOPIC_MARKET_METRICS,
            MetricEvent(
                symbol=symbol,
                metric="daily_bar_collection",
                value=float(new_count),
                timestamp=time.time(),
                source="alphavantage",
                extra={"total_processed": len(db_rows)}
            )
        )
        return new_count

    def set_telegram(self, telegram: Any) -> None:
        """Set telegram bot instance for notifications."""
        self._telegram = telegram

    def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
