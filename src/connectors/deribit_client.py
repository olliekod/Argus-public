"""
Deribit REST API Client
=======================

Public REST client for Deribit options data.
No authentication required for public endpoints.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import aiohttp

from ..core.logger import get_connector_logger
from ..core.events import QuoteEvent, MetricEvent, TOPIC_MARKET_QUOTES, TOPIC_MARKET_METRICS
from ..core.bar_builder import _ts_sane

logger = get_connector_logger('deribit')


class DeribitClient:
    """
    Deribit public API client for options data.
    
    Provides:
    - Options IV data
    - Greeks
    - Volatility index
    
    Uses public endpoints - no API key required for US users.
    """
    
    MAINNET_URL = "https://www.deribit.com/api/v2"
    TESTNET_URL = "https://test.deribit.com/api/v2"
    
    def __init__(self, testnet: bool = True, event_bus=None):
        """
        Initialize Deribit client.

        Args:
            testnet: Use testnet endpoint (recommended for testing)
            event_bus: Optional EventBus for publishing QuoteEvents
        """
        self.base_url = self.TESTNET_URL if testnet else self.MAINNET_URL
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limit = 20  # public rate limit: 20/min unauthenticated
        self._request_count = 0
        self._last_reset = datetime.now(timezone.utc)
        self._event_bus = event_bus
        self.last_message_ts: Optional[float] = None
        self.reconnect_attempts = 0  # REST doesn't "reconnect", but we track errors
        self._request_count_total = 0
        self.last_success_ts: Optional[float] = None
        self.last_error: Optional[str] = None
        self.consecutive_failures: int = 0
        self.error_count: int = 0
        self.last_latency_ms: Optional[float] = None
        self.avg_latency_ms: Optional[float] = None
        self.last_poll_ts: Optional[float] = None
        self.last_http_status: Optional[int] = None

        logger.info(f"Deribit client initialized ({'testnet' if testnet else 'mainnet'})")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=10))
        return self._session
    
    async def close(self) -> None:
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def _request(self, method: str, params: Dict = None) -> Dict:
        """Make a public API request."""
        # Simple rate limiting: 20 requests per 60-second window.
        # Uses total_seconds() (not .seconds) to correctly handle intervals > 60s.
        now = datetime.now(timezone.utc)
        elapsed = (now - self._last_reset).total_seconds()
        if elapsed >= 60:
            self._request_count = 0
            self._last_reset = now

        if self._request_count >= self._rate_limit:
            wait_time = max(0, 60 - (now - self._last_reset).total_seconds())
            logger.warning(f"Rate limit reached, waiting {wait_time:.0f}s")
            await asyncio.sleep(wait_time)
            self._request_count = 0
            self._last_reset = datetime.now(timezone.utc)
        
        self._request_count += 1
        
        session = await self._get_session()
        url = f"{self.base_url}/public/{method}"
        start = asyncio.get_running_loop().time()
        self.last_poll_ts = datetime.now(timezone.utc).timestamp()
        self._request_count_total += 1
        
        try:
            async with session.get(url, params=params) as resp:
                self.last_http_status = resp.status
                data = await resp.json()
                
                if isinstance(data, dict) and 'error' in data:
                    self.last_error = str(data.get('error'))
                    self.error_count += 1
                    self.consecutive_failures += 1
                    logger.warning(f"Deribit API error: {data['error']}")
                else:
                    import time
                    self.last_message_ts = time.time()
                    self.last_success_ts = self.last_message_ts
                    self.consecutive_failures = 0
                    self.last_error = None
                
                latency_ms = (asyncio.get_running_loop().time() - start) * 1000
                self.last_latency_ms = latency_ms
                self.avg_latency_ms = (
                    latency_ms if self.avg_latency_ms is None
                    else (latency_ms * 0.2) + (self.avg_latency_ms * 0.8)
                )
                return data
        except Exception as e:
            self.last_error = str(e)
            self.error_count += 1
            self.consecutive_failures += 1
            logger.error(f"Deribit request failed: {e}")
            return {'error': str(e)}

    @staticmethod
    def _normalize_source_ts(raw: Optional[float]) -> Tuple[Optional[float], Optional[str]]:
        """Normalize Deribit timestamps to epoch seconds."""
        if raw is None:
            return None, "missing"
        try:
            raw_val = float(raw)
        except (TypeError, ValueError):
            return None, "invalid"

        if _ts_sane(raw_val):
            return raw_val, None

        if raw_val > 10_000_000_000_000:
            candidate = raw_val / 1_000_000.0
            if _ts_sane(candidate):
                return candidate, "converted_us"
        elif raw_val > 10_000_000_000:
            candidate = raw_val / 1000.0
            if _ts_sane(candidate):
                return candidate, "converted_ms"

        return None, "out_of_range"

    @staticmethod
    def _describe_payload(data: Any) -> str:
        if isinstance(data, dict):
            return f"dict keys={list(data.keys())}"
        if isinstance(data, list):
            return f"list len={len(data)}"
        return f"{type(data).__name__}"

    @staticmethod
    def _get_dict(data: Any, context: str) -> Dict[str, Any]:
        if isinstance(data, dict):
            return data
        logger.warning("Deribit %s expected dict but got %s", context, type(data).__name__)
        return {}

    @classmethod
    def _extract_source_ts(cls, data: Any) -> Tuple[Optional[float], Optional[str], Optional[str], Optional[float]]:
        """Extract a source timestamp from a Deribit API response."""
        data_dict = cls._get_dict(data, "source_ts")
        result = data_dict.get("result")
        result_dict = result if isinstance(result, dict) else {}
        candidates = [
            ("result.timestamp", result_dict.get("timestamp")),
            ("result.creation_timestamp", result_dict.get("creation_timestamp")),
            ("usOut", data_dict.get("usOut")),
            ("usIn", data_dict.get("usIn")),
        ]
        for label, raw in candidates:
            source_ts, reason = cls._normalize_source_ts(raw)
            if source_ts is not None:
                return source_ts, reason, label, raw
        return None, "missing", None, None

    @classmethod
    def _coerce_result_list(cls, data: Any, context: str) -> List[Dict[str, Any]]:
        """Return a list of result items, handling dict/list payloads safely."""
        if isinstance(data, list):
            logger.warning(
                "Deribit %s response returned a list (%s)",
                context,
                cls._describe_payload(data),
            )
            return data
        if not isinstance(data, dict):
            logger.warning(
                "Deribit %s response unexpected payload (%s)",
                context,
                cls._describe_payload(data),
            )
            return []

        result = data.get("result", [])
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            logger.warning(
                "Deribit %s response had dict result (%s)",
                context,
                cls._describe_payload(result),
            )
            return [result]

        logger.warning(
            "Deribit %s response had unexpected result (%s)",
            context,
            cls._describe_payload(result),
        )
        return []
    
    async def get_ticker(self, instrument_name: str) -> Optional[Dict]:
        """
        Get ticker data including IV for an option.
        
        Args:
            instrument_name: Deribit instrument name (e.g., 'BTC-28JUN24-50000-C')
            
        Returns:
            Ticker data with IV and Greeks or None
        """
        data = await self._request('ticker', {'instrument_name': instrument_name})
        data_dict = self._get_dict(data, "ticker")
        result = data_dict.get("result")
        if isinstance(result, dict):
            return {
                'instrument': instrument_name,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'last_price': result.get('last_price'),
                'mark_price': result.get('mark_price'),
                'mark_iv': result.get('mark_iv'),  # Implied volatility for mark price
                'bid_iv': result.get('bid_iv'),     # IV for best bid
                'ask_iv': result.get('ask_iv'),     # IV for best ask
                'underlying_price': result.get('underlying_price'),
                'underlying_index': result.get('underlying_index'),
                'delta': result.get('greeks', {}).get('delta'),
                'gamma': result.get('greeks', {}).get('gamma'),
                'theta': result.get('greeks', {}).get('theta'),
                'vega': result.get('greeks', {}).get('vega'),
                'rho': result.get('greeks', {}).get('rho'),
                'open_interest': result.get('open_interest'),
                'volume_24h': result.get('stats', {}).get('volume'),
            }
        if result is not None and not isinstance(result, dict):
            logger.warning(
                "Deribit ticker response unexpected result (%s)",
                self._describe_payload(result),
            )
        return None
    
    async def get_book_summary_by_currency(
        self,
        currency: str = "BTC",
        kind: str = "option"
    ) -> List[Dict]:
        """
        Get summary of all instruments for a currency.
        
        Args:
            currency: 'BTC' or 'ETH'
            kind: 'option' or 'future'
            
        Returns:
            List of instrument summaries with IV data
        """
        data = await self._request(
            'get_book_summary_by_currency',
            {'currency': currency, 'kind': kind}
        )
        
        source_ts, ts_reason, ts_label, ts_raw = self._extract_source_ts(data)
        result_items = self._coerce_result_list(data, "book_summary")
        result = []
        for item in result_items:
            if not isinstance(item, dict):
                logger.warning(
                    "Deribit book_summary item unexpected payload (%s)",
                    self._describe_payload(item),
                )
                continue
            result.append({
                'instrument': item.get('instrument_name'),
                'currency': currency,
                'kind': kind,
                'mark_price': item.get('mark_price'),
                'mark_iv': item.get('mark_iv'),
                'bid_price': item.get('bid_price'),
                'ask_price': item.get('ask_price'),
                'volume_24h': item.get('volume'),
                'open_interest': item.get('open_interest'),
                'underlying_price': item.get('underlying_price'),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'source_ts': source_ts,
                'source_ts_reason': ts_reason,
                'source_ts_label': ts_label,
                'source_ts_raw': ts_raw,
            })
        
        return result
    
    async def get_instruments(
        self,
        currency: str = "BTC",
        kind: str = "option",
        expired: bool = False
    ) -> List[Dict]:
        """
        Get all available instruments.
        
        Args:
            currency: 'BTC' or 'ETH'
            kind: 'option' or 'future'
            expired: Include expired instruments
            
        Returns:
            List of instrument definitions
        """
        data = await self._request(
            'get_instruments',
            {'currency': currency, 'kind': kind, 'expired': str(expired).lower()}
        )
        data_dict = self._get_dict(data, "instruments")
        result_items = data_dict.get('result', [])
        if not isinstance(result_items, list):
            logger.warning(
                "Deribit instruments response unexpected result (%s)",
                self._describe_payload(result_items),
            )
            return []

        result = []
        for item in result_items:
            if not isinstance(item, dict):
                logger.warning(
                    "Deribit instruments item unexpected payload (%s)",
                    self._describe_payload(item),
                )
                continue
            result.append({
                'instrument': item.get('instrument_name'),
                'currency': currency,
                'kind': kind,
                'strike': item.get('strike'),
                'option_type': item.get('option_type'),  # 'call' or 'put'
                'expiration_timestamp': item.get('expiration_timestamp'),
                'is_active': item.get('is_active'),
                'min_trade_amount': item.get('min_trade_amount'),
            })
        
        return result
    
    async def get_index_price(self, index_name: str = "btc_usd") -> Optional[Dict]:
        """
        Get current index price.
        
        Args:
            index_name: Index name (e.g., 'btc_usd', 'eth_usd')
            
        Returns:
            Index price data
        """
        data = await self._request('get_index_price', {'index_name': index_name})
        
        if isinstance(data, list):
            logger.warning(
                "Deribit index price response unexpected payload (%s)",
                self._describe_payload(data),
            )
            return None

        if isinstance(data, dict) and 'result' in data:
            source_ts, ts_reason, ts_label, ts_raw = self._extract_source_ts(data)
            return {
                'index_name': index_name,
                'index_price': data['result'].get('index_price'),
                'estimated_delivery_price': data['result'].get('estimated_delivery_price'),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'source_ts': source_ts,
                'source_ts_reason': ts_reason,
                'source_ts_label': ts_label,
                'source_ts_raw': ts_raw,
            }
        return None
    
    async def get_historical_volatility(
        self,
        currency: str = "BTC"
    ) -> Optional[Dict]:
        """
        Get historical volatility data.
        
        Args:
            currency: 'BTC' or 'ETH'
            
        Returns:
            Historical volatility data
        """
        data = await self._request(
            'get_historical_volatility',
            {'currency': currency}
        )
        
        if 'result' in data:
            # Returns list of [timestamp, volatility] pairs
            return {
                'currency': currency,
                'data': data['result'],
                'latest_hv': data['result'][-1][1] if data['result'] else None,
                'timestamp': datetime.now(timezone.utc).isoformat(),
            }
        return None
    
    async def get_atm_iv(self, currency: str = "BTC") -> Optional[Dict]:
        """
        Get ATM implied volatility by finding nearest strike options.
        
        Args:
            currency: 'BTC' or 'ETH'
            
        Returns:
            ATM IV data or None
        """
        # Get index price first
        index_data = await self.get_index_price(f"{currency.lower()}_usd")
        if not index_data:
            return None
        
        index_price = index_data['index_price']
        
        # Get all options
        options = await self.get_book_summary_by_currency(currency, 'option')
        if not options:
            return None
        
        # Find ATM options (closest to current price)
        atm_options = []
        for opt in options:
            if opt['mark_iv'] and opt['mark_iv'] > 0:
                # Parse instrument name to get strike
                # Format: BTC-28JUN24-50000-C
                parts = opt['instrument'].split('-')
                if len(parts) >= 3:
                    try:
                        strike = float(parts[2])
                        distance = abs(strike - index_price) / index_price
                        if distance < 0.1:  # Within 10% of ATM
                            opt['strike'] = strike
                            opt['distance_from_atm'] = distance
                            atm_options.append(opt)
                    except ValueError:
                        pass
        
        if not atm_options:
            return None
        
        # Sort by distance from ATM and get closest
        atm_options.sort(key=lambda x: x['distance_from_atm'])
        closest = atm_options[0]
        
        # Average IV of closest options
        atm_iv_values = [o['mark_iv'] for o in atm_options[:4] if o['mark_iv']]
        avg_atm_iv = sum(atm_iv_values) / len(atm_iv_values) if atm_iv_values else None
        
        # AGGREGATE GREEKS: Fetch ticker for top 2 ATM options to get real Greeks
        agg_greeks_sums = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
        agg_greeks_counts = {"delta": 0, "gamma": 0, "vega": 0, "theta": 0}
        
        # We fetch the Call and Put for the closest strike if possible
        top_instruments = [o['instrument'] for o in atm_options[:2]]
        for inst in top_instruments:
            ticker = await self.get_ticker(inst)
            if ticker:
                for g in agg_greeks_sums:
                    val = ticker.get(g)
                    if val is not None:
                        agg_greeks_sums[g] += float(val)
                        agg_greeks_counts[g] += 1
        
        agg_greeks = {}
        for g, count in agg_greeks_counts.items():
            if count > 0:
                agg_greeks[g] = agg_greeks_sums[g] / count
        
        if not agg_greeks:
            agg_greeks = None

        result = {
            'currency': currency,
            'index_price': index_price,
            'atm_iv': avg_atm_iv,
            'closest_strike': closest['strike'],
            'closest_iv': closest['mark_iv'],
            'sample_size': len(atm_iv_values),
            'greeks': agg_greeks,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'source_ts': index_data.get('source_ts') if index_data else None,
            'source_ts_reason': index_data.get('source_ts_reason') if index_data else None,
            'source_ts_label': index_data.get('source_ts_label') if index_data else None,
            'source_ts_raw': index_data.get('source_ts_raw') if index_data else None,
        }

        # Publish QuoteEvent for the underlying index (price-only)
        if self._event_bus is not None:
            import time as _time
            try:
                source_ts = index_data.get("source_ts") if index_data else None
                source_ts_raw = index_data.get("source_ts_raw") if index_data else None
                source_ts_reason = index_data.get("source_ts_reason") if index_data else "missing"
                source_ts_label = index_data.get("source_ts_label") if index_data else None
                if source_ts is None:
                    logger.warning(
                        "Rejected Deribit quote for %s: %s (raw_ts=%r, field=%s)",
                        f"{currency}-INDEX",
                        source_ts_reason or "missing source_ts",
                        source_ts_raw,
                        source_ts_label,
                    )
                    source_ts = 0.0
                timestamp = source_ts if source_ts else 0.0
                now = _time.time()
                quote = QuoteEvent(
                    symbol=f"{currency}-INDEX",
                    bid=index_price,
                    ask=index_price,
                    mid=index_price,
                    last=index_price,
                    timestamp=timestamp,
                    source='deribit',
                    source_ts=source_ts,
                )
                self._event_bus.publish(TOPIC_MARKET_QUOTES, quote)

                # Publish ATM IV as a separate metric
                if avg_atm_iv is not None:
                    self._event_bus.publish(TOPIC_MARKET_METRICS, MetricEvent(
                        symbol=f"{currency}-INDEX",
                        metric='atm_iv',
                        value=avg_atm_iv,
                        timestamp=now,
                        source='deribit',
                        extra={'sample_size': len(atm_iv_values)},
                    ))
                
                # Publish Aggregate Greeks as metrics
                if agg_greeks:
                    for greek, val in agg_greeks.items():
                        self._event_bus.publish(TOPIC_MARKET_METRICS, MetricEvent(
                            symbol=f"{currency}-INDEX",
                            metric=f"atm_{greek}",
                            value=val,
                            timestamp=now,
                            source='deribit',
                            extra={'sample_size': agg_greeks_counts.get(greek, 0)},
                        ))
            except Exception as e:
                logger.error("QuoteEvent publish error (%s): %s", type(e).__name__, e)
                logger.debug("Deribit QuoteEvent publish error detail", exc_info=True)

        return result
    
    async def poll_options_iv(
        self,
        currency: str = "BTC",
        interval_seconds: int = 60,
        callback=None
    ) -> None:
        """
        Continuously poll ATM IV.
        
        Args:
            currency: Currency to monitor
            interval_seconds: Polling interval
            callback: Function to call with data
        """
        logger.info(f"Starting options IV polling for {currency}")
        
        while True:
            try:
                data = await self.get_atm_iv(currency)
                if data and callback:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
            except Exception as e:
                logger.error("Error polling IV (%s): %s", type(e).__name__, e)
                logger.debug("Deribit IV polling error detail", exc_info=True)
            
            await asyncio.sleep(interval_seconds)

    def get_health_status(self) -> Dict[str, Any]:
        """Return health for dashboard."""
        import time
        now = time.time()
        age = (now - self.last_message_ts) if self.last_message_ts else None
        if self.consecutive_failures > 0:
            status = "degraded"
        elif self.last_success_ts:
            status = "ok"
        else:
            status = "unknown"

        from ..core.status import build_status

        return build_status(
            name="deribit",
            type="rest",
            status=status,
            last_success_ts=self.last_success_ts,
            last_error=self.last_error,
            consecutive_failures=self.consecutive_failures,
            reconnect_attempts=self.reconnect_attempts,
            request_count=self._request_count_total,
            error_count=self.error_count,
            avg_latency_ms=round(self.avg_latency_ms, 2) if self.avg_latency_ms is not None else None,
            last_latency_ms=round(self.last_latency_ms, 2) if self.last_latency_ms is not None else None,
            last_poll_ts=self.last_poll_ts,
            age_seconds=round(age, 1) if age is not None else None,
            extras={
                "connected": self._session is not None and not self._session.closed,
                "rate_limit_remaining": max(self._rate_limit - self._request_count, 0),
                "last_http_status": self.last_http_status,
            },
        )
