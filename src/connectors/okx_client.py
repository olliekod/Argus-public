"""
OKX REST API Client
===================

REST client for OKX market data and funding rates.
"""

import asyncio
import hmac
import hashlib
import base64
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import aiohttp

from ..core.logger import get_connector_logger

logger = get_connector_logger('okx')


class OKXClient:
    """
    OKX REST API client for market data.
    
    Provides:
    - Funding rate data
    - Ticker/price data
    - Open interest
    """
    
    BASE_URL = "https://www.okx.com"
    
    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        passphrase: str = "",
    ):
        """
        Initialize OKX client.
        
        Args:
            api_key: OKX API key
            api_secret: OKX API secret
            passphrase: OKX API passphrase
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limit = 5  # requests per second
        self._last_request_time = 0
        self.last_message_ts: Optional[float] = None
        self.last_success_ts: Optional[float] = None
        self.last_error: Optional[str] = None
        self.consecutive_failures: int = 0
        self.reconnect_attempts: int = 0
        self.request_count: int = 0
        self.error_count: int = 0
        self.last_latency_ms: Optional[float] = None
        self.avg_latency_ms: Optional[float] = None
        self.last_poll_ts: Optional[float] = None
        self.last_http_status: Optional[int] = None
        
        logger.info("OKX client initialized")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self) -> None:
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    def _sign_request(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        """Generate request signature."""
        message = timestamp + method + path + body
        mac = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        )
        return base64.b64encode(mac.digest()).decode()
    
    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict] = None,
        signed: bool = False
    ) -> Dict:
        """Make an API request."""
        # Rate limiting
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < (1 / self._rate_limit):
            await asyncio.sleep((1 / self._rate_limit) - elapsed)
        self._last_request_time = time.time()
        
        session = await self._get_session()
        url = f"{self.BASE_URL}{path}"
        start = asyncio.get_running_loop().time()
        self.last_poll_ts = time.time()
        self.request_count += 1
        
        headers = {
            "Content-Type": "application/json",
        }
        
        if signed and self.api_key:
            timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.') + \
                       f'{datetime.now(timezone.utc).microsecond // 1000:03d}Z'
            signature = self._sign_request(timestamp, method.upper(), path)
            headers.update({
                "OK-ACCESS-KEY": self.api_key,
                "OK-ACCESS-SIGN": signature,
                "OK-ACCESS-TIMESTAMP": timestamp,
                "OK-ACCESS-PASSPHRASE": self.passphrase,
            })
        
        try:
            async with session.request(method, url, params=params, headers=headers) as resp:
                self.last_http_status = resp.status
                data = await resp.json()
                
                if data.get('code') != '0':
                    self.last_error = str(data.get('msg'))
                    self.error_count += 1
                    self.consecutive_failures += 1
                    logger.warning(f"OKX API error: {data.get('msg')}")
                else:
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
            logger.error(f"OKX request failed: {e}")
            return {'code': '-1', 'msg': str(e), 'data': []}
    
    async def get_funding_rate(self, symbol: str) -> Optional[Dict]:
        """
        Get current funding rate for a symbol.
        
        Args:
            symbol: Trading pair (e.g., 'BTC-USDT-SWAP')
            
        Returns:
            Funding rate data dict or None
        """
        # Convert symbol format
        inst_id = self._to_okx_symbol(symbol)
        
        data = await self._request(
            'GET',
            '/api/v5/public/funding-rate',
            params={'instId': inst_id}
        )
        
        if data.get('data'):
            item = data['data'][0]
            return {
                'symbol': symbol,
                'exchange': 'okx',
                'funding_rate': float(item.get('fundingRate', 0)),
                'next_funding_rate': float(item.get('nextFundingRate', 0)) if item.get('nextFundingRate') else None,
                'funding_time': item.get('fundingTime'),
                'timestamp': datetime.now(timezone.utc).isoformat(),
            }
        return None
    
    async def get_funding_rate_history(
        self,
        symbol: str,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get historical funding rates.
        
        Args:
            symbol: Trading pair
            limit: Number of records
            
        Returns:
            List of funding rate records
        """
        inst_id = self._to_okx_symbol(symbol)
        
        data = await self._request(
            'GET',
            '/api/v5/public/funding-rate-history',
            params={'instId': inst_id, 'limit': str(limit)}
        )
        
        result = []
        for item in data.get('data', []):
            result.append({
                'symbol': symbol,
                'exchange': 'okx',
                'funding_rate': float(item.get('fundingRate', 0)),
                'funding_time': item.get('fundingTime'),
                'realized_rate': float(item.get('realizedRate', 0)) if item.get('realizedRate') else None,
            })
        
        return result
    
    async def get_ticker(self, symbol: str) -> Optional[Dict]:
        """
        Get ticker data for a symbol.
        
        Args:
            symbol: Trading pair
            
        Returns:
            Ticker data dict or None
        """
        inst_id = self._to_okx_symbol(symbol)
        
        data = await self._request(
            'GET',
            '/api/v5/market/ticker',
            params={'instId': inst_id}
        )
        
        if data.get('data'):
            item = data['data'][0]
            return {
                'symbol': symbol,
                'exchange': 'okx',
                'last_price': float(item.get('last', 0)),
                'bid_price': float(item.get('bidPx', 0)),
                'ask_price': float(item.get('askPx', 0)),
                'volume_24h': float(item.get('vol24h', 0)),
                'volume_24h_ccy': float(item.get('volCcy24h', 0)),
                'high_24h': float(item.get('high24h', 0)),
                'low_24h': float(item.get('low24h', 0)),
                'timestamp': datetime.now(timezone.utc).isoformat(),
            }
        return None
    
    async def get_open_interest(self, symbol: str) -> Optional[Dict]:
        """
        Get open interest for a symbol.
        
        Args:
            symbol: Trading pair
            
        Returns:
            Open interest data or None
        """
        inst_id = self._to_okx_symbol(symbol)
        
        data = await self._request(
            'GET',
            '/api/v5/public/open-interest',
            params={'instId': inst_id}
        )
        
        if data.get('data'):
            item = data['data'][0]
            return {
                'symbol': symbol,
                'exchange': 'okx',
                'open_interest': float(item.get('oi', 0)),
                'open_interest_ccy': float(item.get('oiCcy', 0)),
                'timestamp': datetime.now(timezone.utc).isoformat(),
            }
        return None
    
    def _to_okx_symbol(self, symbol: str) -> str:
        """
        Convert unified symbol to OKX format.
        
        'BTC/USDT:USDT' -> 'BTC-USDT-SWAP'
        'BTC/USDT' -> 'BTC-USDT'
        """
        is_perp = ':' in symbol
        
        if is_perp:
            symbol = symbol.split(':')[0]
        
        base_quote = symbol.replace('/', '-')
        
        if is_perp:
            return f"{base_quote}-SWAP"
        return base_quote
    
    async def poll_funding_rates(
        self,
        symbols: List[str],
        interval_seconds: int = 300,
        callback=None
    ) -> None:
        """
        Continuously poll funding rates.
        
        Args:
            symbols: List of symbols to poll
            interval_seconds: Polling interval
            callback: Function to call with data
        """
        logger.info(f"Starting funding rate polling for {len(symbols)} symbols")
        
        while True:
            for symbol in symbols:
                try:
                    data = await self.get_funding_rate(symbol)
                    if data and callback:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(data)
                        else:
                            callback(data)
                except Exception as e:
                    logger.error(f"Error polling {symbol}: {e}")
            
            await asyncio.sleep(interval_seconds)

    def get_health_status(self) -> Dict[str, Any]:
        """Return health for dashboard."""
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
            name="okx",
            type="rest",
            status=status,
            last_success_ts=self.last_success_ts,
            last_error=self.last_error,
            consecutive_failures=self.consecutive_failures,
            reconnect_attempts=self.reconnect_attempts,
            request_count=self.request_count,
            error_count=self.error_count,
            avg_latency_ms=round(self.avg_latency_ms, 2) if self.avg_latency_ms is not None else None,
            last_latency_ms=round(self.last_latency_ms, 2) if self.last_latency_ms is not None else None,
            last_poll_ts=self.last_poll_ts,
            age_seconds=round(age, 1) if age is not None else None,
            extras={
                "connected": self._session is not None and not self._session.closed,
                "last_http_status": self.last_http_status,
            },
        )
