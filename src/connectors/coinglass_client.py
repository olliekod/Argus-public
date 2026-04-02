# Created by Oliver Meihls

# Coinglass API Client
#
# REST client for Coinglass liquidation data.

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import aiohttp

from ..core.logger import get_connector_logger

logger = get_connector_logger('coinglass')


class CoinglassClient:
    # Coinglass API client for liquidation data.
    #
    # Provides:
    # - Liquidation events across exchanges
    # - Aggregated liquidation statistics
    #
    # Requires API key (free tier available).
    
    BASE_URL = "https://open-api.coinglass.com/public/v2"
    
    def __init__(self, api_key: str):
        # Initialize Coinglass client.
        #
        # Args:
        # api_key: Coinglass API key
        self.api_key = api_key
        self._session: Optional[aiohttp.ClientSession] = None
        self._upgrade_warning_shown = False  # Only show upgrade warning once
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
        
        logger.info("Coinglass client initialized")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        # Get or create HTTP session.
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=10))
        return self._session
    
    async def close(self) -> None:
        # Close HTTP session.
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def _request(self, endpoint: str, params: Dict = None) -> Dict:
        # Make an API request.
        session = await self._get_session()
        url = f"{self.BASE_URL}/{endpoint}"
        start = asyncio.get_running_loop().time()
        self.last_poll_ts = datetime.now(timezone.utc).timestamp()
        self.request_count += 1
        
        headers = {
            "accept": "application/json",
            "coinglassSecret": self.api_key,
        }
        
        try:
            async with session.get(url, params=params, headers=headers) as resp:
                self.last_http_status = resp.status
                data = await resp.json()
                
                if data.get('code') != '0' and data.get('success') is not True:
                    msg = data.get('msg', '')
                    self.last_error = str(msg)
                    self.error_count += 1
                    self.consecutive_failures += 1
                    # Only show upgrade warning once to avoid log spam
                    if 'Upgrade' in str(msg):
                        if not self._upgrade_warning_shown:
                            logger.warning("Coinglass free tier limited - liquidation data may be incomplete")
                            self._upgrade_warning_shown = True
                    else:
                        logger.warning(f"Coinglass API warning: {msg}")
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
            logger.error(f"Coinglass request failed: {e}")
            return {'code': '-1', 'msg': str(e), 'data': None}
    
    async def get_liquidation_history(
        self,
        symbol: str = "BTC",
        time_type: str = "h1"  # h1, h4, h12, h24
    ) -> List[Dict]:
        # Get liquidation history.
        #
        # Args:
        # symbol: Coin symbol (BTC, ETH, etc.)
        # time_type: Time interval
        #
        # Returns:
        # List of liquidation records
        data = await self._request(
            'liquidation_history',
            {'symbol': symbol, 'time_type': time_type}
        )
        
        result = []
        for item in data.get('data', []) or []:
            result.append({
                'symbol': symbol,
                'timestamp': datetime.fromtimestamp(item.get('time', 0) / 1000).isoformat() if item.get('time') else None,
                'long_liquidations': float(item.get('longLiquidationUsd', 0)),
                'short_liquidations': float(item.get('shortLiquidationUsd', 0)),
                'total_liquidations': float(item.get('longLiquidationUsd', 0)) + float(item.get('shortLiquidationUsd', 0)),
            })
        
        return result
    
    async def get_liquidation_aggregated(
        self,
        symbol: str = "BTC"
    ) -> Optional[Dict]:
        # Get aggregated liquidation data across exchanges.
        #
        # Args:
        # symbol: Coin symbol
        #
        # Returns:
        # Aggregated liquidation data
        data = await self._request('liquidation_info', {'symbol': symbol})
        
        if data.get('data'):
            item = data['data']
            
            total_long = 0
            total_short = 0
            by_exchange = {}
            
            # Parse exchange-level data if available
            for exchange_data in item if isinstance(item, list) else [item]:
                if isinstance(exchange_data, dict):
                    exchange = exchange_data.get('exchangeName', 'unknown')
                    long_liq = float(exchange_data.get('longLiquidationUsd', 0) or 0)
                    short_liq = float(exchange_data.get('shortLiquidationUsd', 0) or 0)
                    
                    total_long += long_liq
                    total_short += short_liq
                    by_exchange[exchange] = {
                        'long': long_liq,
                        'short': short_liq,
                        'total': long_liq + short_liq
                    }
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'total_long_usd': total_long,
                'total_short_usd': total_short,
                'total_usd': total_long + total_short,
                'by_exchange': by_exchange,
            }
        return None
    
    async def get_liquidation_chart(
        self,
        symbol: str = "BTC",
        interval: str = "h1"
    ) -> List[Dict]:
        # Get liquidation chart data.
        #
        # Args:
        # symbol: Coin symbol
        # interval: Time interval
        #
        # Returns:
        # Chart data points
        data = await self._request(
            'liquidation_chart',
            {'symbol': symbol, 'interval': interval}
        )
        
        result = []
        for item in data.get('data', []) or []:
            result.append({
                'symbol': symbol,
                'timestamp': datetime.fromtimestamp(item.get('t', 0) / 1000).isoformat() if item.get('t') else None,
                'price': float(item.get('p', 0)),
                'long_liquidations': float(item.get('l', 0)),
                'short_liquidations': float(item.get('s', 0)),
            })
        
        return result
    
    async def check_liquidation_cascade(
        self,
        symbol: str = "BTC",
        threshold_usd: float = 5_000_000,
        time_window_minutes: int = 5
    ) -> Optional[Dict]:
        # Check if a liquidation cascade is occurring.
        #
        # Args:
        # symbol: Coin symbol
        # threshold_usd: Minimum liquidation amount to trigger
        # time_window_minutes: Time window to check
        #
        # Returns:
        # Cascade info if detected, None otherwise
        history = await self.get_liquidation_history(symbol, 'h1')
        
        if not history:
            return None
        
        # Get recent liquidations (last entry is most recent)
        recent = history[-1] if history else None
        
        if recent and recent['total_liquidations'] >= threshold_usd:
            dominant_side = 'long' if recent['long_liquidations'] > recent['short_liquidations'] else 'short'
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'cascade_detected': True,
                'total_liquidations_usd': recent['total_liquidations'],
                'long_liquidations_usd': recent['long_liquidations'],
                'short_liquidations_usd': recent['short_liquidations'],
                'dominant_side': dominant_side,
                'threshold_usd': threshold_usd,
            }
        
        return None
    
    async def poll_liquidations(
        self,
        symbols: List[str],
        interval_seconds: int = 30,
        callback=None
    ) -> None:
        # Continuously poll for liquidation cascades.
        #
        # Args:
        # symbols: Symbols to monitor
        # interval_seconds: Polling interval
        # callback: Function to call when cascade detected
        logger.info(f"Starting liquidation polling for {len(symbols)} symbols")
        
        while True:
            for symbol in symbols:
                try:
                    cascade = await self.check_liquidation_cascade(symbol)
                    if cascade and callback:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(cascade)
                        else:
                            callback(cascade)
                except Exception as e:
                    logger.error(f"Error polling liquidations for {symbol}: {e}")
            
            await asyncio.sleep(interval_seconds)

    def get_health_status(self) -> Dict[str, Any]:
        # Return health for dashboard.
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
            name="coinglass",
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
