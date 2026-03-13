"""
Data Quality Report
===================

Compute quality metrics per provider+symbol from bar data:
- gap_rate: percentage of expected bars that are missing
- staleness_p50/p95: bar age when received (median and 95th percentile)
- missing_intervals: list of [start_ts, end_ts] gaps
- ohlc_violations: count of OHLC invariant failures

Usage:
    from src.analysis.data_quality import DataQualityReport
    report = DataQualityReport(db)
    metrics = await report.compute_metrics(provider="alpaca", symbol="IBIT", days=7)
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("argus.analysis.data_quality")


@dataclass
class QualityMetrics:
    """Quality metrics for a provider+symbol."""
    provider: str
    symbol: str
    period_start: float  # UTC epoch
    period_end: float  # UTC epoch
    
    # Bar coverage
    expected_bars: int = 0
    actual_bars: int = 0
    gap_rate: float = 0.0  # % missing
    
    # Staleness (time from bar open to receive)
    staleness_p50: Optional[float] = None  # seconds
    staleness_p95: Optional[float] = None  # seconds
    
    # Missing intervals
    missing_intervals: List[Tuple[float, float]] = field(default_factory=list)
    total_gap_minutes: float = 0.0
    
    # OHLC violations
    ohlc_violations: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "symbol": self.symbol,
            "period_start": datetime.fromtimestamp(self.period_start, tz=timezone.utc).isoformat(),
            "period_end": datetime.fromtimestamp(self.period_end, tz=timezone.utc).isoformat(),
            "expected_bars": self.expected_bars,
            "actual_bars": self.actual_bars,
            "gap_rate_pct": round(self.gap_rate * 100, 2),
            "staleness_p50_s": round(self.staleness_p50, 2) if self.staleness_p50 else None,
            "staleness_p95_s": round(self.staleness_p95, 2) if self.staleness_p95 else None,
            "missing_intervals": len(self.missing_intervals),
            "total_gap_minutes": round(self.total_gap_minutes, 1),
            "ohlc_violations": self.ohlc_violations,
        }


class DataQualityReport:
    """
    Compute data quality metrics from bar history.
    
    Parameters
    ----------
    db : Database
        Argus database instance.
    """
    
    def __init__(self, db) -> None:
        self._db = db
    
    async def compute_metrics(
        self,
        provider: str,
        symbol: str,
        days: int = 7,
        bar_duration_s: int = 60,
    ) -> QualityMetrics:
        """
        Compute quality metrics for a provider+symbol.
        
        Parameters
        ----------
        provider : str
            Data provider (e.g., "alpaca", "yahoo", "bybit").
        symbol : str
            Symbol (e.g., "IBIT", "BITO").
        days : int
            Number of days to analyze.
        bar_duration_s : int
            Expected bar duration in seconds (default 60 for 1-min bars).
        
        Returns
        -------
        QualityMetrics
            Quality metrics for the specified period.
        """
        now = datetime.now(timezone.utc)
        end_ts = now.timestamp()
        start_ts = (now - timedelta(days=days)).timestamp()
        
        metrics = QualityMetrics(
            provider=provider,
            symbol=symbol,
            period_start=start_ts,
            period_end=end_ts,
        )
        
        # Query bars from database
        bars = await self._fetch_bars(provider, symbol, start_ts, end_ts)
        
        if not bars:
            logger.warning("No bars found for %s/%s in last %d days", provider, symbol, days)
            # Compute expected bars assuming ~6.5 trading hours per day for equities
            trading_hours_per_day = 6.5
            metrics.expected_bars = int(days * trading_hours_per_day * 60 / (bar_duration_s / 60))
            metrics.gap_rate = 1.0  # 100% missing
            return metrics
        
        # Compute expected bars (using trading hours)
        # For equities: 9:30-16:00 ET = 6.5 hours = 390 minutes per day, weekdays only
        # Simplified: use trading_days * 390 bars per day
        weekdays_in_period = self._count_weekdays(start_ts, end_ts)
        trading_minutes_per_day = 390
        metrics.expected_bars = weekdays_in_period * trading_minutes_per_day // (bar_duration_s // 60)
        metrics.actual_bars = len(bars)
        
        if metrics.expected_bars > 0:
            missing = metrics.expected_bars - metrics.actual_bars
            metrics.gap_rate = max(0, missing) / metrics.expected_bars
        
        # Compute staleness (time from bar open to event_ts)
        staleness_values = []
        for bar in bars:
            bar_ts = bar.get("timestamp", 0)
            event_ts = bar.get("event_ts") or bar.get("receive_time") or bar_ts
            if event_ts > bar_ts:
                staleness_values.append(event_ts - bar_ts)
        
        if staleness_values:
            staleness_values.sort()
            metrics.staleness_p50 = statistics.median(staleness_values)
            idx_95 = int(len(staleness_values) * 0.95)
            metrics.staleness_p95 = staleness_values[min(idx_95, len(staleness_values) - 1)]
        
        # Find missing intervals (gaps > 2 * bar_duration)
        bar_timestamps = sorted(bar.get("timestamp", 0) for bar in bars)
        gap_threshold = bar_duration_s * 2
        
        for i in range(1, len(bar_timestamps)):
            gap = bar_timestamps[i] - bar_timestamps[i - 1]
            if gap > gap_threshold:
                metrics.missing_intervals.append(
                    (bar_timestamps[i - 1], bar_timestamps[i])
                )
                metrics.total_gap_minutes += gap / 60
        
        # Check OHLC violations
        for bar in bars:
            o, h, l, c = bar.get("open", 0), bar.get("high", 0), bar.get("low", 0), bar.get("close", 0)
            if l > min(o, c) or h < max(o, c) or l > h:
                metrics.ohlc_violations += 1
        
        return metrics
    
    async def _fetch_bars(
        self,
        provider: str,
        symbol: str,
        start_ts: float,
        end_ts: float,
    ) -> List[Dict[str, Any]]:
        """Fetch bars from database."""
        start_iso = datetime.fromtimestamp(start_ts, tz=timezone.utc).isoformat()
        end_iso = datetime.fromtimestamp(end_ts, tz=timezone.utc).isoformat()

        def _parse_timestamp(value: Any) -> float:
            if value is None:
                return 0.0
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                ts = value
                if ts.endswith("Z"):
                    ts = ts[:-1] + "+00:00"
                try:
                    parsed = datetime.fromisoformat(ts)
                except ValueError:
                    return 0.0
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed.timestamp()
            return 0.0

        def _normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
            row = dict(row)
            row["timestamp"] = _parse_timestamp(row.get("timestamp"))
            event_ts = row.get("last_source_ts")
            if event_ts is None:
                event_ts = row.get("first_source_ts")
            if event_ts is None:
                event_ts = row.get("event_ts")
            row["event_ts"] = _parse_timestamp(event_ts) if event_ts is not None else row["timestamp"]
            return row

        try:
            # Use the database's bar query method
            # Assumes db has a method like get_bars_by_source
            if hasattr(self._db, "get_bars_by_source"):
                rows = await self._db.get_bars_by_source(
                    source=provider,
                    symbol=symbol,
                    start_timestamp=start_iso,
                    end_timestamp=end_iso,
                )
                return [_normalize_row(row) for row in rows]
            elif hasattr(self._db, "fetch_all"):
                query = """
                    SELECT timestamp, symbol, source, open, high, low, close,
                           first_source_ts, last_source_ts
                    FROM market_bars
                    WHERE source = ? AND symbol = ?
                    AND timestamp >= ? AND timestamp <= ?
                    ORDER BY timestamp
                """
                rows = await self._db.fetch_all(query, (provider, symbol, start_iso, end_iso))
                return [_normalize_row(row) for row in rows]
            elif hasattr(self._db, "query"):
                # Fallback: direct query
                query = """
                    SELECT * FROM market_bars 
                    WHERE source = ? AND symbol = ? 
                    AND timestamp >= ? AND timestamp <= ?
                    ORDER BY timestamp
                """
                rows = await self._db.query(query, (provider, symbol, start_iso, end_iso))
                return [_normalize_row(row) for row in rows]
            else:
                logger.warning("Database doesn't support bar queries")
                return []
        except Exception as e:
            logger.error("Failed to fetch bars: %s", e)
            return []
    
    @staticmethod
    def _count_weekdays(start_ts: float, end_ts: float) -> int:
        """Count weekdays (Mon-Fri) between two timestamps."""
        start = datetime.fromtimestamp(start_ts, tz=timezone.utc)
        end = datetime.fromtimestamp(end_ts, tz=timezone.utc)
        count = 0
        current = start
        while current < end:
            if current.weekday() < 5:  # Mon=0, Fri=4
                count += 1
            current += timedelta(days=1)
        return count
    
    async def generate_report(
        self,
        providers: Optional[List[str]] = None,
        symbols: Optional[List[str]] = None,
        days: int = 7,
    ) -> Dict[str, Any]:
        """
        Generate a complete data quality report.
        
        Parameters
        ----------
        providers : list of str or None
            Providers to analyze. None = all known.
        symbols : list of str or None
            Symbols to analyze. None = all known.
        days : int
            Number of days to analyze.
        
        Returns
        -------
        dict
            Complete report with metrics per provider+symbol.
        """
        providers = providers or ["alpaca", "yahoo", "bybit", "tastytrade"]
        symbols = symbols or ["IBIT", "BITO"]
        
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "period_days": days,
            "metrics": [],
        }
        
        for provider in providers:
            for symbol in symbols:
                try:
                    metrics = await self.compute_metrics(provider, symbol, days)
                    report["metrics"].append(metrics.to_dict())
                except Exception as e:
                    logger.error("Failed to compute metrics for %s/%s: %s", provider, symbol, e)
        
        return report
