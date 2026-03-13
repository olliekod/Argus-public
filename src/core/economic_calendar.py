"""
Economic Calendar
=================

Tracks high-impact economic events for trading blackout periods.
Prevents entering trades before FOMC, CPI, and Jobs reports.
"""

import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class EventRisk(Enum):
    """Risk level for economic events."""
    CRITICAL = "CRITICAL"   # FOMC - 48h blackout
    HIGH = "HIGH"           # CPI - 24h blackout
    MEDIUM = "MEDIUM"       # Jobs - 12h blackout


@dataclass
class EconomicEvent:
    """Represents an economic event."""
    name: str
    date: datetime
    risk: EventRisk
    blackout_hours: int
    description: str = ""


class EconomicCalendar:
    """
    Tracks economic events and blackout periods.
    
    Blackout periods:
    - FOMC: 48 hours before meeting end (Wednesday 2 PM EST)
    - CPI: 24 hours before release (8:30 AM EST)
    - Jobs: 12 hours before release (8:30 AM EST first Friday)
    """
    
    # 2025 FOMC Meeting Dates (month, day) - Wednesday end dates
    # Source: Federal Reserve Schedule
    FOMC_2025 = [
        (1, 29),   # Jan 28-29
        (3, 19),   # Mar 18-19
        (5, 7),    # May 6-7
        (6, 18),   # Jun 17-18
        (7, 30),   # Jul 29-30
        (9, 17),   # Sep 16-17
        (11, 5),   # Nov 4-5
        (12, 17),  # Dec 16-17
    ]
    
    # 2026 FOMC Meeting Dates (month, day) - Wednesday end dates
    # Source: Federal Reserve Schedule
    FOMC_2026 = [
        (1, 28),   # Jan 27-28
        (3, 18),   # Mar 17-18
        (5, 6),    # May 5-6
        (6, 17),   # Jun 16-17
        (7, 29),   # Jul 28-29
        (9, 16),   # Sep 15-16
        (11, 4),   # Nov 3-4
        (12, 16),  # Dec 15-16
    ]
    
    # 2025 CPI Release Dates (approximate)
    CPI_2025 = [
        (1, 15), (2, 12), (3, 12), (4, 10),
        (5, 13), (6, 11), (7, 11), (8, 13),
        (9, 11), (10, 10), (11, 13), (12, 11),
    ]
    
    # 2026 CPI Release Dates (approximate)
    CPI_2026 = [
        (1, 14), (2, 12), (3, 11), (4, 15),
        (5, 13), (6, 10), (7, 15), (8, 12),
        (9, 16), (10, 14), (11, 12), (12, 9),
    ]
    
    # Map years to their event schedules
    FOMC_DATES = {2025: FOMC_2025, 2026: FOMC_2026}
    CPI_DATES = {2025: CPI_2025, 2026: CPI_2026}

    # Jobs Report: First Friday of each month, 8:30 AM EST
    # This is calculated dynamically

    def __init__(self):
        """Initialize economic calendar."""
        self._events_cache: Optional[List[EconomicEvent]] = None
        self._events_cache_year: Optional[int] = None
        logger.info("Economic Calendar initialized")

    @classmethod
    def _estimate_fomc_dates(cls, year: int) -> List[tuple]:
        """
        P3: Estimate FOMC meeting dates for years beyond hardcoded data.

        The Fed typically meets 8 times per year roughly every 6 weeks.
        This provides a reasonable approximation based on historical patterns.
        """
        # Typical FOMC months and approximate Wednesday end dates
        return [
            (1, 29), (3, 19), (5, 7), (6, 18),
            (7, 30), (9, 17), (11, 5), (12, 17),
        ]

    @classmethod
    def _estimate_cpi_dates(cls, year: int) -> List[tuple]:
        """
        P3: Estimate CPI release dates for years beyond hardcoded data.

        CPI is typically released on the 2nd or 3rd Tuesday-Wednesday
        of each month. This approximates mid-month.
        """
        return [
            (1, 14), (2, 12), (3, 12), (4, 10),
            (5, 13), (6, 11), (7, 15), (8, 12),
            (9, 11), (10, 14), (11, 12), (12, 10),
        ]
    
    def _get_first_friday(self, year: int, month: int) -> datetime:
        """Get first Friday of a given month."""
        first_day = datetime(year, month, 1)
        # weekday(): Monday=0, Tuesday=1, ..., Friday=4
        days_until_friday = (4 - first_day.weekday()) % 7
        return first_day + timedelta(days=days_until_friday)
    
    def get_all_events(self, year: int = 2026) -> List[EconomicEvent]:
        """
        Get all economic events for the year.
        
        Args:
            year: Year to get events for
            
        Returns:
            List of EconomicEvent objects sorted by date
        """
        if self._events_cache and self._events_cache_year == year:
            return self._events_cache

        events = []

        # P3: Dynamic FOMC dates — use hardcoded if available, else estimate
        fomc_dates = self.FOMC_DATES.get(year, None) or self._estimate_fomc_dates(year)
        for month, day in fomc_dates:
            try:
                dt = datetime(year, month, day, 14, 0)  # 2 PM
                events.append(EconomicEvent(
                    name="FOMC",
                    date=dt,
                    risk=EventRisk.CRITICAL,
                    blackout_hours=48,
                    description="Federal Reserve rate decision"
                ))
            except ValueError:
                continue
        
        # P3: Dynamic CPI dates — use hardcoded if available, else estimate
        cpi_dates = self.CPI_DATES.get(year, None) or self._estimate_cpi_dates(year)
        for month, day in cpi_dates:
            try:
                dt = datetime(year, month, day, 8, 30)
                events.append(EconomicEvent(
                    name="CPI",
                    date=dt,
                    risk=EventRisk.HIGH,
                    blackout_hours=24,
                    description="Consumer Price Index"
                ))
            except ValueError:
                continue
        
        # Jobs reports (first Friday, 8:30 AM EST)
        for month in range(1, 13):
            friday = self._get_first_friday(year, month)
            friday = friday.replace(hour=8, minute=30)
            events.append(EconomicEvent(
                name="Jobs",
                date=friday,
                risk=EventRisk.MEDIUM,
                blackout_hours=12,
                description="Non-Farm Payrolls"
            ))
        
        # Sort by date
        events.sort(key=lambda e: e.date)
        self._events_cache = events
        self._events_cache_year = year

        return events
    
    def get_next_event(self, from_date: datetime = None) -> Optional[EconomicEvent]:
        """
        Get the next upcoming economic event.
        
        Args:
            from_date: Date to search from (default: now)
            
        Returns:
            Next EconomicEvent or None
        """
        if from_date is None:
            from_date = datetime.now()
        
        events = self.get_all_events(from_date.year)
        
        for event in events:
            if event.date > from_date:
                return event
        
        # Check next year
        next_year_events = self.get_all_events(from_date.year + 1)
        for event in next_year_events:
            if event.date > from_date:
                return event
        
        return None
    
    def is_blackout_period(self, check_time: datetime = None) -> Tuple[bool, Optional[EconomicEvent]]:
        """
        Check if we're in a blackout period.
        
        Args:
            check_time: Time to check (default: now)
            
        Returns:
            Tuple of (is_blackout, event_causing_blackout)
        """
        if check_time is None:
            check_time = datetime.now()
        
        events = self.get_all_events(check_time.year)
        
        for event in events:
            blackout_start = event.date - timedelta(hours=event.blackout_hours)
            
            if blackout_start <= check_time <= event.date:
                return (True, event)
        
        return (False, None)
    
    def get_blackout_warning(self, check_time: datetime = None) -> Optional[str]:
        """
        Get a warning message if in or near a blackout period.
        
        Args:
            check_time: Time to check (default: now)
            
        Returns:
            Warning string or None
        """
        is_blackout, event = self.is_blackout_period(check_time)
        
        if is_blackout:
            hours_until = (event.date - (check_time or datetime.now())).total_seconds() / 3600
            return f"⚠️ {event.name} in {hours_until:.0f}h - REDUCE POSITION SIZE OR WAIT"
        
        # Check if within 72 hours of any event (extended caution)
        next_event = self.get_next_event(check_time)
        if next_event:
            hours_until = (next_event.date - (check_time or datetime.now())).total_seconds() / 3600
            if hours_until <= 72:
                return f"⚠️ {next_event.name} in {hours_until:.0f}h - BE CAUTIOUS"
        
        return None
    
    def get_events_this_week(self, from_date: datetime = None) -> List[EconomicEvent]:
        """
        Get all events in the next 7 days.
        
        Args:
            from_date: Start date (default: now)
            
        Returns:
            List of events in the next 7 days
        """
        if from_date is None:
            from_date = datetime.now()
        
        end_date = from_date + timedelta(days=7)
        events = self.get_all_events(from_date.year)
        
        return [e for e in events if from_date <= e.date <= end_date]
    
    def format_weekly_summary(self, from_date: datetime = None) -> str:
        """
        Format a weekly summary of upcoming events.
        
        Returns:
            Formatted string for Telegram/console
        """
        events = self.get_events_this_week(from_date)
        
        if not events:
            return "📅 *This Week:* No major events"
        
        lines = ["📅 *This Week's Events:*"]
        
        for event in events:
            risk_emoji = {
                EventRisk.CRITICAL: "🔴",
                EventRisk.HIGH: "🟠",
                EventRisk.MEDIUM: "🟡",
            }.get(event.risk, "⚪")
            
            date_str = event.date.strftime("%a %m/%d %I:%M %p")
            lines.append(f"  {risk_emoji} {event.name}: {date_str}")
        
        return "\n".join(lines)


# Test function
def test_calendar():
    """Test the economic calendar."""
    print("=" * 60)
    print("ECONOMIC CALENDAR TEST")
    print("=" * 60)
    
    cal = EconomicCalendar()
    
    # Get all events
    events = cal.get_all_events()
    print(f"\n2026 Events: {len(events)} total")
    
    # Show first 10
    print("\nUpcoming Events:")
    for event in events[:10]:
        print(f"  {event.date.strftime('%Y-%m-%d %H:%M')} - {event.name} ({event.risk.value})")
    
    # Next event
    next_event = cal.get_next_event()
    if next_event:
        print(f"\nNext Event: {next_event.name} on {next_event.date}")
    
    # Blackout check
    is_blackout, event = cal.is_blackout_period()
    print(f"\nCurrently in blackout: {is_blackout}")
    if event:
        print(f"  Caused by: {event.name}")
    
    # Warning
    warning = cal.get_blackout_warning()
    if warning:
        print(f"\nWarning: {warning}")
    
    # Weekly summary
    print(f"\n{cal.format_weekly_summary()}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_calendar()
