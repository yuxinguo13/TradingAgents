"""US equity market clock: what session are we in, and when does it change.

The monitor loop is always on, but *what it should do* depends entirely on the
session. Pre-market news matters and cannot be traded on at Investopedia's
simulated fills; a 15:55 signal is a different trade from a 10:05 signal; and
polling yfinance every 30s over a weekend is pure waste.

Holidays are the NYSE calendar, hardcoded through 2027. A hardcoded table beats
a dependency here: the list is short, changes once a year, and a wrong holiday
costs one idle day rather than a bad fill. ``pandas_market_calendars`` would be
correct too, but it is a heavy dep for a lookup table.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

PREMARKET_OPEN = time(4, 0)
REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)
AFTERHOURS_CLOSE = time(20, 0)

# NYSE full-day closures. Half-days (1pm close) are listed separately because
# the agent must not sit waiting for a 16:00 close that never comes.
HOLIDAYS: frozenset[date] = frozenset({
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
    date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7),
    date(2026, 11, 26), date(2026, 12, 25),
    date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15), date(2027, 3, 26),
    date(2027, 5, 31), date(2027, 6, 18), date(2027, 7, 5), date(2027, 9, 6),
    date(2027, 11, 25), date(2027, 12, 24),
})

HALF_DAYS: frozenset[date] = frozenset({
    date(2026, 11, 27), date(2026, 12, 24),
    date(2027, 11, 26),
})

# Session names, ordered through the day.
CLOSED = "closed"
PRE = "premarket"
OPEN = "open"
AFTER = "afterhours"


@dataclass(frozen=True)
class MarketState:
    session: str
    now: datetime          # timezone-aware, Eastern
    trading_day: date      # the session date this belongs to
    minutes_to_close: float | None
    minutes_from_open: float | None

    @property
    def is_open(self) -> bool:
        return self.session == OPEN

    @property
    def is_tradeable(self) -> bool:
        """Investopedia fills at the regular-session price; only OPEN counts."""
        return self.session == OPEN

    def phase(self) -> str:
        """Coarse intraday phase, used to bias how aggressive the desk is.

        The open and the close are structurally different from midday: the
        first 30 minutes carry overnight gap repricing and the widest spreads,
        the last 30 carry index rebalancing and closing auctions. Naming the
        phase lets the reasoning layer know which one it is looking at instead
        of inferring it from a raw timestamp.
        """
        if self.session != OPEN:
            return self.session
        mo = self.minutes_from_open or 0
        mc = self.minutes_to_close or 0
        if mo <= 30:
            return "opening_drive"
        if mc <= 30:
            return "closing_drive"
        if mc <= 90:
            return "late_session"
        return "midday"


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in HOLIDAYS


def close_time(d: date) -> time:
    return time(13, 0) if d in HALF_DAYS else REGULAR_CLOSE


def market_state(now: datetime | None = None) -> MarketState:
    """Classify ``now`` (default: real time) into a session."""
    now = now.astimezone(ET) if now else datetime.now(ET)
    d, t = now.date(), now.time()

    if not is_trading_day(d):
        return MarketState(CLOSED, now, d, None, None)

    end = close_time(d)
    if t < PREMARKET_OPEN:
        return MarketState(CLOSED, now, d, None, None)
    if t < REGULAR_OPEN:
        return MarketState(PRE, now, d, None, None)
    if t < end:
        open_dt = datetime.combine(d, REGULAR_OPEN, tzinfo=ET)
        close_dt = datetime.combine(d, end, tzinfo=ET)
        return MarketState(
            OPEN, now, d,
            minutes_to_close=(close_dt - now).total_seconds() / 60,
            minutes_from_open=(now - open_dt).total_seconds() / 60,
        )
    if t < AFTERHOURS_CLOSE:
        return MarketState(AFTER, now, d, None, None)
    return MarketState(CLOSED, now, d, None, None)


def next_open(now: datetime | None = None) -> datetime:
    """The next regular-session open at or after ``now``."""
    now = now.astimezone(ET) if now else datetime.now(ET)
    d = now.date()
    if is_trading_day(d) and now.time() < REGULAR_OPEN:
        return datetime.combine(d, REGULAR_OPEN, tzinfo=ET)
    d += timedelta(days=1)
    while not is_trading_day(d):
        d += timedelta(days=1)
    return datetime.combine(d, REGULAR_OPEN, tzinfo=ET)


def seconds_until_open(now: datetime | None = None) -> float:
    now = now.astimezone(ET) if now else datetime.now(ET)
    return max(0.0, (next_open(now) - now).total_seconds())


def last_trading_day(now: datetime | None = None) -> date:
    """Most recent day whose regular session has *started*.

    This is the date the data layer should be asked for: before 09:30 the
    current day has no bar yet, so asking for it returns yesterday's data
    labelled with today's date, which is exactly the look-ahead confusion the
    verified loader exists to prevent.
    """
    now = now.astimezone(ET) if now else datetime.now(ET)
    d = now.date()
    if is_trading_day(d) and now.time() >= REGULAR_OPEN:
        return d
    d -= timedelta(days=1)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d
