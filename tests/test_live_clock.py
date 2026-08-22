"""Market-clock classification: the loop's cadence depends entirely on this."""

from datetime import date, datetime

import pytest

from tradingagents.live import clock


def et(y, m, d, hh, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=clock.ET)


@pytest.mark.unit
class TestSessions:
    def test_regular_session_is_open(self):
        s = clock.market_state(et(2026, 8, 20, 11, 0))
        assert s.session == clock.OPEN
        assert s.is_tradeable

    def test_premarket_is_not_tradeable(self):
        s = clock.market_state(et(2026, 8, 20, 7, 30))
        assert s.session == clock.PRE
        assert not s.is_tradeable

    def test_afterhours_is_not_tradeable(self):
        s = clock.market_state(et(2026, 8, 20, 17, 0))
        assert s.session == clock.AFTER
        assert not s.is_tradeable

    def test_weekend_is_closed(self):
        assert clock.market_state(et(2026, 8, 22, 11, 0)).session == clock.CLOSED

    def test_holiday_is_closed(self):
        # Christmas 2026 falls on a Friday; a weekday check alone would pass it.
        assert clock.market_state(et(2026, 12, 25, 11, 0)).session == clock.CLOSED

    def test_half_day_closes_early(self):
        # Black Friday 2026: 13:00 close. 14:00 must be after-hours, not open.
        assert clock.market_state(et(2026, 11, 27, 14, 0)).session == clock.AFTER
        assert clock.market_state(et(2026, 11, 27, 12, 0)).session == clock.OPEN


@pytest.mark.unit
class TestPhases:
    def test_opening_drive(self):
        assert clock.market_state(et(2026, 8, 20, 9, 45)).phase() == "opening_drive"

    def test_midday(self):
        assert clock.market_state(et(2026, 8, 20, 12, 0)).phase() == "midday"

    def test_closing_drive(self):
        assert clock.market_state(et(2026, 8, 20, 15, 45)).phase() == "closing_drive"

    def test_late_session(self):
        assert clock.market_state(et(2026, 8, 20, 15, 0)).phase() == "late_session"


@pytest.mark.unit
class TestDataDate:
    def test_before_open_uses_previous_session(self):
        # 08:00 Thursday: today has no bar yet, so the data date is Wednesday.
        assert clock.last_trading_day(et(2026, 8, 20, 8, 0)) == date(2026, 8, 19)

    def test_after_open_uses_today(self):
        assert clock.last_trading_day(et(2026, 8, 20, 10, 0)) == date(2026, 8, 20)

    def test_weekend_walks_back_to_friday(self):
        assert clock.last_trading_day(et(2026, 8, 23, 10, 0)) == date(2026, 8, 21)

    def test_next_open_skips_weekend(self):
        assert clock.next_open(et(2026, 8, 21, 17, 0)) == et(2026, 8, 24, 9, 30)

    def test_next_open_skips_holiday(self):
        # 2026-12-24 is a half day; 12-25 is a holiday, so the next open is 12-28.
        assert clock.next_open(et(2026, 12, 24, 14, 0)) == et(2026, 12, 28, 9, 30)
