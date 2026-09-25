"""The morning wait: wall-clock time, and the three answers the launcher reads."""

from datetime import datetime, timedelta

import pytest

from tradingagents.live import clock, waitopen

ET = clock.ET


class FakeClock:
    """A wall clock that can jump forward the way a sleeping Mac does."""

    def __init__(self, start: datetime, jump: timedelta | None = None):
        self.t = start
        self.jump = jump          # a single oversleep applied on the first sleep
        self.slept: list[float] = []

    def now(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        if self.jump is not None:
            self.t += self.jump
            self.jump = None
        else:
            self.t += timedelta(seconds=s)


def run(fc):
    lines = []
    rc = waitopen.wait_for_open(now=fc.now, sleep=fc.sleep, out=lines.append)
    return rc, lines


@pytest.mark.unit
class TestWaitForOpen:
    def test_a_morning_start_waits_until_five_past_the_open(self):
        fc = FakeClock(datetime(2026, 9, 24, 9, 25, tzinfo=ET))   # Thursday
        rc, _ = run(fc)
        assert rc == waitopen.PLACE
        assert fc.t >= datetime(2026, 9, 24, 9, 35, tzinfo=ET)
        assert max(fc.slept) <= waitopen.SLICE

    def test_a_start_mid_session_places_at_once(self):
        fc = FakeClock(datetime(2026, 9, 24, 11, 0, tzinfo=ET))
        rc, _ = run(fc)
        assert rc == waitopen.PLACE and fc.slept == []

    def test_a_holiday_places_nothing(self):
        fc = FakeClock(datetime(2026, 11, 26, 9, 25, tzinfo=ET))  # Thanksgiving
        rc, lines = run(fc)
        assert rc == waitopen.NO_SESSION and "nothing placed" in lines[0]

    def test_a_start_after_the_close_places_nothing(self):
        fc = FakeClock(datetime(2026, 9, 24, 16, 20, tzinfo=ET))
        rc, _ = run(fc)
        assert rc == waitopen.NO_SESSION

    def test_sleeping_through_the_session_is_reported_as_missed(self):
        """2026-09-15: ten minutes of waiting became seven hours."""
        fc = FakeClock(datetime(2026, 9, 24, 9, 25, tzinfo=ET), jump=timedelta(hours=7))
        rc, lines = run(fc)
        assert rc == waitopen.MISSED and "read-only" in lines[-1]

    def test_an_oversleep_that_lands_inside_the_session_still_places(self):
        fc = FakeClock(datetime(2026, 9, 24, 9, 25, tzinfo=ET), jump=timedelta(hours=2))
        rc, _ = run(fc)
        assert rc == waitopen.PLACE
