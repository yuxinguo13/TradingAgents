"""Wait for the open to settle, against the wall clock, and say what happened.

``desk_cron.sh submit`` starts before the open and calls this first. The old
inline wait was one ``time.sleep``, and on macOS that counts only seconds the
machine was awake: on 2026-09-15 a ten-minute wait took seven hours and
``execute`` started twenty minutes after the close. This version sleeps in
short slices and re-reads the wall clock after each one, so a machine that
slept through the target notices as soon as it wakes.

Exit codes, read by the launcher:

    0  the session is open and settled — place
    3  no session opens within twenty minutes (a holiday, a wake after the
       close) — place nothing
    4  the session ended while the machine slept — reconcile read-only
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from datetime import datetime, timedelta

from tradingagents.live import clock

SETTLE = 5 * 60          # seconds after 09:30 ET before placing
MAX_WAIT = 20 * 60       # how far ahead an open may be and still be waited for
SLICE = 30.0             # longest single sleep; the wall clock is re-read after each

PLACE, NO_SESSION, MISSED = 0, 3, 4


def target_time(now: datetime) -> datetime | None:
    """When to place, or None when no session opens soon enough to wait for."""
    st = clock.market_state(now)
    if st.is_open:
        return st.now + timedelta(seconds=max(0.0, SETTLE - (st.minutes_from_open or 0.0) * 60))
    if clock.seconds_until_open(now) > MAX_WAIT:
        return None
    return clock.next_open(now) + timedelta(seconds=SETTLE)


def wait_for_open(
    now: Callable[[], datetime] = lambda: datetime.now(clock.ET),
    sleep: Callable[[float], None] = time.sleep,
    out: Callable[[str], None] = lambda s: print(s, flush=True),
) -> int:
    start = now()
    target = target_time(start)
    if target is None:
        out(f"no session opens within {MAX_WAIT // 60} min "
            f"(next {clock.next_open(start):%a %F %H:%M %Z}); nothing placed")
        return NO_SESSION

    out(f"waiting {max(0.0, (target - start).total_seconds()) / 60:.1f} min "
        f"for the open to settle (until {target:%H:%M %Z})")
    while (remaining := (target - now()).total_seconds()) > 0:
        sleep(min(SLICE, remaining))

    woke = now()
    if clock.market_state(woke).is_open:
        return PLACE
    out(f"the session ended while the machine slept (awake again {woke:%a %H:%M %Z}); "
        "reconciling read-only")
    return MISSED


if __name__ == "__main__":
    sys.exit(wait_for_open())
