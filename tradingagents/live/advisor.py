"""One considered list a day: what to buy at the next open, and what to sell.

The monitor loop and this module answer different questions, and running one of
them as a mode of the other makes both worse.

:mod:`monitor` is reactive. It wakes every two minutes, and its unit of work is
an event — a stop breached, a headline that just landed, a name that moved 1.4
ATR. That is the right shape for managing risk on an open book and the wrong
shape for the thing the user actually asked for: *"everyday give me a list of
stocks that I should buy ... give me the price and amount to buy, we can also
set the limit price. Also sell them."* A list is a considered artefact produced
once, off the full day's data, and read once. Produced continuously it stops
being a list and becomes a feed, and a feed of buy ideas is how an account ends
up holding forty names nobody chose.

So this runs once a day, after the close, against the last completed session's
bars, and prints a page for a human to act on at the next open. Those are two
different sessions and the report prints both dates: an idea dated for Monday's
open was produced from Friday's close, and a page that named one date for both
would be claiming it read Friday's close before Friday's open. Nothing here
places an order. Every idea is written into :mod:`recommendations` at the levels
it was issued with, and every exit the review calls for is written back to the
same book, which is what makes the record checkable a month later.

Three ordering decisions in :meth:`DailyAdvisor.run` are load-bearing:

* **Sells are decided before buys.** Two reasons. The proceeds of a sale are
  what fund a purchase, so the buy budget is not knowable until the exits are
  known; and an agent that generates ideas before managing what it already
  holds will accumulate positions forever, because nothing in the buy path ever
  asks what the book already contains.
* **The R filter runs before sizing.** R is made of the three levels and
  nothing else, so no share count can change it. Checking it first rejects a
  name for the price of three subtractions instead of a full sizing pass, and
  the number the report rejects on is the same number it would have ranked on.
* **Policy tilts the ranking; it never creates or destroys a candidate.**
  :func:`~.policy.sector_pressure` says so about itself, and it is right: the
  sector map is a table of hand-written priors, not a model. Here it subtracts
  at most ``MAX_TILT_SHIFT`` from a rank, which reorders neighbours and can
  never put a name on the list that the screen did not.

The honest caveat, spelled out in ``CAVEAT`` and printed at the foot of every
report: this is a shortlist produced by a model reading public information. It
is not advice, it knows nothing about the reader's circumstances, and every
convention in it — the two-ATR stop, the trend-extrapolated target, the minimum
R — is written down in this repository rather than established by anyone. The
track record at the foot of the report is the only part that is checkable,
which is why every idea goes into the book whether it works or not, and why
every exit goes in with it.

    python -m tradingagents.live.advisor --top 8 --risk-pct 1.0
    python -m tradingagents.live.advisor --use-cache --no-llm --dry-run
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import logging
import re
import math
import os
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import date as _date, datetime, timedelta
from pathlib import Path

from . import clock
from .brain import Panel, Snapshot, Trigger, build_evidence, snapshot, triggers
from .broker import BUY as VENUE_BUY, Account
from .newsfeed import NOISE_CAP, NewsItem, NewsMonitor
from .policy import PolicyEvent, PolicyMonitor, policy_brief, sector_pressure
from .recommendations import (
    BUY as ADVICE_BUY,
    DEFAULT_CONVICTION,
    REASON_MANUAL,
    ExitRules,
    ExitSignal,
    Recommendation,
    RecommendationBook,
    TrackRecord,
    format_track_record,
    make_id,
)
from .secretary import RiskLimits, Secretary, TradeLedger
from .sizing import DEFAULT_RISK_PCT, r_multiple, size_position, stop_from_atr

logger = logging.getLogger(__name__)

# Report filenames are built from this; anything else is rejected.
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")

# --- windows ----------------------------------------------------------------
# The user's requirement, spelled as a constant because it is a promise the
# report makes rather than a tuning knob: "You should monitor the news in 24
# hours."
NEWS_WINDOW_HOURS = 24.0

# Policy gets a longer window on purpose. A company headline is repriced inside
# the session it lands in; an export rule or a rate decision is a standing
# condition that keeps acting on a sector until something reverses it, so a
# tariff announced the morning before yesterday is still part of today's
# backdrop. Both numbers are conventions, not results.
POLICY_WINDOW_HOURS = 48.0

# --- ranking ----------------------------------------------------------------
# What a sector tilt of +/-1.0 subtracts from a name's rank. Bounded rather
# than weighted: an additive score adjustment can promote the fortieth name
# over the first if the tilt is large enough, and this table of hand-written
# priors has not earned that.
#
# The bound is on the adjustment, not on the displacement, and the difference
# is worth stating because the two are not the same number: other names move
# the other way at the same time. A name is overtaken only by names whose
# screen rank is fewer than MAX_TILT_SHIFT * (their tilt - its tilt) below its
# own, so on the screen's consecutive ranking, with tilts of +1 and -1 pulling
# against each other, a name falls at most 2 * MAX_TILT_SHIFT - 1 places. That
# is still neighbours reordered rather than the list rewritten, which is what
# sector_pressure's own docstring says the tilt is for.
MAX_TILT_SHIFT = 5.0

# --- levels -----------------------------------------------------------------
DEFAULT_ATR_STOP_MULT = 2.0        # same convention as sizing.stop_from_atr
DEFAULT_MIN_R = 1.5
DEFAULT_HORIZON_DAYS = 30

# The target is extrapolated from the trend, and the extrapolation is bounded
# at both ends. The floor stops a flat name producing a target inside the
# spread; the cap stops a name that tripled in a quarter claiming it will do it
# again. Neither number is fitted.
TARGET_MOVE_FLOOR = 0.03
TARGET_MOVE_CAP = 0.40

# An R this high from this construction is a statement about the stop being
# unusually tight relative to the trend, not a claim that the trade is five
# times better than a 4R one. The target is trimmed so the ranking cannot be
# won on that number alone.
MAX_CREDIBLE_R = 5.0

# Sessions per calendar day, used to convert a horizon a human said in days
# into the bar count the trend was measured in. Weekdays only; holidays are
# ignored because the error is under 4% and the target is an extrapolation, not
# a forecast.
SESSIONS_PER_CALENDAR_DAY = 5.0 / 7.0
TREND_LOOKBACK_SESSIONS = 63.0     # brain.Snapshot.ret_3m's window
SHORT_TREND_SESSIONS = 21.0        # brain.Snapshot.ret_1m's window

# --- execution --------------------------------------------------------------
# A marketable limit: priced far enough through the last close that it fills
# like a market order on an ordinary open, and not one cent further. A retail
# user acting at 09:30 is not watching the tape, and a market order sent into a
# gapped open pays whatever the first print asks — which on a name that gapped
# 6% on overnight news is a different trade from the one this report described.
# The limit caps that. The cost of the cap is a miss on the days the gap is
# real, and a missed entry is a recoverable mistake in a way that a 6% worse
# entry on a 1R-risk position is not.
DEFAULT_LIMIT_BUFFER = 0.003

BENCHMARK = "SPY"

CAVEAT = (
    "This is a shortlist produced by a model reading public information. It is "
    "not advice: nobody here knows your circumstances, your tax position or "
    "your other holdings. Every level in it — the two-ATR stop, the "
    "trend-extrapolated target, the minimum R — is a convention written down "
    "in this repository, not a result anyone has established. The track record "
    "below is the only checkable part, which is why every idea above is "
    "written into it whether it works or not."
)

REFERENCE_NOTE = (
    "R and P&L are measured from the reference price (the last close). Filling "
    "at the limit costs up to the buffer above that, and the record does not "
    "carry it."
)

def _md_catalyst(rec) -> str:
    """Markdown cell: the same line, hyperlinked when a URL survived.

    Pipes are escaped because a headline containing one would silently split
    the row into extra columns.
    """
    text = catalyst_line(rec).replace("|", "\\|")
    url = getattr(rec, "catalyst_url", "")
    if not url:
        return text
    return f"[{text}]({url})"


def catalyst_line(rec, width: int | None = None) -> str:
    """The catalyst with its provenance attached, or bare when none is claimed.

    One function for both renderers: a headline that reads as sourced in the
    terminal and unsourced in the markdown would be worse than either.
    """
    text = rec.catalyst or rec.rationale or ""
    bits = []
    if getattr(rec, "catalyst_source", ""):
        bits.append(rec.catalyst_source)
    age = rec.catalyst_age_hours() if hasattr(rec, "catalyst_age_hours") else float("nan")
    if age == age:
        bits.append(f"{age:.0f}h")
    if bits:
        text = f"{text} [{', '.join(bits)}]"
    return text if width is None else text[:width]


def _pctn(x: float) -> str:
    """Compact signed percent for the fixed-width table; blank when absent."""
    try:
        if x is None or math.isnan(float(x)):
            return "-"
    except (TypeError, ValueError):
        return "-"
    return f"{float(x) * 100:+.1f}"


def _pct(x: float) -> str:
    """A fraction as a signed percentage, or an em dash when it is not a number.

    Blank would read as zero in a markdown table; an em dash reads as absent,
    which is what a missing factor actually means.
    """
    try:
        if x is None or math.isnan(float(x)):
            return "—"
    except (TypeError, ValueError):
        return "—"
    return f"{float(x) * 100:+.1f}%"


WATCHLIST_NOTE = (
    "These are followed every day by request, not proposed by the screen. A "
    "name marked BELOW 200 would have been filtered out of the buy list "
    "today — it is shown because a drawdown in something you follow is "
    "information, and a report that hides it exactly then is worth less than "
    "no report. Nothing in this section is sized, and nothing is recorded in "
    "the book."
)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _num(value: object, default: float = float("nan")) -> float:
    """Coerce anything to a finite float, or ``default``.

    Values arrive from a screen DataFrame, a hand-editable CSV and LLM JSON, so
    ``None``, ``""`` and NaN are ordinary inputs rather than programming errors.
    """
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def _home() -> Path:
    return Path(os.getenv("TRADINGAGENTS_HOME", Path.home() / ".tradingagents"))


def reports_dir() -> Path:
    return _home() / "reports"


def screens_dirs() -> list[Path]:
    """Where a saved screen CSV might be, most authoritative first.

    ``screener.save_results`` hardcodes ``~/.tradingagents/screens`` and does
    not read TRADINGAGENTS_HOME, so both are searched: the env-var location
    first because a user who set it meant it, and the literal path second
    because that is where the screener actually put the file.
    """
    out = [_home() / "screens"]
    literal = Path.home() / ".tradingagents" / "screens"
    if literal != out[0]:
        out.append(literal)
    return out


def last_completed_session(now: datetime | None = None) -> _date:
    """The most recent session whose regular hours have ended.

    Deliberately not :func:`clock.last_trading_day`, which returns a session
    that has merely *started*: at 12:00 its bars are half a day of trading that
    has not happened yet, and a report built on them would rank names on an
    unfinished close and call it the close. Half-days go through
    ``clock.close_time``, so the Friday after Thanksgiving is complete at 13:00.
    """
    et_now = now.astimezone(clock.ET) if now else datetime.now(clock.ET)
    d = et_now.date()
    if clock.is_trading_day(d) and et_now.time() >= clock.close_time(d):
        return d
    d -= timedelta(days=1)
    for _ in range(10):
        if clock.is_trading_day(d):
            return d
        d -= timedelta(days=1)
    return d


def sessions_for(when: str | _date | None = None, now: datetime | None = None
                 ) -> tuple[_date, _date]:
    """(the session the orders are for, the session the data comes from).

    ``when`` names the session the orders are FOR and defaults to the next
    open, so a Saturday run is dated Monday rather than dated a Saturday with
    no session. The data is always the last completed session, which is why the
    two dates are never equal.

    Raises ValueError when the data session is not strictly before the order
    session — the case of a ``when`` whose session has already closed. Answered
    instead of refused, it would hand back the bars of the very session it
    claims to be advising on, which is the look-ahead this report exists to
    keep out of the record.
    """
    data_day = last_completed_session(now)
    if when is None or when == "":
        order_day = clock.next_open(now).date()
    else:
        if isinstance(when, datetime):
            order_day = when.date()
        elif isinstance(when, _date):
            order_day = when
        else:
            try:
                order_day = _date.fromisoformat(str(when).strip())
            except ValueError:
                raise ValueError(
                    f"{when!r} is not an ISO date (YYYY-MM-DD)") from None
        for _ in range(10):
            if clock.is_trading_day(order_day):
                break
            order_day += timedelta(days=1)
    if data_day >= order_day:
        raise ValueError(
            f"the {order_day} session has already closed (the last completed "
            f"session is {data_day}): this report is produced from a completed "
            f"session for the next open, and cannot be dated backwards")
    return order_day, data_day


def estimated_cost(rec: Recommendation) -> float:
    """What the buy costs at the price the instruction would fill at."""
    px = _num(rec.limit_price, _num(rec.reference_price))
    return 0.0 if math.isnan(px) else px * rec.shares


def planned_risk(rec: Recommendation) -> float:
    """Dollars lost if the stop as issued fills. NaN when it cannot be read."""
    return rec.risk_amount()


def _wrapped(text: str, width: int, indent: str = "  ") -> list[str]:
    """Fold one paragraph to the report width. Never returns an empty list."""
    return textwrap.wrap(text, width=max(20, width), initial_indent=indent,
                         subsequent_indent=indent) or [indent + text]


def _no_stop_reason(entry: float, atr_pct: float, k: float) -> str:
    """Which of :func:`stop_from_atr`'s refusals actually happened.

    It refuses for a missing ATR, a non-positive entry or multiple, a stop that
    a very wide ATR pushes to or below zero, and a stop that rounds inside the
    tick. The message always named the last one, so a name with no ATR at all
    was reported as "ATR is nan% of price, which puts the stop inside the tick".
    """
    e, a, mult = _num(entry), _num(atr_pct), _num(k)
    if math.isnan(e) or e <= 0:
        return (f"no stop could be derived: the reference price {entry!r} is not "
                f"a positive number")
    if math.isnan(mult) or mult <= 0:
        return (f"no stop could be derived: the ATR multiple {k!r} is not a "
                f"positive number")
    if math.isnan(a) or a <= 0:
        return ("no stop could be derived: this name has no usable ATR, so there "
                "is no measure of its own noise to place a stop outside of")
    if e - mult * a * e <= 0:
        return (f"no stop could be derived: {mult:g} ATRs of {a:.2%} is the whole "
                f"price, so the stop lands at or below zero")
    return (f"no stop could be derived: an ATR of {a:.2%} of ${e:,.2f} puts the "
            f"{mult:g}-ATR stop inside the cent the shares quote in")


def _poll_standing(monitor, call):
    """Poll a stateful monitor for what is standing now, without eating its backlog.

    Both monitors answer "what is new since I last looked" and this report needs
    "what is standing right now" — run twice in one morning it must not have an
    empty backdrop the second time. Clearing ``seen`` gets that, but ``poll``
    persists the set it ends with, so clearing an *injected* monitor writes the
    emptied set to that monitor's own state file and its next cycle replays days
    of coverage as breaking: the exact failure ``NewsMonitor.prime`` exists to
    prevent. The previous fingerprints are therefore put back and re-persisted.
    """
    previous = dict(getattr(monitor, "seen", None) or {})
    monitor.seen = {}
    try:
        return call()
    finally:
        merged = dict(getattr(monitor, "seen", None) or {})
        for fingerprint, stamp in previous.items():
            # setdefault, not update: a fingerprint seen again in this poll keeps
            # its newer timestamp and so survives the monitor's age prune longer.
            merged.setdefault(fingerprint, stamp)
        monitor.seen = merged
        save = getattr(monitor, "_save", None)
        if previous and callable(save):
            try:
                save()
            except Exception as exc:
                logger.warning("could not restore the seen-set of %s: %s",
                               type(monitor).__name__, exc)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

@dataclass
class AdvisorConfig:
    """Everything the daily report is allowed to vary.

    ``cap_fraction`` defaults to the Secretary's own new-position cap rather
    than to a second number, so the report cannot recommend a size the live
    risk gate would refuse.
    """

    top: int = 8                      # ideas printed and recorded, at most
    max_candidates: int = 8           # LLM budget: panels run per report
    screen_top: int = 40              # names taken off the universe screen
    exchange: str = "nasdaq"
    use_cache: bool = False           # never rescan; read a saved screen
    dry_run: bool = False             # do not write to the recommendation book

    risk_pct: float = DEFAULT_RISK_PCT
    cap_fraction: float = RiskLimits().max_new_position_weight
    min_r: float = DEFAULT_MIN_R
    atr_stop_mult: float = DEFAULT_ATR_STOP_MULT
    horizon_days: int = DEFAULT_HORIZON_DAYS
    limit_buffer: float = DEFAULT_LIMIT_BUFFER
    min_price: float = RiskLimits().min_price

    max_new_per_sector: int = 2       # one policy theme must not own the list
    news_window_hours: float = NEWS_WINDOW_HOURS
    policy_window_hours: float = POLICY_WINDOW_HOURS
    max_news_symbols: int = 25        # RSS calls are the slow part of a run
    feed_pause: float = 0.4           # throttle; see NewsMonitor.poll

    # Used only when the venue cannot be reached. Sizing needs an account
    # value, and refusing to produce a report because a broker is down would
    # be a worse answer than producing one against a stated assumption.
    fallback_account_value: float = 100_000.0

    @classmethod
    def from_env(cls) -> AdvisorConfig:
        cfg = cls()
        raw = os.getenv("TRADINGAGENTS_ACCOUNT_VALUE")
        if raw:
            v = _num(raw)
            if not math.isnan(v) and v > 0:
                cfg.fallback_account_value = v
        return cfg


# ---------------------------------------------------------------------------
# candidates
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    """One name off the universe screen, plus everything gathered about it."""

    symbol: str
    rank: int = 0
    name: str = ""
    sector: str = "Unknown"
    score: float = float("nan")
    screen_price: float = float("nan")
    tilt: float = 0.0
    adjusted_rank: float = 0.0
    snap: Snapshot | None = None
    news: list[NewsItem] = field(default_factory=list)
    reason: str = ""                  # why it was skipped, when it was


def _candidate_from_row(symbol: str, row) -> Candidate:
    """One screen row → a Candidate, tolerant of both a DataFrame and a CSV."""
    def get(key, default=""):
        try:
            v = row[key]
        except (KeyError, IndexError, TypeError):
            return default
        return default if v is None else v

    sector = str(get("sector", "") or "").strip() or "Unknown"
    return Candidate(
        symbol=str(symbol).strip().upper(),
        rank=int(_num(get("rank", 0), 0.0)),
        name=str(get("name", "") or ""),
        sector=sector,
        score=_num(get("score")),
        screen_price=_num(get("price")),
    )


def candidates_from_frame(frame) -> list[Candidate]:
    """Rows of a :func:`screener.screen` result, in its own rank order."""
    out: list[Candidate] = []
    try:
        rows = list(frame.iterrows())
    except Exception:
        return out
    for symbol, row in rows:
        try:
            out.append(_candidate_from_row(symbol, row))
        except Exception as exc:
            # One unreadable row must not cost the other forty-nine.
            logger.warning("skipped an unreadable screen row for %r: %s", symbol, exc)
    return out


def candidates_from_csv(path: Path) -> list[Candidate]:
    """A saved screen CSV, read without pandas.

    Stdlib ``csv`` on purpose: this is the fallback path, taken when a fresh
    scan is unwanted or has just failed, and it should depend on as little as
    possible. The first column is the ticker, as ``DataFrame.to_csv`` writes it.
    """
    out: list[Candidate] = []
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            first = (reader.fieldnames or [""])[0]
            for row in reader:
                symbol = (row.get(first) or row.get("symbol") or "").strip()
                if not symbol:
                    continue
                out.append(_candidate_from_row(symbol, row))
    except Exception as exc:
        logger.error("could not read the saved screen %s: %s", path, exc)
        return []
    return out


def find_saved_screen(when: _date, exchange: str) -> tuple[Path | None, _date | None]:
    """The newest saved screen at or before ``when``, and the date it is from.

    The date is returned rather than swallowed because a screen from last
    Tuesday is a usable ranking and a silent one is not: the report says how
    old it is, and the reader decides.
    """
    best: tuple[Path, _date] | None = None
    for directory in screens_dirs():
        try:
            paths = list(directory.glob(f"screen_{exchange}_*.csv"))
        except Exception:
            continue
        for path in paths:
            stamp = path.stem.rsplit("_", 1)[-1]
            try:
                d = _date.fromisoformat(stamp)
            except ValueError:
                continue
            if d > when:
                continue
            if best is None or d > best[1]:
                best = (path, d)
    return (best[0], best[1]) if best else (None, None)


def run_screen(when: str, exchange: str, top: int, log=logger.debug):
    """Seam over :func:`tradingagents.trading.screener.screen`.

    Imported inside the call, as :mod:`monitor` does, so that ``--help`` and a
    cached run do not pull requests, yfinance and the whole universe download.
    Exists as a named function so tests can replace the scan without touching
    the screener.
    """
    from tradingagents.trading.screener import save_results, screen
    frame, stats = screen(when, exchange=exchange, top=top, log=log)
    try:
        save_results(frame, when, exchange)
    except Exception as exc:
        # A screen that ran and could not be filed is still a usable screen.
        logger.warning("could not save the screen CSV: %s", exc)
    return frame, stats


def run_watchlist(when: str, exchange: str = "nasdaq", log=logger.debug):
    """Seam over :func:`screener.screen_watchlist`.

    Separate from :func:`run_screen` because the watchlist must survive every
    path the screen can take — cached, skipped, or failed. A name you follow
    daily that only appears on days the universe scan happened is not a name
    you follow daily.
    """
    from tradingagents.trading.screener import screen_watchlist
    return screen_watchlist(when, exchange=exchange, log=log)


def watch_rows_from_frame(frame, log=logger.debug) -> list[WatchRow]:
    """Rows of the screener's watchlist frame, bases first then tag and symbol."""
    out: list[WatchRow] = []
    try:
        rows = list(frame.iterrows())
        bases = list(getattr(frame, "attrs", {}).get("bases") or [])
    except Exception:
        return out

    def _f(row, key):
        try:
            return _num(row.get(key))
        except Exception:
            return float("nan")

    for symbol, row in rows:
        try:
            out.append(WatchRow(
                symbol=str(symbol),
                tag=str(row.get("tag") or ""),
                name=str(row.get("name") or ""),
                price=_f(row, "price"),
                ret_1m=_f(row, "ret_1m"),
                ret_3m=_f(row, "ret_3m"),
                off_high=_f(row, "off_high"),
                ext_200=_f(row, "ext_200"),
                above_50=bool(row.get("above_50")),
                above_200=bool(row.get("above_200")),
                dollar_vol=_f(row, "dollar_vol_50"),
                is_base=bool(row.get("is_base")),
                passes_filter=bool(row.get("passes_filter")),
                fail_reason=str(row.get("fail_reason") or ""),
                excess_1m={b: _f(row, f"vs_{b}_1m") for b in bases},
                screen_rank=_f(row, "screen_rank"),
            ))
        except Exception as exc:
            logger.warning("skipped an unreadable watchlist row for %r: %s", symbol, exc)
    # Bases first: every other row is read against them, so they have to be
    # on the page before the rows that reference them.
    out.sort(key=lambda w: (not w.is_base, w.tag, w.symbol))
    return out


# ---------------------------------------------------------------------------
# levels
# ---------------------------------------------------------------------------

def project_target(entry: float, snap: Snapshot, horizon_days: int,
                   floor: float = TARGET_MOVE_FLOOR,
                   cap: float = TARGET_MOVE_CAP) -> float | None:
    """A target from the trend, deliberately not from the stop.

    This is the one level that must come from somewhere other than ATR. The
    stop is two ATRs out; a target built the same way would make every idea's
    R multiple the same number, the minimum-R filter would reject nothing, and
    ranking by R would be ranking by noise. Deriving the target from the
    three-month return instead makes R read as *trend against noise*: how far
    the name has been travelling per unit of its own daily range. A name whose
    recent trend, extended over the holding period, does not cover twice its
    stop distance is a name whose expected move is smaller than the move it
    has to survive.

    Extrapolating a past return is a convention and a weak one — trends end,
    and nothing here has back-tested this. It is written down rather than
    hidden so a reader can disagree with it. Returns None when there is no
    trend to extend, which is a refusal, not a zero.
    """
    e = _num(entry)
    if math.isnan(e) or e <= 0 or snap is None:
        return None

    sessions = max(1.0, horizon_days * SESSIONS_PER_CALENDAR_DAY)
    trend = _num(snap.ret_3m)
    lookback = TREND_LOOKBACK_SESSIONS
    if math.isnan(trend) or trend <= 0:
        # A shorter history, or a name whose quarter is flat but whose month is
        # not. Falling back to the 1-month return keeps a recent breakout
        # rankable; both being absent or negative is a refusal.
        trend = _num(snap.ret_1m)
        lookback = SHORT_TREND_SESSIONS
    if math.isnan(trend) or trend <= 0:
        return None

    move = trend * (sessions / lookback)
    move = max(floor, min(cap, move))
    target = round(e * (1.0 + move), 2)
    return target if target > e else None


def limit_price(reference: float, buffer: float = DEFAULT_LIMIT_BUFFER) -> float | None:
    """A buy limit priced ``buffer`` through the reference. See DEFAULT_LIMIT_BUFFER."""
    r, b = _num(reference), _num(buffer)
    if math.isnan(r) or r <= 0 or math.isnan(b) or b < 0:
        return None
    px = round(r * (1.0 + b), 2)
    # Rounding a cheap stock's buffer to zero would print a limit at the
    # reference, which is not marketable and would sit unfilled all day.
    return max(px, round(r + 0.01, 2))


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------

@dataclass
class WatchRow:
    """One always-analysed name, whether or not the screen liked it today.

    Carried separately from :class:`Candidate` on purpose: a candidate is
    something the screen proposed, a watch row is something you asked to see.
    Merging them would let a name you follow drift into a name you are being
    advised to buy.
    """

    symbol: str
    tag: str = ""
    name: str = ""
    price: float = float("nan")
    # The screener's own column names. Naming these ret_1d/ret_21d produced two
    # columns of NaN that rendered as an em dash in every row — the report
    # looked complete and said nothing.
    ret_1m: float = float("nan")
    ret_3m: float = float("nan")
    off_high: float = float("nan")
    ext_200: float = float("nan")
    above_50: bool = False
    above_200: bool = False
    dollar_vol: float = float("nan")
    is_base: bool = False
    passes_filter: bool = False
    fail_reason: str = ""
    screen_rank: float = float("nan")
    # {base symbol: excess return over that base, one month}. Empty when no
    # watchlist entry is tagged as a base.
    excess_1m: dict = field(default_factory=dict)


@dataclass
class DailyReport:
    """One day's output. Rendered by :func:`format_report` and :func:`to_markdown`."""

    date: str
    generated_at: str = ""
    data_date: str = ""

    buys: list[Recommendation] = field(default_factory=list)
    sells: list[ExitSignal] = field(default_factory=list)
    # What the sells above actually did to the book. Empty on a dry run, and
    # empty when a signal named an idea the book no longer has open.
    closed: list[Recommendation] = field(default_factory=list)

    policy_summary: str = ""
    policy_events: list[PolicyEvent] = field(default_factory=list)
    sector_tilt: dict[str, float] = field(default_factory=dict)
    market_context: str = ""

    track_record: TrackRecord | None = None
    candidates: list[Candidate] = field(default_factory=list)
    watchlist: list[WatchRow] = field(default_factory=list)
    marks: dict[str, float] = field(default_factory=dict)

    account_value: float = 0.0
    cash: float = 0.0
    buy_budget: float = 0.0
    risk_pct: float = DEFAULT_RISK_PCT
    dry_run: bool = False
    panel_ran: bool = False

    # Two lists, not one. A note is a decision the report made and is willing
    # to explain ("AVGO skipped: already open in the book"). A warning is a
    # source that did not answer, which changes how much of the report can be
    # trusted — collapsing them would bury the second in the first.
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # Set when no report could be produced at all — a --date whose session has
    # already closed. Rendered as a refusal rather than returned as an empty
    # page, because an empty page reads exactly like a day with nothing to do.
    refused: str = ""

    @property
    def total_cost(self) -> float:
        return sum(estimated_cost(r) for r in self.buys)

    @property
    def total_risk(self) -> float:
        return sum(v for v in (planned_risk(r) for r in self.buys) if not math.isnan(v))

    def sells_closing(self) -> list[ExitSignal]:
        return [s for s in self.sells if s.closes_position]


# ---------------------------------------------------------------------------
# the advisor
# ---------------------------------------------------------------------------

class DailyAdvisor:
    """Assembles one :class:`DailyReport`. Nothing here places an order.

    Every collaborator is injectable because every one of them is either slow,
    networked or stateful, and a report generator that can only be exercised
    against a live venue and a live LLM is a report generator nobody tests.
    """

    def __init__(self, cfg: AdvisorConfig | None = None, *, book: RecommendationBook | None = None,
                 llm=None, broker=None, secretary: Secretary | None = None,
                 news: NewsMonitor | None = None, policy_monitor: PolicyMonitor | None = None,
                 exit_rules: ExitRules | None = None):
        self.cfg = cfg or AdvisorConfig.from_env()
        self.book = book if book is not None else RecommendationBook()
        self.llm = llm
        self.broker = broker
        # The Secretary is here only for its order parser, which is what turns
        # a persona's reply into an Order. Its risk gate is deliberately not
        # run: that gate guards the venue, and it rejects everything when the
        # market is closed — which is precisely when this report is produced.
        # The discipline that applies to a recommendation is the sizing rule
        # and the exposure cap, and cap_fraction defaults to the Secretary's
        # own so the two cannot drift apart.
        self.secretary = secretary or Secretary(limits=RiskLimits.from_env(),
                                                ledger=TradeLedger())
        self.panel = Panel(llm, self.secretary) if llm is not None else None
        self.news = news if news is not None else NewsMonitor(
            state_path=_home() / "advisor_news_seen.json")
        self.policy = policy_monitor if policy_monitor is not None else PolicyMonitor(
            state_path=_home() / "advisor_policy_seen.json",
            max_age_hours=int(self.cfg.policy_window_hours))
        self.exit_rules = exit_rules or ExitRules()
        self._snaps: dict[str, Snapshot] = {}

    # --- data -------------------------------------------------------------

    def snapshot(self, symbol: str, when: str) -> Snapshot:
        """Memoised per run: a symbol on the shortlist is also often in the book."""
        key = symbol.upper()
        if key not in self._snaps:
            try:
                self._snaps[key] = snapshot(key, when)
            except Exception as exc:
                # brain.snapshot already swallows its own failures; this is the
                # guard for the ones it cannot, such as an import blowing up.
                s = Snapshot(symbol=key)
                s.error = f"{type(exc).__name__}: {exc}"
                self._snaps[key] = s
        return self._snaps[key]

    def account(self, report: DailyReport) -> Account:
        """The venue's account, or a stated assumption when it cannot be read.

        Refusing to produce a report because a broker is unreachable would be
        the wrong trade: the shortlist and the exits are still worth having,
        and the only thing the account supplies is the number the sizes are
        scaled by. So a failure degrades to ``fallback_account_value`` and says
        so on the page, where the reader can see that every share count in the
        report is against an assumed balance.
        """
        acct: Account | None = None
        if self.broker is not None:
            try:
                acct = self.broker.account()
            except Exception as exc:
                report.warnings.append(f"could not read the account: {exc}")
        else:
            try:
                from .broker import configured_venue, open_broker
                venue = configured_venue()
                with open_broker(venue) as b:
                    if b.is_logged_in():
                        acct = b.account()
                    else:
                        report.warnings.append(
                            f"{venue} is not signed in; sizes are against an "
                            f"assumed account value")
            except Exception as exc:
                report.warnings.append(
                    f"could not reach the venue ({type(exc).__name__}: {exc}); "
                    f"sizes are against an assumed account value")

        if acct is None or (acct.account_value or 0) <= 0:
            av = self.cfg.fallback_account_value
            report.warnings.append(
                f"account value assumed at ${av:,.0f} (set TRADINGAGENTS_ACCOUNT_VALUE "
                f"or configure a venue); every share count below scales with it")
            return Account(account_value=av, cash=av, buying_power=av)
        return acct

    def poll_policy(self, report: DailyReport) -> list[PolicyEvent]:
        """Policy events inside the window. A dead feed yields none, loudly."""
        try:
            events = _poll_standing(
                self.policy, lambda: self.policy.poll(pause=self.cfg.feed_pause))
        except Exception as exc:
            report.warnings.append(
                f"policy feeds unavailable ({type(exc).__name__}: {exc}); the "
                f"report has no political backdrop and no sector tilt")
            return []
        if not events:
            report.notes.append("no policy or political events cleared the filter today")
        return events

    def poll_news(self, symbols: list[str], report: DailyReport) -> list[NewsItem]:
        """Headlines for ``symbols`` plus the macro feeds, inside the window."""
        try:
            items = _poll_standing(
                self.news,
                lambda: self.news.poll(symbols, macro=True, pause=self.cfg.feed_pause))
        except Exception as exc:
            report.warnings.append(
                f"news feeds unavailable ({type(exc).__name__}: {exc}); exits "
                f"were checked on price alone and no headline informed a buy")
            return []
        window = self.cfg.news_window_hours
        fresh = [i for i in items if i.age_hours() <= window]
        if len(fresh) < len(items):
            report.notes.append(
                f"{len(items) - len(fresh)} headlines older than {window:.0f}h were "
                f"dropped; a policy search is not chronological and returns "
                f"back-coverage alongside this morning's")
        return fresh

    # --- 1. the sell side -------------------------------------------------

    def review_exits(self, news_by_symbol: dict[str, list[NewsItem]],
                     data_day: _date, report: DailyReport) -> list[ExitSignal]:
        """Price every open idea and say which are done.

        Runs before a single buy is generated. The proceeds of these sales are
        the budget the buy side spends, and an advisor that produced ideas
        first would never look at what it already holds — which is how a book
        becomes forty names nobody chose.

        The review is dated by the data session, not by the session the orders
        are for: the marks are that session's closes, and dating it by the next
        open would add up to three calendar days to every horizon over a
        weekend and time-stop ideas a day or two early.
        """
        data_date = data_day.isoformat()
        prices: dict[str, float] = {}
        for rec in self.book.open_recommendations():
            snap = self.snapshot(rec.symbol, data_date)
            if snap.ok and snap.price > 0:
                prices[rec.symbol] = snap.price
                report.marks[rec.symbol] = snap.price
        try:
            signals = self.book.review(
                prices, news_by_symbol, as_of=data_day, rules=self.exit_rules,
                # A dry run must not move a stop on disk. review applies the
                # trailing rule in memory either way, so the printed page is
                # the same page a real run would print.
                persist=not self.cfg.dry_run,
            )
        except Exception as exc:
            report.warnings.append(
                f"the exit review failed ({type(exc).__name__}: {exc}); no open "
                f"recommendation was checked today")
            return []
        return signals

    def apply_exits(self, signals: list[ExitSignal], data_day: _date,
                    report: DailyReport) -> list[Recommendation]:
        """Write the closing signals back to the book. Never raises.

        Without this the book only ever grows: buys are recorded unconditionally
        as if taken and sells never are, and that asymmetry breaks three things
        at once by the second day. ``track_record.closed`` stays 0, so the one
        checkable part of the report permanently prints "nothing has been closed
        yet"; every symbol ever recommended stays banned from the shortlist, so
        the candidate pool only shrinks; and a stopped-out idea re-emits the
        same SELL every morning forever.

        A closing signal with no usable price is expired rather than closed: it
        has no exit price, and a closed row with no P&L counts in the sample
        while contributing nothing to it.
        """
        if not signals:
            return []
        closing = [s for s in signals if s.closes_position]
        if not closing:
            return []
        if self.cfg.dry_run:
            report.notes.append(
                "dry run: the book was not touched, so "
                + ", ".join(sorted({s.symbol for s in closing}))
                + " stay open and will be signalled again tomorrow")
            return []

        closed: list[Recommendation] = []
        for sig in closing:
            px = _num(sig.price)
            try:
                if math.isnan(px) or px <= 0:
                    rec = self.book.expire(
                        sig.rec_id, reason=(sig.exit_reason or sig.reason)[:200], exit_date=data_day)
                else:
                    rec = self.book.close(
                        sig.rec_id, px, sig.exit_reason or REASON_MANUAL,
                        exit_date=data_day)
            except Exception as exc:
                report.warnings.append(
                    f"{sig.symbol} could not be closed in the book "
                    f"({type(exc).__name__}: {exc}); it stays open and the same "
                    f"exit will be signalled again tomorrow")
                continue
            if rec is None:
                report.warnings.append(
                    f"{sig.symbol} ({sig.rec_id}) was not open in the book, so the "
                    f"exit above changed nothing")
                continue
            closed.append(rec)
        if closed:
            report.notes.append(
                f"{len(closed)} recommendation(s) closed in the book: "
                + ", ".join(f"{r.symbol} ({r.exit_reason or 'exit'})" for r in closed))
        return closed

    # --- 2. candidates ----------------------------------------------------

    def candidates(self, data_day: _date, report: DailyReport) -> list[Candidate]:
        """The universe cut, from a cached screen where one will do.

        A screen is built from daily bars, so a second scan of the same session
        returns the same ranking at the full cost of the universe download —
        the argument :meth:`monitor.LiveDesk.refresh_screen` makes for running
        at most once a day. So a saved CSV for this exact session is used even
        when ``use_cache`` is off; ``use_cache`` additionally accepts an older
        one rather than scanning.

        The comparison is against the *data* session, never against the session
        the orders are for. Those two are different dates on every run, and
        comparing against the latter never matched: it re-downloaded the whole
        universe for a session already saved on disk, and then, when the rescan
        failed, reported that same up-to-date screen as a degraded fallback.
        """
        exchange = self.cfg.exchange
        data_date = data_day.isoformat()
        path, saved_date = find_saved_screen(data_day, exchange)

        if path is not None and saved_date == data_day:
            report.notes.append(
                f"screen: reusing the saved scan of the {data_date} session "
                f"({path.name})")
            return candidates_from_csv(path)

        if self.cfg.use_cache:
            if path is None:
                report.warnings.append(
                    f"--use-cache was asked for but no saved {exchange} screen exists "
                    f"in {screens_dirs()[0]}; there are no buy candidates today")
                return []
            stale = (data_day - saved_date).days if saved_date else 0
            report.warnings.append(
                f"screen is {stale} day(s) old ({path.name}); the ranking is from "
                f"{saved_date} and nothing in it has been re-scored since")
            return candidates_from_csv(path)

        try:
            frame, stats = run_screen(data_date, exchange, self.cfg.screen_top)
        except Exception as exc:
            report.warnings.append(
                f"the universe screen failed ({type(exc).__name__}: {exc})")
            if path is None:
                report.warnings.append("and no saved screen exists to fall back on")
                return []
            report.warnings.append(f"falling back to the saved screen from {saved_date}")
            return candidates_from_csv(path)

        # A stub or a future screener may return anything as its second value,
        # and a report must not die on a counter it only prints.
        counts = stats if isinstance(stats, dict) else {}
        report.notes.append(
            f"screen: {counts.get('universe', '?')} listed names → "
            f"{counts.get('passed_filters', '?')} passed the filters")
        return candidates_from_frame(frame)

    def watch(self, data_date: str, report: DailyReport) -> list[WatchRow]:
        """The always-analysed names, scored fresh every run.

        Runs whatever the screen did — a cached screen, a skipped one, a failed
        one. A failure here costs the section and nothing else: the buy list
        does not depend on it.
        """
        try:
            frame = run_watchlist(data_date, self.cfg.exchange)
        except Exception as exc:
            report.warnings.append(
                f"the watchlist could not be scored ({type(exc).__name__}: {exc})")
            return []
        rows = watch_rows_from_frame(frame)
        if not rows:
            return []
        missing = list(getattr(frame, "attrs", {}).get("missing") or [])
        if missing:
            report.notes.append(
                f"watchlist: no price data for {', '.join(missing)}")
        # Bases are excluded from both sides of the count: a yardstick is not
        # one of the names being judged, and counting it would disagree with
        # the section header two lines below.
        judged = [r for r in rows if not r.is_base]
        weak = [r.symbol for r in judged if not r.passes_filter]
        if weak:
            report.notes.append(
                f"watchlist: {len(weak)} of {len(judged)} are below the screen's own "
                f"bar today ({', '.join(weak)}) — shown anyway, which is the point")
        return rows

    def rank(self, cands: list[Candidate], tilt: dict[str, float]) -> list[Candidate]:
        """Apply the policy tilt to the screen's order, bounded by MAX_TILT_SHIFT.

        A bounded shift in rank space rather than a weighted score. The bound is
        on the adjustment, not on the displacement — see MAX_TILT_SHIFT. On the
        screen's consecutive ranking a name falls at most 2 * MAX_TILT_SHIFT - 1
        places, and only against a sector tilted the opposite way at full
        strength. Policy therefore reorders neighbours, and can never put a name
        on the list that the screen did not.
        """
        for i, c in enumerate(cands, start=1):
            if not c.rank:
                c.rank = i
            c.tilt = float(tilt.get(c.sector, 0.0))
            c.adjusted_rank = c.rank - MAX_TILT_SHIFT * c.tilt
        return sorted(cands, key=lambda c: (c.adjusted_rank, c.rank, c.symbol))

    # --- 3. the buy side ---------------------------------------------------

    def _eligible(self, cands: list[Candidate], account: Account,
                  data_date: str, report: DailyReport) -> list[Candidate]:
        """The candidates worth spending an LLM call on, cheapest tests first.

        Everything rejected here is rejected on arithmetic, before any model is
        asked anything. The two that matter most are the two that stop the book
        growing: a name already open in the recommendation book, and a name
        already held in the account. Both are *add-to-a-position* decisions,
        which is a different question from the one this report answers, and an
        advisor that re-issues its own open ideas every morning compounds one
        view into an accidental concentration.

        A name this same report just closed is rejected too. Closing an idea
        frees its symbol the instant the book is written, and a page that says
        SELL AAA above and BUY AAA below is not a considered list.
        """
        open_ideas = {r.symbol for r in self.book.open_recommendations()}
        exited_today = {r.symbol for r in report.closed}
        held = {h.symbol for h in account.holdings}
        out: list[Candidate] = []
        for c in cands:
            if not c.symbol:
                continue
            if c.symbol in open_ideas:
                c.reason = "already open in the recommendation book"
            elif c.symbol in exited_today:
                c.reason = ("exited on this same report; re-entering it in the "
                            "same breath is a different decision")
            elif c.symbol in held:
                c.reason = "already held; adding to a position is a different decision"
            else:
                snap = self.snapshot(c.symbol, data_date)
                c.snap = snap
                if not snap.ok:
                    c.reason = f"no usable price history ({snap.error or 'unknown'})"
                elif snap.price < self.cfg.min_price:
                    c.reason = (f"${snap.price:,.2f} is below the ${self.cfg.min_price:,.2f} "
                                f"price floor")
                else:
                    out.append(c)
                    continue
            report.notes.append(f"{c.symbol} skipped: {c.reason}")
        return out

    def deliberate(self, cand: Candidate, account: Account, news: list[NewsItem],
                   macro: list[NewsItem], events: list[PolicyEvent],
                   data_date: str) -> tuple[bool, float, str]:
        """Put one candidate in front of the panel. Returns (approved, conviction, why).

        The evidence pack is :func:`brain.build_evidence` with the policy brief
        appended rather than folded in. :mod:`brain` has no policy dependency
        and should not gain one — the same separation :mod:`recommendations`
        keeps by writing out a materiality constant instead of importing brain
        and its numeric stack.
        """
        if self.panel is None:
            # No panel configured. The idea is still produced — a missing API
            # key must not turn the daily list into an empty page — but it is
            # marked, so a later reader of the book can separate ideas a panel
            # reviewed from ideas only the screen and the sizing rule saw.
            # Checked first: with no panel the evidence pack is built and
            # thrown away, and it is the most expensive thing in this method.
            return True, DEFAULT_CONVICTION, (
                f"unreviewed: no panel ran. Rank #{cand.rank} on the "
                f"{self.cfg.exchange} momentum and accumulation screen.")

        snap = cand.snap or self.snapshot(cand.symbol, data_date)
        try:
            trigs = triggers(cand.symbol, snap, news, account, cand.rank)
        except Exception as exc:
            # brain reaches pandas and the news scorer; a report that runs
            # unattended must not end on one of their edge cases.
            logger.warning("triggers failed on %s: %s", cand.symbol, exc)
            trigs = []
        if not trigs:
            trigs = [Trigger(cand.symbol, "daily_review",
                             f"rank #{cand.rank} on the {self.cfg.exchange} screen"
                             + (f", sector tilt {cand.tilt:+.2f}" if cand.tilt else ""),
                             urgency=0)]
        try:
            evidence = build_evidence(cand.symbol, snap, news, account, trigs, macro,
                                      phase="daily report, for the next open")
            if events:
                evidence += "\n\n" + policy_brief(events)
        except Exception as exc:
            # Not taken rather than taken unreviewed: the panel is configured,
            # and an idea it never saw must not be printed as one it approved.
            return False, 0.0, (f"the evidence pack could not be built "
                                f"({type(exc).__name__}: {exc}); nothing was put "
                                f"to the panel")

        try:
            result = self.panel.deliberate(cand.symbol, evidence, account, snap.price)
        except Exception as exc:
            logger.warning("panel failed on %s: %s", cand.symbol, exc)
            return False, 0.0, f"panel failed ({type(exc).__name__}: {exc})"

        logger.info("%s", result.summary())
        if result.order is None or result.consensus != VENUE_BUY:
            why = result.concern or f"panel consensus was {result.consensus}"
            return False, 0.0, why
        # The panel's own share count is discarded on purpose. It is a view
        # expressed in the wrong unit: the panel does not know where the stop
        # is, and the stop is the only thing that decides how many shares a
        # fixed risk budget buys.
        return True, result.order.confidence, result.order.rationale

    def size(self, cand: Candidate, account: Account, issued: _date,
             conviction: float, rationale: str) -> tuple[Recommendation | None, str]:
        """Levels, share count and R for one approved candidate.

        Returns the recommendation *unrecorded*, or None and the sentence that
        explains the refusal. Every refusal here is a number, not a judgement.
        """
        snap = cand.snap
        if snap is None or not snap.ok:
            return None, "no usable price history"
        entry = snap.price

        stop = stop_from_atr(entry, atr_pct=snap.atr_pct, k=self.cfg.atr_stop_mult)
        if stop is None:
            return None, _no_stop_reason(entry, snap.atr_pct, self.cfg.atr_stop_mult)

        target = project_target(entry, snap, self.cfg.horizon_days)
        if target is None:
            return None, "no positive trend to extrapolate a target from"
        # Trim rather than reject: see MAX_CREDIBLE_R.
        risk_per_share = entry - stop
        ceiling = round(entry + MAX_CREDIBLE_R * risk_per_share, 2)
        if target > ceiling:
            target = ceiling

        r = r_multiple(entry, stop, target)
        if math.isnan(r):
            return None, f"levels are incoherent (entry {entry:.2f}, stop {stop:.2f}, target {target:.2f})"
        # A target closer than the stop is a losing bet however often it works:
        # at R below 1 the trade has to win more than half the time merely to
        # break even, and nothing in this report claims to know a win rate.
        # 1.5R needs 40%; that is the bar, and it is stated rather than tuned.
        if r < self.cfg.min_r:
            return None, (f"{r:.2f}R is below the {self.cfg.min_r:.2f}R minimum "
                          f"(it would have to win {1 / (1 + r):.0%} of the time to "
                          f"break even)")

        sized = size_position(account.account_value, entry, stop,
                              risk_pct=self.cfg.risk_pct,
                              cap_fraction=self.cfg.cap_fraction)
        if not sized:
            return None, sized.reason

        lim = limit_price(entry, self.cfg.limit_buffer)
        catalyst, cat_source, cat_url, cat_at = self._catalyst(cand)

        rec = Recommendation(
            id=make_id(cand.symbol, issued),
            issued_date=issued.isoformat(),
            symbol=cand.symbol,
            action=ADVICE_BUY,
            shares=sized.quantity,
            reference_price=entry,
            stop_price=stop,
            target_price=target,
            limit_price=lim,
            horizon_days=self.cfg.horizon_days,
            conviction=conviction,
            rationale=rationale[:1000],
            sector=cand.sector,
            catalyst=catalyst,
            catalyst_source=cat_source,
            catalyst_url=cat_url,
            catalyst_at=cat_at,
            issued_at=datetime.now().isoformat(),
        )
        return rec, sized.reason

    def _catalyst(self, cand: Candidate) -> tuple[str, str, str, str]:
        """The strongest checkable reason to look at this name, or none claimed.

        This used to be ``cand.news[0].title`` — the first headline the feed
        happened to return. Institutional-flow filings are capped at
        ``NOISE_CAP`` materiality rather than dropped (they belong in a news
        listing, just not in a trigger), so the first item was routinely
        "76,221 Shares in Roku Bought by ..." or "Meros Investment Management
        LP Invests $1.2 Million in BJ's Restaurants". A 13F filing is not a
        reason to buy anything, and printing one in the WHY column dresses a
        rank-driven entry as a news-driven one.

        Returns ``(text, source, url, published)``. Source, link and the
        absolute publication time travel with the headline rather than being
        formatted into it: an undated headline cannot be told apart from one
        that predates the move it is offered as the explanation for, and a
        claim the reader cannot open is one they have to go and re-find.

        The three trailing values are empty for a rank-driven idea, which is
        itself the signal that no news is being claimed.
        """
        rank_line = (f"rank #{cand.rank} on the {self.cfg.exchange} screen"
                     + (f"; sector tilt {cand.tilt:+.2f}" if cand.tilt else ""))
        usable = [n for n in cand.news if getattr(n, "materiality", 0) > NOISE_CAP]
        if not usable:
            if cand.news:
                return (f"{rank_line} — no company news above filing noise", "", "", "")
            return (rank_line, "", "", "")
        # Materiality first, then freshness. A 9 from yesterday beats a 7 from
        # an hour ago; two 9s are separated by which one is newer.
        best = max(usable, key=lambda n: (n.materiality, -n.age_hours()))
        return (best.title[:200], str(getattr(best, "source", "") or ""),
                str(getattr(best, "link", "") or ""),
                str(getattr(best, "published", "") or ""))

    def _record(self, rec: Recommendation) -> Recommendation:
        """Write one idea into the book and return the stored object.

        The stored copy is returned rather than the proposed one because the
        book assigns the id — a second idea on one name in one day gets a
        counter appended, and the report must print the id the book will answer
        to when the user quotes it back.
        """
        return self.book.add(
            rec.symbol, rec.action, rec.shares, rec.reference_price,
            rec.stop_price, rec.target_price,
            limit_price=rec.limit_price, horizon_days=rec.horizon_days,
            conviction=rec.conviction, rationale=rec.rationale,
            sector=rec.sector, catalyst=rec.catalyst,
            issued_date=rec.issued_date,
        )

    def generate_buys(self, cands: list[Candidate], account: Account,
                      news: list[NewsItem], macro: list[NewsItem],
                      events: list[PolicyEvent], budget: float,
                      issued: _date, data_date: str,
                      report: DailyReport) -> list[Recommendation]:
        """The shortlist: filter, deliberate, size, rank by R, then fit the budget."""
        eligible = self._eligible(cands, account, data_date, report)
        if not eligible:
            return []

        considered = eligible[:max(0, self.cfg.max_candidates)]
        if len(eligible) > len(considered):
            report.notes.append(
                f"{len(eligible) - len(considered)} eligible candidates were not "
                f"reviewed: the panel budget is {self.cfg.max_candidates} per report")

        proposals: list[Recommendation] = []
        for cand in considered:
            approved, conviction, why = self.deliberate(
                cand, account, news, macro, events, data_date)
            if not approved:
                report.notes.append(f"{cand.symbol} not taken: {why}")
                continue
            rec, reason = self.size(cand, account, issued, conviction, why)
            if rec is None:
                report.notes.append(f"{cand.symbol} sized to nothing: {reason}")
                continue
            proposals.append(rec)

        # Ranked by R, which is made only of the three levels the idea itself
        # asserts and needs no opinion about how often it works — the argument
        # sizing.py makes at length for preferring it to expectancy.
        proposals.sort(key=lambda r: (-r.planned_r(), r.symbol))

        by_sector: dict[str, int] = {}
        remaining = budget
        out: list[Recommendation] = []
        for rec in proposals:
            if len(out) >= self.cfg.top:
                report.notes.append(
                    f"{rec.symbol} ({rec.planned_r():.2f}R) cut: the list is capped "
                    f"at {self.cfg.top}")
                continue
            sector = rec.sector or "Unknown"
            if by_sector.get(sector, 0) >= self.cfg.max_new_per_sector:
                report.notes.append(
                    f"{rec.symbol} cut: {sector} already has "
                    f"{self.cfg.max_new_per_sector} ideas on this list")
                continue
            cost = estimated_cost(rec)
            if cost > remaining:
                report.notes.append(
                    f"{rec.symbol} cut: ${cost:,.0f} does not fit the ${remaining:,.0f} "
                    f"left of the buy budget")
                continue
            by_sector[sector] = by_sector.get(sector, 0) + 1
            remaining -= cost
            out.append(rec)

        if self.cfg.dry_run:
            report.notes.append("dry run: nothing was written to the recommendation book")
            return out
        recorded: list[Recommendation] = []
        for rec in out:
            try:
                recorded.append(self._record(rec))
            except Exception as exc:
                # The idea is still printed. Losing the record is bad; losing
                # the report because the record could not be written is worse.
                report.warnings.append(
                    f"{rec.symbol} could not be written to the book "
                    f"({type(exc).__name__}: {exc}); it is not in the track record")
                recorded.append(rec)
        return recorded

    # --- the run ----------------------------------------------------------

    def market_context(self, data_date: str, now: datetime | None = None) -> str:
        """Two lines about the tape, or a sentence saying they are unavailable."""
        state = clock.market_state(now)
        nxt = clock.next_open(now)
        lines = [f"Session: {state.session} · next open {nxt:%a %Y-%m-%d %H:%M} ET "
                 f"· bars from {data_date}"]
        snap = self.snapshot(BENCHMARK, data_date)
        if snap.ok:
            head = f"{BENCHMARK} {snap.price:,.2f} ({snap.change_pct:+.2%})"
            trend = []
            if not math.isnan(snap.sma50):
                trend.append(f"{'above' if snap.price > snap.sma50 else 'below'} SMA50")
            if not math.isnan(snap.sma200):
                trend.append(f"{'above' if snap.price > snap.sma200 else 'below'} SMA200")
            if trend:
                head += " · " + " and ".join(trend)
            lines.insert(0, head)
        else:
            lines.insert(0, f"{BENCHMARK} unavailable ({snap.error or 'no data'}); "
                            f"this report has no read on the broad tape")
        return "\n".join(lines)

    def run(self, when: str | _date | None = None,
            now: datetime | None = None) -> DailyReport:
        """Produce one day's report. Never raises.

        ``when`` is the session the orders are FOR, defaulting to the next open;
        the numbers always come from the last completed session. The order below
        is the argument of the module docstring in code: read the account,
        gather, decide the exits, write them back, and only then look for
        something to buy with what the exits freed.
        """
        try:
            order_day, data_day = sessions_for(when, now)
        except ValueError as exc:
            # A refusal is rendered, not raised: main() prints whatever comes
            # back, and an empty report reads exactly like a quiet day.
            asked = when.isoformat() if isinstance(when, (_date, datetime)) else str(when)
            report = DailyReport(
                date=asked, data_date=last_completed_session(now).isoformat(),
                generated_at=datetime.now().isoformat(),
                risk_pct=self.cfg.risk_pct, dry_run=self.cfg.dry_run,
                panel_ran=self.panel is not None, refused=str(exc),
            )
            report.warnings.append(str(exc))
            return report
        data_date = data_day.isoformat()

        report = DailyReport(
            date=order_day.isoformat(), data_date=data_date,
            generated_at=datetime.now().isoformat(),
            risk_pct=self.cfg.risk_pct, dry_run=self.cfg.dry_run,
            panel_ran=self.panel is not None,
        )
        if self.panel is None:
            report.warnings.append(
                "no LLM configured: the ideas below were not reviewed by the panel, "
                "only screened and sized")

        account = self.account(report)
        report.account_value = account.account_value
        report.cash = account.cash

        events = self.poll_policy(report)
        report.policy_events = events
        try:
            report.policy_summary = policy_brief(events)
            report.sector_tilt = sector_pressure(events) if events else {}
        except Exception as exc:
            # Both read a hand-written table over feed text. A malformed event
            # costs the backdrop and the tilt, never the report.
            report.warnings.append(
                f"the policy backdrop could not be summarised "
                f"({type(exc).__name__}: {exc}); the ranking has no sector tilt")
            report.policy_summary = ""
            report.sector_tilt = {}

        cands = self.rank(self.candidates(data_day, report), report.sector_tilt)
        report.candidates = cands
        report.watchlist = self.watch(data_date, report)

        # One poll for both halves of the report. Open ideas come first in the
        # list: if the cap bites, it must bite on a candidate whose news is
        # nice to have, never on a position whose thesis may have broken.
        open_syms = [r.symbol for r in self.book.open_recommendations()]
        wanted: list[str] = []
        for sym in open_syms + [c.symbol for c in cands]:
            if sym and sym not in wanted:
                wanted.append(sym)
        polled = wanted[:max(1, self.cfg.max_news_symbols)]
        if len(wanted) > len(polled):
            report.notes.append(
                f"news was polled for {len(polled)} of {len(wanted)} symbols "
                f"(max_news_symbols); the rest were judged on price alone")
        news = self.poll_news(polled, report)
        macro = [n for n in news if not n.ticker]
        by_symbol: dict[str, list[NewsItem]] = {}
        for item in news:
            if item.ticker:
                by_symbol.setdefault(item.ticker, []).append(item)
        for cand in cands:
            cand.news = by_symbol.get(cand.symbol, [])

        # --- sells first ---
        report.sells = self.review_exits(by_symbol, data_day, report)
        report.closed = self.apply_exits(report.sells, data_day, report)

        # Only a closing signal frees capital. A TRIM does too, but partially,
        # and this book stores one exit per idea and so cannot tell whether a
        # trim was taken — counting it would spend money that may still be in
        # the position.
        proceeds = 0.0
        for sig in report.sells_closing():
            px = _num(sig.price)
            if not math.isnan(px):
                proceeds += px * sig.shares
        budget = max(0.0, (account.buying_power or account.cash)) + proceeds
        report.buy_budget = budget
        if proceeds:
            report.notes.append(
                f"${proceeds:,.0f} of the buy budget is proceeds from the sells above, "
                f"which only exist if you take them")

        # --- then buys ---
        report.buys = self.generate_buys(cands, account, news, macro, events,
                                         budget, order_day, data_date, report)

        for rec in report.buys:
            report.marks[rec.symbol] = rec.reference_price
        try:
            # Marked and aged as of the data session: the open ideas are marked
            # at that session's closes, and dating them by the next open would
            # add days to every holding period the record reports.
            report.track_record = self.book.track_record(report.marks, data_day)
        except Exception as exc:
            report.warnings.append(f"the track record could not be computed: {exc}")

        try:
            report.market_context = self.market_context(data_date, now)
        except Exception as exc:
            report.warnings.append(
                f"the market context could not be read ({type(exc).__name__}: "
                f"{exc}); this report has no read on the broad tape")
        return report


# ----------------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------------

def _sess(report: "DailyReport") -> str:
    """The two dates that must never be confused.

    The report is built from a *completed* session's data and acted on at the
    *next* open, so these are always two different sessions. Printing only one
    of them is how a track record later becomes unauditable — "did it know this
    before or after the move" is the only question that matters when reviewing
    a call.
    """
    return (f"data through the {report.data_date} close"
            f"   →   orders for the {report.date} open")


def format_report(report: "DailyReport") -> str:
    """The morning read. Terminal-friendly, aligned, no colour."""
    W = 104
    if report.refused:
        return "\n".join(["=" * W, "  DAILY ADVISOR — no report", "-" * W]
                         + _wrapped(report.refused, W - 4) + ["=" * W])

    out = [
        "=" * W,
        f"  DAILY ADVISOR — {report.date}",
        f"  {_sess(report)}",
        "=" * W,
    ]

    if report.market_context:
        out += ["", report.market_context.strip()]

    if report.policy_summary:
        out += ["", "POLICY BACKDROP", "-" * W, report.policy_summary.strip()]

    tilt = {k: v for k, v in (report.sector_tilt or {}).items() if abs(v) > 0.05}
    if tilt:
        out.append("  sector tilt: " + "  ".join(
            f"{k} {v:+.2f}" for k, v in sorted(tilt.items(), key=lambda kv: -abs(kv[1]))))

    # Sells first: freed capital funds the buys, and a book nobody prunes only
    # ever grows.
    out += ["", f"SELL / EXIT  ({len(report.sells)})", "-" * W]
    if report.sells:
        out.append(f"  {'SYMBOL':<8}{'ACTION':<7}{'SHARES':>8}{'PRICE':>11}"
                   f"{'P&L':>12}{'R':>7}  REASON")
        for s in sorted(report.sells, key=lambda x: -x.urgency):
            flag = "!" if s.urgency >= 3 else " "
            out.append(f" {flag}{s.symbol:<8}{s.action:<7}{s.shares:>8.0f}"
                       f"{s.price:>11,.2f}{s.pnl:>+12,.2f}{s.r_multiple:>+7.2f}"
                       f"  {s.reason[:44]}")
        if report.closed:
            out.append("  closed in the book: " + ", ".join(
                f"{r.symbol} ({r.exit_reason or 'exit'})" for r in report.closed)
                + " — these now score in the track record below")
    else:
        out.append("  (nothing to exit)")

    out += ["", f"BUY  ({len(report.buys)})", "-" * W]
    if report.buys:
        out.append(f"  {'SYMBOL':<8}{'SHARES':>7}{'REF':>10}{'LIMIT':>10}"
                   f"{'STOP':>10}{'TARGET':>10}{'R':>6}{'COST':>11}{'RISK':>9}  WHY")
        for r in report.buys:
            # planned_r, not the live stop: the trailing rule moves stop_price,
            # and the R this table shows must be the R the list was ranked on.
            rr = r.planned_r()
            lim = f"{r.limit_price:,.2f}" if r.limit_price else "MKT"
            out.append(
                f"  {r.symbol:<8}{r.shares:>7.0f}{r.reference_price:>10,.2f}"
                f"{lim:>10}{r.stop_price:>10,.2f}{r.target_price:>10,.2f}"
                f"{rr:>6.2f}{estimated_cost(r):>11,.2f}{planned_risk(r):>9,.2f}"
                f"  {catalyst_line(r, 46)}")
        out.append("-" * W)
        out.append(f"  {'TOTAL':<8}{'':>7}{'':>10}{'':>10}{'':>10}{'':>10}{'':>6}"
                   f"{report.total_cost:>11,.2f}{report.total_risk:>9,.2f}")
        out += _wrapped(REFERENCE_NOTE, W - 4)
    else:
        out.append("  (no candidate cleared the bar — this is a normal outcome)")

    if report.watchlist:
        rows = report.watchlist
        bases = [r.symbol for r in rows if r.is_base]
        weak = sum(1 for r in rows if not r.is_base and not r.passes_filter)
        watched = sum(1 for r in rows if not r.is_base)
        out += ["", f"WATCHLIST  ({watched}, {weak} below the screen's bar)", "-" * W]
        head = (f"  {'SYMBOL':<8}{'TAG':<6}{'PRICE':>10}{'1M%':>7}{'3M%':>7}")
        for b in bases:
            head += f"{'vs ' + b:>9}"
        head += f"{'OFF HIGH':>10}{'vs200':>7}  STATUS"
        out.append(head)
        for i, r in enumerate(rows):
            if i and rows[i - 1].is_base and not r.is_base:
                out.append("  " + "·" * (W - 4))
            status = (r.fail_reason.upper() if not r.is_base and not r.passes_filter
                      and r.fail_reason
                      else "benchmark" if r.is_base
                      else "above 50 & 200" if r.above_50 and r.above_200
                      else "above 200" if r.above_200
                      else "BELOW 200")
            if not math.isnan(r.screen_rank):
                status += f"   screen #{int(r.screen_rank)}"
            line = (f"  {r.symbol:<8}{r.tag:<6}{r.price:>10,.2f}"
                    f"{_pctn(r.ret_1m):>7}{_pctn(r.ret_3m):>7}")
            for b in bases:
                line += f"{_pctn(r.excess_1m.get(b, float('nan'))):>9}"
            line += (f"{_pctn(r.off_high):>10}{_pctn(r.ext_200):>7}  {status}")
            out.append(line)
        out += _wrapped(WATCHLIST_NOTE, W - 4)

    out += ["", "ACCOUNT", "-" * W,
            f"  equity ${report.account_value:,.2f}   cash ${report.cash:,.2f}   "
            f"buy budget ${report.buy_budget:,.2f}   "
            f"risk budget {report.risk_pct:.2f}%/trade"]
    if report.buys:
        # The budget is not what the rows risk: the position cap trims any
        # trade whose stop is near, which is most of them, so the header
        # percentage alone reads as a promise the table does not keep.
        risk = report.total_risk
        share = (f" ({risk / report.account_value:.2%} of equity)"
                 if report.account_value > 0 else "")
        out.append(f"  planned risk ${risk:,.2f}{share} across "
                   f"{len(report.buys)} idea(s), after the position cap")

    if report.track_record:
        out += ["", "TRACK RECORD", "-" * W]
        try:
            out.append(format_track_record(report.track_record))
        except Exception:
            out.append(f"  {report.track_record}")

    if report.notes:
        out += ["", "NOTES", "-" * W] + [f"  - {n}" for n in report.notes]
    if report.warnings:
        out += ["", "WARNINGS", "-" * W] + [f"  ! {w}" for w in report.warnings]
    if report.dry_run:
        out += ["", "  [DRY RUN — nothing was written to the recommendation book]"]

    out += ["", "=" * W] + _wrapped(CAVEAT, W - 4) + ["=" * W]
    return "\n".join(out)


def to_markdown(report: "DailyReport") -> str:
    if report.refused:
        return "\n".join(["# Daily Advisor — no report", "", report.refused, ""])

    md = [f"# Daily Advisor — {report.date}", "", f"_{_sess(report)}_", ""]
    if report.market_context:
        md += ["## Market", "", report.market_context.strip(), ""]
    if report.policy_summary:
        md += ["## Policy backdrop", "", report.policy_summary.strip(), ""]

    md += [f"## Sell / exit ({len(report.sells)})", ""]
    if report.sells:
        md += ["| Symbol | Action | Shares | Price | P&L | R | Reason |",
               "|---|---|---:|---:|---:|---:|---|"]
        md += [f"| {s.symbol} | {s.action} | {s.shares:.0f} | {s.price:,.2f} | "
               f"{s.pnl:+,.2f} | {s.r_multiple:+.2f} | {s.reason} |"
               for s in sorted(report.sells, key=lambda x: -x.urgency)]
        if report.closed:
            md += ["", "Closed in the book: " + ", ".join(
                f"{r.symbol} ({r.exit_reason or 'exit'})" for r in report.closed)]
    else:
        md.append("_Nothing to exit._")

    md += ["", f"## Buy ({len(report.buys)})", ""]
    if report.buys:
        md += ["| Symbol | Shares | Ref | Limit | Stop | Target | R | Cost | Risk | Why |",
               "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        for r in report.buys:
            # See format_report: the R shown is the R as issued.
            rr = r.planned_r()
            lim = f"{r.limit_price:,.2f}" if r.limit_price else "MKT"
            md.append(f"| {r.symbol} | {r.shares:.0f} | {r.reference_price:,.2f} | {lim} | "
                      f"{r.stop_price:,.2f} | {r.target_price:,.2f} | {rr:.2f} | "
                      f"{estimated_cost(r):,.2f} | {planned_risk(r):,.2f} | "
                      f"{_md_catalyst(r)} |")
        md += ["", f"_{REFERENCE_NOTE}_"]
    else:
        md.append("_No candidate cleared the bar._")

    if report.watchlist:
        rows = report.watchlist
        bases = [r.symbol for r in rows if r.is_base]
        weak = sum(1 for r in rows if not r.is_base and not r.passes_filter)
        watched = sum(1 for r in rows if not r.is_base)
        md += ["", f"## Watchlist ({watched})", "",
               f"_{weak} of {watched} would not have cleared the screen's own bar today._", ""]
        header = "| Symbol | Tag | Price | 1M | 3M |"
        divide = "|---|---|---:|---:|---:|"
        for b in bases:
            header += f" vs {b} |"
            divide += "---:|"
        header += " Off high | vs SMA200 | Status |"
        divide += "---:|---:|---|"
        md += [header, divide]
        for r in rows:
            status = ("_benchmark_" if r.is_base
                      else f"**{r.fail_reason}**" if not r.passes_filter and r.fail_reason
                      else "above 50 & 200" if r.above_50 and r.above_200
                      else "above 200" if r.above_200
                      else "**below 200**")
            if not math.isnan(r.screen_rank):
                status += f" · screen #{int(r.screen_rank)}"
            row = (f"| {r.symbol} | {r.tag} | {r.price:,.2f} | "
                   f"{_pct(r.ret_1m)} | {_pct(r.ret_3m)} |")
            for b in bases:
                row += f" {_pct(r.excess_1m.get(b, float('nan')))} |"
            row += f" {_pct(r.off_high)} | {_pct(r.ext_200)} | {status} |"
            md.append(row)
        md += ["", f"_{WATCHLIST_NOTE}_"]

    md += ["", "## Account", "",
           f"- Equity **${report.account_value:,.2f}**, cash ${report.cash:,.2f}",
           f"- Buy budget ${report.buy_budget:,.2f}, risk budget "
           f"{report.risk_pct:.2f}%/trade"]
    if report.buys:
        share = (f" ({report.total_risk / report.account_value:.2%} of equity)"
                 if report.account_value > 0 else "")
        md.append(f"- Planned risk ${report.total_risk:,.2f}{share} across "
                  f"{len(report.buys)} idea(s), after the position cap")
    if report.notes:
        md += ["", "## Notes", ""] + [f"- {n}" for n in report.notes]
    if report.warnings:
        md += ["", "## Warnings", ""] + [f"- {w}" for w in report.warnings]
    md += ["", "---", "", f"_{CAVEAT}_"]
    return "\n".join(md)


def save_report(report: "DailyReport") -> Path:
    """Write the page, atomically.

    Same tmp+replace as :meth:`RecommendationBook.save`: a run killed mid-write
    would otherwise leave a truncated page that reads as a complete one, and
    the reader has no way to tell which half is missing.
    """
    # report.date is normally an ISO date, but the refusal path echoes back the
    # raw --date the user typed, and that string reaches the filename. A
    # traversal ("../../etc/passwd") would then escape the reports directory.
    # Validating here rather than only in the caller keeps the guard attached to
    # the operation it protects.
    stem = str(report.date or "").strip()
    if not _ISO_DATE.fullmatch(stem):
        stem = f"invalid-date-{_dt.datetime.now():%Y%m%d-%H%M%S}"
        logger.warning("report date %r is not an ISO date; writing as %s",
                       report.date, stem)
    path = reports_dir() / f"{stem}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(to_markdown(report), encoding="utf-8")
    os.replace(tmp, path)
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="advisor",
        description="Daily buy/sell list for the next session.")
    p.add_argument("--date", default=None,
                   help="Session the orders are FOR (default: the next open). "
                        "Data is always taken from the last completed session, so "
                        "a session that has already closed is refused.")
    p.add_argument("--top", type=int, default=None,
                   help="ideas printed and recorded, at most (default 8); the "
                        "panel budget follows it unless --max-candidates is given")
    p.add_argument("--risk-pct", type=float, default=None,
                   help="percent of equity risked per trade (default 1.0)")
    p.add_argument("--min-r", type=float, default=None,
                   help="reject anything below this reward/risk (default 1.5)")
    p.add_argument("--max-candidates", type=int, default=None,
                   help="LLM budget: how many names reach the panel")
    p.add_argument("--exchange", default=None, choices=["nasdaq", "all"])
    p.add_argument("--use-cache", action="store_true",
                   help="reuse the most recent saved screen instead of rescanning")
    p.add_argument("--no-llm", action="store_true",
                   help="rules only — no panel, no API calls")
    p.add_argument("--dry-run", action="store_true",
                   help="print the report but do not record it to the book")
    p.add_argument("-q", "--quiet", action="store_true")
    a = p.parse_args(argv)

    logging.basicConfig(level=logging.WARNING if a.quiet else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("urllib3", "yfinance", "peewee", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # from_env, not the bare default: the report tells the reader to set
    # TRADINGAGENTS_ACCOUNT_VALUE, and every share count on the page scales
    # with the number it was ignoring.
    cfg = AdvisorConfig.from_env()
    for src, dst in (("top", "top"), ("risk_pct", "risk_pct"), ("min_r", "min_r"),
                     ("max_candidates", "max_candidates"), ("exchange", "exchange")):
        v = getattr(a, src, None)
        if v is not None and hasattr(cfg, dst):
            setattr(cfg, dst, v)
    # A list of N ideas needs at least N names in front of the panel:
    # generate_buys cuts to max_candidates before a single idea is produced, so
    # --top 20 on its own printed at most the default 8 and said nothing.
    if a.max_candidates is None and cfg.top > cfg.max_candidates:
        cfg.max_candidates = cfg.top
    for flag, attr in (("use_cache", "use_cache"), ("dry_run", "dry_run")):
        if getattr(a, flag, False) and hasattr(cfg, attr):
            setattr(cfg, attr, True)

    llm = None
    if not a.no_llm:
        try:
            from .monitor import build_llm
            llm = build_llm()
        except Exception as exc:
            print(f"LLM unavailable ({exc}); ranking on rules alone.")

    report = DailyAdvisor(cfg, llm=llm).run(a.date)
    print()
    print(format_report(report))
    if report.refused:
        return 2
    if not report.dry_run:
        # Guarded: the ideas are already printed and already in the book, and a
        # read-only reports directory must not end the run in a traceback that
        # looks like the report itself failed.
        try:
            print(f"\nsaved: {save_report(report)}")
        except Exception as exc:
            print(f"\nthe report could not be saved ({type(exc).__name__}: {exc}); "
                  f"the ideas above are still in the recommendation book")
    return 0


if __name__ == "__main__":
    sys.exit(main())
