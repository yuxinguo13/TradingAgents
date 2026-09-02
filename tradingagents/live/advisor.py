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

from . import charting, clock, deepdive, execute, fundamentals as fund, horizons, research
from .brain import Panel, Snapshot, Trigger, build_evidence, snapshot, triggers
from .earnings import Earnings, EarningsBook, summarise as summarise_earnings
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
from .zhnames import DERIVED_MARK, ZhNames

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
    # And the same cap across the *whole* open book, not just today's page.
    # max_new_per_sector counts one report at a time, so two healthcare ideas a
    # day for three days is six healthcare positions and never trips it — which
    # is exactly what happened: a six-slot swing book ended up holding six
    # biotech names, six bets on one financing environment wearing the costume
    # of a diversified book. Half the slots is the convention.
    max_open_per_sector: int = 3

    # --- horizons ---------------------------------------------------------
    # The swing book's slot count is the churn control: new ideas only fill
    # empty slots, so a full book proposes nothing and the page stops looking
    # like a fresh portfolio every morning. See :mod:`~.horizons`.
    swing_slots: int = 6
    core_seed: bool = True            # write a first core.json when none exists
    daytrade_top: int = 5             # names published as intraday levels, at most

    # --- the pages --------------------------------------------------------
    write_pages: bool = True          # one deep-dive markdown page per symbol
    with_fundamentals: bool = True    # pull statements for the pages
    # Statements are one network call per symbol on a cold cache. Buys, sells,
    # open ideas and the core are always covered; the watchlist fills whatever
    # is left, and the report says so when it ran out.
    max_fundamentals: int = 45
    news_window_hours: float = NEWS_WINDOW_HOURS
    policy_window_hours: float = POLICY_WINDOW_HOURS
    max_news_symbols: int = 25        # RSS calls are the slow part of a run
    feed_pause: float = 0.4           # throttle; see NewsMonitor.poll

    # Used only when the venue cannot be reached. Sizing needs an account
    # value, and refusing to produce a report because a broker is down would
    # be a worse answer than producing one against a stated assumption.
    fallback_account_value: float = 100_000.0

    # Fields settable from the environment, and the type each is read as.
    # Everything here changes *what the report proposes*, so it belongs
    # alongside the risk limits in being adjustable without editing code —
    # a knob whose only interface is a source edit is a knob nobody turns.
    _ENV_FIELDS = {
        "top": int, "max_candidates": int, "screen_top": int,
        "risk_pct": float, "min_r": float, "atr_stop_mult": float,
        "horizon_days": int, "limit_buffer": float,
        "swing_slots": int, "max_new_per_sector": int, "max_open_per_sector": int,
        "daytrade_top": int, "max_fundamentals": int, "max_news_symbols": int,
        "core_seed": bool, "write_pages": bool, "with_fundamentals": bool,
    }

    @classmethod
    def from_env(cls) -> AdvisorConfig:
        cfg = cls()
        raw = os.getenv("TRADINGAGENTS_ACCOUNT_VALUE")
        if raw:
            v = _num(raw)
            if not math.isnan(v) and v > 0:
                cfg.fallback_account_value = v
        for field_name, kind in cls._ENV_FIELDS.items():
            raw = os.getenv(f"TRADINGAGENTS_ADVISOR_{field_name.upper()}")
            if raw is None or not raw.strip():
                continue
            try:
                if kind is bool:
                    value = raw.strip().lower() in ("1", "true", "yes", "on")
                else:
                    value = kind(_num(raw))
                    # A negative slot count or a zero risk budget is a typo, and
                    # silently accepting it produces a report that is empty for
                    # no stated reason.
                    if value < 0 or (kind is float and value == 0
                                     and field_name in ("risk_pct", "min_r")):
                        raise ValueError(raw)
            except (TypeError, ValueError):
                logger.warning("ignoring TRADINGAGENTS_ADVISOR_%s=%r: not a %s",
                               field_name.upper(), raw, kind.__name__)
                continue
            setattr(cfg, field_name, value)
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
    earnings: Earnings | None = None
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
    # symbol → Earnings, for every name the report priced. Kept on the report
    # rather than only on the candidates so the watchlist section can use it.
    earnings: dict = field(default_factory=dict)
    marks: dict[str, float] = field(default_factory=dict)

    # --- the three horizons -------------------------------------------
    # Carried separately from ``buys`` because they answer different
    # questions on different clocks; see :mod:`~.horizons`.
    open_ideas: list = field(default_factory=list)     # horizons.OpenIdea
    core: list = field(default_factory=list)           # horizons.CoreLine
    core_seeded: bool = False
    core_review_day: bool = False
    daytrade: list = field(default_factory=list)       # horizons.DayTradeIdea
    swing_slots: int = horizons.DEFAULT_SWING_SLOTS
    # What the venue actually holds against what the book says it should.
    # None when the account could not be read — see DailyAdvisor.account.
    reconcile: object = None

    # symbol -> deepdive.SymbolAnalysis, for every name the page links to.
    # Each analysis carries its own ``page`` path once write_pages has run;
    # until then page_link() emits nothing, so the daily page never links to a
    # file nobody wrote.
    analysis: dict = field(default_factory=dict)

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
                 exit_rules: ExitRules | None = None, fundamentals=None,
                 bars_loader=None):
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
        self._bars: dict[str, deepdive.Bars] = {}
        # Both are seams for the same reason the venue and the feeds are:
        # statements and OHLCV are network reads, and a report generator that
        # can only be exercised online is one nobody tests.
        self._fundamentals = fundamentals
        self._bars_loader = bars_loader
        # Kept so the core section can read position weights without a second
        # venue call; set by run() and never used for a decision.
        self._account = None
        self._account_live = False
        self._news_items: list = []

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
        # Whether the numbers below came from a venue or from an assumption.
        # The reconciliation section is only meaningful against a real account:
        # run against the fallback it would report every open idea as missing.
        self._account_live = False
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

        if acct is not None and (acct.account_value or 0) > 0:
            self._account_live = True
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

    def _with_earnings(self, cand: Candidate, catalyst: str, ref: _date) -> str:
        """Append the earnings facts that change how this idea should be read.

        Two of them, and only these two, because only these two are facts
        rather than opinions about a number:

        * a report inside the holding horizon — the stop cannot cover the gap;
        * the last surprise — a name topping a momentum screen days after a
          large miss is momentum running against the fundamentals.

        Silence when the calendar is unknown. "No earnings due" and "we could
        not find out" must not read the same, and the missing names are listed
        separately in the report's notes.
        """
        e = cand.earnings
        if e is None or not (e.next_date or e.last_date):
            return catalyst
        bits = []
        days = e.days_to_next(ref)
        if days == days and 0 <= days <= self.cfg.horizon_days:
            bits.append(f"reports in {days:.0f}d, inside the {self.cfg.horizon_days}d "
                        f"horizon — the stop does not cover a gap")
        beat = e.beat()
        since = e.days_since_last(ref)
        if beat is not None and since == since and since <= 45:
            word = "beat" if beat else "missed"
            bits.append(f"{word} by {abs(e.surprise_pct):.0f}% {since:.0f}d ago")
        return f"{catalyst}　| {'; '.join(bits)}" if bits else catalyst

    def attach_earnings(self, cands: list[Candidate], report: DailyReport,
                        data_day: _date, order_day: _date) -> None:
        """Earnings facts for the candidates and the watchlist.

        A two-ATR stop is not a defence against an earnings gap — the gap opens
        through it — so a report that sizes a position without knowing the date
        is sizing against a risk it cannot see. And a name topping a momentum
        screen right after a large miss is momentum running against the
        fundamentals, which is worth saying out loud rather than leaving for
        the reader to discover next quarter.

        Never fatal: the buy list does not depend on this, and a calendar that
        cannot be read is reported as a gap rather than assumed benign.
        """
        symbols = [c.symbol for c in cands] + [
            w.symbol for w in report.watchlist if not w.is_base]
        symbols = list(dict.fromkeys(s for s in symbols if s))
        if not symbols:
            return
        try:
            book = EarningsBook().get(symbols, data_day, log=logger.debug)
        except Exception as exc:
            report.warnings.append(
                f"earnings dates could not be read ({type(exc).__name__}: {exc}); "
                f"nothing below accounts for a report landing inside its holding period")
            return

        report.earnings = book
        for c in cands:
            c.earnings = book.get(c.symbol)

        # Fetched as of the data session — a report published after the data
        # cut must not be visible as "the last one". Counted from the order
        # session, which is when the holding period actually starts.
        stats = summarise_earnings(book, order_day, self.cfg.horizon_days)
        if stats["missing"]:
            report.notes.append(
                f"no earnings calendar for {', '.join(stats['missing'])} — "
                f"those names are sized as if no report were due, which is an "
                f"assumption, not a finding")

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
        # ``issued`` — the session the order is for — not the data session:
        # the countdown a reader acts on starts when they own the position.
        # (Fetching still uses the data session; see attach_earnings.)
        catalyst = self._with_earnings(cand, catalyst, issued)

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
                      report: DailyReport, limit: int | None = None) -> list[Recommendation]:
        """The shortlist: filter, deliberate, size, rank by R, then fit the budget.

        ``limit`` is the number of *free swing slots*, not a display cap. A book
        already carrying its full complement proposes nothing, and that is the
        intended behaviour: the alternative is a position count that grows every
        day the screen has an opinion, which is how a considered list becomes a
        feed.
        """
        cap = self.cfg.top if limit is None else max(0, min(self.cfg.top, limit))
        if cap <= 0:
            report.notes.append(
                f"没有新增波段建议：{self.cfg.swing_slots} 个仓位槽已经占满。"
                f"这不是「今天没机会」，是「先把手上的处理完」")
            return []
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

        # Seeded from what the book already holds, so the cap counts positions
        # rather than page rows. See AdvisorConfig.max_open_per_sector.
        by_sector: dict[str, int] = {}
        held_by_sector: dict[str, int] = {}
        closed_today = {r.id for r in report.closed}
        for open_rec in self.book.open_recommendations():
            if getattr(open_rec, "id", None) in closed_today:
                continue
            key = open_rec.sector or "Unknown"
            held_by_sector[key] = held_by_sector.get(key, 0) + 1
        remaining = budget
        out: list[Recommendation] = []
        for rec in proposals:
            if len(out) >= cap:
                report.notes.append(
                    f"{rec.symbol} ({rec.planned_r():.2f}R) cut: the list is capped "
                    f"at {cap}" + ("" if limit is None or cap == self.cfg.top
                                   else f" (free swing slots, of {self.cfg.swing_slots})"))
                continue
            sector = rec.sector or "Unknown"
            if by_sector.get(sector, 0) >= self.cfg.max_new_per_sector:
                report.notes.append(
                    f"{rec.symbol} cut: {sector} already has "
                    f"{self.cfg.max_new_per_sector} ideas on this list")
                continue
            open_here = held_by_sector.get(sector, 0) + by_sector.get(sector, 0)
            if open_here >= self.cfg.max_open_per_sector:
                report.notes.append(
                    f"{rec.symbol} cut: 账本里 {sector} 已经有 {open_here} 个仓位"
                    f"（上限 {self.cfg.max_open_per_sector}/{self.cfg.swing_slots}）——"
                    f"同一个板块占满半数仓位就不是分散，是同一个赌注下了几次")
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

    # --- horizons and the pages -------------------------------------------

    def bars(self, symbol: str, when: str) -> deepdive.Bars:
        """Memoised daily history. Same disk cache :meth:`snapshot` reads."""
        key = (symbol or "").upper()
        if key not in self._bars:
            self._bars[key] = deepdive.load_bars(key, when, loader=self._bars_loader)
        return self._bars[key]

    def facts(self, symbols: list[str], when: str) -> dict:
        """``{symbol: facts}`` for the horizon modules, computed once per name."""
        out: dict[str, dict] = {}
        for sym in symbols:
            key = (sym or "").upper()
            if not key or key in out:
                continue
            try:
                got = self.bars(key, when).facts()
            except Exception as exc:
                logger.debug("facts for %s failed: %s", key, exc)
                got = {}
            if got:
                out[key] = got
        return out

    def review_open(self, data_day: _date, report: DailyReport) -> list:
        """The swing positions still open after today's exits were applied.

        This section is the answer to "why is it a different list every day":
        it was not — the report simply never printed what it was already
        holding. Nothing here decides anything; the exit engine owns that.
        """
        try:
            closed_today = {r.id for r in report.closed}
            live = [r for r in self.book.open_recommendations()
                    if getattr(r, "id", None) not in closed_today]
            marks = dict(report.marks)
            for rec in live:
                if rec.symbol not in marks:
                    snap = self.snapshot(rec.symbol, data_day.isoformat())
                    if snap.ok and snap.price > 0:
                        marks[rec.symbol] = snap.price
            return horizons.open_swing(live, marks, data_day)
        except Exception as exc:
            report.warnings.append(
                f"the open-position section could not be built "
                f"({type(exc).__name__}: {exc}); the swing book is still in "
                f"recommendations.json")
            return []

    def core_section(self, report: DailyReport, data_date: str,
                     order_day: _date) -> list:
        """The long-term book: read every day, acted on monthly.

        Seeds a first ``core.json`` when the reader has none, because an empty
        section teaches nothing and a proposal can be edited. The seeding is
        announced in the notes and the file is never touched again.
        """
        try:
            holdings = horizons.load_core()
            report.core_review_day = horizons.is_review_day(order_day)
            if not holdings and self.cfg.core_seed:
                seeded = self.seed_core(report, data_date)
                if seeded and not self.cfg.dry_run:
                    path = horizons.save_core(seeded)
                    report.core_seeded = True
                    report.notes.append(
                        f"核心长仓名单是空的，已按长期规则初选 {len(seeded)} 只"
                        f"（合计 {sum(h.weight for h in seeded) * 100:.0f}% 仓位）"
                        f"写入 {path}——这是一份草稿，请按你自己的判断改写它，"
                        f"之后本报告只跟踪不重选")
                holdings = seeded
            if not holdings:
                return []
            facts = self.facts([h.symbol for h in holdings], data_date)
            weights = {}
            try:
                for h in getattr(self._account, "holdings", []) or []:
                    value = _num(getattr(h, "market_value", None))
                    if not math.isnan(value) and report.account_value > 0:
                        weights[h.symbol] = value / report.account_value
            except Exception:
                weights = {}
            return horizons.review_core(holdings, facts, held_weights=weights,
                                        review_day=report.core_review_day)
        except Exception as exc:
            report.warnings.append(
                f"the core section could not be built ({type(exc).__name__}: {exc})")
            return []

    def seed_core(self, report: DailyReport, data_date: str) -> list:
        """A first core list, in two passes.

        The price filters are free and the statements are not, so the cheap
        pass runs first and the fetch is paid only for the names it already
        likes. The second pass is the one that decides: it adds the two tests
        price cannot make — the company earns money, and no sector takes more
        than :data:`~.horizons.CORE_MAX_PER_SECTOR` slots.

        The invested fraction is derived from the swing book's own claim on the
        account rather than fixed, so the two books cannot add up to more than
        the account holds.
        """
        pool = [w.symbol for w in report.watchlist if not w.is_base]
        pool += [c.symbol for c in report.candidates[:40]]
        facts = self.facts(pool, data_date)

        shortlist = [h.symbol for h in horizons.propose_core(
            pool, facts, count=30, require_profit=False, max_per_sector=0)]
        if not shortlist:
            return []

        stats: dict = {}
        try:
            book = self._fundamentals
            if book is None:
                book = self._fundamentals = fund.FundamentalsBook()
            stats = book.get(shortlist, log=logger.debug)
        except Exception as exc:
            # Without statements the profitability test cannot be made, and
            # _earns() reads "not checked" as "no". Seeding nothing is the
            # right answer: an unfiltered list would be exactly the one this
            # method exists to stop producing.
            report.warnings.append(
                f"核心名单没有初始化：拿不到财报数据（{type(exc).__name__}: {exc}），"
                f"无法确认盈利。请自己编辑 core.json")
            return []

        sectors = {c.symbol: c.sector for c in report.candidates if c.sector}
        for sym, f in stats.items():
            if getattr(f, "sector", ""):
                sectors[sym] = f.sector

        seeded = horizons.propose_core(
            shortlist, facts, count=8, fundamentals=stats, sectors=sectors,
            invested=horizons.core_budget(self.cfg.swing_slots, self.cfg.cap_fraction))
        dropped = [s for s in shortlist[:12] if s not in {h.symbol for h in seeded}]
        if dropped:
            report.notes.append(
                "核心初选里被剔除的（未盈利，或所属行业已占满 "
                f"{horizons.CORE_MAX_PER_SECTOR} 个名额）：{'、'.join(dropped[:8])}")
        return seeded

    def daytrade_section(self, report: DailyReport, data_date: str) -> list:
        """Levels for the next session, off the watchlist and today's screen.

        Deliberately drawn from names already on the page rather than from a
        fresh scan: an intraday list assembled from a different universe than
        the rest of the report is a second desk, not a section of this one.
        """
        try:
            pool = [w.symbol for w in report.watchlist if not w.is_base]
            pool += [c.symbol for c in report.candidates[:30]]
            return horizons.daytrade_candidates(
                self.facts(pool, data_date), count=self.cfg.daytrade_top)
        except Exception as exc:
            report.warnings.append(
                f"the intraday section could not be built "
                f"({type(exc).__name__}: {exc})")
            return []

    def reconcile_account(self, report: DailyReport) -> object:
        """What the venue holds, against what the book says it should.

        On the page every day rather than behind a command, because this is the
        gap nobody notices: the advisor records decisions, ``--no-llm`` keeps
        the monitor from acting on anything, and the track record goes on
        scoring ideas as taken while the account holds something else. It cost
        this desk eleven days and three books with no symbol in common.

        Read-only. :mod:`~.execute` places the orders, and only when asked.
        """
        if not self._account_live or self._account is None:
            return None
        try:
            return execute.plan(self.book, self._account, exits=report.sells,
                                as_of=_date.fromisoformat(report.data_date),
                                quote=getattr(self.broker, "quote", None))
        except Exception as exc:
            report.warnings.append(
                f"账本与账户对不上账（{type(exc).__name__}: {exc}）；"
                f"报告本身不受影响，但今天没有核对过持仓")
            return None

    def _page_symbols(self, report: DailyReport) -> list[str]:
        """Every name the daily page will link to, in priority order.

        Priority is the order the fundamentals budget is spent in: the names
        being acted on first, the ones merely watched last.
        """
        order: list[str] = []
        for group in ([r.symbol for r in report.buys],
                      [s.symbol for s in report.sells],
                      [o.symbol for o in report.open_ideas],
                      [c.holding.symbol for c in report.core],
                      [d.symbol for d in report.daytrade],
                      [w.symbol for w in report.watchlist]):
            for sym in group:
                sym = (sym or "").upper()
                if sym and sym not in order:
                    order.append(sym)
        return order

    def _roles(self, symbol: str, report: DailyReport) -> list[str]:
        roles = []
        if any(r.symbol == symbol for r in report.buys):
            roles.append("buy")
        if any(s.symbol == symbol for s in report.sells):
            roles.append("sell")
        if any(o.symbol == symbol for o in report.open_ideas):
            roles.append("open")
        if any(c.holding.symbol == symbol for c in report.core):
            roles.append("core")
        if any(d.symbol == symbol for d in report.daytrade):
            roles.append("daytrade")
        if any(w.symbol == symbol for w in report.watchlist):
            roles.append("watch")
        return roles

    def build_analyses(self, report: DailyReport, data_date: str) -> dict:
        """Assemble one :class:`~.deepdive.SymbolAnalysis` per linked name.

        Runs last and swallows everything: a page that fails to build must cost
        that page and nothing else. The daily report is already complete by the
        time this is called.
        """
        wanted = self._page_symbols(report)
        if not wanted:
            return {}
        names = ZhNames()
        by_symbol: dict[str, list] = {}
        for item in getattr(self, "_news_items", []) or []:
            if getattr(item, "ticker", ""):
                by_symbol.setdefault(item.ticker, []).append(item)

        stats = {}
        if self.cfg.with_fundamentals:
            budget = wanted[:max(0, self.cfg.max_fundamentals)]
            if len(wanted) > len(budget):
                report.notes.append(
                    f"财报数据只拉取了 {len(budget)}/{len(wanted)} 只"
                    f"（max_fundamentals）；其余个股页面没有财报一节")
            try:
                book = self._fundamentals
                if book is None:
                    book = self._fundamentals = fund.FundamentalsBook()
                stats = book.get(budget, log=logger.debug)
            except Exception as exc:
                report.warnings.append(
                    f"财报数据源不可用（{type(exc).__name__}: {exc}）；"
                    f"个股页面只有价格与消息")

        watch_by = {w.symbol: w for w in report.watchlist}
        cand_by = {c.symbol: c for c in report.candidates}
        core_by = {c.holding.symbol: c for c in report.core}
        out: dict[str, deepdive.SymbolAnalysis] = {}
        for sym in wanted:
            try:
                cand = cand_by.get(sym)
                watch = watch_by.get(sym)
                english = (getattr(cand, "name", "") or getattr(watch, "name", "")
                           or getattr(stats.get(sym), "name", "") or "")
                bars = self.bars(sym, data_date)
                snap = self.snapshot(sym, data_date) if bars.ok else None
                excess = dict(getattr(watch, "excess_1m", {}) or {})
                a = deepdive.SymbolAnalysis(
                    symbol=sym,
                    zh=names.get(sym, english),
                    roles=self._roles(sym, report),
                    sector=(getattr(cand, "sector", "") or
                            getattr(stats.get(sym), "sector", "") or ""),
                    tag=getattr(watch, "tag", "") or "",
                    bars=bars,
                    snap=snap,
                    fundamentals=stats.get(sym),
                    earnings=(report.earnings or {}).get(sym),
                    news=by_symbol.get(sym, []),
                    rec=next((r for r in report.buys if r.symbol == sym), None),
                    open_idea=next((o for o in report.open_ideas if o.symbol == sym), None),
                    exit_signal=next((s for s in report.sells if s.symbol == sym), None),
                    watch=watch,
                    core=core_by.get(sym),
                    daytrade=next((d for d in report.daytrade if d.symbol == sym), None),
                    tilt=_num(getattr(cand, "tilt", 0.0), 0.0),
                    screen_rank=_num(getattr(watch, "screen_rank", float("nan"))),
                    excess=excess,
                )
                if bars.ok:
                    a.trend = deepdive.build_trend(bars, snap, excess,
                                                   self.cfg.atr_stop_mult)
                out[sym] = a
            except Exception as exc:
                logger.warning("could not analyse %s: %s", sym, exc)
        return out

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
            panel_ran=self.panel is not None, swing_slots=self.cfg.swing_slots,
        )
        if self.panel is None:
            report.warnings.append(
                "no LLM configured: the ideas below were not reviewed by the panel, "
                "only screened and sized")

        account = self.account(report)
        self._account = account
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
        self.attach_earnings(cands, report, data_day, order_day)

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
        self._news_items = news
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
        # Before the buys, because the number still open is what decides how
        # many new ones the book has room for.
        report.open_ideas = self.review_open(data_day, report)

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
        slots = horizons.free_slots(len(report.open_ideas), self.cfg.swing_slots)
        if report.open_ideas:
            report.notes.append(
                f"波段仓位槽：{len(report.open_ideas)}/{self.cfg.swing_slots} 已占用，"
                f"今天最多再开 {slots} 个")
        report.buys = self.generate_buys(cands, account, news, macro, events,
                                         budget, order_day, data_date, report,
                                         limit=slots)

        inside = []
        for rec in report.buys:
            e = report.earnings.get(rec.symbol) if report.earnings else None
            if e is not None and e.reports_within(self.cfg.horizon_days, order_day):
                inside.append(f"{rec.symbol}({e.days_to_next(order_day):.0f}d)")
        if inside:
            report.warnings.append(
                f"earnings land inside the holding horizon for {', '.join(inside)}: "
                f"a two-ATR stop does not survive a gap, so the risk shown for "
                f"those is a floor, not a bound")

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

        # The long and the intraday books, last: both read the sections above,
        # and neither may change a single number in them.
        report.reconcile = self.reconcile_account(report)
        report.core = self.core_section(report, data_date, order_day)
        report.daytrade = self.daytrade_section(report, data_date)
        try:
            report.analysis = self.build_analyses(report, data_date)
        except Exception as exc:
            report.warnings.append(
                f"个股详情页无法生成（{type(exc).__name__}: {exc}）；"
                f"当日报告本身不受影响")
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


# --- shared by both renderers ----------------------------------------------

# The swing section's own horizon, printed so the reader knows which clock a
# row is on. See :mod:`~.horizons`.
SWING_NOTE = (
    "这一节的持有期是 1–4 周，不是当天。仓位槽满了就不再新增——"
    "报告每天看起来「换了一批股票」，通常是因为它只印了当天新发的建议，"
    "而没有印它一直拿着的那些。上面那一节就是它一直拿着的。"
)

CORE_NOTE = (
    "核心长仓不参与每日排名，也不因为今天跌了就动。它只在两种情况下改变："
    "每月复核时权重偏离超过 ±25%，或者触发成文的破位规则"
    "（收在 200 日线下方超过 4%，或近一年跌超 15%）。"
    "规则触发只意味着「该重新想一遍」，不等于卖出指令。"
)

NAME_NOTE = (
    f"公司名带 {DERIVED_MARK} 的，是按英文名后缀机械转写的（例如 Therapeutics→制药），"
    f"不是通用译名；想改成你习惯的叫法，编辑 ~/.tradingagents/company_names_zh.json。"
)


RECONCILE_NOTE = (
    "这一节比对的是「账本说该持有什么」和「券商账户里实际有什么」。它每天都在，"
    "因为这个缺口不会自己报错：advisor 只记录决策不下单，monitor 带 --no-llm 时"
    "永远不决策，于是战绩照常按「已成交」计分，而账户里可能是完全不同的东西。"
    "要真的补上差额，用 python -m tradingagents.live.execute --submit。"
)


def _names(report: "DailyReport") -> ZhNames:
    """One resolver per render, reused across every section of the page."""
    cached = getattr(report, "_zh_resolver", None)
    if cached is None:
        cached = ZhNames()
        report._zh_resolver = cached          # noqa: SLF001 - render-local cache
    return cached


def zh_label(report: "DailyReport", symbol: str) -> str:
    """The Chinese name for a symbol, from the analysis if it was built."""
    a = (report.analysis or {}).get((symbol or "").upper())
    if a is not None and a.zh is not None:
        return a.zh.label()
    return _names(report).label(symbol)


def page_link(report: "DailyReport", symbol: str) -> str:
    """Relative path to the symbol's page, or "" when no page was written.

    Read off the analysis rather than rebuilt from the date, because
    :func:`write_pages` sets it only for the symbols whose page actually
    rendered. A link to a file that does not exist is worse than no link: it
    reads as a promise the report keeps everywhere else.
    """
    a = (report.analysis or {}).get((symbol or "").upper())
    return getattr(a, "page", "") if a is not None else ""


def _linked(report: "DailyReport", symbol: str, text: str = "") -> str:
    """``text`` (default: the Chinese name) as a link to the symbol's page."""
    label = (text or zh_label(report, symbol) or symbol).replace("|", "\\|")
    href = page_link(report, symbol)
    return f"[{label}]({href})" if href else label


def _spark(report: "DailyReport", symbol: str, width: int = 14) -> str:
    a = (report.analysis or {}).get((symbol or "").upper())
    if a is None or a.bars is None or not a.bars.closes:
        return ""
    return charting.sparkline(a.bars.closes[-63:], width)


def _risks(analysis, limit: int) -> list[str]:
    """The bear case for one name, or nothing.

    Guarded because both renderers are the last thing standing: ``main`` prints
    :func:`format_report` unwrapped, and a report that raises while formatting a
    single risk bullet loses the whole page — including the sells, which are the
    part with a deadline. The absence of a bullet is a small loss; the absence
    of the page is not.
    """
    if analysis is None:
        return []
    try:
        return deepdive.risks(analysis)[:limit]
    except Exception as exc:
        logger.warning("could not read the risks for %s: %s",
                       getattr(analysis, "symbol", "?"), exc)
        return []


def _breadth(rows: list) -> list[str]:
    """What the watchlist as a whole is doing — the read a row-by-row table hides."""
    watched = [r for r in rows if not r.is_base]
    if not watched:
        return []
    rets = sorted((r.ret_1m for r in watched if not math.isnan(_num(r.ret_1m))))
    above200 = sum(1 for r in watched if r.above_200)
    both = sum(1 for r in watched if r.above_50 and r.above_200)
    out = [f"跟踪的 {len(watched)} 只里，{above200} 只在 200 日线上方、"
           f"{both} 只同时站上 50 与 200 日线。"]
    if rets:
        mid = rets[len(rets) // 2]
        out.append(f"近一月收益的中位数是 {mid * 100:+.1f}%，"
                   f"区间 {rets[0] * 100:+.1f}% 到 {rets[-1] * 100:+.1f}%——"
                   f"{'离散度很大，说明这是选股行情而不是板块行情' if rets[-1] - rets[0] > 0.4 else '离散度不大，板块内部走势比较一致'}。")
    by_1m = sorted((r for r in watched if not math.isnan(_num(r.ret_1m))),
                   key=lambda r: -r.ret_1m)
    if len(by_1m) >= 3:
        lead = "、".join(f"{r.symbol} {r.ret_1m * 100:+.0f}%" for r in by_1m[:3])
        lag = "、".join(f"{r.symbol} {r.ret_1m * 100:+.0f}%" for r in by_1m[-3:])
        out.append(f"领涨：{lead}；落后：{lag}。")
    tags = {}
    for r in watched:
        if r.tag and not math.isnan(_num(r.ret_1m)):
            tags.setdefault(r.tag, []).append(r.ret_1m)
    if len(tags) > 1:
        parts = [f"{tag} 组均值 {sum(v) / len(v) * 100:+.1f}%（{len(v)} 只）"
                 for tag, v in sorted(tags.items(),
                                      key=lambda kv: -sum(kv[1]) / len(kv[1]))]
        out.append("按标签分组：" + "、".join(parts) + "。")
    return out


def _summary(report: "DailyReport") -> list[str]:
    """The paragraph that says what today's page actually is.

    Written because the previous page opened with a policy dump and a table:
    a reader had to assemble the answer to "what changed and what do I do"
    from six sections. This states it once, in order of what needs a decision.
    """
    out: list[str] = []
    ctx = (report.market_context or "").strip().splitlines()
    if ctx:
        out.append(f"**大盘**：{ctx[0].strip()}")

    urgent = [s for s in report.sells if s.urgency >= 3]
    breached = [c for c in report.core if getattr(c, "breached", False)]
    near_stop = [o for o in report.open_ideas if o.status == "贴近止损"]
    todo = []
    if urgent:
        todo.append(f"{len(urgent)} 笔紧急离场（{'、'.join(s.symbol for s in urgent)}）")
    if report.sells and not urgent:
        todo.append(f"{len(report.sells)} 笔常规离场")
    if report.buys:
        todo.append(f"{len(report.buys)} 笔新增波段建议")
    if near_stop:
        todo.append(f"{len(near_stop)} 个在场仓位贴近止损")
    if breached:
        todo.append(f"{len(breached)} 只核心长仓触发破位规则"
                    f"（{'、'.join(c.holding.symbol for c in breached)}）")
    out.append("**今天需要动的**：" + ("；".join(todo) + "。" if todo
               else "没有。持有的继续持有，这是最常见的一天。"))

    held = len(report.open_ideas)
    out.append(f"**书上的仓位**：核心长仓 {len(report.core)} 只（月度复核，"
               f"{'本月是复核月' if report.core_review_day else '本月不复核'}）、"
               f"在场波段 {held}/{report.swing_slots} 个仓位槽、"
               f"今日新增 {len(report.buys)} 个。核心与波段是两本账，不要互相挪用仓位。")

    if report.buys:
        risk = report.total_risk
        share = (f"，占净值 {risk / report.account_value * 100:.2f}%"
                 if report.account_value > 0 else "")
        rs = [r.planned_r() for r in report.buys if not math.isnan(r.planned_r())]
        rline = (f"，R 从 {min(rs):.2f} 到 {max(rs):.2f}" if rs else "")
        out.append(f"**新增建议的风险**：全部止损同时被打掉会亏约 ${risk:,.0f}{share}"
                   f"{rline}。成本合计 ${report.total_cost:,.0f}。")

    tilt = {k: v for k, v in (report.sector_tilt or {}).items() if abs(v) > 0.2}
    if tilt:
        best = sorted(tilt.items(), key=lambda kv: -kv[1])
        up = "、".join(f"{k} {v:+.2f}" for k, v in best[:3] if v > 0)
        down = "、".join(f"{k} {v:+.2f}" for k, v in best[-3:] if v < 0)
        out.append(f"**政策倾斜**：顺风 {up or '无'}；逆风 {down or '无'}。"
                   f"它只挪动排名，不会凭空造出或否决一个候选。")

    if report.watchlist:
        out += ["**关注列表的结构**：" + " ".join(_breadth(report.watchlist))]

    out.append("**这份报告不知道的事**：你的税、你的其它持仓、你的现金需求，"
               "以及任何没有出现在价格、财报和过去 24 小时新闻里的东西。"
               "每一节下面都留了原始数据和外部链接，是给你自己复核用的。")
    return out


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

    summary = _summary(report)
    if summary:
        out += ["", "今日综述 / SUMMARY", "-" * W]
        for line in summary:
            out += _wrapped(line.replace("**", ""), W - 4)

    if report.policy_summary:
        out += ["", "POLICY BACKDROP", "-" * W, report.policy_summary.strip()]

    tilt = {k: v for k, v in (report.sector_tilt or {}).items() if abs(v) > 0.05}
    if tilt:
        out.append("  sector tilt: " + "  ".join(
            f"{k} {v:+.2f}" for k, v in sorted(tilt.items(), key=lambda kv: -abs(kv[1]))))

    # Core first: it is the book that should not change, and printing it above
    # the day's activity is the whole point of separating the horizons.
    if report.core:
        out += ["", f"CORE — 核心长仓 ({len(report.core)}, "
                    f"{'本月复核' if report.core_review_day else '月度复核，本月不动'})",
                "-" * W,
                f"  {'SYMBOL':<8}{'PRICE':>10}{'vs200':>8}{'6M':>8}{'12M':>8}"
                f"  {'TREND':<16}{'STATUS':<10}ACTION / 公司"]
        for c in sorted(report.core, key=lambda x: (not x.breached, x.holding.symbol)):
            flag = "!" if c.breached else " "
            out.append(f" {flag}{c.holding.symbol:<8}{c.price:>10,.2f}"
                       f"{_pctn(c.ext_200):>8}{_pctn(c.ret_6m):>8}{_pctn(c.ret_12m):>8}"
                       f"  {(c.spark or '')[:14]:<16}{c.status:<10}"
                       f"{c.action}  {zh_label(report, c.holding.symbol)}")
        out += _wrapped(CORE_NOTE, W - 4)

    if report.open_ideas:
        out += ["", f"OPEN SWING — 在场的波段建议 ({len(report.open_ideas)})", "-" * W,
                f"  {'SYMBOL':<8}{'DAYS':>5}{'PRICE':>10}{'STOP':>10}{'TARGET':>10}"
                f"{'R':>7}{'→STOP':>8}{'→TGT':>8}  STATUS / 公司"]
        for o in report.open_ideas:
            out.append(f"  {o.symbol:<8}{o.days_held:>5}{o.price:>10,.2f}"
                       f"{_num(getattr(o.rec, 'stop_price', None)):>10,.2f}"
                       f"{_num(getattr(o.rec, 'target_price', None)):>10,.2f}"
                       f"{o.r_now:>+7.2f}{_pctn(o.to_stop):>8}{_pctn(o.to_target):>8}"
                       f"  {o.status}  {zh_label(report, o.symbol)}")
        out += _wrapped(SWING_NOTE, W - 4)

    # Sells before buys: freed capital funds the buys, and a book nobody prunes
    # only ever grows.
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
        # The narrative, after the table on purpose: the table is what a reader
        # acts on, the paragraphs are what they argue with.
        for r in report.buys:
            out += [""] + _buy_terminal(report, r, W)
    else:
        out.append("  (no candidate cleared the bar — this is a normal outcome)")

    if report.daytrade:
        out += ["", f"DAY TRADE — 日内盯盘 ({len(report.daytrade)})", "-" * W,
                f"  {'SYMBOL':<8}{'CLOSE':>10}{'PDH':>10}{'PDL':>10}{'TRIGGER':>10}"
                f"{'STOP':>10}{'TARGET':>10}{'ATR%':>7}{'RVOL':>6}  公司"]
        for d in report.daytrade:
            out.append(f"  {d.symbol:<8}{d.price:>10,.2f}{d.prev_high:>10,.2f}"
                       f"{d.prev_low:>10,.2f}{d.long_trigger:>10,.2f}"
                       f"{d.long_stop:>10,.2f}{d.long_target:>10,.2f}"
                       f"{_pctn(d.atr_pct):>7}{d.rvol:>6.1f}  {zh_label(report, d.symbol)}")
        out += _wrapped(horizons.DAYTRADE_CAVEAT, W - 4)

    if report.watchlist:
        rows = report.watchlist
        bases = [r.symbol for r in rows if r.is_base]
        weak = sum(1 for r in rows if not r.is_base and not r.passes_filter)
        watched = sum(1 for r in rows if not r.is_base)
        out += ["", f"WATCHLIST  ({watched}, {weak} below the screen's bar)", "-" * W]
        for line in _breadth(rows):
            out += _wrapped(line, W - 4)
        out.append("")
        head = (f"  {'SYMBOL':<8}{'TAG':<6}{'PRICE':>10}{'1M%':>7}{'3M%':>7}")
        for b in bases:
            head += f"{'vs ' + b:>9}"
        head += f"{'OFF HIGH':>10}{'vs200':>7}  {'TREND':<16}STATUS"
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
            line += (f"{_pctn(r.off_high):>10}{_pctn(r.ext_200):>7}  "
                     f"{_spark(report, r.symbol, 14):<16}{status}")
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

    rc = report.reconcile
    if rc is not None:
        n = len(rc.to_open) + len(rc.to_close)
        out += ["", f"BOOK vs ACCOUNT — 账本与账户", "-" * W]
        if rc.clean and not rc.unmanaged:
            out.append("  一致：账户持有的正是账本上还开着的那些。")
        if rc.to_close:
            out.append(f"  要卖出 {len(rc.to_close)}：" + "、".join(
                f"{i.symbol} {i.shares}股" for i in rc.to_close))
        if rc.to_open:
            out.append(f"  要建仓 {len(rc.to_open)}：" + "、".join(
                f"{i.symbol} {i.shares}股" for i in rc.to_open))
        if rc.stale:
            out.append(f"  过期未成交 {len(rc.stale)}（不会自动补单）：" + "、".join(
                f"{i.symbol} 放了{age}天" for i, age, *_ in rc.stale))
        if rc.drift:
            out.append("  股数对不上：" + "、".join(
                f"{s_} 账本{w}/账户{h_}" for s_, w, h_ in rc.drift))
        if rc.unmanaged:
            out.append(f"  账户里有、账本不认识 {len(rc.unmanaged)}（不会动）：" + "、".join(
                h.symbol for h in rc.unmanaged))
        out += _wrapped(RECONCILE_NOTE, W - 4)

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


def _buy_terminal(report: "DailyReport", rec, width: int) -> list[str]:
    """One buy's reasoning, for the terminal. The markdown gets a richer version."""
    a = (report.analysis or {}).get(rec.symbol)
    head = f"  {rec.symbol} · {zh_label(report, rec.symbol)}"
    if a is not None and a.spark:
        head += f"   {a.spark}"
    out = [head]
    if a is not None and a.trend is not None and not a.trend.error:
        out += _wrapped(f"图形：{a.trend.ma_stack}｜{a.trend.structure}｜"
                        f"{a.trend.momentum}", width - 4)
        out += _wrapped(f"读图结论：{a.trend.verdict}", width - 4)
    r = rec.planned_r()
    if not math.isnan(r):
        out += _wrapped(
            f"算术：{r:.2f}R，盈亏平衡胜率 {1 / (1 + r) * 100:.0f}%；"
            f"止损 {rec.stop_price:,.2f} 在参考价下方 "
            f"{(1 - rec.stop_price / rec.reference_price) * 100:.1f}%，"
            f"打掉约亏 ${planned_risk(rec):,.0f}", width - 4)
    for line in _risks(a, 1):
        out += _wrapped(f"风险：{line}", width - 4)
    link = page_link(report, rec.symbol)
    if link:
        out.append(f"    完整分析：{link}")
    return out


def _md_buy_narrative(report: "DailyReport", rec) -> list[str]:
    """The paragraph under the buy table: what the row cannot say."""
    sym = rec.symbol
    a = (report.analysis or {}).get(sym)
    out = [f"#### {sym} · {zh_label(report, sym)}", ""]
    if a is not None and a.spark:
        out += [f"近三个月走势 `{a.spark}`  ", ""]
    if a is not None and a.trend is not None and not a.trend.error:
        out += [f"> {a.trend.verdict}", ""]
        for k, v in a.trend.bullets()[:4]:
            out.append(f"- **{k}**：{v}")
    r = rec.planned_r()
    if not math.isnan(r) and rec.reference_price:
        out.append(
            f"- **这笔交易的算术**：R = {r:.2f}，盈亏平衡胜率 "
            f"{1 / (1 + r) * 100:.0f}%；止损 {rec.stop_price:,.2f} 在参考价下方 "
            f"{(1 - rec.stop_price / rec.reference_price) * 100:.1f}%，"
            f"止损被打掉约亏 ${planned_risk(rec):,.0f}"
            + (f"（净值的 {planned_risk(rec) / report.account_value * 100:.2f}%）"
               if report.account_value > 0 else "") + "。")
    cat = catalyst_line(rec)
    if cat:
        url = getattr(rec, "catalyst_url", "")
        out.append(f"- **催化剂**：{f'[{cat}]({url})' if url else cat}")
    e = (report.earnings or {}).get(sym)
    if e is not None and getattr(e, "next_date", ""):
        out.append(f"- **财报日历**：下次 {e.next_date}（约 {e.days_to_next():.0f} 天后）"
                   f"{'——落在持有期内，跳空会直接穿过止损' if e.reports_within(rec.horizon_days) else ''}")
    for line in _risks(a, 2):
        out.append(f"- **反方**：{line}")
    link = page_link(report, sym)
    if link:
        out += ["", f"[→ {sym} 完整分析：图形、财报、消息、原始数据与外部链接]({link})"]
    out.append("")
    return out


def to_markdown(report: "DailyReport") -> str:
    if report.refused:
        return "\n".join(["# Daily Advisor — no report", "", report.refused, ""])

    md = [f"# Daily Advisor — {report.date} · 每日投研简报", "",
          f"_{_sess(report)}_", ""]

    summary = _summary(report)
    if summary:
        md += ["## 今日综述", ""] + [f"- {line}" for line in summary] + [""]

    if report.market_context:
        md += ["## 市场环境 Market", "", "```text",
               report.market_context.strip(), "```", ""]
    if report.policy_summary:
        md += ["## 政策与政治背景 Policy backdrop", "",
               report.policy_summary.strip(), ""]

    # --- 核心长仓 ---
    md += [f"## 一、核心长仓（长期 · {'本月复核' if report.core_review_day else '月度复核，本月不动'}）", ""]
    if report.core:
        md += ["| 代码 | 公司 | 目标权重 | 现价 | 近 3 月走势 | 距 200 日 | 近 6 月 | 近 12 月 | 状态 | 动作 |",
               "|---|---|---:|---:|---|---:|---:|---:|---|---|"]
        for c in sorted(report.core, key=lambda x: (not x.breached, x.holding.symbol)):
            sym = c.holding.symbol
            md.append(f"| {sym} | {_linked(report, sym)} | "
                      f"{c.holding.weight * 100:.1f}% | {c.price:,.2f} | "
                      f"`{_spark(report, sym, 16) or ' '}` | {_pct(c.ext_200)} | "
                      f"{_pct(c.ret_6m)} | {_pct(c.ret_12m)} | "
                      f"{'**' + c.status + '**' if c.breached else c.status} | {c.action} |")
        breached = [c for c in report.core if c.breached]
        if breached:
            md += [""] + [f"- **{c.holding.symbol}**：{c.note}" for c in breached]
        md += ["", f"_{CORE_NOTE}_", ""]
    else:
        md += ["_还没有核心长仓名单。_ 在 `~/.tradingagents/core.json` 里写上你要长期持有的"
               "代码与目标权重（例如 `{\"MSFT\": 0.08}`），这一节就会每天跟踪它们，"
               "并且只在月度复核或破位时才提出改动。", ""]

    # --- 在场波段 ---
    md += [f"## 二、在场的波段建议（{len(report.open_ideas)} 个仓位，继续持有）", ""]
    if report.open_ideas:
        md += ["| 代码 | 公司 | 建议日 | 持有 | 现价 | 止损 | 目标 | 浮动 R | 距止损 | 距目标 | 状态 |",
               "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
        for o in report.open_ideas:
            rec = o.rec
            md.append(f"| {o.symbol} | {_linked(report, o.symbol)} | "
                      f"{getattr(rec, 'issued_date', '—')} | {o.days_held}天 | "
                      f"{o.price:,.2f} | {_num(getattr(rec, 'stop_price', None)):,.2f} | "
                      f"{_num(getattr(rec, 'target_price', None)):,.2f} | "
                      f"{o.r_now:+.2f} | {_pct(o.to_stop)} | {_pct(o.to_target)} | "
                      f"{o.status} |")
        md += ["", f"_{SWING_NOTE}_", ""]
    else:
        md += ["_书上没有在场的波段建议。_", ""]

    # --- 卖出 ---
    md += [f"## 三、卖出 / 离场 Sell ({len(report.sells)})", ""]
    if report.sells:
        md += ["| Symbol | 公司 | Action | Shares | Price | P&L | R | Reason |",
               "|---|---|---|---:|---:|---:|---:|---|"]
        md += [f"| {s.symbol} | {_linked(report, s.symbol)} | {s.action} | "
               f"{s.shares:.0f} | {s.price:,.2f} | "
               f"{s.pnl:+,.2f} | {s.r_multiple:+.2f} | {s.reason} |"
               for s in sorted(report.sells, key=lambda x: -x.urgency)]
        if report.closed:
            md += ["", "Closed in the book: " + ", ".join(
                f"{r.symbol} ({r.exit_reason or 'exit'})" for r in report.closed)]
    else:
        md.append("_Nothing to exit._")

    # --- 新增波段 ---
    md += ["", f"## 四、新增波段建议 Buy ({len(report.buys)}) · 持有 1–4 周", ""]
    if report.buys:
        md += ["| Symbol | 公司 | 走势 | Shares | Ref | Limit | Stop | Target | R | Cost | Risk | Why |",
               "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        for r in report.buys:
            # See format_report: the R shown is the R as issued.
            rr = r.planned_r()
            lim = f"{r.limit_price:,.2f}" if r.limit_price else "MKT"
            md.append(f"| {r.symbol} | {_linked(report, r.symbol)} | "
                      f"`{_spark(report, r.symbol, 12) or ' '}` | "
                      f"{r.shares:.0f} | {r.reference_price:,.2f} | {lim} | "
                      f"{r.stop_price:,.2f} | {r.target_price:,.2f} | {rr:.2f} | "
                      f"{estimated_cost(r):,.2f} | {planned_risk(r):,.2f} | "
                      f"{_md_catalyst(r)} |")
        md += ["", f"_{REFERENCE_NOTE}_", "", "### 逐只分析", ""]
        for r in report.buys:
            md += _md_buy_narrative(report, r)
    else:
        md.append("_No candidate cleared the bar._")

    # --- 日内 ---
    if report.daytrade:
        md += ["", f"## 五、日内盯盘 Day trade ({len(report.daytrade)})", "",
               "| 代码 | 公司 | 收盘 | 昨日高 | 昨日低 | 向上触发 | 止损 | 目标 | 向下触发 | ATR% | 相对量 |",
               "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for d in report.daytrade:
            md.append(f"| {d.symbol} | {_linked(report, d.symbol)} | {d.price:,.2f} | "
                      f"{d.prev_high:,.2f} | {d.prev_low:,.2f} | "
                      f"**{d.long_trigger:,.2f}** | {d.long_stop:,.2f} | "
                      f"{d.long_target:,.2f} | {d.short_trigger:,.2f} | "
                      f"{_pct(d.atr_pct).lstrip('+')} | {d.rvol:.1f}× |")
        md += ["", f"_{horizons.DAYTRADE_CAVEAT}_", ""]

    # --- 关注列表 ---
    if report.watchlist:
        rows = report.watchlist
        bases = [r.symbol for r in rows if r.is_base]
        weak = sum(1 for r in rows if not r.is_base and not r.passes_filter)
        watched = sum(1 for r in rows if not r.is_base)
        md += ["", f"## 六、长期关注列表 Watchlist ({watched})", ""]
        md += [f"- {line}" for line in _breadth(rows)]
        md += ["", f"_{weak} of {watched} would not have cleared the screen's own bar today._", ""]
        header = "| Symbol | 公司 | Tag | Price | 走势 | 1M | 3M |"
        divide = "|---|---|---|---:|---|---:|---:|"
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
            row = (f"| {r.symbol} | {_linked(report, r.symbol)} | {r.tag} | "
                   f"{r.price:,.2f} | `{_spark(report, r.symbol, 12) or ' '}` | "
                   f"{_pct(r.ret_1m)} | {_pct(r.ret_3m)} |")
            for b in bases:
                row += f" {_pct(r.excess_1m.get(b, float('nan')))} |"
            row += f" {_pct(r.off_high)} | {_pct(r.ext_200)} | {status} |"
            md.append(row)
        md += ["", f"_{WATCHLIST_NOTE}_"]

    md += ["", "## 账户 Account", "",
           f"- Equity **${report.account_value:,.2f}**, cash ${report.cash:,.2f}",
           f"- Buy budget ${report.buy_budget:,.2f}, risk budget "
           f"{report.risk_pct:.2f}%/trade"]
    if report.buys:
        share = (f" ({report.total_risk / report.account_value:.2%} of equity)"
                 if report.account_value > 0 else "")
        md.append(f"- Planned risk ${report.total_risk:,.2f}{share} across "
                  f"{len(report.buys)} idea(s), after the position cap")
    rc = report.reconcile
    if rc is not None:
        md += ["", "## 账本与账户 Book vs account", ""]
        if rc.clean and not rc.unmanaged:
            md.append("_一致：账户持有的正是账本上还开着的那些。_")
        rows = []
        for i in rc.to_close:
            rows.append(("要卖出", i.symbol, f"{i.shares:,} 股", i.reason))
        for i in rc.to_open:
            rows.append(("要建仓", i.symbol, f"{i.shares:,} 股",
                         (f"限价 {i.limit:,.2f}" if i.limit else "市价") + "　" + i.reason))
        for i, age, r_now, px, to_stop in rc.stale:
            rows.append(("过期未成交", i.symbol, f"放了 {age} 天",
                         f"现价 {px:,.2f} · 现在 {r_now:.2f}R · 距止损 "
                         f"{to_stop * 100:+.1f}% — {execute._stale_verdict(r_now, to_stop)}"
                         if math.isfinite(px) else "定不了价"))
        for sym, want, have in rc.drift:
            rows.append(("股数对不上", sym, f"账本 {want:,} / 账户 {have:,}", "手工核对"))
        for h in rc.unmanaged:
            rows.append(("账本不认识", h.symbol, f"{_num(h.quantity, 0):,.0f} 股",
                         "不会被动——仅因不在这本账里就卖掉，等于这座桥认为整个账户都归它管"))
        if rows:
            md += ["| 类别 | 代码 | 数量 | 说明 |", "|---|---|---|---|"]
            md += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
        md += ["", f"_{RECONCILE_NOTE}_"]

    if report.track_record:
        try:
            md += ["", "## 战绩 Track record", "", "```text",
                   format_track_record(report.track_record), "```"]
        except Exception:
            pass
    if report.notes:
        md += ["", "## 说明 Notes", ""] + [f"- {n}" for n in report.notes]
    if report.warnings:
        md += ["", "## 警告 Warnings", ""] + [f"- {w}" for w in report.warnings]
    md += ["", "---", "", f"_{NAME_NOTE}_", "", f"_{CAVEAT}_"]
    return "\n".join(md)


def write_pages(report: "DailyReport", directory: Path) -> list[Path]:
    """One deep-dive page per analysed symbol. Returns what was written.

    Written *before* the daily page and recorded on the report, because
    :func:`page_link` refuses to link to a file that does not exist: a broken
    link in the one section a reader is told to click is worse than no link.

    A page that fails is skipped and logged. The daily report is already
    complete by this point and must not be lost to a rendering error in one
    symbol's chart.
    """
    written: list[Path] = []
    if not report.analysis:
        return written
    directory.mkdir(parents=True, exist_ok=True)
    back = f"../{report.date}.md"
    for symbol, analysis in report.analysis.items():
        try:
            text = deepdive.render_page(
                analysis, report_date=report.date, data_date=report.data_date,
                account_value=report.account_value,
                atr_stop_mult=DEFAULT_ATR_STOP_MULT, back_link=back)
        except Exception as exc:
            logger.warning("could not render the page for %s: %s", symbol, exc)
            continue
        path = directory / f"{symbol}.md"
        tmp = path.with_name(path.name + ".tmp")
        try:
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, path)
        except Exception as exc:
            logger.warning("could not write %s: %s", path, exc)
            continue
        analysis.page = f"{directory.name}/{symbol}.md"
        written.append(path)
    return written


def save_report(report: "DailyReport") -> Path:
    """Write the page and its per-symbol pages, atomically.

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

    # The symbol pages first: the daily page links to them, and page_link()
    # emits a link only for a symbol that has one.
    if report.analysis and not report.refused:
        try:
            write_pages(report, path.parent / stem)
        except Exception as exc:
            logger.warning("the per-symbol pages could not be written: %s", exc)

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
    p.add_argument("--swing-slots", type=int, default=None,
                   help="concurrent swing positions the book may carry (default 6); "
                        "new ideas only fill free slots")
    p.add_argument("--no-pages", action="store_true",
                   help="skip the per-symbol deep-dive pages and their statement fetch")
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
                     ("max_candidates", "max_candidates"), ("exchange", "exchange"),
                     ("swing_slots", "swing_slots")):
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
    if getattr(a, "no_pages", False):
        cfg.write_pages = False
        cfg.with_fundamentals = False

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
