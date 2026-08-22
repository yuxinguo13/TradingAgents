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

So this runs once a day, after the close, against the day's finished bars, and
prints a page for a human to act on at the next open. Nothing here places an
order. Every idea is written into :mod:`recommendations` at the levels it was
issued with, which is what makes it checkable a month later.

Three ordering decisions in :meth:`DailyAdvisor.run` are load-bearing:

* **Sells are decided before buys.** Two reasons. The proceeds of a sale are
  what fund a purchase, so the buy budget is not knowable until the exits are
  known; and an agent that generates ideas before managing what it already
  holds will accumulate positions forever, because nothing in the buy path ever
  asks what the book already contains.
* **The R filter runs after sizing, not before.** A share count comes from the
  stop distance, and the stop is also half of R — computing them in the other
  order means rejecting on a number the size was not derived from.
* **Policy tilts the ranking; it never creates or destroys a candidate.**
  :func:`~.policy.sector_pressure` says so about itself, and it is right: the
  sector map is a table of hand-written priors, not a model. Here it may move a
  name a few places, bounded by ``MAX_TILT_SHIFT``, and nothing more.

The honest caveat, stated once here and printed on every report: this is a
shortlist produced by a model reading public information. It is not advice, it
knows nothing about the reader's circumstances, and every convention in it —
the two-ATR stop, the trend-extrapolated target, the minimum R — is written
down in this repository rather than established by anyone. The track record at
the foot of the report is the only part that is checkable, which is why every
idea goes into the book whether it works or not.

    python -m tradingagents.live.advisor --top 8 --risk-pct 1.0
    python -m tradingagents.live.advisor --use-cache --no-llm --dry-run
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import date as _date, datetime, timedelta
from pathlib import Path

from . import clock
from .brain import Panel, Snapshot, Trigger, build_evidence, snapshot, triggers
from .broker import BUY as VENUE_BUY, Account
from .newsfeed import NewsItem, NewsMonitor
from .policy import PolicyEvent, PolicyMonitor, policy_brief, sector_pressure
from .recommendations import (
    BUY as ADVICE_BUY,
    DEFAULT_CONVICTION,
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
# The most places a sector tilt of +/-1.0 may move a name on the shortlist.
# Bounded rather than weighted: an additive score adjustment can promote the
# fortieth name over the first if the tilt is large enough, and this table of
# hand-written priors has not earned that. Five places reorders neighbours,
# which is exactly what sector_pressure's own docstring says it is for.
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


def data_date_for(report_day: _date, now: datetime | None = None) -> _date:
    """The session whose bars this report may read.

    For today this is :func:`clock.last_trading_day`, which is *not* today
    before 09:30: no bar exists yet, and asking for one returns yesterday's
    data labelled with today's date — the look-ahead confusion the verified
    loader exists to prevent. For a past date it is that date, walked back to
    the previous session if it was a weekend or a holiday.
    """
    if report_day >= _date.today():
        return clock.last_trading_day(now)
    d = report_day
    for _ in range(10):
        if clock.is_trading_day(d):
            return d
        d -= timedelta(days=1)
    return d


def estimated_cost(rec: Recommendation) -> float:
    """What the buy costs at the price the instruction would fill at."""
    px = _num(rec.limit_price, _num(rec.reference_price))
    return 0.0 if math.isnan(px) else px * rec.shares


def planned_risk(rec: Recommendation) -> float:
    """Dollars lost if the stop as issued fills. NaN when it cannot be read."""
    return rec.risk_amount()


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
class DailyReport:
    """One day's output. Rendered by :func:`format_report` and :func:`to_markdown`."""

    date: str
    generated_at: str = ""
    data_date: str = ""

    buys: list[Recommendation] = field(default_factory=list)
    sells: list[ExitSignal] = field(default_factory=list)

    policy_summary: str = ""
    policy_events: list[PolicyEvent] = field(default_factory=list)
    sector_tilt: dict[str, float] = field(default_factory=dict)
    market_context: str = ""

    track_record: TrackRecord | None = None
    candidates: list[Candidate] = field(default_factory=list)
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
        # Cleared rather than loaded, for the same reason the news seen-set is:
        # the loop's question is "what is new since I last looked" and this
        # report's is "what is standing right now". A report that ran twice in
        # one morning must not have an empty backdrop the second time.
        self.policy.seen = {}
        try:
            events = self.policy.poll(pause=self.cfg.feed_pause)
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
        self.news.seen = {}
        try:
            items = self.news.poll(symbols, macro=True, pause=self.cfg.feed_pause)
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

    def review_exits(self, news_by_symbol: dict[str, list[NewsItem]], when: _date,
                     data_date: str, report: DailyReport) -> list[ExitSignal]:
        """Price every open idea and say which are done.

        Runs before a single buy is generated. The proceeds of these sales are
        the budget the buy side spends, and an advisor that produced ideas
        first would never look at what it already holds — which is how a book
        becomes forty names nobody chose.
        """
        prices: dict[str, float] = {}
        for rec in self.book.open_recommendations():
            snap = self.snapshot(rec.symbol, data_date)
            if snap.ok and snap.price > 0:
                prices[rec.symbol] = snap.price
                report.marks[rec.symbol] = snap.price
        try:
            signals = self.book.review(
                prices, news_by_symbol, as_of=when, rules=self.exit_rules,
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

    # --- 2. candidates ----------------------------------------------------

    def candidates(self, when: _date, data_date: str, report: DailyReport) -> list[Candidate]:
        """The universe cut, from a cached screen where one will do.

        A screen is built from daily bars, so a second scan of the same session
        returns the same ranking at the full cost of the universe download —
        the argument :meth:`monitor.LiveDesk.refresh_screen` makes for running
        at most once a day. So a saved CSV for this exact session is used even
        when ``use_cache`` is off; ``use_cache`` additionally accepts an older
        one rather than scanning.
        """
        exchange = self.cfg.exchange
        path, saved_date = find_saved_screen(when, exchange)

        if path is not None and saved_date == when:
            report.notes.append(f"screen: reusing today's saved scan ({path.name})")
            return candidates_from_csv(path)

        if self.cfg.use_cache:
            if path is None:
                report.warnings.append(
                    f"--use-cache was asked for but no saved {exchange} screen exists "
                    f"in {screens_dirs()[0]}; there are no buy candidates today")
                return []
            stale = (when - saved_date).days if saved_date else 0
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

        report.notes.append(
            f"screen: {stats.get('universe', '?')} listed names → "
            f"{stats.get('passed_filters', '?')} passed the filters")
        return candidates_from_frame(frame)

    def rank(self, cands: list[Candidate], tilt: dict[str, float]) -> list[Candidate]:
        """Apply the policy tilt to the screen's order, bounded by MAX_TILT_SHIFT.

        A bounded shift in rank space rather than a weighted score, so the
        arithmetic can be stated in one sentence a reader can check: policy
        moves a name at most five places, and it can never put a name on the
        list that the screen did not.
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
        """
        open_ideas = {r.symbol for r in self.book.open_recommendations()}
        held = {h.symbol for h in account.holdings}
        out: list[Candidate] = []
        for c in cands:
            if not c.symbol:
                continue
            if c.symbol in open_ideas:
                c.reason = "already open in the recommendation book"
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
        snap = cand.snap or self.snapshot(cand.symbol, data_date)
        trigs = triggers(cand.symbol, snap, news, account, cand.rank)
        if not trigs:
            trigs = [Trigger(cand.symbol, "daily_review",
                             f"rank #{cand.rank} on the {self.cfg.exchange} screen"
                             + (f", sector tilt {cand.tilt:+.2f}" if cand.tilt else ""),
                             urgency=0)]
        evidence = build_evidence(cand.symbol, snap, news, account, trigs, macro,
                                  phase="daily report, for the next open")
        if events:
            evidence += "\n\n" + policy_brief(events)

        if self.panel is None:
            # No panel configured. The idea is still produced — a missing API
            # key must not turn the daily list into an empty page — but it is
            # marked, so a later reader of the book can separate ideas a panel
            # reviewed from ideas only the screen and the sizing rule saw.
            return True, DEFAULT_CONVICTION, (
                f"unreviewed: no panel ran. Rank #{cand.rank} on the "
                f"{self.cfg.exchange} momentum and accumulation screen.")

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
            return None, (f"no stop could be derived: ATR is "
                          f"{snap.atr_pct:.2%} of price, which puts the stop inside "
                          f"the tick the shares quote in")

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
        catalyst = cand.news[0].title[:140] if cand.news else (
            f"rank #{cand.rank} on the {self.cfg.exchange} screen"
            + (f"; sector tilt {cand.tilt:+.2f}" if cand.tilt else ""))

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
            issued_at=datetime.now().isoformat(),
        )
        return rec, sized.reason

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

    def run(self, when: str | _date | None = None) -> DailyReport:
        """Produce one day's report. Never raises.

        The order below is the argument of the module docstring in code: read
        the account, gather, decide the exits, and only then look for something
        to buy with what the exits freed.
        """
        if isinstance(when, _date):
            report_day = when
        elif when:
            try:
                report_day = _date.fromisoformat(str(when))
            except ValueError:
                logger.warning("unreadable date %r; using today", when)
                report_day = _date.today()
        else:
            report_day = _date.today()
        data_day = data_date_for(report_day)
        data_date = data_day.isoformat()

        report = DailyReport(
            date=report_day.isoformat(), data_date=data_date,
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
        report.policy_summary = policy_brief(events)
        report.sector_tilt = sector_pressure(events) if events else {}

        cands = self.rank(self.candidates(report_day, data_date, report),
                          report.sector_tilt)
        report.candidates = cands

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
        report.sells = self.review_exits(by_symbol, report_day, data_date, report)

        # Only a closing signal frees capital. A TRIM does too, but partially,
        # and this book stores one exit per idea and so cannot tell whether a
        # trim was taken — counting it would spend money that may still be in
        # the position.
        proceeds = 0.0
        for sig in report.sells:
            if sig.closes_position and not math.isnan(sig.price):
                proceeds += sig.price * sig.shares
        budget = max(0.0, (account.buying_power or account.cash)) + proceeds
        report.buy_budget = budget
        if proceeds:
            report.notes.append(
                f"${proceeds:,.0f} of the buy budget is proceeds from the sells above, "
                f"which only exist if you take them")

        # --- then buys ---
        report.buys = self.generate_buys(cands, account, news, macro, events,
                                         budget, report_day, data_date, report)

        for rec in report.buys:
            report.marks[rec.symbol] = rec.reference_price
        try:
            report.track_record = self.book.track_record(report.marks, report_day)
        except Exception as exc:
            report.warnings.append(f"the track record could not be computed: {exc}")

        report.market_context = self.market_context(data_date)
        return report


# ----------------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------------

def _sess(report: "DailyReport") -> str:
    """The two dates that must never be confused.

    The report is built from a *completed* session's data and acted on at the
    *next* open. Printing only one of them is how a track record later becomes
    unauditable — "did it know this before or after the move" is the only
    question that matters when reviewing a call.
    """
    return (f"数据截至 analysis as of {report.data_date} close"
            f"   →   下单于 orders for {report.date} open")


def format_report(report: "DailyReport") -> str:
    """The morning read. Terminal-friendly, aligned, no colour."""
    W = 104
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
    else:
        out.append("  (nothing to exit)")

    out += ["", f"BUY  ({len(report.buys)})", "-" * W]
    if report.buys:
        out.append(f"  {'SYMBOL':<8}{'SHARES':>7}{'REF':>10}{'LIMIT':>10}"
                   f"{'STOP':>10}{'TARGET':>10}{'R':>6}{'COST':>11}{'RISK':>9}  WHY")
        for r in report.buys:
            rr = r_multiple(r.reference_price, r.stop_price, r.target_price)
            lim = f"{r.limit_price:,.2f}" if r.limit_price else "MKT"
            out.append(
                f"  {r.symbol:<8}{r.shares:>7.0f}{r.reference_price:>10,.2f}"
                f"{lim:>10}{r.stop_price:>10,.2f}{r.target_price:>10,.2f}"
                f"{rr:>6.2f}{estimated_cost(r):>11,.2f}{planned_risk(r):>9,.2f}"
                f"  {(r.catalyst or r.rationale)[:32]}")
        out.append("-" * W)
        out.append(f"  {'TOTAL':<8}{'':>7}{'':>10}{'':>10}{'':>10}{'':>10}{'':>6}"
                   f"{sum(estimated_cost(r) for r in report.buys):>11,.2f}"
                   f"{sum(planned_risk(r) for r in report.buys):>9,.2f}")
    else:
        out.append("  (no candidate cleared the bar — this is a normal outcome)")

    out += ["", "ACCOUNT", "-" * W,
            f"  equity ${report.account_value:,.2f}   cash ${report.cash:,.2f}   "
            f"buy budget ${report.buy_budget:,.2f}   risk/trade {report.risk_pct:.2f}%"]

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

    out += ["", "=" * W,
            "  Not financial advice. A model reading public information.",
            "  The track record above is the only thing that makes it checkable.",
            "=" * W]
    return "\n".join(out)


def to_markdown(report: "DailyReport") -> str:
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
    else:
        md.append("_Nothing to exit._")

    md += ["", f"## Buy ({len(report.buys)})", ""]
    if report.buys:
        md += ["| Symbol | Shares | Ref | Limit | Stop | Target | R | Cost | Risk | Why |",
               "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        for r in report.buys:
            rr = r_multiple(r.reference_price, r.stop_price, r.target_price)
            lim = f"{r.limit_price:,.2f}" if r.limit_price else "MKT"
            md.append(f"| {r.symbol} | {r.shares:.0f} | {r.reference_price:,.2f} | {lim} | "
                      f"{r.stop_price:,.2f} | {r.target_price:,.2f} | {rr:.2f} | "
                      f"{estimated_cost(r):,.2f} | {planned_risk(r):,.2f} | "
                      f"{(r.catalyst or r.rationale)} |")
    else:
        md.append("_No candidate cleared the bar._")

    md += ["", "## Account", "",
           f"- Equity **${report.account_value:,.2f}**, cash ${report.cash:,.2f}",
           f"- Buy budget ${report.buy_budget:,.2f}, risk/trade {report.risk_pct:.2f}%"]
    if report.notes:
        md += ["", "## Notes", ""] + [f"- {n}" for n in report.notes]
    if report.warnings:
        md += ["", "## Warnings", ""] + [f"- {w}" for w in report.warnings]
    md += ["", "---", "",
           "_Not financial advice. A model reading public information._"]
    return "\n".join(md)


def save_report(report: "DailyReport") -> Path:
    path = reports_dir() / f"{report.date}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_markdown(report), encoding="utf-8")
    return path


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        prog="advisor",
        description="Daily buy/sell list for the next session.")
    p.add_argument("--date", default=None,
                   help="Session the orders are FOR (default: the next one). "
                        "Data is always taken from the last completed session.")
    p.add_argument("--top", type=int, default=None, help="candidates to consider")
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

    cfg = AdvisorConfig()
    for src, dst in (("top", "top"), ("risk_pct", "risk_pct"), ("min_r", "min_r"),
                     ("max_candidates", "max_candidates"), ("exchange", "exchange")):
        v = getattr(a, src, None)
        if v is not None and hasattr(cfg, dst):
            setattr(cfg, dst, v)
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
    if not report.dry_run:
        print(f"\nsaved: {save_report(report)}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
