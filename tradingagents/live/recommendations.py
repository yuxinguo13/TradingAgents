"""The recommendation book: what the desk advised, and how the advice turned out.

Deliberately a different book from the portfolio. A portfolio records what was
*filled*; this records what was *advised*, and conflating the two destroys both
records at once:

* The user may not act on a recommendation at all. Booked through a portfolio,
  a skipped idea leaves no trace — so an idea that would have made 3R and an
  idea nobody ever had are the same entry: none. You can no longer tell a bad
  idea from a good idea the user passed on.
* The user may size differently, or act two days late at a worse price. Scored
  through the fill, a good call taken badly and a bad call look identical.
* The portfolio also holds positions the desk never recommended — legacy
  holdings, the user's own trades. Scoring those as the desk's output credits
  or blames it for someone else's decisions.

So the books are kept apart. The portfolio answers "what do I own". This one
answers "was the advice any good", and it answers from the levels the advice
itself named: an entry, a stop and a target, all fixed at issue time.

Fixed is the whole design. A recommendation whose stop may be edited afterwards
has no record at all — every loss becomes "I would have been out by then" and
every win keeps its full R. ``initial_stop_price`` is therefore written once
and never touched, and realised R is always measured against it, even after the
trailing rule has moved the live stop to breakeven.

The second half of the module is the half the user actually asked for: exits. A
recommendation with no exit rule is not advice, it is an opinion.
:meth:`RecommendationBook.review` prices every open idea and says which are
done — stop hit, target reached, horizon spent without the thesis working, or a
fresh headline that breaks it.

    from tradingagents.live.recommendations import RecommendationBook
    book = RecommendationBook()
    book.add("NVDA", BUY, 40, 182.50, stop_price=168.00, target_price=215.00)
    for sig in book.review({"NVDA": 166.10}):
        print(sig)
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field, fields
from datetime import date, datetime
from pathlib import Path

from .sizing import LONG, SHORT, r_multiple

logger = logging.getLogger(__name__)

# --- vocabulary -------------------------------------------------------------
#
# These are NOT broker.BUY / broker.SELL, which spell the same words as "Buy"
# and "Sell" and belong to the venue's four-verb order vocabulary. A
# recommendation is advice, not an order: it names no order type, no venue and
# no time in force. Spelling the actions in caps keeps the two apart, so that
# passing a recommendation's action straight into ``place_order`` fails loudly
# at the adapter instead of quietly meaning something slightly different.
BUY = "BUY"
SELL = "SELL"
ACTIONS = (BUY, SELL)

# BUY is a long idea. SELL is the mirror — advice to get out of, or stay out
# of, a name — and it is scored as a short would be: it worked if the price
# fell. That is the only honest way to grade "we told you to sell at 200" once
# the stock is at 180.
_DIRECTIONS = {BUY: LONG, SELL: SHORT}

OPEN = "open"
CLOSED = "closed"
EXPIRED = "expired"
SUPERSEDED = "superseded"
STATUSES = (OPEN, CLOSED, EXPIRED, SUPERSEDED)

# Exit signal kinds. RAISE_STOP moves no shares; it is included because the
# caller has to be told the stop changed, and because a report that shows only
# sells hides the rule doing most of the work.
SELL_ALL = "SELL"
TRIM = "TRIM"
RAISE_STOP = "RAISE_STOP"
UNAVAILABLE = "UNAVAILABLE"

# Exit reasons, stored on the closed recommendation so the track record can be
# broken down by how ideas ended rather than only by whether they won.
REASON_STOP = "stop"
REASON_TARGET = "target"
REASON_TIME = "time_stop"
REASON_THESIS = "thesis_break"
REASON_MANUAL = "manual"

DEFAULT_HORIZON_DAYS = 30
DEFAULT_CONVICTION = 0.5

# Matches brain.NEWS_MATERIALITY_TRIGGER by value and is written out rather
# than imported: brain pulls pandas, numpy and the OHLCV loader, and this
# module must stay importable for a report command that needs none of them. If
# one moves, move the other.
THESIS_BREAK_MATERIALITY = 7
THESIS_BREAK_MAX_AGE_HOURS = 24.0

BREAKEVEN_TRIGGER_R = 1.0
DEFAULT_TRIM_FRACTION = 0.5

# A desk convention, not a result: nobody here has established the sample size
# at which a win rate stops being noise. It is here so the report says "this
# record is too short to read" rather than presenting six trades as evidence.
MIN_MEANINGFUL_CLOSED = 20
OPEN_SHARE_WARN = 0.50
EXPIRED_SHARE_WARN = 0.25
FLATTERY_WARN_R = 0.10


def book_path() -> Path:
    """Default location. Overridable so a second book can run alongside.

    Mirrors :func:`tradingagents.live.paper.portfolio_path`: two experiments
    sharing one file would merge two track records into one meaningless number.
    """
    env = os.getenv("TRADINGAGENTS_RECOMMENDATIONS_PATH")
    if env:
        return Path(env).expanduser()
    home = Path(os.getenv("TRADINGAGENTS_HOME", Path.home() / ".tradingagents"))
    return home / "recommendations.json"


# ---------------------------------------------------------------------------
# coercion
# ---------------------------------------------------------------------------

def _num(value: object, default: float = float("nan")) -> float:
    """Coerce anything to a finite float, or ``default``.

    Values arrive here from LLM JSON, from a hand-edited book file and from
    quote dictionaries whose misses are ``None``. All of those are ordinary
    inputs rather than programming errors, so they collapse onto one case.
    """
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def _action(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return {"BUY": BUY, "LONG": BUY, "B": BUY,
            "SELL": SELL, "SHORT": SELL, "SELL SHORT": SELL,
            "S": SELL}.get(value.strip().upper())


def _jsonable(value: object) -> object:
    """NaN and infinity out, ``None`` in.

    ``json.dumps`` writes a bare ``NaN`` token by default. Python reads it back
    happily and nothing else does — jq, a browser, a spreadsheet and every
    strict parser reject the file. Since this book is meant to be readable and
    hand-editable, non-finite floats are stored as null and restored to NaN on
    load.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def make_id(symbol: str, issued: str | date | None = None) -> str:
    """``NVDA-20260821``: readable, greppable, and stable across restarts.

    Deterministic rather than random because the id is the handle the user
    quotes back ("close NVDA-20260821") and the string that appears in the log.
    A uuid would be unique and useless. Same-day repeats on one name are
    disambiguated by :meth:`RecommendationBook._unique_id`, which appends a
    counter, so the readable form survives the common case.
    """
    sym = (symbol or "?").strip().upper()
    if isinstance(issued, date):
        d = issued
    else:
        try:
            d = date.fromisoformat(str(issued)) if issued else date.today()
        except ValueError:
            d = date.today()
    return f"{sym}-{d:%Y%m%d}"


# ---------------------------------------------------------------------------
# the recommendation
# ---------------------------------------------------------------------------

@dataclass
class Recommendation:
    """One issued idea and everything needed to grade it later.

    ``limit_price`` sits after the three levels rather than between the
    reference price and the stop, where it belongs conceptually: a dataclass
    cannot put a defaulted field before required ones, and the stop and target
    are required — an idea issued without them cannot be scored, which is the
    thing this book exists to do.
    """

    id: str
    issued_date: str                  # ISO date; the day the advice was given
    symbol: str
    action: str                       # BUY | SELL
    shares: int
    reference_price: float            # the price the advice was priced off
    stop_price: float                 # live stop; the trailing rule may move it
    target_price: float

    limit_price: float | None = None  # None means "at the market"
    horizon_days: int = DEFAULT_HORIZON_DAYS
    conviction: float = DEFAULT_CONVICTION
    rationale: str = ""
    sector: str = ""
    catalyst: str = ""                # what should make it work, in one line

    status: str = OPEN
    exit_date: str = ""
    exit_price: float | None = None
    exit_reason: str = ""
    realized_pnl: float | None = None
    realized_r: float | None = None

    # --- written once, never edited ---
    # The stop as issued. Realised R is measured against this and only this;
    # see the module docstring. Without it, moving a stop rewrites history.
    initial_stop_price: float = float("nan")
    issued_at: str = ""               # full timestamp, for ordering within a day

    # --- review state, maintained by RecommendationBook.review ---
    stop_raised: bool = False         # the breakeven rule has already fired
    target_taken: bool = False        # the target instruction has been issued once
    peak_r: float = 0.0               # best R this idea ever reached
    last_price: float | None = None
    last_reviewed: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(_num(self.initial_stop_price)):
            self.initial_stop_price = self.stop_price

    # --- direction and levels ----------------------------------------------

    @property
    def direction(self) -> str:
        """The sizing module's direction, so its arithmetic can be reused."""
        return _DIRECTIONS.get(self.action, LONG)

    @property
    def is_open(self) -> bool:
        return self.status == OPEN

    def problem(self) -> str:
        """Empty when this is a gradeable recommendation, else why it is not.

        Mirrors :meth:`tradingagents.live.sizing.Trade.problem`. Note that a
        recommendation with a problem is still stored — see
        :meth:`RecommendationBook.add`.
        """
        if self.action not in ACTIONS:
            return f"unknown action {self.action!r}; expected {BUY} or {SELL}"
        if self.shares <= 0:
            return f"shares {self.shares!r} is not a positive whole number"
        if math.isnan(_num(self.reference_price)) or self.reference_price <= 0:
            return f"reference price {self.reference_price!r} is not a positive number"
        if math.isnan(self.planned_r()):
            return (f"levels do not describe a {self.action} idea (entry "
                    f"{self.reference_price!r}, stop {self.initial_stop_price!r}, "
                    f"target {self.target_price!r})")
        return ""

    def planned_r(self) -> float:
        """Reward over risk as issued. NaN when the levels are incoherent.

        Delegates to :func:`tradingagents.live.sizing.r_multiple` — the ordering
        check that rejects a long whose target sits below its entry lives there
        and is not worth having twice.
        """
        return r_multiple(self.reference_price, self.initial_stop_price,
                          self.target_price, self.direction)

    def risk_per_share(self) -> float:
        """Distance from entry to the stop *as issued*. NaN if unusable."""
        e, s = _num(self.reference_price), _num(self.initial_stop_price)
        if math.isnan(e) or math.isnan(s) or e <= 0 or s <= 0 or e == s:
            return float("nan")
        return abs(e - s)

    def risk_amount(self) -> float:
        """Dollars at stake if the stop as issued had filled. NaN if unusable."""
        rps = self.risk_per_share()
        return float("nan") if math.isnan(rps) else rps * self.shares

    # --- marks --------------------------------------------------------------

    def pnl_at(self, price: float) -> float:
        """Dollar P&L of the advice at ``price``. NaN when it cannot be priced."""
        p, e = _num(price), _num(self.reference_price)
        if math.isnan(p) or math.isnan(e) or p <= 0:
            return float("nan")
        move = (p - e) if self.direction == LONG else (e - p)
        return move * self.shares

    def r_at(self, price: float) -> float:
        """P&L in R units, against the risk as issued. NaN when unpriceable."""
        pnl, risk = self.pnl_at(price), self.risk_amount()
        if math.isnan(pnl) or math.isnan(risk) or risk <= 0:
            return float("nan")
        return pnl / risk

    def days_held(self, as_of: date | None = None) -> int | None:
        """Calendar days since issue, or None if the issue date is unreadable.

        Calendar days, not trading days: a "30-day idea" is a sentence a human
        said about a calendar, and converting it to sessions would silently
        stretch every horizon by two days a week.
        """
        try:
            issued = date.fromisoformat(self.issued_date)
        except (TypeError, ValueError):
            return None
        return max(0, ((as_of or date.today()) - issued).days)

    # --- serialisation ------------------------------------------------------

    def to_dict(self) -> dict:
        return {k: _jsonable(v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, data: object) -> Recommendation | None:
        """Rebuild from stored JSON, or None if the row is not a recommendation.

        Tolerant on purpose. The file is small, readable and meant to be edited
        by hand, and a field added in a later version must not make an older
        book unloadable — the book is the track record, so losing it to a schema
        change would be the worst possible failure of this module.
        """
        if not isinstance(data, dict):
            return None
        known = {f.name for f in fields(cls)}
        d = {k: v for k, v in data.items() if k in known}

        symbol = str(d.get("symbol", "")).strip().upper()
        if not symbol:
            return None
        issued_date = str(d.get("issued_date", "") or "")
        action = _action(d.get("action")) or BUY

        rec = cls(
            id=str(d.get("id") or make_id(symbol, issued_date)),
            issued_date=issued_date,
            symbol=symbol,
            action=action,
            shares=int(_num(d.get("shares"), 0.0)),
            reference_price=_num(d.get("reference_price")),
            stop_price=_num(d.get("stop_price")),
            target_price=_num(d.get("target_price")),
        )
        lim = _num(d.get("limit_price"))
        rec.limit_price = None if math.isnan(lim) else lim
        rec.horizon_days = int(_num(d.get("horizon_days"), float(DEFAULT_HORIZON_DAYS)))
        rec.conviction = _num(d.get("conviction"), DEFAULT_CONVICTION)
        rec.rationale = str(d.get("rationale", "") or "")
        rec.sector = str(d.get("sector", "") or "")
        rec.catalyst = str(d.get("catalyst", "") or "")

        status = str(d.get("status", OPEN) or OPEN).strip().lower()
        rec.status = status if status in STATUSES else OPEN
        rec.exit_date = str(d.get("exit_date", "") or "")
        exit_price = _num(d.get("exit_price"))
        rec.exit_price = None if math.isnan(exit_price) else exit_price
        rec.exit_reason = str(d.get("exit_reason", "") or "")
        pnl = _num(d.get("realized_pnl"))
        rec.realized_pnl = None if math.isnan(pnl) else pnl
        realized_r = _num(d.get("realized_r"))
        rec.realized_r = None if math.isnan(realized_r) else realized_r

        # Books written before the trailing rule existed carry no initial stop.
        # Falling back to the stored stop is the only available answer and it is
        # correct for every such row, because nothing had moved a stop yet.
        rec.initial_stop_price = _num(d.get("initial_stop_price"), rec.stop_price)
        rec.issued_at = str(d.get("issued_at", "") or "")
        rec.stop_raised = bool(d.get("stop_raised", False))
        rec.target_taken = bool(d.get("target_taken", False))
        rec.peak_r = _num(d.get("peak_r"), 0.0)
        last = _num(d.get("last_price"))
        rec.last_price = None if math.isnan(last) else last
        rec.last_reviewed = str(d.get("last_reviewed", "") or "")
        return rec

    def __str__(self) -> str:
        r = self.planned_r()
        rs = "n/a" if math.isnan(r) else f"{r:.2f}R"
        return (f"{self.id} {self.action} {self.shares:,} {self.symbol} @ "
                f"{self.reference_price:,.2f} stop {self.stop_price:,.2f} "
                f"target {self.target_price:,.2f} ({rs}, {self.status})")


# ---------------------------------------------------------------------------
# exits
# ---------------------------------------------------------------------------

@dataclass
class ExitSignal:
    """One instruction about an open recommendation.

    ``urgency`` uses the same 1/2/3 scale as :class:`brain.Trigger` — 1 is
    routine, 3 is act now — so a caller can merge the two streams and sort them
    together without a translation table.
    """

    rec_id: str
    symbol: str
    action: str                      # SELL | TRIM | RAISE_STOP | UNAVAILABLE
    shares: int
    reason: str
    urgency: int = 1
    price: float = float("nan")
    pnl: float = float("nan")
    r_multiple: float = float("nan")
    new_stop: float | None = None    # RAISE_STOP only
    exit_reason: str = ""            # the REASON_* to record if this is taken

    @property
    def is_urgent(self) -> bool:
        return self.urgency >= 3

    @property
    def closes_position(self) -> bool:
        return self.action == SELL_ALL

    def __str__(self) -> str:
        head = f"[{self.urgency}] {self.action} {self.shares:,} {self.symbol}"
        if not math.isnan(self.r_multiple):
            head += f" ({self.r_multiple:+.2f}R)"
        return f"{head} — {self.reason}"


@dataclass(frozen=True)
class ExitRules:
    """The thresholds the exit engine uses. Edit deliberately.

    None of these are fitted numbers. They are conventions with a stated
    reason, exactly like ``stop_from_atr``'s k=2, and nothing in this repo has
    back-tested any of them.
    """

    breakeven_at_r: float = BREAKEVEN_TRIGGER_R
    trim_at_target: bool = True
    trim_fraction: float = DEFAULT_TRIM_FRACTION
    time_stop: bool = True
    thesis_break: bool = True
    thesis_break_materiality: int = THESIS_BREAK_MATERIALITY
    thesis_break_max_age_hours: float = THESIS_BREAK_MAX_AGE_HOURS


def _stop_hit(rec: Recommendation, price: float) -> bool:
    stop = _num(rec.stop_price)
    if math.isnan(stop) or stop <= 0:
        return False
    return price <= stop if rec.direction == LONG else price >= stop


def _target_hit(rec: Recommendation, price: float) -> bool:
    target = _num(rec.target_price)
    if math.isnan(target) or target <= 0:
        return False
    return price >= target if rec.direction == LONG else price <= target


def _headline(item: object) -> tuple[int, str, str, float] | None:
    """(materiality, lean, title, age_hours) from a NewsItem or a mapping.

    ``news_by_symbol`` is assembled by the caller from a poll that may have
    partly failed, so a malformed entry is an ordinary input. Returning None
    for the bad entry keeps the rest of the headlines — and the other twelve
    positions — reviewable.
    """
    try:
        if isinstance(item, dict):
            mat = int(_num(item.get("materiality"), 0.0))
            lean = str(item.get("lean", "neutral"))
            title = str(item.get("title", ""))
            age = _num(item.get("age_hours"), 0.0)
        else:
            mat = int(_num(getattr(item, "materiality", 0), 0.0))
            lean = str(getattr(item, "lean", "neutral"))
            title = str(getattr(item, "title", ""))
            raw_age = getattr(item, "age_hours", 0.0)
            age = _num(raw_age(), 0.0) if callable(raw_age) else _num(raw_age, 0.0)
    except Exception:
        return None
    return mat, lean, title, (0.0 if math.isnan(age) else age)


def _thesis_break(rec: Recommendation, news: object,
                  rules: ExitRules) -> tuple[str, int] | None:
    """The worst fresh headline that argues against holding, or None.

    A BUY is broken by bearish news and a SELL by bullish news, which is the
    only reading that makes the rule symmetric. Freshness is checked because
    Google News returns a quarter of back-coverage: without an age bound this
    rule would close a position on last quarter's earnings the first time the
    seen-set was pruned. That is the same guard ``brain.triggers`` applies, for
    the same reason.
    """
    if not rules.thesis_break or not isinstance(news, (list, tuple)):
        return None
    against = "bearish" if rec.direction == LONG else "bullish"
    worst: tuple[str, int] | None = None
    for item in news:
        parsed = _headline(item)
        if parsed is None:
            continue
        mat, lean, title, age = parsed
        if lean != against or mat < rules.thesis_break_materiality:
            continue
        if age > rules.thesis_break_max_age_hours:
            continue
        if worst is None or mat > worst[1]:
            worst = (f"[{mat}/{lean}] {title[:120]} ({age:.0f}h ago)", mat)
    return worst


# ---------------------------------------------------------------------------
# track record
# ---------------------------------------------------------------------------

@dataclass
class TrackRecord:
    """The book's own scorecard, computed two ways on purpose.

    Every field ending in ``_with_open`` includes the open recommendations
    marked to market. The gap between the two is not a rounding detail: it is
    the exact size of the book's self-flattery, because a book closes its
    winners and lets its losers stay open.
    """

    closed: int = 0
    wins: int = 0
    losses: int = 0
    scratches: int = 0
    win_rate: float = float("nan")
    avg_win: float = float("nan")
    avg_loss: float = float("nan")
    avg_win_r: float = float("nan")
    avg_loss_r: float = float("nan")
    expectancy_r: float = float("nan")
    expectancy_dollars: float = float("nan")
    profit_factor: float = float("nan")
    total_pnl: float = 0.0
    best: Recommendation | None = None
    worst: Recommendation | None = None

    open_count: int = 0
    expired_count: int = 0
    superseded_count: int = 0
    open_share: float = 0.0
    open_marked: int = 0
    open_unrealized: float = float("nan")
    unpriced: list[str] = field(default_factory=list)

    win_rate_with_open: float = float("nan")
    expectancy_r_with_open: float = float("nan")
    total_pnl_with_open: float = float("nan")
    flattery_r: float = float("nan")

    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {k: _jsonable(v) for k, v in asdict(self).items()
             if k not in ("best", "worst")}
        for k, rec in (("best", self.best), ("worst", self.worst)):
            d[k] = None if rec is None else {"id": rec.id, "symbol": rec.symbol,
                                             "pnl": _jsonable(rec.realized_pnl),
                                             "r": _jsonable(rec.realized_r)}
        return d


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


# ---------------------------------------------------------------------------
# the book
# ---------------------------------------------------------------------------

class RecommendationBook:
    """Every recommendation ever issued, plus the rules that end them."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else book_path()
        self.recommendations: list[Recommendation] = self._load()

    # --- persistence --------------------------------------------------------

    def _load(self) -> list[Recommendation]:
        """Never raises. A missing or corrupt file yields an empty book.

        Losing the file to a parse error would be bad; refusing to start
        because of one would be worse — the loop that raises here is the loop
        that stops watching every open stop.
        """
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except Exception as exc:
            logger.error("could not read the recommendation book at %s: %s",
                         self.path, exc)
            return []
        rows = raw.get("recommendations", []) if isinstance(raw, dict) else raw
        if not isinstance(rows, list):
            return []
        out = []
        for row in rows:
            rec = Recommendation.from_dict(row)
            if rec is None:
                logger.warning("skipped an unreadable row in %s", self.path)
                continue
            out.append(rec)
        return out

    def save(self) -> Path:
        """Atomic write, matching secretary.TradeLedger and paper's Portfolio.

        A half-written book is worse than no book: the track record is the only
        thing here that cannot be recomputed from anywhere else.
        """
        payload = {
            "updated": datetime.now().isoformat(),
            "recommendations": [r.to_dict() for r in self.recommendations],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)
        return self.path

    def _save_quietly(self) -> None:
        """Persist from inside the review loop, where nothing may raise."""
        try:
            self.save()
        except Exception as exc:
            logger.error("could not save the recommendation book: %s", exc)

    # --- lookups ------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.recommendations)

    def get(self, rec_id: str) -> Recommendation | None:
        return next((r for r in self.recommendations if r.id == rec_id), None)

    def open_recommendations(self, symbol: str | None = None) -> list[Recommendation]:
        """The live objects, not copies — :meth:`review` mutates review state."""
        out = [r for r in self.recommendations if r.status == OPEN]
        if symbol:
            sym = symbol.strip().upper()
            out = [r for r in out if r.symbol == sym]
        return out

    def closed_recommendations(self) -> list[Recommendation]:
        return [r for r in self.recommendations if r.status == CLOSED]

    def for_symbol(self, symbol: str) -> list[Recommendation]:
        sym = symbol.strip().upper()
        return [r for r in self.recommendations if r.symbol == sym]

    def _unique_id(self, base: str) -> str:
        if not self.get(base):
            return base
        # Two ideas on one name in one day are rare and usually a supersede.
        # When they are genuinely two, a counter keeps the readable prefix.
        n = 2
        while self.get(f"{base}-{n}"):
            n += 1
        return f"{base}-{n}"

    # --- issuing ------------------------------------------------------------

    def add(
        self,
        symbol: str,
        action: str,
        shares: int,
        reference_price: float,
        stop_price: float,
        target_price: float,
        *,
        limit_price: float | None = None,
        horizon_days: int = DEFAULT_HORIZON_DAYS,
        conviction: float = DEFAULT_CONVICTION,
        rationale: str = "",
        sector: str = "",
        catalyst: str = "",
        issued_date: str | date | None = None,
        rec_id: str | None = None,
        save: bool = True,
    ) -> Recommendation:
        """Record an issued recommendation and return it.

        A badly-specified recommendation is stored anyway, with
        :meth:`Recommendation.problem` describing what is wrong. Refusing it
        would quietly delete the desk's worst output, and the record of the bad
        advice is the part of a track record most worth keeping. What an
        incoherent recommendation loses is only the level-dependent exit rules,
        and :meth:`review` says so out loud rather than staying silent.
        """
        issued = issued_date if isinstance(issued_date, str) else None
        if issued is None:
            d = issued_date if isinstance(issued_date, date) else date.today()
            issued = d.isoformat()

        sym = (symbol or "").strip().upper()
        act = _action(action) or str(action).strip().upper()
        rec = Recommendation(
            id=rec_id or self._unique_id(make_id(sym, issued)),
            issued_date=issued,
            symbol=sym,
            action=act,
            shares=int(_num(shares, 0.0)),
            reference_price=_num(reference_price),
            stop_price=_num(stop_price),
            target_price=_num(target_price),
            limit_price=None if limit_price is None else _num(limit_price),
            horizon_days=int(_num(horizon_days, float(DEFAULT_HORIZON_DAYS))),
            conviction=_num(conviction, DEFAULT_CONVICTION),
            rationale=str(rationale or "")[:1000],
            sector=str(sector or ""),
            catalyst=str(catalyst or "")[:300],
            issued_at=datetime.now().isoformat(),
        )
        problem = rec.problem()
        if problem:
            logger.warning("recording %s with a problem: %s", rec.id, problem)
        self.recommendations.append(rec)
        if save:
            self._save_quietly()
        return rec

    # --- ending -------------------------------------------------------------

    def close(self, rec_id: str, exit_price: float, exit_reason: str = REASON_MANUAL,
              exit_date: str | date | None = None, save: bool = True
              ) -> Recommendation | None:
        """Close an open recommendation and freeze its realised P&L and R.

        Returns None when the id is unknown or the recommendation is not open;
        both are logged. Closing twice would double-count the idea in the track
        record, which is the one number this module exists to keep honest.
        """
        rec = self.get(rec_id)
        if rec is None:
            logger.warning("cannot close unknown recommendation %r", rec_id)
            return None
        if rec.status != OPEN:
            logger.warning("cannot close %s: already %s", rec.id, rec.status)
            return None

        px = _num(exit_price)
        rec.status = CLOSED
        rec.exit_price = None if math.isnan(px) else px
        rec.exit_reason = str(exit_reason or REASON_MANUAL)
        if isinstance(exit_date, date):
            rec.exit_date = exit_date.isoformat()
        else:
            rec.exit_date = str(exit_date) if exit_date else date.today().isoformat()

        pnl = rec.pnl_at(px)
        rec.realized_pnl = None if math.isnan(pnl) else pnl
        # Against the risk as issued, never against a stop the trailing rule
        # moved. Measuring against the live stop would score a breakeven exit
        # as a division by zero and, worse, would let a raised stop shrink the
        # denominator and inflate every winner.
        r = rec.r_at(px)
        rec.realized_r = None if math.isnan(r) else r
        if save:
            self._save_quietly()
        return rec

    def expire(self, rec_id: str, reason: str = "horizon passed",
               save: bool = True) -> Recommendation | None:
        """End an idea with no exit price: it was never actionable or priceable.

        Kept distinct from :meth:`close` because an expired idea has no P&L and
        must not be averaged into one. The track record counts expiries
        separately and warns when there are many, since "expire the losers,
        close the winners" is the easiest way to fake a record.
        """
        rec = self.get(rec_id)
        if rec is None or rec.status != OPEN:
            logger.warning("cannot expire %r (%s)", rec_id,
                           "unknown" if rec is None else rec.status)
            return None
        rec.status = EXPIRED
        rec.exit_reason = str(reason or "")
        rec.exit_date = date.today().isoformat()
        if save:
            self._save_quietly()
        return rec

    def supersede(self, rec_id: str, by: str = "", save: bool = True
                  ) -> Recommendation | None:
        """Replace an idea with a newer one on the same name.

        Also unscored, and for the same reason: the view changed, the trade did
        not happen twice, and counting both would double the sample.
        """
        rec = self.get(rec_id)
        if rec is None or rec.status != OPEN:
            logger.warning("cannot supersede %r (%s)", rec_id,
                           "unknown" if rec is None else rec.status)
            return None
        rec.status = SUPERSEDED
        rec.exit_reason = f"superseded by {by}" if by else "superseded"
        rec.exit_date = date.today().isoformat()
        if save:
            self._save_quietly()
        return rec

    def close_from(self, signal: ExitSignal, save: bool = True
                   ) -> Recommendation | None:
        """Act on a position-closing signal. Trims and stop raises are ignored.

        A TRIM changes the size of a live idea, not its outcome, and this book
        stores one entry and one exit per recommendation; partial exits would
        need a fill history, which is the portfolio's job, not this one.
        """
        if not signal.closes_position:
            return None
        return self.close(signal.rec_id, signal.price,
                          signal.exit_reason or REASON_MANUAL, save=save)

    # --- the exit engine ----------------------------------------------------

    def review(
        self,
        prices: dict[str, float] | None = None,
        news_by_symbol: dict[str, object] | None = None,
        *,
        as_of: date | None = None,
        rules: ExitRules | None = None,
        persist: bool = True,
    ) -> list[ExitSignal]:
        """What to do about every open recommendation, most urgent first.

        The five rules, in the order they are checked:

        1. **Stop hit.** Mechanical and terminal. The level was named at issue
           precisely so that this decision would not be taken while the
           position is losing money.
        2. **Thesis break.** A fresh, material headline pointing against the
           idea. Scored by the newsfeed's keyword table, which that module is
           explicit about being crude triage, so the urgency tracks materiality
           rather than being pinned at "act now".
        3. **Target hit.** Trims by default rather than selling everything: the
           target was chosen before anything was known about how the move would
           behave, and closing the whole position caps a trade that is working.
           The remainder rides behind a breakeven stop.
        4. **Time stop.** Past the horizon without ever reaching 1R. Capital
           tied up in a thesis that has not worked is the cost nobody accounts
           for: it never shows up as a loss, so it is never counted, but it is
           the reason a book with a decent win rate goes nowhere.
        5. **Trailing stop to breakeven at 1R.** The single highest-value rule
           in this set, because it removes the one outcome that ruins a record:
           a position that was up 2R and closed at -1R. Once the stop is at the
           entry the idea can no longer lose money, so what is left is a free
           option on the rest of the move. Nothing here has back-tested the 1R
           threshold; it is a convention.

        This is the one method that mutates a recommendation: it applies the
        trailing stop and updates ``peak_r`` and the last mark. The raise is
        applied rather than merely suggested because a stop that depends on the
        caller remembering to write it back is not a stop — one forgetful cycle
        and the winner is exposed to a full-R loss again.
        """
        rules = rules or ExitRules()
        when = as_of or date.today()
        prices = prices or {}
        news_by_symbol = news_by_symbol or {}

        out: list[ExitSignal] = []
        changed = False
        for rec in self.open_recommendations():
            try:
                signals, moved = self._review_one(rec, prices, news_by_symbol,
                                                  when, rules)
            except Exception as exc:
                # One malformed recommendation must not stop the other twelve
                # from being checked. A raise here silently stops watching
                # every stop in the book.
                logger.error("could not review %s: %s", rec.id, exc)
                out.append(ExitSignal(
                    rec_id=rec.id, symbol=rec.symbol, action=UNAVAILABLE,
                    shares=0, urgency=2,
                    reason=f"review failed ({type(exc).__name__}: {exc}); "
                           f"this recommendation was not checked"))
                continue
            out += signals
            changed = changed or moved

        if changed and persist:
            self._save_quietly()
        out.sort(key=lambda s: (-s.urgency, s.symbol, s.rec_id))
        return out

    def _review_one(self, rec: Recommendation, prices: dict[str, float],
                    news_by_symbol: dict[str, object], when: date,
                    rules: ExitRules) -> tuple[list[ExitSignal], bool]:
        """Signals for one recommendation, and whether its state changed."""
        signals: list[ExitSignal] = []
        changed = False

        price = _num(prices.get(rec.symbol))
        priced = not math.isnan(price) and price > 0
        levels_ok = not rec.problem()

        r_now = rec.r_at(price) if priced else float("nan")
        pnl_now = rec.pnl_at(price) if priced else float("nan")

        if priced:
            rec.last_price = price
            rec.last_reviewed = when.isoformat()
            changed = True
            if not math.isnan(r_now) and r_now > rec.peak_r:
                rec.peak_r = r_now

        def signal(action: str, shares: int, reason: str, urgency: int,
                   exit_reason: str = "", new_stop: float | None = None) -> ExitSignal:
            return ExitSignal(
                rec_id=rec.id, symbol=rec.symbol, action=action, shares=shares,
                reason=reason, urgency=urgency, price=price if priced else float("nan"),
                pnl=pnl_now, r_multiple=r_now, new_stop=new_stop,
                exit_reason=exit_reason,
            )

        # An unpriceable name is reported, not skipped. Silence about it reads
        # exactly like "nothing to do", and this is the one case where a stop
        # may already have been blown through unseen.
        if not priced:
            signals.append(signal(
                UNAVAILABLE, 0,
                f"no price for {rec.symbol}; the stop at "
                f"{rec.stop_price:,.2f} could not be checked", 2))
        elif not levels_ok:
            signals.append(signal(
                UNAVAILABLE, 0,
                f"levels unusable ({rec.problem()}); the stop, target and "
                f"breakeven rules were skipped", 2))

        # 1. stop hit
        if priced and levels_ok and _stop_hit(rec, price):
            at_breakeven = rec.stop_raised
            signals.append(signal(
                SELL_ALL, rec.shares,
                (f"stop at {rec.stop_price:,.2f} hit at {price:,.2f}"
                 + (" — this is the breakeven stop, so the idea comes off flat"
                    if at_breakeven else
                    " — the level the idea was said to be wrong at")),
                3, REASON_STOP))
            return signals, changed

        # 2. thesis break
        broken = _thesis_break(rec, news_by_symbol.get(rec.symbol), rules)
        if broken:
            headline, materiality = broken
            signals.append(signal(
                SELL_ALL, rec.shares,
                f"thesis break — {headline}", 3 if materiality >= 9 else 2,
                REASON_THESIS))
            return signals, changed

        # 3. target hit
        #
        # Issued once, not every cycle. This book stores one entry and one exit
        # per idea, so it cannot see whether a trim happened; repeating the same
        # instruction every two minutes teaches the reader to skim the report,
        # which is paid for by the urgent signals losing their urgency. After
        # the instruction has been given the position rides on its breakeven
        # stop, which is a rule the book *can* still enforce.
        if priced and levels_ok and not rec.target_taken and _target_hit(rec, price):
            rec.target_taken = True
            changed = True
            if rules.trim_at_target and rec.shares > 1:
                n = int(round(rec.shares * rules.trim_fraction))
                n = max(1, min(rec.shares - 1, n))
                # Free by construction: the price is beyond the target, which
                # is beyond the entry, so moving the stop to the entry cannot
                # be a stop already hit. Done even when planned R was below 1
                # and the breakeven rule below never fired.
                if not rec.stop_raised:
                    rec.stop_price = rec.reference_price
                    rec.stop_raised = True
                    changed = True
                signals.append(signal(
                    TRIM, n,
                    f"target {rec.target_price:,.2f} reached at {price:,.2f}; take "
                    f"{n:,} of {rec.shares:,} off and let the rest run behind a "
                    f"breakeven stop at {rec.reference_price:,.2f}",
                    2, REASON_TARGET, new_stop=rec.reference_price))
            else:
                signals.append(signal(
                    SELL_ALL, rec.shares,
                    f"target {rec.target_price:,.2f} reached at {price:,.2f}; the "
                    f"plan said this is where it comes off", 2, REASON_TARGET))
            return signals, changed

        # 4. time stop
        held = rec.days_held(when)
        if (rules.time_stop and held is not None and held > rec.horizon_days
                and rec.peak_r < 1.0):
            signals.append(signal(
                SELL_ALL, rec.shares,
                f"time stop — {held} calendar days against a {rec.horizon_days}-day "
                f"horizon and never reached 1R (best {rec.peak_r:+.2f}R); the "
                f"capital is doing nothing and that cost is never counted",
                1, REASON_TIME))
            return signals, changed

        # 5. trailing stop to breakeven
        if (priced and levels_ok and not rec.stop_raised
                and not math.isnan(r_now) and r_now >= rules.breakeven_at_r):
            rec.stop_price = rec.reference_price
            rec.stop_raised = True
            changed = True
            signals.append(signal(
                RAISE_STOP, 0,
                f"up {r_now:+.2f}R — stop moved from {rec.initial_stop_price:,.2f} "
                f"to breakeven at {rec.reference_price:,.2f}; from here the idea "
                f"cannot lose money", 1, new_stop=rec.reference_price))

        return signals, changed

    # --- track record -------------------------------------------------------

    def track_record(self, prices: dict[str, float] | None = None,
                     as_of: date | None = None) -> TrackRecord:
        """Score the book, closed and open reported separately.

        The separation is the point. Closed recommendations are the only ones
        with a realised number, and a book that reports only those is reporting
        the ideas that resolved — and winners resolve first, because a winner
        hits its target while a loser sits open hoping. So the same statistics
        are computed a second time with open positions marked to market, and
        the difference between the two is published as ``flattery_r``.
        """
        prices = prices or {}
        tr = TrackRecord()

        closed = self.closed_recommendations()
        open_recs = self.open_recommendations()
        tr.closed = len(closed)
        tr.open_count = len(open_recs)
        tr.expired_count = sum(1 for r in self.recommendations if r.status == EXPIRED)
        tr.superseded_count = sum(1 for r in self.recommendations
                                  if r.status == SUPERSEDED)

        scored = [r for r in closed if r.realized_pnl is not None]
        wins = [r for r in scored if (r.realized_pnl or 0.0) > 0]
        losses = [r for r in scored if (r.realized_pnl or 0.0) < 0]
        tr.wins, tr.losses = len(wins), len(losses)
        tr.scratches = len(scored) - tr.wins - tr.losses
        tr.total_pnl = sum(r.realized_pnl or 0.0 for r in scored)

        if scored:
            tr.win_rate = tr.wins / len(scored)
            tr.expectancy_dollars = tr.total_pnl / len(scored)
            tr.best = max(scored, key=lambda r: r.realized_pnl or 0.0)
            tr.worst = min(scored, key=lambda r: r.realized_pnl or 0.0)
        tr.avg_win = _mean([r.realized_pnl or 0.0 for r in wins])
        tr.avg_loss = _mean([r.realized_pnl or 0.0 for r in losses])

        closed_r = [r.realized_r for r in scored if r.realized_r is not None]
        tr.expectancy_r = _mean(closed_r)
        tr.avg_win_r = _mean([r.realized_r for r in wins if r.realized_r is not None])
        tr.avg_loss_r = _mean([r.realized_r for r in losses if r.realized_r is not None])

        gross_win = sum(r.realized_pnl or 0.0 for r in wins)
        gross_loss = -sum(r.realized_pnl or 0.0 for r in losses)
        if gross_loss > 0:
            tr.profit_factor = gross_win / gross_loss
        elif gross_win > 0:
            # No closed loser yet. Infinity is the arithmetic answer and it is
            # not a result; the warning below says so.
            tr.profit_factor = float("inf")

        # --- open, marked to market ---
        open_pnl = 0.0
        open_r: list[float] = []
        open_wins = 0
        for rec in open_recs:
            px = _num(prices.get(rec.symbol), _num(rec.last_price))
            pnl = rec.pnl_at(px) if not math.isnan(px) else float("nan")
            if math.isnan(pnl):
                tr.unpriced.append(rec.symbol)
                continue
            tr.open_marked += 1
            open_pnl += pnl
            if pnl > 0:
                open_wins += 1
            r = rec.r_at(px)
            if not math.isnan(r):
                open_r.append(r)
        tr.open_unrealized = open_pnl if tr.open_marked else float("nan")
        resolved = tr.closed + tr.open_count
        tr.open_share = tr.open_count / resolved if resolved else 0.0

        if tr.open_marked:
            both = len(scored) + tr.open_marked
            tr.win_rate_with_open = (tr.wins + open_wins) / both if both else float("nan")
            tr.total_pnl_with_open = tr.total_pnl + open_pnl
            all_r = closed_r + open_r
            tr.expectancy_r_with_open = _mean(all_r)
            if not math.isnan(tr.expectancy_r) and not math.isnan(tr.expectancy_r_with_open):
                tr.flattery_r = tr.expectancy_r - tr.expectancy_r_with_open

        tr.warnings = self._record_warnings(tr)
        return tr

    @staticmethod
    def _record_warnings(tr: TrackRecord) -> list[str]:
        """Plain sentences naming every way this record could be read wrong."""
        out: list[str] = []
        if tr.closed == 0:
            out.append("Nothing has been closed yet, so there is no record — only "
                       "open positions, which are a forecast and not a result.")
        elif tr.closed < MIN_MEANINGFUL_CLOSED:
            out.append(f"{tr.closed} closed recommendation"
                       f"{'' if tr.closed == 1 else 's'} is too few to read as a "
                       f"win rate. {MIN_MEANINGFUL_CLOSED} is this desk's "
                       f"convention for a minimum, not a result anyone has "
                       f"established.")
        if tr.open_share >= OPEN_SHARE_WARN and tr.open_count:
            out.append(f"{tr.open_share:.0%} of the book is still open. A record "
                       f"made only of closed ideas is the record of the ideas that "
                       f"resolved, and losers resolve last — they sit open while a "
                       f"winner hits its target and closes.")
        if not math.isnan(tr.flattery_r) and tr.flattery_r > FLATTERY_WARN_R:
            out.append(f"Closed-only expectancy is {tr.flattery_r:+.2f}R per idea "
                       f"better than the same book with open positions marked to "
                       f"market. That gap is the self-flattery, measured.")
        if tr.unpriced:
            names = ", ".join(sorted(set(tr.unpriced))[:8])
            out.append(f"{len(tr.unpriced)} open recommendation"
                       f"{'' if len(tr.unpriced) == 1 else 's'} could not be priced "
                       f"({names}); they are excluded from the marked-to-market "
                       f"figures rather than counted as flat.")
        ended = tr.closed + tr.expired_count + tr.superseded_count
        if ended and tr.expired_count / ended >= EXPIRED_SHARE_WARN:
            out.append(f"{tr.expired_count} of {ended} finished recommendations were "
                       f"expired rather than closed, and an expired idea carries no "
                       f"P&L. Expiring the ones that went wrong and closing the ones "
                       f"that went right produces any record you like.")
        if math.isinf(tr.profit_factor):
            out.append("No closed recommendation has lost money yet, so the profit "
                       "factor is infinite. That is arithmetic, not evidence.")
        return out


# ---------------------------------------------------------------------------
# terminal reports
# ---------------------------------------------------------------------------

def _money(x: float | None, width: int = 10) -> str:
    v = _num(x)
    return f"{'n/a':>{width}}" if math.isnan(v) else f"{v:>+{width},.0f}"


def _r(x: float | None, width: int = 7) -> str:
    v = _num(x)
    if math.isnan(v):
        return f"{'n/a':>{width}}"
    if math.isinf(v):
        return f"{'inf':>{width}}"
    return f"{v:>+{width - 1}.2f}R"


def _ratio(x: float | None, width: int = 12) -> str:
    v = _num(x)
    if math.isnan(v):
        return f"{'n/a':>{width}}"
    return f"{'inf':>{width}}" if math.isinf(v) else f"{v:>{width},.2f}"


def format_open_book(recs: list[Recommendation],
                     prices: dict[str, float] | None = None,
                     as_of: date | None = None) -> str:
    """The open ideas, one line each.

    No separate symbol column: the id already begins with the ticker, and a
    duplicated column costs width the levels need.
    """
    if not recs:
        return "(no open recommendations)"
    prices = prices or {}
    when = as_of or date.today()
    hdr = (f"{'ID':<18}{'Act':<5}{'Shr':>6}{'Entry':>9}{'Stop':>9}{'Target':>9}"
           f"{'Last':>9}{'P&L':>10}{'R':>7}{'Peak':>7}{'Held':>8}  Catalyst")
    lines = [hdr, "-" * len(hdr)]
    for rec in sorted(recs, key=lambda r: (r.symbol, r.id)):
        px = _num(prices.get(rec.symbol), _num(rec.last_price))
        held = rec.days_held(when)
        held_s = "n/a" if held is None else f"{held}/{rec.horizon_days}"
        # A star marks a stop the trailing rule has already moved, so a reader
        # can tell a breakeven stop from the one the idea was issued with.
        flag = "*" if rec.stop_raised else " "
        last_s = "n/a" if math.isnan(px) else f"{px:,.2f}"
        lines.append(
            f"{rec.id:<18}{rec.action:<5}{rec.shares:>6,}"
            f"{_num(rec.reference_price, 0.0):>9,.2f}"
            f"{_num(rec.stop_price, 0.0):>8,.2f}{flag}"
            f"{_num(rec.target_price, 0.0):>9,.2f}{last_s:>9}"
            f"{_money(rec.pnl_at(px))}{_r(rec.r_at(px))}{_r(rec.peak_r)}"
            f"{held_s:>8}  {rec.catalyst[:40]}"
        )
    if any(r.stop_raised for r in recs):
        lines.append("  * stop already moved to breakeven")
    return "\n".join(lines)


def format_exit_signals(signals: list[ExitSignal]) -> str:
    """What to sell and why, urgent first."""
    if not signals:
        return "(no exits due)"
    lines = []
    for s in signals:
        mark = {3: "!!", 2: " !", 1: "  "}.get(s.urgency, "  ")
        head = f"{mark} {s.action:<11}{s.shares:>6,} {s.symbol:<6}"
        tail = "" if math.isnan(s.r_multiple) else f" [{s.r_multiple:+.2f}R]"
        lines.append(f"{head}{s.reason}{tail}")
    return "\n".join(lines)


def format_track_record(tr: TrackRecord) -> str:
    """The scorecard, with the open book kept visibly apart from the closed one."""
    lines = [f"Closed recommendations   {tr.closed:>8,}"]
    if tr.closed:
        lines += [
            f"  win rate               {tr.win_rate:>8.0%}   "
            f"({tr.wins}W / {tr.losses}L"
            + (f" / {tr.scratches} scratch" if tr.scratches else "") + ")",
            f"  average win        {_money(tr.avg_win)}   {_r(tr.avg_win_r).strip()}",
            f"  average loss       {_money(tr.avg_loss)}   {_r(tr.avg_loss_r).strip()}",
            f"  expectancy         {_money(tr.expectancy_dollars)}   "
            f"{_r(tr.expectancy_r).strip()} per idea",
            f"  profit factor      {_ratio(tr.profit_factor)}",
            f"  total P&L          {_money(tr.total_pnl)}",
        ]
        if tr.best:
            lines.append(f"  best   {tr.best.id:<18}{_money(tr.best.realized_pnl)}"
                         f"{_r(tr.best.realized_r)}")
        if tr.worst:
            lines.append(f"  worst  {tr.worst.id:<18}{_money(tr.worst.realized_pnl)}"
                         f"{_r(tr.worst.realized_r)}")

    lines += ["", f"Open recommendations     {tr.open_count:>8,}   "
                  f"({tr.open_share:.0%} of the resolved book, "
                  f"{tr.open_marked} priced)"]
    if tr.open_marked:
        lines.append(f"  marked to market   {_money(tr.open_unrealized)}")
        lines += ["", "Same book, including open positions marked to market",
                  f"  win rate               {tr.win_rate_with_open:>8.0%}",
                  f"  expectancy         {_r(tr.expectancy_r_with_open, 10)} per idea",
                  f"  total P&L          {_money(tr.total_pnl_with_open)}"]
        if not math.isnan(tr.flattery_r):
            lines.append(f"  self-flattery      {_r(tr.flattery_r, 10)}   "
                         f"closed-only minus marked")
    if tr.expired_count or tr.superseded_count:
        lines.append(f"\nUnscored: {tr.expired_count} expired, "
                     f"{tr.superseded_count} superseded (no P&L either way)")
    if tr.warnings:
        lines.append("")
        lines += [f"NOTE  {w}" for w in tr.warnings]
    return "\n".join(lines)


def format_report(book: RecommendationBook,
                  prices: dict[str, float] | None = None,
                  signals: list[ExitSignal] | None = None,
                  as_of: date | None = None) -> str:
    """The whole book as one terminal page.

    Takes already-computed ``signals`` rather than calling :meth:`review`
    itself, because review moves stops and persists — printing a report must
    not change the book.
    """
    when = as_of or date.today()
    parts = [f"# Recommendation book — {when:%Y-%m-%d} ({book.path})", ""]
    parts += ["## Open ideas", format_open_book(book.open_recommendations(),
                                                prices, when)]
    if signals is not None:
        parts += ["", "## Exits due", format_exit_signals(signals)]
    parts += ["", "## Track record",
              format_track_record(book.track_record(prices, when))]
    return "\n".join(parts)
