"""What the last report said, and when the next one lands.

The desk reasons about price, volume and news. None of those knows that a
company reports in four days, and a two-ATR stop is not a defence against an
earnings gap — the gap opens through it. Nor do they know that the name making
new highs did so *after missing badly*, which is momentum running against the
fundamentals rather than with them.

Two facts per symbol, both cheap and both checkable:

* **days to the next report** — a risk fact. Inside the holding horizon, the
  stop is decorative for one session.
* **the last report's surprise** — a context fact. Post-earnings drift is one
  of the better-documented anomalies, and its sign is the sign of the surprise.

Deliberately not here: estimates, guidance, margins, or any judgement about
whether a number was "good". This module reports what was published.

Look-ahead is the failure this file must not have. Everything is filtered
against the caller's ``as_of`` date, so a run for a past session cannot see a
report that had not happened yet.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

# Earnings calendars move rarely; a day of staleness costs nothing and saves a
# network round trip per symbol on every re-run.
CACHE_TTL_HOURS = 20.0


def _home() -> Path:
    return Path(os.getenv("TRADINGAGENTS_HOME", Path.home() / ".tradingagents"))


def cache_path() -> Path:
    return _home() / "earnings.json"


@dataclass
class Earnings:
    """One symbol's earnings facts as of a stated date. All fields optional."""

    symbol: str
    as_of: str = ""
    next_date: str = ""          # ISO date of the next scheduled report
    last_date: str = ""          # ISO date of the most recent completed report
    eps_estimate: float = float("nan")
    eps_actual: float = float("nan")
    surprise_pct: float = float("nan")
    error: str = ""              # why this is empty, when it is

    def days_to_next(self, ref: date | None = None) -> float:
        if not self.next_date:
            return float("nan")
        try:
            return (date.fromisoformat(self.next_date) - (ref or date.today())).days
        except ValueError:
            return float("nan")

    def days_since_last(self, ref: date | None = None) -> float:
        if not self.last_date:
            return float("nan")
        try:
            return ((ref or date.today()) - date.fromisoformat(self.last_date)).days
        except ValueError:
            return float("nan")

    def beat(self) -> bool | None:
        """True on a beat, False on a miss, None when it cannot be told.

        None is not False. A missing estimate means the question was not
        answered, and answering it anyway is how a gap becomes a claim.
        """
        s = self.surprise_pct
        if s != s:
            return None
        return s > 0

    def reports_within(self, days: float, ref: date | None = None) -> bool:
        d = self.days_to_next(ref)
        return d == d and 0 <= d <= days


def _num(v) -> float:
    try:
        f = float(v)
        return f
    except (TypeError, ValueError):
        return float("nan")


def fetch(symbol: str, as_of: date, *, log=logger.debug) -> Earnings:
    """One symbol, live. Never raises: a missing calendar is a named gap."""
    out = Earnings(symbol=symbol, as_of=as_of.isoformat())
    try:
        import yfinance as yf
        frame = yf.Ticker(symbol).get_earnings_dates(limit=12)
    except Exception as exc:                      # network, parse, upstream shape
        out.error = f"{type(exc).__name__}: {exc}"
        return out
    if frame is None or getattr(frame, "empty", True):
        out.error = "no earnings calendar"
        return out

    try:
        future = [d for d in frame.index if d.date() > as_of]
        past = [d for d in frame.index if d.date() <= as_of]
        if future:
            out.next_date = min(future).date().isoformat()
        if past:
            when = max(past)
            out.last_date = when.date().isoformat()
            row = frame.loc[when]
            out.eps_estimate = _num(row.get("EPS Estimate"))
            out.eps_actual = _num(row.get("Reported EPS"))
            out.surprise_pct = _num(row.get("Surprise(%)"))
    except Exception as exc:
        out.error = f"unreadable calendar ({type(exc).__name__}: {exc})"
    return out


@dataclass
class EarningsBook:
    """Cached earnings facts, keyed by (symbol, as_of)."""

    path: Path = field(default_factory=cache_path)
    _rows: dict = field(default_factory=dict)
    _loaded: bool = False

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                self._rows = raw.get("rows", {}) if "rows" in raw else raw
        except Exception:
            self._rows = {}

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(json.dumps({"rows": self._rows}, indent=0), encoding="utf-8")
            os.replace(tmp, self.path)
        except Exception as exc:
            logger.debug("could not save the earnings cache: %s", exc)

    def get(self, symbols: list[str], as_of: date, *, refresh: bool = False,
            log=logger.debug) -> dict[str, Earnings]:
        """Facts for every symbol, fetching only what the cache lacks."""
        self._load()
        out: dict[str, Earnings] = {}
        fetched = 0
        for sym in symbols:
            key = f"{sym}|{as_of.isoformat()}"
            cached = None if refresh else self._rows.get(key)
            if cached:
                try:
                    out[sym] = Earnings(**cached)
                    continue
                except TypeError:
                    pass                       # shape changed; refetch
            e = fetch(sym, as_of, log=log)
            fetched += 1
            self._rows[key] = asdict(e)
            out[sym] = e
        if fetched:
            # Keep only the two most recent as_of dates; the file is a cache.
            dates = sorted({k.split("|", 1)[1] for k in self._rows}, reverse=True)[:2]
            self._rows = {k: v for k, v in self._rows.items()
                          if k.split("|", 1)[1] in dates}
            self._save()
            log(f"[earnings] fetched {fetched} of {len(symbols)} symbols")
        return out


def summarise(book: dict[str, Earnings], as_of: date, horizon_days: int) -> dict:
    """Counts a report can state without re-deriving them at the call site."""
    known = [e for e in book.values() if e.next_date or e.last_date]
    inside = [e for e in book.values() if e.reports_within(horizon_days, as_of)]
    missing = sorted(s for s, e in book.items() if not (e.next_date or e.last_date))
    return {"known": len(known), "total": len(book),
            "reporting_within_horizon": sorted(e.symbol for e in inside),
            "missing": missing}
