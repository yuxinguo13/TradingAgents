"""Earnings facts: the date guard, the last surprise, and the gaps.

A two-ATR stop does not survive an earnings gap — the gap opens through it —
so a report that sizes a position without knowing the date is sizing against a
risk it cannot see. These tests pin the two things that could go quietly
wrong: reading a report that had not happened yet, and letting "unknown"
render as "nothing due".
"""

import datetime as dt
import json

import pytest

from tradingagents.live.earnings import Earnings, EarningsBook, summarise

AS_OF = dt.date(2026, 8, 28)


def _e(**kw):
    return Earnings(symbol=kw.pop("symbol", "X"), as_of=AS_OF.isoformat(), **kw)


# ---------------------------------------------------------------------------
# the facts themselves
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_days_to_next_and_since_last():
    e = _e(next_date="2026-09-02", last_date="2026-06-03")
    assert e.days_to_next(AS_OF) == 5
    assert e.days_since_last(AS_OF) == 86


@pytest.mark.unit
def test_unknown_dates_are_nan_not_zero():
    # 0 would mean "reports today" and "reported today"; both are claims.
    e = _e()
    assert e.days_to_next(AS_OF) != e.days_to_next(AS_OF)      # NaN
    assert e.days_since_last(AS_OF) != e.days_since_last(AS_OF)


@pytest.mark.unit
def test_an_unparseable_date_does_not_raise():
    e = _e(next_date="not-a-date", last_date="also-not")
    assert e.days_to_next(AS_OF) != e.days_to_next(AS_OF)
    assert e.days_since_last(AS_OF) != e.days_since_last(AS_OF)


@pytest.mark.unit
def test_beat_returns_none_when_it_cannot_be_told():
    # None is not False. A missing estimate means the question was not
    # answered; answering it anyway turns a gap into a claim.
    assert _e(surprise_pct=6.2).beat() is True
    assert _e(surprise_pct=-87.2).beat() is False
    assert _e().beat() is None


@pytest.mark.unit
def test_reports_within_is_false_when_the_date_is_unknown():
    # The dangerous default: "we don't know" must not read as "nothing due",
    # which is why the report lists unknown calendars separately.
    assert _e(next_date="2026-09-02").reports_within(30, AS_OF) is True
    assert _e(next_date="2026-11-04").reports_within(30, AS_OF) is False
    assert _e().reports_within(30, AS_OF) is False


@pytest.mark.unit
def test_a_past_date_is_not_counted_as_upcoming():
    assert _e(next_date="2026-08-01").reports_within(30, AS_OF) is False


# ---------------------------------------------------------------------------
# look-ahead
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_fetch_never_reads_a_report_published_after_the_cut(monkeypatch):
    # The failure this module must not have: a run for a past session seeing a
    # result that had not been published yet.
    import pandas as pd

    from tradingagents.live import earnings as mod

    idx = pd.to_datetime(["2026-06-03", "2026-09-02"]).tz_localize("America/New_York")
    frame = pd.DataFrame({"EPS Estimate": [2.40, 2.55],
                          "Reported EPS": [2.44, 9.99],      # the future one
                          "Surprise(%)": [1.74, 300.0]}, index=idx)

    class FakeTicker:
        def __init__(self, sym): pass
        def get_earnings_dates(self, limit=12): return frame

    monkeypatch.setattr(mod, "fetch", mod.fetch)   # keep the real function
    import sys
    import types
    fake_yf = types.ModuleType("yfinance")
    fake_yf.Ticker = FakeTicker
    monkeypatch.setitem(sys.modules, "yfinance", fake_yf)

    got = mod.fetch("AVGO", AS_OF)
    assert got.last_date == "2026-06-03"
    assert got.next_date == "2026-09-02"
    assert got.eps_actual == pytest.approx(2.44)   # not 9.99
    assert got.surprise_pct == pytest.approx(1.74)


@pytest.mark.unit
def test_a_missing_calendar_is_a_named_gap_not_a_crash(monkeypatch):
    import sys
    import types

    from tradingagents.live import earnings as mod

    class Boom:
        def __init__(self, sym): pass
        def get_earnings_dates(self, limit=12):
            raise RuntimeError("upstream said no")

    fake_yf = types.ModuleType("yfinance")
    fake_yf.Ticker = Boom
    monkeypatch.setitem(sys.modules, "yfinance", fake_yf)
    got = mod.fetch("XYZ", AS_OF)
    assert got.error
    assert got.next_date == "" and got.last_date == ""


# ---------------------------------------------------------------------------
# the cache
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_the_cache_avoids_a_second_fetch(tmp_path, monkeypatch):
    from tradingagents.live import earnings as mod
    calls = []

    def fake_fetch(symbol, as_of, log=None):
        calls.append(symbol)
        return Earnings(symbol=symbol, as_of=as_of.isoformat(), next_date="2026-11-04")

    monkeypatch.setattr(mod, "fetch", fake_fetch)
    book = mod.EarningsBook(path=tmp_path / "e.json")
    book.get(["A", "B"], AS_OF, log=lambda m: None)
    assert calls == ["A", "B"]
    book2 = mod.EarningsBook(path=tmp_path / "e.json")
    book2.get(["A", "B"], AS_OF, log=lambda m: None)
    assert calls == ["A", "B"], "the second book should have read the file"


@pytest.mark.unit
def test_a_different_as_of_is_a_different_cache_entry(tmp_path, monkeypatch):
    # Keying on the symbol alone would serve yesterday's "next report" forever.
    from tradingagents.live import earnings as mod
    calls = []

    def fake_fetch(symbol, as_of, log=None):
        calls.append((symbol, as_of))
        return Earnings(symbol=symbol, as_of=as_of.isoformat())

    monkeypatch.setattr(mod, "fetch", fake_fetch)
    book = mod.EarningsBook(path=tmp_path / "e.json")
    book.get(["A"], AS_OF, log=lambda m: None)
    book.get(["A"], AS_OF + dt.timedelta(days=1), log=lambda m: None)
    assert len(calls) == 2


@pytest.mark.unit
def test_a_corrupt_cache_file_is_ignored_not_fatal(tmp_path, monkeypatch):
    from tradingagents.live import earnings as mod
    p = tmp_path / "e.json"
    p.write_text("{ not json", encoding="utf-8")
    monkeypatch.setattr(mod, "fetch",
                        lambda s, a, log=None: Earnings(symbol=s, as_of=a.isoformat()))
    got = mod.EarningsBook(path=p).get(["A"], AS_OF, log=lambda m: None)
    assert "A" in got
    assert json.loads(p.read_text(encoding="utf-8"))["rows"]


# ---------------------------------------------------------------------------
# the summary a report prints
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_summary_separates_known_from_missing():
    book = {
        "SOON": _e(symbol="SOON", next_date="2026-09-02"),
        "LATER": _e(symbol="LATER", next_date="2026-11-04"),
        "UNKNOWN": _e(symbol="UNKNOWN"),
    }
    s = summarise(book, AS_OF, 30)
    assert s["reporting_within_horizon"] == ["SOON"]
    assert s["missing"] == ["UNKNOWN"]
    assert s["known"] == 2 and s["total"] == 3
