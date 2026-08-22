"""The daily advisor: the two dates it prints, and the book it writes back to.

Two failures are worth more than the rest of this file put together, and both
are silent — the report renders, the numbers add up, and the page is wrong.

The first is the pair of dates. The report is built from a completed session
and acted on at the next open, and a page that names one date for both is
claiming it read Friday's close before Friday's open. Every idea in the book
carries that date, so a track record built on it cannot answer the only
question worth asking of a call: did it know this before or after the move.

The second is the book itself. Buys were written into it unconditionally, as if
taken, and the exits never were. Nothing in the package ever closed a
recommendation, so ``closed`` stayed 0 forever while the report claimed the
track record was the only checkable part of it, every symbol ever recommended
stayed banned from the shortlist, and a stopped-out idea re-emitted the same
SELL every morning.

Nothing here touches a network, a browser or an LLM: the screen, the price
snapshots, the news feeds, the policy feeds and the venue are all injected.
"""

import math
import os
from datetime import date, datetime
from pathlib import Path

import pytest

from tradingagents.live import advisor
from tradingagents.live import clock
from tradingagents.live.advisor import (
    AdvisorConfig,
    CAVEAT,
    DailyAdvisor,
    DailyReport,
    MAX_TILT_SHIFT,
    REFERENCE_NOTE,
    Candidate,
    format_report,
    last_completed_session,
    save_report,
    sessions_for,
    to_markdown,
)
from tradingagents.live.brain import Snapshot
from tradingagents.live.broker import Account, Holding
from tradingagents.live.recommendations import (
    CLOSED,
    EXPIRED,
    OPEN,
    REASON_STOP,
    SELL_ALL,
    ExitSignal,
    RecommendationBook,
)

# 2026-08-21 is a Friday, 08-22 a Saturday, 08-24 the Monday after it.
FRIDAY_AFTER_CLOSE = datetime(2026, 8, 21, 17, 0, tzinfo=clock.ET)
FRIDAY_MIDDAY = datetime(2026, 8, 21, 12, 0, tzinfo=clock.ET)
SATURDAY_NOON = datetime(2026, 8, 22, 12, 0, tzinfo=clock.ET)
MONDAY_AFTER_CLOSE = datetime(2026, 8, 24, 17, 0, tzinfo=clock.ET)
TUESDAY_AFTER_CLOSE = datetime(2026, 8, 25, 17, 0, tzinfo=clock.ET)

FRIDAY = date(2026, 8, 21)
MONDAY = date(2026, 8, 24)


# ---------------------------------------------------------------------------
# stubs: everything slow, networked or stateful
# ---------------------------------------------------------------------------

class StubMonitor:
    """A news or policy monitor with no RSS behind it.

    ``poll`` repopulates ``seen`` the way both real monitors do, which is what
    makes the backlog-suppression test meaningful.
    """

    def __init__(self, items=(), seen=None):
        self.items = list(items)
        self.seen = dict(seen or {})
        self.saved: list[dict] = []

    def poll(self, *args, **kwargs):
        for item in self.items:
            self.seen[getattr(item, "fingerprint", None) or str(item)] = "now"
        return list(self.items)

    def _save(self):
        self.saved.append(dict(self.seen))


class StubBroker:
    def __init__(self, account):
        self._account = account

    def account(self):
        return self._account


class StubFrame:
    """Just the ``iterrows`` that :func:`candidates_from_frame` asks for."""

    def __init__(self, rows):
        self.rows = rows

    def iterrows(self):
        return iter(self.rows)


class StubPanel:
    def __init__(self, result=None):
        self.result = result
        self.calls: list[str] = []

    def deliberate(self, symbol, evidence, account, price):
        self.calls.append(symbol)
        return self.result


def snap(symbol, price, atr_pct=0.03, ret_3m=0.60, **kw):
    return Snapshot(symbol=symbol, price=price, atr_pct=atr_pct, ret_3m=ret_3m,
                    ok=True, **kw)


ROWS = [
    ("AAA", {"rank": 1, "name": "A Co", "sector": "Technology", "score": 9.0,
             "price": 100.0}),
    ("BBB", {"rank": 2, "name": "B Co", "sector": "Energy", "score": 8.0,
             "price": 50.0}),
]


class Desk:
    """One advisor plus the mutable stubs behind it."""

    def __init__(self, tmp_path, cfg=None, holdings=()):
        self.tmp_path = tmp_path
        self.snaps = {
            "SPY": snap("SPY", 500.0, atr_pct=0.01, sma50=490.0, sma200=450.0),
            "AAA": snap("AAA", 100.0),
            "BBB": snap("BBB", 50.0, atr_pct=0.02, ret_3m=0.40),
        }
        self.rows = list(ROWS)
        self.stats = {"universe": 3000, "passed_filters": 42}
        self.screens: list[str] = []
        self.news = StubMonitor()
        self.policy = StubMonitor()
        self.account = Account(account_value=100_000.0, cash=100_000.0,
                               buying_power=100_000.0, holdings=list(holdings))
        self.cfg = cfg or AdvisorConfig()

    def screen(self, when, exchange, top, log=None):
        self.screens.append(when)
        return StubFrame(self.rows), self.stats

    def snapshot(self, symbol, when):
        return self.snaps.get(symbol, Snapshot(symbol=symbol, error="no data"))

    def advisor(self, **kw):
        return DailyAdvisor(self.cfg, broker=StubBroker(self.account),
                            news=self.news, policy_monitor=self.policy, **kw)

    def run(self, when=None, now=FRIDAY_AFTER_CLOSE, **kw):
        return self.advisor(**kw).run(when, now=now)


@pytest.fixture
def desk(tmp_path, monkeypatch):
    """An advisor whose every outside dependency is a stub inside tmp_path.

    ``screens_dirs`` is redirected too: it deliberately also searches the
    literal ~/.tradingagents/screens, which on a developer machine holds real
    saved screens and would make these tests read the user's own files.
    """
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path))
    d = Desk(tmp_path)
    monkeypatch.setattr(advisor, "screens_dirs", lambda: [tmp_path / "screens"])
    monkeypatch.setattr(advisor, "snapshot", d.snapshot)
    monkeypatch.setattr(advisor, "run_screen", d.screen)
    return d


def save_screen(tmp_path, when: date, exchange: str = "nasdaq") -> Path:
    """A screen CSV on disk, in the layout ``DataFrame.to_csv`` writes."""
    directory = tmp_path / "screens"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"screen_{exchange}_{when.isoformat()}.csv"
    path.write_text("symbol,rank,name,sector,score,price\n"
                    "AAA,1,A Co,Technology,9.0,100.0\n"
                    "BBB,2,B Co,Energy,8.0,50.0\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# the two dates
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSessionDates:
    def test_orders_are_never_dated_for_the_session_the_data_came_from(self, desk):
        """The report claimed it read Friday's close before Friday's open.

        Both dates defaulted to today, so a Friday-evening run printed "analysis
        as of 2026-08-21 close -> orders for 2026-08-21 open" — the open that
        happened seven hours before the close it was derived from.
        """
        report = desk.run()
        assert report.data_date == "2026-08-21"
        assert report.date == "2026-08-24"
        assert report.data_date < report.date

    def test_a_weekend_run_is_dated_for_the_next_real_session(self, desk):
        """A Saturday run named a Saturday open, and Saturday has no session."""
        report = desk.run(now=SATURDAY_NOON)
        assert report.date == "2026-08-24"
        assert clock.is_trading_day(date.fromisoformat(report.date))

    def test_data_is_a_completed_session_not_one_that_has_merely_opened(self, desk):
        """Midday bars are half a session of trading that has not happened yet.

        clock.last_trading_day returns a session that has *started*, so a run at
        12:00 ranked names on an unfinished close and called it the close.
        """
        assert clock.last_trading_day(FRIDAY_MIDDAY) == FRIDAY
        assert last_completed_session(FRIDAY_MIDDAY) == date(2026, 8, 20)
        assert desk.run(now=FRIDAY_MIDDAY).data_date == "2026-08-20"

    def test_a_half_day_is_complete_at_its_own_close(self):
        """The Friday after Thanksgiving ends at 13:00, not at 16:00."""
        half_day = date(2026, 11, 27)
        assert half_day in clock.HALF_DAYS
        noon = datetime(2026, 11, 27, 12, 0, tzinfo=clock.ET)
        assert last_completed_session(noon) == date(2026, 11, 25)
        after = datetime(2026, 11, 27, 13, 30, tzinfo=clock.ET)
        assert last_completed_session(after) == half_day

    def test_a_holiday_is_never_named_as_the_order_session(self):
        """Thanksgiving is a closed day; orders placed at its open cannot fill."""
        wednesday = datetime(2026, 11, 25, 17, 0, tzinfo=clock.ET)
        order_day, data_day = sessions_for(now=wednesday)
        assert order_day == date(2026, 11, 27)          # Thursday is the holiday
        assert data_day == date(2026, 11, 25)

    def test_date_names_the_session_the_orders_are_for(self, desk):
        """--date is the session acted on, not the session read."""
        report = desk.run(when="2026-08-25", now=MONDAY_AFTER_CLOSE)
        assert report.date == "2026-08-25"
        assert report.data_date == "2026-08-24"

    def test_a_date_on_a_closed_day_moves_to_the_session_after_it(self, desk):
        report = desk.run(when="2026-08-22", now=FRIDAY_AFTER_CLOSE)
        assert report.date == "2026-08-24"

    def test_a_session_that_has_already_closed_is_refused(self, desk):
        """Answering it would hand back the bars of the session it advises on.

        --date 2026-08-21 run after that session closed used to return the
        08-21 bars dated for the 08-21 open: a perfect look-ahead, written into
        the book as advice.
        """
        report = desk.run(when="2026-08-21", now=FRIDAY_AFTER_CLOSE)
        assert report.refused
        assert "already closed" in report.refused
        assert report.buys == [] and report.sells == []
        assert len(RecommendationBook(desk.tmp_path / "recommendations.json")) == 0
        assert "no report" in format_report(report)

    def test_a_date_further_in_the_past_is_refused_too(self, desk):
        assert desk.run(when="2026-06-01", now=FRIDAY_AFTER_CLOSE).refused

    def test_an_unreadable_date_is_refused_rather_than_silently_today(self, desk):
        report = desk.run(when="last tuesday")
        assert report.refused and "ISO date" in report.refused

    def test_the_header_names_both_sessions(self, desk):
        text = format_report(desk.run())
        assert "data through the 2026-08-21 close" in text
        assert "orders for the 2026-08-24 open" in text

    def test_the_english_report_contains_no_chinese(self, desk):
        """The session line was the only CJK in the package, printed always.

        It rendered into the terminal report and every saved .md regardless of
        output_language, which is the mixed-language report tests/
        test_i18n_coverage.py exists to forbid.
        """
        report = desk.run()
        for text in (format_report(report), to_markdown(report)):
            assert not any("一" <= ch <= "鿿" for ch in text)


# ---------------------------------------------------------------------------
# exits reach the book
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestExitsReachTheBook:
    def _issued(self, desk):
        """One run that issues AAA and BBB, then AAA stopped out."""
        first = desk.run(now=FRIDAY_AFTER_CLOSE)
        assert [r.symbol for r in first.buys] == ["AAA", "BBB"]
        desk.snaps["AAA"] = snap("AAA", 90.0)      # through the 94.00 stop
        return first

    def test_a_stopped_out_idea_is_closed_in_the_book(self, desk):
        """review() emitted the SELL and nothing ever acted on it.

        No caller in the package invoked close_from, close or expire, so the
        idea stayed open forever with the exit printed on a page and nowhere
        else.
        """
        self._issued(desk)
        report = desk.run(now=MONDAY_AFTER_CLOSE)
        assert [s.symbol for s in report.sells_closing()] == ["AAA"]
        assert [r.symbol for r in report.closed] == ["AAA"]

        book = RecommendationBook(desk.tmp_path / "recommendations.json")
        aaa = next(r for r in book.recommendations if r.symbol == "AAA")
        assert aaa.status == CLOSED
        assert aaa.exit_reason == REASON_STOP
        assert aaa.exit_price == 90.0
        assert aaa.exit_date == "2026-08-24"        # the session the mark is from
        assert aaa.realized_pnl == pytest.approx(-800.0)

    def test_a_closed_idea_stops_re_emitting_its_sell(self, desk):
        """Every stopped-out idea used to re-emit the same SELL every morning."""
        self._issued(desk)
        desk.run(now=MONDAY_AFTER_CLOSE)
        again = desk.run(now=TUESDAY_AFTER_CLOSE)
        assert [s.symbol for s in again.sells] == []

    def test_a_closed_idea_appears_in_the_track_record(self, desk):
        """closed stayed 0 forever, so the one checkable part had nothing in it."""
        self._issued(desk)
        report = desk.run(now=MONDAY_AFTER_CLOSE)
        assert report.track_record.closed == 1
        assert report.track_record.losses == 1
        assert report.track_record.total_pnl == pytest.approx(-800.0)
        assert "Nothing has been closed yet" not in format_report(report)

    def test_a_closed_symbol_can_be_a_candidate_again(self, desk):
        """_eligible banned every symbol ever recommended, so the pool only shrank.

        open_recommendations never lost a member, and the ban is keyed on it.
        """
        self._issued(desk)
        desk.run(now=MONDAY_AFTER_CLOSE)
        later = desk.run(now=TUESDAY_AFTER_CLOSE)
        assert "AAA" in [r.symbol for r in later.buys]

    def test_the_same_report_does_not_buy_back_what_it_just_sold(self, desk):
        """Closing frees the symbol immediately; SELL AAA above, BUY AAA below."""
        self._issued(desk)
        report = desk.run(now=MONDAY_AFTER_CLOSE)
        assert "AAA" not in [r.symbol for r in report.buys]
        assert any("exited on this same report" in n for n in report.notes)

    def test_a_dry_run_closes_nothing(self, desk):
        """--dry-run promises the book is untouched, on the sell side too."""
        self._issued(desk)
        desk.cfg.dry_run = True
        report = desk.run(now=MONDAY_AFTER_CLOSE)
        assert [s.symbol for s in report.sells_closing()] == ["AAA"]
        assert report.closed == []
        book = RecommendationBook(desk.tmp_path / "recommendations.json")
        assert all(r.status == OPEN for r in book.recommendations)
        assert any("dry run" in n for n in report.notes)

    def test_an_unpriceable_exit_expires_instead_of_closing_with_no_pnl(self, desk):
        """A thesis break can fire with no price; closing it would fake a sample.

        close() with a NaN price stores no P&L, and the row would still count in
        track_record.closed while contributing nothing to it.
        """
        adv = desk.advisor()
        rec = adv.book.add("CCC", "BUY", 10, 100.0, 90.0, 130.0)
        report = DailyReport(date="2026-08-24", data_date="2026-08-21")
        signal = ExitSignal(rec_id=rec.id, symbol="CCC", action=SELL_ALL, shares=10,
                            reason="thesis break — recall", exit_reason="thesis_break")
        assert math.isnan(signal.price)

        closed = adv.apply_exits([signal], FRIDAY, report)
        assert [r.status for r in closed] == [EXPIRED]
        assert adv.book.track_record().closed == 0

    def test_a_book_that_cannot_be_written_warns_instead_of_raising(self, desk, monkeypatch):
        """Nothing may raise out of a report meant to run unattended for weeks."""
        adv = desk.advisor()
        rec = adv.book.add("CCC", "BUY", 10, 100.0, 90.0, 130.0)

        def boom(*args, **kwargs):
            raise OSError("read-only file system")

        monkeypatch.setattr(adv.book, "close", boom)
        report = DailyReport(date="2026-08-24", data_date="2026-08-21")
        signal = ExitSignal(rec_id=rec.id, symbol="CCC", action=SELL_ALL, shares=10,
                            reason="stop hit", price=89.0, exit_reason=REASON_STOP)
        assert adv.apply_exits([signal], FRIDAY, report) == []
        assert any("could not be closed" in w for w in report.warnings)

    def test_a_signal_for_an_idea_the_book_lost_is_reported(self, desk):
        adv = desk.advisor()
        report = DailyReport(date="2026-08-24", data_date="2026-08-21")
        signal = ExitSignal(rec_id="GONE-20260101", symbol="GONE", action=SELL_ALL,
                            shares=5, reason="stop hit", price=10.0)
        assert adv.apply_exits([signal], FRIDAY, report) == []
        assert any("was not open in the book" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# the screen cache
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestScreenCache:
    def test_a_screen_saved_for_the_data_session_is_reused(self, desk):
        """The cache compared the saved date against the *order* session.

        Those two are different dates on every run, so the check never matched:
        the whole universe was re-downloaded for a session already on disk.
        """
        save_screen(desk.tmp_path, FRIDAY)
        report = desk.run(now=FRIDAY_AFTER_CLOSE)
        assert desk.screens == []
        assert any("reusing the saved scan of the 2026-08-21 session" in n
                   for n in report.notes)
        assert not any("falling back" in w for w in report.warnings)

    def test_a_failed_rescan_falls_back_to_an_older_screen(self, desk, monkeypatch):
        save_screen(desk.tmp_path, date(2026, 8, 18))

        def boom(*args, **kwargs):
            raise RuntimeError("nasdaqtrader down")

        monkeypatch.setattr(advisor, "run_screen", boom)
        report = desk.run(now=FRIDAY_AFTER_CLOSE)
        assert any("falling back to the saved screen from 2026-08-18" in w
                   for w in report.warnings)
        assert [c.symbol for c in report.candidates] == ["AAA", "BBB"]

    def test_use_cache_reports_the_age_against_the_data_session(self, desk):
        save_screen(desk.tmp_path, date(2026, 8, 18))
        desk.cfg.use_cache = True
        report = desk.run(now=FRIDAY_AFTER_CLOSE)
        assert desk.screens == []
        assert any("screen is 3 day(s) old" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# the CLI
# ---------------------------------------------------------------------------

def _capture_cfg(monkeypatch):
    """Run main() against a stub advisor and hand back the config it built."""
    seen = []

    class Stub:
        def __init__(self, cfg=None, **kwargs):
            seen.append(cfg)

        def run(self, when=None, now=None):
            return DailyReport(date="2026-08-24", data_date="2026-08-21",
                               dry_run=True)

    monkeypatch.setattr(advisor, "DailyAdvisor", Stub)
    return seen


@pytest.mark.unit
class TestCommandLine:
    def test_the_account_value_env_var_is_read(self, monkeypatch):
        """The report told the reader to set it, then built a config ignoring it.

        DailyAdvisor falls back to from_env() only when cfg is falsy, and a
        dataclass instance never is — so every share count on the page was
        scaled by the default $100,000.
        """
        monkeypatch.setenv("TRADINGAGENTS_ACCOUNT_VALUE", "250000")
        seen = _capture_cfg(monkeypatch)
        assert advisor.main(["--no-llm", "--dry-run"]) == 0
        assert seen[0].fallback_account_value == 250_000.0

    def test_top_above_the_panel_budget_raises_the_panel_budget(self, monkeypatch):
        """--top 20 printed at most 8 ideas and said nothing about it.

        generate_buys cuts to max_candidates before a single idea is produced,
        and --top was wired only to cfg.top.
        """
        seen = _capture_cfg(monkeypatch)
        advisor.main(["--no-llm", "--dry-run", "--top", "20"])
        assert seen[0].top == 20 and seen[0].max_candidates == 20

    def test_an_explicit_panel_budget_still_wins(self, monkeypatch):
        seen = _capture_cfg(monkeypatch)
        advisor.main(["--no-llm", "--dry-run", "--top", "20",
                      "--max-candidates", "3"])
        assert seen[0].top == 20 and seen[0].max_candidates == 3

    def test_top_below_the_default_does_not_shrink_the_panel_budget(self, monkeypatch):
        seen = _capture_cfg(monkeypatch)
        advisor.main(["--no-llm", "--dry-run", "--top", "2"])
        assert seen[0].top == 2 and seen[0].max_candidates == 8

    def test_a_refused_date_exits_non_zero(self, desk, monkeypatch):
        monkeypatch.setattr(advisor, "DailyAdvisor",
                            lambda *a, **k: desk.advisor())
        assert advisor.main(["--no-llm", "--date", "1999-01-04"]) == 2

    def test_a_report_that_cannot_be_saved_does_not_end_in_a_traceback(
            self, desk, monkeypatch):
        """The ideas are already printed and already in the book by then."""
        monkeypatch.setattr(advisor, "DailyAdvisor",
                            lambda *a, **k: desk.advisor())

        def boom(report):
            raise OSError("read-only file system")

        monkeypatch.setattr(advisor, "save_report", boom)
        assert advisor.main(["--no-llm"]) == 0


# ---------------------------------------------------------------------------
# the policy tilt
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPolicyTilt:
    def _cands(self, n=12, sector_of=lambda i: "Technology"):
        return [Candidate(symbol=f"S{i}", rank=i, sector=sector_of(i))
                for i in range(1, n + 1)]

    def test_a_name_falls_no_further_than_twice_the_shift(self, desk):
        """"Moves a name at most five places" was never a bound on places.

        MAX_TILT_SHIFT bounds the score adjustment, and opposing tilts move the
        rest of the list the other way at the same time: with one sector at
        -1.0 and the rest at +1.0 the top name fell to last on a ten-name list.
        The real bound is 2 * MAX_TILT_SHIFT - 1 places, and that is what the
        docstring now claims.
        """
        cands = self._cands(30, sector_of=lambda i: "Energy" if i == 1 else "Technology")
        order = desk.advisor().rank(cands, {"Energy": -1.0, "Technology": 1.0})
        moved = [c.symbol for c in order].index("S1")
        assert 0 < moved <= 2 * MAX_TILT_SHIFT - 1

    def test_no_name_moves_further_than_the_bound_in_either_direction(self, desk):
        cands = self._cands(30, sector_of=lambda i: "Energy" if i % 3 else "Technology")
        order = desk.advisor().rank(cands, {"Energy": -1.0, "Technology": 1.0})
        for place, cand in enumerate(order, start=1):
            assert abs(place - cand.rank) <= 2 * MAX_TILT_SHIFT - 1

    def test_the_tilt_never_adds_or_removes_a_name(self, desk):
        cands = self._cands(12, sector_of=lambda i: "Energy" if i % 2 else "Utilities")
        order = desk.advisor().rank(cands, {"Energy": -1.0, "Utilities": 1.0})
        assert sorted(c.symbol for c in order) == sorted(c.symbol for c in cands)


# ---------------------------------------------------------------------------
# sizing refusals
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSizingRefusals:
    def _size(self, desk, cand_snap, **cfg):
        for key, value in cfg.items():
            setattr(desk.cfg, key, value)
        cand = Candidate(symbol="AAA", rank=1, sector="Technology", snap=cand_snap)
        return desk.advisor().size(cand, desk.account, MONDAY, 0.5, "because")

    def test_a_name_with_no_atr_is_not_blamed_on_the_tick(self, desk):
        """The refusal named one of five causes unconditionally.

        With no ATR the message read "ATR is nan% of price, which puts the stop
        inside the tick the shares quote in", which is not what happened.
        """
        rec, reason = self._size(desk, snap("AAA", 100.0, atr_pct=float("nan")))
        assert rec is None
        assert "no usable ATR" in reason
        assert "nan%" not in reason and "tick" not in reason

    def test_a_tick_sized_stop_is_still_reported_as_one(self, desk):
        rec, reason = self._size(desk, snap("AAA", 100.0, atr_pct=0.00001))
        assert rec is None and "inside the cent" in reason

    def test_an_atr_wider_than_the_price_says_so(self, desk):
        rec, reason = self._size(desk, snap("AAA", 100.0, atr_pct=0.80))
        assert rec is None and "at or below zero" in reason

    def test_the_r_filter_runs_before_sizing(self, desk, monkeypatch):
        """The docstring claimed the opposite order for a load-bearing decision.

        R is made of the three levels and nothing else, so no share count can
        change it; a name below the minimum is rejected without a sizing pass.
        """
        def boom(*args, **kwargs):
            raise AssertionError("sizing must not run on a rejected R")

        monkeypatch.setattr(advisor, "size_position", boom)
        rec, reason = self._size(desk, snap("AAA", 100.0), min_r=99.0)
        assert rec is None and "below the 99.00R minimum" in reason


# ---------------------------------------------------------------------------
# what the page says
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRendering:
    def test_the_caveat_is_printed_on_every_report(self, desk):
        """It was defined, documented as printed on every report, and dead."""
        report = desk.run()
        opening = CAVEAT.split(".")[0]
        assert opening in " ".join(format_report(report).split())
        assert opening in " ".join(to_markdown(report).split())

    def test_the_reference_note_discloses_the_limit_buffer(self, desk):
        """RISK 480.00 is measured at the reference; a fill at 100.30 risks 504.

        REFERENCE_NOTE was written to disclose exactly that gap and never
        reached the page.
        """
        report = desk.run()
        text = " ".join(format_report(report).split())
        assert " ".join(REFERENCE_NOTE.split()) in text
        aaa = next(r for r in report.buys if r.symbol == "AAA")
        assert aaa.limit_price > aaa.reference_price

    def test_the_r_column_is_the_r_the_list_was_ranked_on(self, desk):
        """The renderers recomputed R against the live stop the trailing rule moves.

        generate_buys ranks on planned_r, which uses initial_stop_price because
        moving a stop rewrites history; the two agreed only on issue day.
        """
        report = desk.run()
        aaa = next(r for r in report.buys if r.symbol == "AAA")
        planned = aaa.planned_r()
        aaa.stop_price = aaa.reference_price          # the breakeven rule fires

        # Assert against AAA's own row, not the whole page. A substring search
        # over the report passed against the broken code, because another row
        # happened to carry the same R and matched while AAA rendered "nan".
        def row(text: str, sep: str) -> str:
            line = next(ln for ln in text.splitlines()
                        if ln.strip().startswith(sep + "AAA") or " AAA " in ln
                        or ln.strip().startswith("AAA"))
            return line

        assert f"{planned:.2f}" in row(format_report(report), "")
        assert "nan" not in row(format_report(report), "").lower()
        assert f"| {planned:.2f} |" in row(to_markdown(report), "|")

    def test_an_unreadable_risk_does_not_make_the_total_unreadable(self, desk):
        """format_report summed with a bare sum(), dropping total_risk's NaN filter."""
        report = desk.run()
        report.buys[0].initial_stop_price = float("nan")   # risk_amount() -> NaN
        assert math.isnan(advisor.planned_risk(report.buys[0]))
        assert not math.isnan(report.total_risk)
        total = next(line for line in format_report(report).splitlines()
                     if line.strip().startswith("TOTAL"))
        assert "nan" not in total.lower() and "320.00" in total

    def test_the_account_line_calls_the_risk_a_budget_and_prints_the_real_one(
            self, desk):
        """"risk/trade 1.00%" described rows that risk 0.80% after the position cap."""
        report = desk.run()
        text = format_report(report)
        assert "risk budget 1.00%/trade" in text
        assert f"planned risk ${report.total_risk:,.2f}" in text
        assert "0.80% of equity" in text

    def test_a_refusal_renders_as_a_refusal_in_both_formats(self, desk):
        report = desk.run(when="2026-08-21", now=FRIDAY_AFTER_CLOSE)
        assert "no report" in format_report(report)
        assert "no report" in to_markdown(report)
        assert "BUY" not in to_markdown(report)


# ---------------------------------------------------------------------------
# nothing raises, nothing is destroyed
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestNothingRaises:
    def test_a_policy_summariser_that_raises_costs_only_the_backdrop(
            self, desk, monkeypatch):
        """run() promised "never raises" with policy_brief outside every try."""
        def boom(*args, **kwargs):
            raise ValueError("malformed event")

        monkeypatch.setattr(advisor, "policy_brief", boom)
        report = desk.run()
        assert report.buys and report.policy_summary == ""
        assert any("policy backdrop could not be summarised" in w
                   for w in report.warnings)

    def test_a_market_context_that_raises_costs_only_the_context(
            self, desk, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("zoneinfo missing")

        monkeypatch.setattr(clock, "market_state", boom)
        report = desk.run()
        assert report.buys and report.market_context == ""
        assert any("market context could not be read" in w for w in report.warnings)

    def test_screen_stats_that_are_not_a_mapping_do_not_end_the_run(self, desk):
        desk.stats = None
        report = desk.run()
        assert report.buys
        assert any("listed names" in n for n in report.notes)

    def test_an_evidence_pack_that_raises_leaves_the_name_untaken(
            self, desk, monkeypatch):
        """A configured panel that never saw the idea must not appear to approve it."""
        def boom(*args, **kwargs):
            raise ValueError("pandas exploded")

        monkeypatch.setattr(advisor, "build_evidence", boom)
        panel = StubPanel()
        adv = desk.advisor()
        adv.panel = panel
        report = adv.run(now=FRIDAY_AFTER_CLOSE)
        assert report.buys == [] and panel.calls == []
        assert any("evidence pack could not be built" in n for n in report.notes)

    def test_no_panel_means_no_evidence_pack_is_built_at_all(self, desk, monkeypatch):
        """With no panel it was assembled and thrown away, once per candidate."""
        def boom(*args, **kwargs):
            raise AssertionError("no panel, no evidence pack")

        monkeypatch.setattr(advisor, "build_evidence", boom)
        assert len(desk.run().buys) == 2


@pytest.mark.unit
class TestInjectedMonitors:
    def test_an_injected_monitors_backlog_survives_the_report(self, desk):
        """Clearing seen made poll() persist the emptied set to the monitor's file.

        news= and policy_monitor= are advertised injection points; handing the
        live monitor's own instance to a report wiped its suppression, and its
        next cycle replayed days of coverage as breaking — the failure
        NewsMonitor.prime exists to prevent.
        """
        desk.news = StubMonitor(seen={"old-story": "2026-08-20T12:00:00+00:00"})
        desk.policy = StubMonitor(seen={"old-tariff": "2026-08-19T12:00:00+00:00"})
        desk.run()
        assert "old-story" in desk.news.seen
        assert "old-tariff" in desk.policy.seen
        # And re-persisted, not merely restored in memory.
        assert desk.news.saved and "old-story" in desk.news.saved[-1]
        assert desk.policy.saved and "old-tariff" in desk.policy.saved[-1]

    def test_the_backdrop_is_still_what_is_standing_now(self, desk):
        """The clearing exists so a second run in one morning is not empty."""
        seen = {"old-story": "2026-08-20T12:00:00+00:00"}
        desk.news = StubMonitor(seen=dict(seen))
        polls = []
        original = desk.news.poll

        def record(*args, **kwargs):
            polls.append(dict(desk.news.seen))
            return original(*args, **kwargs)

        desk.news.poll = record
        desk.run()
        assert polls and polls[0] == {}


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSaveReport:
    def test_the_page_is_written_whole_or_not_at_all(self, desk, monkeypatch):
        """A plain write_text leaves a truncated page that reads as a complete one."""
        report = desk.run()
        path = save_report(report)
        first = path.read_text(encoding="utf-8")
        assert first.startswith("# Daily Advisor")
        assert list(path.parent.glob("*.tmp")) == []

        def boom(src, dst):
            raise OSError("no space left on device")

        monkeypatch.setattr(advisor.os, "replace", boom)
        report.notes.append("a second run with more to say")
        with pytest.raises(OSError):
            save_report(report)
        assert path.read_text(encoding="utf-8") == first

    def test_the_page_is_named_for_the_session_the_orders_are_for(self, desk):
        report = desk.run()
        assert save_report(report).name == "2026-08-24.md"
