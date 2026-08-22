"""The recommendation book: the exit engine, and the arithmetic that grades it.

Nothing here touches a network. Prices arrive as a dict the caller assembles,
and headlines as objects or mappings — so a test that wants a 200-day-old
bearish headline builds one, and the only "feed" is the literal it was built
from.

The tests are grouped by the failure each one prevents. Most of them exist
because the module's whole stated purpose is to measure the desk's
self-flattery, which makes every quiet over-statement in here a defect of the
worst kind: a book that flatters itself while claiming to be the thing that
catches flattery.
"""

import json
import math
import re
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone

import pytest

from datetime import date as _date

from tradingagents.live.newsfeed import NewsItem
from tradingagents.live.recommendations import (
    BUY,
    MAX_MARK_AGE_DAYS,
    REASON_TARGET,
    SELL,
    SELL_ALL,
    TRIM,
    UNAVAILABLE,
    ExitRules,
    Recommendation,
    RecommendationBook,
    format_open_book,
    format_track_record,
)


@pytest.fixture()
def book(tmp_path):
    """A book on its own file, so no test can read or write the real one."""
    return RecommendationBook(tmp_path / "recommendations.json")


def _hours_ago(hours: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()


def _news(materiality: int = 9, lean: str = "bearish", age_hours: float = 2.0) -> NewsItem:
    return NewsItem(ticker="CCC", title="CCC misses estimates", link="",
                    source="test", published=_hours_ago(age_hours),
                    materiality=materiality, lean=lean)


# ---------------------------------------------------------------------------
# defect 2: the trim at the target
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestTargetTrimAccounting:
    """A trim that is not booked leaves the idea scored on shares it sold.

    The worked examples are the reviewer's: BUY 100 NVDA at 100.00, stop 90.00,
    target 120.00, so $1,000 of risk and $10 a share.
    """

    def _nvda(self, book) -> Recommendation:
        return book.add("NVDA", BUY, 100, 100.0, 90.0, 120.0, rec_id="NVDA-1")

    def test_trim_reduces_the_shares_still_held(self, book):
        """Without this the book keeps 100 shares of a position it just halved.

        Every later signal and close() then price the whole position, so the
        50 shares the desk's own instruction sold are scored twice.
        """
        rec = self._nvda(book)
        signals = book.review({"NVDA": 121.0})

        trim = [s for s in signals if s.action == TRIM]
        assert len(trim) == 1
        assert trim[0].shares == 50
        assert rec.shares == 50
        assert rec.trimmed_shares == 50
        assert rec.trimmed_pnl == 1_000.0
        assert rec.initial_shares == 100

    def test_trim_then_breakeven_stop_records_975_not_minus_50(self, book):
        """The reviewer's first row: the book recorded -$50 for a +$975 idea.

        Trim at 121, then the breakeven stop fills at 99.50. Following the
        book's own instructions returns 50x(120-100) + 50x(99.50-100) = +$975,
        i.e. +0.975R. Scoring the full 100 shares at 99.50 gives -$50.
        """
        rec = self._nvda(book)
        book.review({"NVDA": 121.0})

        stopped = book.review({"NVDA": 99.50})
        sell = [s for s in stopped if s.action == SELL_ALL]
        assert len(sell) == 1
        assert sell[0].shares == 50          # what is left, not what was issued
        book.close_from(sell[0])

        assert rec.realized_pnl == pytest.approx(975.0)
        assert rec.realized_r == pytest.approx(0.975)

    def test_trim_then_manual_close_at_150_records_3500_not_5000(self, book):
        """The reviewer's second row: +$5,000 booked for a +$3,500 idea, 43% over.

        Trim at 121, then a manual close at 150: 50x(120-100) + 50x(150-100) =
        +$3,500, i.e. +3.50R against $1,000 of risk. The unbooked trim scored
        all 100 shares at 150 and printed +5.00R.
        """
        rec = self._nvda(book)
        book.review({"NVDA": 121.0})
        book.close(rec.id, 150.0)

        assert rec.realized_pnl == pytest.approx(3_500.0)
        assert rec.realized_r == pytest.approx(3.50)

    def test_r_stays_measured_against_the_risk_as_issued(self, book):
        """Halving the shares must not halve the R denominator.

        risk_amount is entry-to-stop times the size *as issued*. Recomputed
        from the remaining 50 shares it would be $500, and the +$3,500 close
        above would print +7.00R — a worse overstatement than the one this
        change fixes.
        """
        rec = self._nvda(book)
        book.review({"NVDA": 121.0})

        assert rec.shares == 50
        assert rec.risk_amount() == pytest.approx(1_000.0)
        assert rec.r_at(150.0) == pytest.approx(3.50)

    def test_trim_is_booked_at_the_target_not_at_a_price_that_ran_past_it(self, book):
        """A poll that catches a print above the target did not earn the gap.

        The instruction was "sell at 120". Booking the trim at whatever the
        poll saw — 180 here — credits the advice with a move it never named,
        which is the exact habit this book exists to measure.
        """
        rec = self._nvda(book)
        book.review({"NVDA": 180.0})

        assert rec.trimmed_pnl == pytest.approx(1_000.0)   # 50 x (120 - 100)

    def test_open_mark_carries_what_the_trim_locked_in(self, book):
        """Booking the trim must not make its profit vanish from the book.

        Back at the entry price the remaining half is flat, and the idea is
        still $1,000 up on the half already sold. Marking only what is left
        would report it as flat.
        """
        rec = self._nvda(book)
        book.review({"NVDA": 121.0})

        assert rec.pnl_at(100.0) == pytest.approx(1_000.0)
        tr = book.track_record({"NVDA": 100.0})
        assert tr.open_marked == 1
        assert tr.open_unrealized == pytest.approx(1_000.0)

    def test_short_side_trim_is_the_same_arithmetic_mirrored(self, book):
        """A SELL idea is graded as a short, so its trim has to book the same way."""
        rec = book.add("TSLA", SELL, 100, 100.0, 110.0, 80.0, rec_id="TSLA-1")
        book.review({"TSLA": 79.0})

        assert rec.shares == 50
        assert rec.trimmed_pnl == pytest.approx(1_000.0)   # 50 x (100 - 80)
        book.close(rec.id, 50.0)
        assert rec.realized_pnl == pytest.approx(3_500.0)
        assert rec.realized_r == pytest.approx(3.50)

    def test_the_target_instruction_is_still_issued_only_once(self, book):
        """The trim must not repeat every cycle while the price sits at the target."""
        rec = self._nvda(book)
        book.review({"NVDA": 121.0})

        again = book.review({"NVDA": 122.0})
        assert [s.action for s in again if s.action == TRIM] == []
        assert rec.shares == 50

    def test_selling_the_whole_position_at_the_target_still_works(self, book):
        """With trim_at_target off nothing is booked partially, so shares stand."""
        rec = self._nvda(book)
        signals = book.review({"NVDA": 121.0}, rules=ExitRules(trim_at_target=False))

        assert [(s.action, s.shares) for s in signals] == [(SELL_ALL, 100)]
        assert signals[0].exit_reason == REASON_TARGET
        assert rec.shares == 100
        assert rec.trimmed_pnl == 0.0

    def test_a_booked_trim_survives_a_save_and_reload(self, tmp_path):
        """A trim held only in memory is a trim that unwinds on the next restart."""
        path = tmp_path / "recommendations.json"
        book = RecommendationBook(path)
        rec = book.add("NVDA", BUY, 100, 100.0, 90.0, 120.0, rec_id="NVDA-1")
        book.review({"NVDA": 121.0})
        book.save()

        reloaded = RecommendationBook(path).get("NVDA-1")
        assert reloaded is not None
        assert (reloaded.shares, reloaded.trimmed_shares) == (50, 100 - 50)
        assert reloaded.trimmed_pnl == pytest.approx(1_000.0)
        assert reloaded.initial_shares == 100
        assert reloaded.risk_amount() == pytest.approx(1_000.0)
        assert rec.trimmed_pnl == reloaded.trimmed_pnl

    def test_a_book_written_before_trims_were_booked_still_loads(self, tmp_path):
        """No initial_shares in the file means nothing had ever reduced shares."""
        path = tmp_path / "recommendations.json"
        path.write_text(json.dumps({"recommendations": [{
            "id": "OLD-1", "issued_date": "2026-01-02", "symbol": "OLD",
            "action": "BUY", "shares": 100, "reference_price": 100.0,
            "stop_price": 90.0, "target_price": 120.0, "status": "open",
        }]}))

        rec = RecommendationBook(path).get("OLD-1")
        assert rec is not None
        assert rec.initial_shares == 100
        assert rec.trimmed_shares == 0
        assert rec.risk_amount() == pytest.approx(1_000.0)


# ---------------------------------------------------------------------------
# defect 3: a datetime where a date was expected
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDatetimeAsOf:
    """datetime is a subclass of date, so it passes the annotation and then raises.

    ``datetime - date`` is a TypeError. Contained by the per-recommendation
    try/except, it turned every open idea into UNAVAILABLE — the whole book
    silently going unchecked, which is the failure that try/except exists to
    prevent.
    """

    def test_review_with_a_datetime_checks_every_recommendation(self, book):
        rec = book.add("BBB", BUY, 10, 100.0, 90.0, 120.0,
                       issued_date="2026-08-01", rec_id="BBB-1")
        signals = book.review({"BBB": 101.0}, as_of=datetime(2026, 8, 21, 15, 0))

        assert [s for s in signals if s.action == UNAVAILABLE] == []
        assert rec.last_price == 101.0

    def test_review_with_a_datetime_writes_a_plain_date(self, book):
        """last_reviewed dates the mark; a timestamp there is a field other
        writers fill with a date, and _usable_mark has to read it back."""
        rec = book.add("BBB", BUY, 10, 100.0, 90.0, 120.0,
                       issued_date="2026-08-01", rec_id="BBB-1")
        book.review({"BBB": 101.0}, as_of=datetime(2026, 8, 21, 15, 0))

        assert rec.last_reviewed == "2026-08-21"

    def test_days_held_accepts_a_datetime(self):
        rec = Recommendation(id="X", issued_date="2026-08-01", symbol="X",
                             action=BUY, shares=1, reference_price=100.0,
                             stop_price=90.0, target_price=120.0)
        assert rec.days_held(datetime(2026, 8, 21, 15, 0)) == 20
        assert rec.days_held(date(2026, 8, 21)) == 20

    def test_time_stop_still_fires_when_as_of_is_a_datetime(self, book):
        """The horizon rule is the one that needs the date arithmetic most."""
        book.add("BBB", BUY, 10, 100.0, 90.0, 120.0,
                 issued_date="2026-07-01", horizon_days=30, rec_id="BBB-1")
        signals = book.review({"BBB": 101.0}, as_of=datetime(2026, 8, 21, 15, 0))

        assert [(s.action, s.exit_reason) for s in signals] == [(SELL_ALL, "time_stop")]

    def test_track_record_and_open_book_accept_a_datetime(self, book):
        book.add("BBB", BUY, 10, 100.0, 90.0, 120.0,
                 issued_date="2026-08-01", rec_id="BBB-1")
        tr = book.track_record({"BBB": 101.0}, as_of=datetime(2026, 8, 21, 15, 0))

        assert tr.open_marked == 1
        assert "20/30" in format_open_book(book.open_recommendations(),
                                           {"BBB": 101.0},
                                           datetime(2026, 8, 21, 15, 0))


# ---------------------------------------------------------------------------
# defect 4: how old a headline is
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestHeadlineFreshness:
    """An unknown age must mean stale, or the age bound fails open.

    The bound is what stops last quarter's earnings from closing a position the
    first time the newsfeed's seen-set is pruned.
    """

    def _book(self, book):
        book.add("CCC", BUY, 10, 100.0, 90.0, 120.0, rec_id="CCC-1")
        return book

    def test_asdict_of_an_old_newsitem_does_not_break_the_thesis(self, book):
        """asdict() drops age_hours — it is a method — leaving only published.

        A 200-day-old bearish headline read as "published now" and sold the
        position, while the same headline as an object was correctly ignored.
        """
        self._book(book)
        old = asdict(_news(age_hours=200 * 24))

        assert book.review({"CCC": 101.0}, {"CCC": [old]}) == []

    def test_a_mapping_with_no_age_at_all_is_treated_as_stale(self, book):
        """LLM JSON with no timestamp cannot be dated, so it cannot be fresh."""
        self._book(book)
        item = {"materiality": 9, "lean": "bearish", "title": "CCC misses"}

        assert book.review({"CCC": 101.0}, {"CCC": [item]}) == []

    def test_an_unreadable_published_is_treated_as_stale(self, book):
        self._book(book)
        item = {"materiality": 9, "lean": "bearish", "title": "CCC misses",
                "published": "last Tuesday"}

        assert book.review({"CCC": 101.0}, {"CCC": [item]}) == []

    def test_a_fresh_mapping_still_breaks_the_thesis(self, book):
        """The fix must not close the rule to real breaking news."""
        self._book(book)
        fresh = asdict(_news(age_hours=2.0))
        signals = book.review({"CCC": 101.0}, {"CCC": [fresh]})

        assert [(s.action, s.exit_reason) for s in signals] == [(SELL_ALL, "thesis_break")]

    def test_an_explicit_age_hours_key_is_still_honoured(self, book):
        self._book(book)
        item = {"materiality": 9, "lean": "bearish", "title": "CCC misses",
                "age_hours": 2.0}

        assert len(book.review({"CCC": 101.0}, {"CCC": [item]})) == 1

    def test_an_explicit_age_hours_key_wins_over_a_fresh_timestamp(self, book):
        """Whoever computed age_hours knew when it was polled; published is a fallback."""
        self._book(book)
        item = asdict(_news(age_hours=2.0)) | {"age_hours": 4_800.0}

        assert book.review({"CCC": 101.0}, {"CCC": [item]}) == []

    def test_the_object_form_is_unchanged(self, book):
        """NewsItem.age_hours is a method, and it stays the authority for objects."""
        self._book(book)
        assert book.review({"CCC": 101.0}, {"CCC": [_news(age_hours=200 * 24)]}) == []
        assert len(book.review({"CCC": 101.0}, {"CCC": [_news(age_hours=2.0)]})) == 1

    def test_a_malformed_entry_does_not_lose_the_rest_of_the_headlines(self, book):
        self._book(book)
        signals = book.review({"CCC": 101.0},
                              {"CCC": [None, "not a headline", _news(age_hours=1.0)]})

        assert [s.action for s in signals] == [SELL_ALL]


# ---------------------------------------------------------------------------
# defect 5: a live stop that is not a level
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestUnusableLiveStop:
    """A blank stop in the hand-edited file used to produce no signal at all.

    problem() checked initial_stop_price, so levels_ok stayed true, _stop_hit
    quietly answered "not hit", and the open book rendered the missing stop as
    0.00 — a position shown as protected by a level that does not exist.
    """

    def _blank_stop_book(self, tmp_path) -> RecommendationBook:
        path = tmp_path / "recommendations.json"
        path.write_text(json.dumps({"recommendations": [{
            "id": "XYZ-1", "issued_date": date.today().isoformat(), "symbol": "XYZ",
            "action": "BUY", "shares": 100, "reference_price": 100.0,
            "stop_price": None, "target_price": 120.0, "initial_stop_price": 90.0,
            "status": "open",
        }]}))
        return RecommendationBook(path)

    def test_a_missing_live_stop_is_reported_as_a_problem(self, tmp_path):
        rec = self._blank_stop_book(tmp_path).get("XYZ-1")
        assert rec is not None
        assert "live stop" in rec.problem()

    def test_review_says_the_stop_could_not_be_checked(self, tmp_path):
        """Price 50 against a 90 stop used to produce silence, which reads as
        "nothing to do"."""
        book = self._blank_stop_book(tmp_path)
        signals = book.review({"XYZ": 50.0})

        assert [s.action for s in signals] == [UNAVAILABLE]
        assert "live stop" in signals[0].reason

    def test_the_open_book_prints_na_rather_than_a_stop_at_zero(self, tmp_path):
        book = self._blank_stop_book(tmp_path)
        page = format_open_book(book.open_recommendations(), {"XYZ": 50.0})
        row = next(line for line in page.splitlines() if line.startswith("XYZ-1"))

        # The stop column sits between the entry and the target: 0.00 there
        # reads as a stop at zero, which is a position shown as protected.
        assert re.search(r"100\.00\s+n/a\s+120\.00", row)

    def test_a_usable_stop_is_still_unremarkable(self, book):
        book.add("OK", BUY, 10, 100.0, 90.0, 120.0, rec_id="OK-1")
        assert book.review({"OK": 101.0}) == []


# ---------------------------------------------------------------------------
# defect 6: infinity survives to the report
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestInfiniteProfitFactor:
    """The table printed n/a directly above a note calling the number infinite.

    _num maps every non-finite value onto its default, so the isinf branches
    downstream of it could never fire.
    """

    def test_profit_factor_prints_inf_when_nothing_has_lost_money(self, book):
        book.add("W", BUY, 10, 100.0, 90.0, 120.0, rec_id="W-1")
        book.close("W-1", 120.0)
        tr = book.track_record({})

        assert math.isinf(tr.profit_factor)
        page = format_track_record(tr)
        assert "profit factor" in page
        line = next(x for x in page.splitlines() if "profit factor" in x)
        assert "inf" in line and "n/a" not in line
        assert any("infinite" in w for w in tr.warnings)

    def test_a_real_profit_factor_is_still_a_number(self, book):
        book.add("W", BUY, 10, 100.0, 90.0, 120.0, rec_id="W-1")
        book.add("L", BUY, 10, 100.0, 90.0, 120.0, rec_id="L-1")
        book.close("W-1", 120.0)     # +200
        book.close("L-1", 90.0)      # -100
        tr = book.track_record({})

        assert tr.profit_factor == pytest.approx(2.0)
        assert "2.00" in format_track_record(tr)

    def test_an_absent_number_still_prints_na(self, book):
        """Nothing closed at all is not an infinite profit factor."""
        book.add("O", BUY, 10, 100.0, 90.0, 120.0, rec_id="O-1")
        tr = book.track_record({"O": 101.0})

        assert math.isnan(tr.profit_factor)
        assert "profit factor" not in format_track_record(tr)


# ---------------------------------------------------------------------------
# defects 7 and 9: how old the mark is, and what as_of is for
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestStaleMarks:
    """An arbitrarily old last_price used to be marked to market as if current.

    last_reviewed was written on every review and read by nothing, so an idea
    last priced in January still contributed an open profit in August — to
    total_pnl_with_open, to win_rate_with_open and to flattery_r, the number
    published specifically to expose optimistic marking. as_of was declared and
    never used; it is what dates the mark.
    """

    def _amd(self, book) -> Recommendation:
        rec = book.add("AMD", BUY, 100, 100.0, 90.0, 120.0,
                       issued_date="2026-01-02", rec_id="AMD-1")
        book.review({"AMD": 118.0}, as_of=date(2026, 1, 2))
        return rec

    def test_a_mark_from_months_ago_is_not_marked_to_market(self, book):
        """+$1,800 from a January price counted as an open win every day since."""
        self._amd(book)
        tr = book.track_record({}, as_of=date(2026, 8, 21))

        assert tr.open_marked == 0
        assert tr.unpriced == ["AMD"]
        assert tr.stale_marks == ["AMD"]
        assert math.isnan(tr.open_unrealized)
        assert math.isnan(tr.flattery_r)

    def test_as_of_is_what_decides_whether_the_mark_is_current(self, book):
        """The same book, the same mark, two report dates — as_of is not dead."""
        self._amd(book)

        fresh = book.track_record({}, as_of=date(2026, 1, 2) +
                                  timedelta(days=MAX_MARK_AGE_DAYS))
        stale = book.track_record({}, as_of=date(2026, 1, 2) +
                                  timedelta(days=MAX_MARK_AGE_DAYS + 1))

        assert fresh.open_marked == 1
        assert fresh.open_unrealized == pytest.approx(1_800.0)
        assert stale.open_marked == 0
        assert stale.stale_marks == ["AMD"]

    def test_a_current_price_always_wins_over_the_mark(self, book):
        """The staleness bound must not reject a price the caller just fetched."""
        self._amd(book)
        tr = book.track_record({"AMD": 95.0}, as_of=date(2026, 8, 21))

        assert tr.open_marked == 1
        assert tr.stale_marks == []
        assert tr.open_unrealized == pytest.approx(-500.0)

    def test_a_mark_that_cannot_be_dated_is_stale(self, tmp_path):
        """A last_price with no last_reviewed has no age, and unknown means stale."""
        path = tmp_path / "recommendations.json"
        path.write_text(json.dumps({"recommendations": [{
            "id": "AMD-1", "issued_date": "2026-01-02", "symbol": "AMD",
            "action": "BUY", "shares": 100, "reference_price": 100.0,
            "stop_price": 90.0, "target_price": 120.0, "status": "open",
            "last_price": 118.0,
        }]}))
        tr = RecommendationBook(path).track_record({}, as_of=date(2026, 8, 21))

        assert tr.open_marked == 0
        assert tr.stale_marks == ["AMD"]

    def test_the_stale_mark_is_named_in_the_warnings(self, book):
        self._amd(book)
        tr = book.track_record({}, as_of=date(2026, 8, 21))

        assert any("AMD" in w and "days old" in w for w in tr.warnings)

    def test_an_idea_never_priced_at_all_is_unpriced_but_not_stale(self, book):
        book.add("NEW", BUY, 10, 100.0, 90.0, 120.0, rec_id="NEW-1")
        tr = book.track_record({})

        assert tr.unpriced == ["NEW"]
        assert tr.stale_marks == []

    def test_the_open_book_flags_a_stale_mark_rather_than_hiding_it(self, book):
        """The last price seen is worth printing; presenting it as today's is not."""
        self._amd(book)
        page = format_open_book(book.open_recommendations(), {}, date(2026, 8, 21))

        assert "118.00?" in page
        assert f"more than {MAX_MARK_AGE_DAYS} days old" in page

    def test_a_fresh_mark_is_not_flagged(self, book):
        self._amd(book)
        page = format_open_book(book.open_recommendations(), {}, date(2026, 1, 3))

        assert "118.00" in page
        assert "?" not in page


# ---------------------------------------------------------------------------
# defect 8: a close with no price
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestUnscoredClose:
    """Closing at a price that cannot be read ends an idea with no P&L.

    It counts in tr.closed and in nothing else, which is the same hole an
    expiry leaves — and expiries are warned about precisely because "expire the
    losers, close the winners" produces any record you like. This route is the
    default one for a losing position nobody can price.
    """

    def test_a_close_with_no_price_is_counted_and_warned(self, book):
        book.add("U", BUY, 10, 100.0, 90.0, 120.0, rec_id="U-1")
        book.close("U-1", float("nan"))
        tr = book.track_record({})

        assert (tr.closed, tr.unscored_closed) == (1, 1)
        assert any("could not be read" in w for w in tr.warnings)

    def test_the_time_stop_route_reaches_the_same_warning(self, book):
        """No manual step needed: an unpriced idea past its horizon takes it.

        Rule 4 emits SELL_ALL with a NaN price, and close_from takes it.
        """
        book.add("U", BUY, 10, 100.0, 90.0, 120.0,
                 issued_date=(date.today() - timedelta(days=60)).isoformat(),
                 rec_id="U-1")
        signals = book.review({})
        sell = [s for s in signals if s.closes_position]
        assert len(sell) == 1

        book.close_from(sell[0])
        tr = book.track_record({})
        assert tr.unscored_closed == 1
        assert any("could not be read" in w for w in tr.warnings)

    def test_the_page_prints_na_rather_than_a_nan_win_rate(self, book):
        """"Closed recommendations 1" above "win rate nan%" reads as a broken
        report rather than as the absence of a number."""
        book.add("U", BUY, 10, 100.0, 90.0, 120.0, rec_id="U-1")
        book.close("U-1", float("nan"))
        page = format_track_record(book.track_record({}))

        assert "nan" not in page
        assert "win rate" in page and "n/a" in page
        assert "unscored" in page

    def test_a_scored_close_is_not_counted_as_unscored(self, book):
        book.add("S", BUY, 10, 100.0, 90.0, 120.0, rec_id="S-1")
        book.close("S-1", 110.0)
        tr = book.track_record({})

        assert tr.unscored_closed == 0
        assert not any("could not be read" in w for w in tr.warnings)
        assert tr.total_pnl == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# the rest of the engine, which had no tests at all
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestExitRulesFire:
    def test_stop_hit_is_urgent_and_terminal(self, book):
        book.add("A", BUY, 10, 100.0, 90.0, 120.0, rec_id="A-1")
        signals = book.review({"A": 89.0})

        assert [(s.action, s.urgency, s.exit_reason) for s in signals] == [
            (SELL_ALL, 3, "stop")]

    def test_breakeven_stop_moves_at_one_r_and_only_once(self, book):
        """The rule that removes the outcome that ruins a record: up 2R, out at -1R."""
        rec = book.add("A", BUY, 10, 100.0, 90.0, 130.0, rec_id="A-1")
        first = book.review({"A": 110.0})

        assert [s.action for s in first] == ["RAISE_STOP"]
        assert rec.stop_price == 100.0
        assert rec.stop_raised is True
        assert book.review({"A": 111.0}) == []

    def test_realised_r_is_measured_against_the_stop_as_issued(self, book):
        """Measured against a raised stop, a breakeven exit is a division by zero
        and every winner's denominator shrinks."""
        rec = book.add("A", BUY, 10, 100.0, 90.0, 130.0, rec_id="A-1")
        book.review({"A": 110.0})
        book.close("A-1", 100.0)

        assert rec.initial_stop_price == 90.0
        assert rec.realized_r == pytest.approx(0.0)

    def test_time_stop_needs_the_horizon_and_a_thesis_that_never_worked(self, book):
        book.add("A", BUY, 10, 100.0, 90.0, 120.0, horizon_days=30,
                 issued_date=(date.today() - timedelta(days=31)).isoformat(),
                 rec_id="A-1")
        signals = book.review({"A": 101.0})

        assert [(s.action, s.exit_reason) for s in signals] == [(SELL_ALL, "time_stop")]

    def test_an_idea_that_reached_one_r_is_not_time_stopped(self, book):
        rec = book.add("A", BUY, 10, 100.0, 90.0, 130.0, horizon_days=30,
                       issued_date=(date.today() - timedelta(days=31)).isoformat(),
                       rec_id="A-1")
        rec.peak_r = 1.5
        rec.stop_raised = True

        assert book.review({"A": 101.0}) == []

    def test_an_unpriced_idea_is_reported_not_skipped(self, book):
        """Silence about an unpriceable name reads exactly like "nothing to do",
        and this is the case where a stop may already have been blown through."""
        book.add("A", BUY, 10, 100.0, 90.0, 120.0, rec_id="A-1")
        signals = book.review({"A": None})

        assert [(s.action, s.urgency) for s in signals] == [(UNAVAILABLE, 2)]

    def test_signals_come_back_most_urgent_first(self, book):
        book.add("A", BUY, 10, 100.0, 90.0, 120.0, rec_id="A-1")
        book.add("B", BUY, 10, 100.0, 90.0, 130.0, rec_id="B-1")
        signals = book.review({"A": 89.0, "B": 110.0})

        assert [s.urgency for s in signals] == [3, 1]


@pytest.mark.unit
class TestNothingEscapesTheLoop:
    """This loop is meant to run unattended for weeks; a raise stops every stop
    in the book from being watched."""

    @pytest.mark.parametrize("prices", [["not", "a", "mapping"], "abc", 3, object()])
    def test_a_junk_prices_argument_degrades_to_unavailable(self, book, prices):
        book.add("A", BUY, 10, 100.0, 90.0, 120.0, rec_id="A-1")
        signals = book.review(prices)

        assert [s.action for s in signals] == [UNAVAILABLE]

    @pytest.mark.parametrize("news", [{"A": "headline"}, {"A": 3}, {"A": None}])
    def test_junk_news_is_ignored_rather_than_raising(self, book, news):
        book.add("A", BUY, 10, 100.0, 90.0, 120.0, rec_id="A-1")
        assert book.review({"A": 101.0}, news) == []

    def test_an_unreadable_issue_date_does_not_stop_the_review(self, book):
        rec = book.add("A", BUY, 10, 100.0, 90.0, 120.0, rec_id="A-1")
        rec.issued_date = "not a date"

        assert rec.days_held() is None
        assert book.review({"A": 89.0})[0].action == SELL_ALL

    def test_a_corrupt_file_yields_an_empty_book(self, tmp_path):
        path = tmp_path / "recommendations.json"
        path.write_text("{not json")

        assert len(RecommendationBook(path)) == 0

    def test_saving_leaves_no_temporary_file_behind(self, tmp_path):
        """Atomic write: a half-written book is worse than no book, because the
        track record cannot be recomputed from anywhere else."""
        path = tmp_path / "sub" / "recommendations.json"
        book = RecommendationBook(path)
        book.add("A", BUY, 10, 100.0, 90.0, 120.0, rec_id="A-1")

        assert path.exists()
        assert list(path.parent.glob("*.tmp")) == []
        assert json.loads(path.read_text())["recommendations"][0]["id"] == "A-1"

    def test_a_non_finite_level_is_stored_as_null_not_as_a_nan_token(self, tmp_path):
        """A bare NaN token is readable by Python and by nothing else, and this
        file is meant to be hand-edited."""
        path = tmp_path / "recommendations.json"
        book = RecommendationBook(path)
        book.add("A", BUY, 10, float("nan"), float("inf"), 120.0, rec_id="A-1")

        assert "NaN" not in path.read_text()
        assert json.loads(path.read_text())["recommendations"][0]["stop_price"] is None


@pytest.mark.unit
class TestUnscoredEndsKeepBookedTrims:
    """An expiry scores nothing on the shares still held — but a trim that
    already happened is money that was actually made, and dropping it
    understates the record in the one module built to catch overstatement.
    """

    def _trimmed(self, tmp_path):
        book = RecommendationBook(path=tmp_path / "r.json")
        rec = book.add("NVDA", BUY, 100, 100.0, 90.0, 120.0,
                       issued_date="2026-08-01")
        rec.take_off(50, 120.0)          # books +$1,000 on half the position
        return book, rec

    def test_expiry_keeps_the_realised_trim(self, tmp_path):
        book, rec = self._trimmed(tmp_path)
        book.expire(rec.id, "horizon passed")
        assert rec.realized_pnl == pytest.approx(1_000.0)
        assert rec.realized_r == pytest.approx(1.0)

    def test_supersede_keeps_the_realised_trim(self, tmp_path):
        book, rec = self._trimmed(tmp_path)
        book.supersede(rec.id, by="a newer NVDA idea")
        assert rec.realized_pnl == pytest.approx(1_000.0)

    def test_an_untrimmed_expiry_stays_unscored(self, tmp_path):
        # Only a *realised* trim is carried; an idea that simply ran out of
        # time still has no P&L, and inventing one would be the flattery.
        book = RecommendationBook(path=tmp_path / "r.json")
        rec = book.add("AAA", BUY, 10, 100.0, 90.0, 120.0, issued_date="2026-08-01")
        book.expire(rec.id, "horizon passed")
        assert rec.realized_pnl is None and rec.realized_r is None

    def test_an_expiry_can_be_dated_by_the_session_not_the_wall_clock(self, tmp_path):
        # A book that dates closes by the data session and expiries by
        # date.today() puts two ends of the same ledger on different calendars.
        book = RecommendationBook(path=tmp_path / "r.json")
        rec = book.add("AAA", BUY, 10, 100.0, 90.0, 120.0, issued_date="2026-08-01")
        book.expire(rec.id, "horizon passed", exit_date=_date(2026, 8, 24))
        assert rec.exit_date == "2026-08-24"

    def test_the_exit_date_still_defaults_to_today(self, tmp_path):
        book = RecommendationBook(path=tmp_path / "r.json")
        rec = book.add("AAA", BUY, 10, 100.0, 90.0, 120.0, issued_date="2026-08-01")
        book.expire(rec.id, "horizon passed")
        assert rec.exit_date == _date.today().isoformat()
