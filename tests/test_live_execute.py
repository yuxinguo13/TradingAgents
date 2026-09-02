"""The bridge between the book and the account, and the two ways it can lie.

The gap this module closes is silent by construction: the advisor writes a full
book of decisions, the monitor with ``--no-llm`` never decides, and neither ever
errors — so the track record scores ideas as taken while the venue holds
something else entirely. Most of this file is about the two ways a bridge that
"just reads the book" makes that worse: it places six-day-old limits as though
they were this morning's, and it sells whatever the book does not recognise.
"""

from datetime import date
from types import SimpleNamespace

import math
import pytest

from tradingagents.live import execute
from tradingagents.live.broker import BUY, SELL, Account, Holding, OrderResult
from tradingagents.live.execute import Intent, plan, submit
from tradingagents.live.recommendations import CLOSED, OPEN
from tradingagents.live.secretary import Order, RiskLimits, Secretary, TradeLedger, Verdict


class Rec:
    """The subset of Recommendation the bridge reads."""

    def __init__(self, symbol, shares=100, entry=100.0, stop=95.0, target=115.0,
                 limit=100.3, issued="2026-09-02", rid=None, status=OPEN,
                 exit_date="", exit_reason=""):
        self.id = rid or f"{symbol}-{issued.replace('-', '')}"
        self.symbol = symbol
        self.shares = shares
        self.reference_price = entry
        self.initial_stop_price = stop
        self.stop_price = stop
        self.target_price = target
        self.limit_price = limit
        self.issued_date = issued
        self.status = status
        self.exit_date = exit_date
        self.exit_reason = exit_reason

    def planned_r(self):
        return (self.target_price - self.reference_price) / (
            self.reference_price - self.initial_stop_price)


def exited(symbol, on, reason="stop hit", **kw):
    """A recommendation the book has already closed."""
    return Rec(symbol, status=CLOSED, exit_date=on, exit_reason=reason, **kw)


class Book:
    """Open rows, plus the closed ones a real book keeps alongside them.

    ``recommendations`` is the full list, which is what tells a closed exit
    apart from a position this desk never chose.
    """

    def __init__(self, recs, closed=()):
        self._recs = list(recs)
        self.recommendations = list(recs) + list(closed)

    def open_recommendations(self):
        return list(self._recs)


class Signal:
    def __init__(self, symbol, shares=100, action="SELL", reason="stop hit", urgency=3):
        self.symbol, self.shares, self.action = symbol, shares, action
        self.reason, self.urgency = reason, urgency

    @property
    def closes_position(self):
        return self.action == "SELL"


def account(holdings=(), cash=100_000.0, equity=100_000.0):
    return Account(account_value=equity, cash=cash, buying_power=cash,
                   holdings=list(holdings))


TODAY = date(2026, 9, 2)


@pytest.mark.unit
class TestPlan:
    def test_an_idea_in_the_book_with_no_position_becomes_a_buy(self):
        got = plan(Book([Rec("AAA")]), account(), as_of=TODAY)
        assert [(i.action, i.symbol, i.shares) for i in got.to_open] == [(BUY, "AAA", 100)]
        assert got.to_open[0].limit == 100.3

    def test_a_matching_position_produces_no_order(self):
        got = plan(Book([Rec("AAA")]),
                   account([Holding(symbol="AAA", quantity=100.0)]), as_of=TODAY)
        assert got.to_open == [] and got.drift == []
        assert got.matched == [("AAA", 100)] and got.clean

    def test_a_share_count_that_disagrees_is_reported_not_corrected(self):
        """Part fill and a hand trade look identical here and need different fixes."""
        got = plan(Book([Rec("AAA", shares=100)]),
                   account([Holding(symbol="AAA", quantity=60.0)]), as_of=TODAY)
        assert got.drift == [("AAA", 100, 60)]
        assert got.to_open == [] and not got.clean

    def test_a_sub_share_difference_is_rounding_not_drift(self):
        got = plan(Book([Rec("AAA", shares=100)]),
                   account([Holding(symbol="AAA", quantity=100.4)]), as_of=TODAY)
        assert got.drift == [] and got.matched == [("AAA", 100)]

    def test_a_position_the_book_does_not_know_is_never_touched(self):
        """Selling it because one book omits it is the bridge claiming the account."""
        got = plan(Book([]), account([Holding(symbol="ZZZ", quantity=50.0)]),
                   as_of=TODAY)
        assert [h.symbol for h in got.unmanaged] == ["ZZZ"]
        assert got.to_close == [] and got.to_open == []

    def test_an_exit_signal_becomes_a_sell_of_what_is_actually_held(self):
        got = plan(Book([Rec("AAA", shares=100)]),
                   account([Holding(symbol="AAA", quantity=70.0)]),
                   exits=[Signal("AAA", shares=100)], as_of=TODAY)
        assert [(i.action, i.symbol, i.shares) for i in got.to_close] == [(SELL, "AAA", 70)]

    def test_an_exit_on_a_position_nobody_holds_is_a_note_not_an_order(self):
        got = plan(Book([Rec("AAA")]), account(), exits=[Signal("AAA")], as_of=TODAY)
        assert got.to_close == []
        assert any("本来就没有仓位" in n for n in got.notes)

    def test_without_exit_signals_the_plan_says_so(self):
        got = plan(Book([Rec("AAA")]), account(), as_of=TODAY)
        assert any("没有传入离场信号" in n for n in got.notes)

    def test_exits_are_ordered_before_entries_and_by_urgency(self):
        got = plan(Book([Rec("AAA"), Rec("BBB"), Rec("CCC")]),
                   account([Holding(symbol="AAA", quantity=100.0),
                            Holding(symbol="BBB", quantity=100.0)]),
                   exits=[Signal("AAA", urgency=1), Signal("BBB", urgency=3)],
                   as_of=TODAY)
        assert [i.symbol for i in got.intents] == ["BBB", "AAA", "CCC"]


@pytest.mark.unit
class TestExitsThatNeverReachedTheVenue:
    """The failure this module exists for, in the shape it actually took.

    The advisor writes its exits back to the book *before* the reconciliation
    runs, so by then today's sell is closed there and its symbol is gone from
    the open rows. Matched on open rows alone it comes back as a position the
    book does not recognise — the one bucket the bridge promises never to
    touch. The report says sell it, the section below says never touch it, the
    record books the loss as cut, and the account keeps riding it.
    """

    def test_todays_exit_is_a_sell_even_though_the_book_already_closed_it(self):
        got = plan(Book([], closed=[exited("AAA", "2026-09-02")]),
                   account([Holding(symbol="AAA", quantity=100.0)]),
                   exits=[Signal("AAA")], as_of=TODAY)
        assert [(i.symbol, i.shares) for i in got.to_close] == [("AAA", 100)]
        assert got.unmanaged == []

    def test_a_sell_is_of_what_is_held_not_what_the_book_remembers(self):
        got = plan(Book([], closed=[exited("AAA", "2026-09-02")]),
                   account([Holding(symbol="AAA", quantity=40.0)]),
                   exits=[Signal("AAA", shares=100)], as_of=TODAY)
        assert got.to_close[0].shares == 40

    def test_an_exit_from_a_previous_run_is_still_an_order(self):
        """No signal today: the rule fired days ago and nothing was placed."""
        got = plan(Book([], closed=[exited("AAA", "2026-08-31", "stop")]),
                   account([Holding(symbol="AAA", quantity=100.0)]),
                   exits=[], as_of=TODAY)
        assert [i.symbol for i in got.to_close] == ["AAA"]
        assert got.to_close[0].urgency == 3
        assert "2026-08-31" in got.to_close[0].reason

    def test_an_old_exit_is_named_but_not_ordered(self):
        """A name exited months ago and since bought back is not this bridge's
        to sell — but it is not allowed to go quiet either."""
        got = plan(Book([], closed=[exited("AAA", "2026-06-01")]),
                   account([Holding(symbol="AAA", quantity=100.0)]),
                   exits=[], as_of=TODAY)
        assert got.to_close == []
        assert [h.symbol for h in got.unmanaged] == ["AAA"]
        assert any("2026-06-01" in n for n in got.notes)

    def test_a_reentered_name_is_reconciled_against_its_open_row(self):
        """An old exit must not outvote the position the book currently carries."""
        got = plan(Book([Rec("AAA", shares=100)],
                        closed=[exited("AAA", "2026-08-31")]),
                   account([Holding(symbol="AAA", quantity=100.0)]),
                   exits=[], as_of=TODAY)
        assert got.to_close == [] and got.matched == [("AAA", 100)]

    def test_the_newest_exit_wins_when_a_name_was_traded_twice(self):
        got = plan(Book([], closed=[exited("AAA", "2026-06-01", "old"),
                                    exited("AAA", "2026-09-01", "recent")]),
                   account([Holding(symbol="AAA", quantity=10.0)]),
                   exits=[], as_of=TODAY)
        assert [i.reason.count("2026-09-01") for i in got.to_close] == [1]

    def test_an_exit_with_an_unreadable_date_orders_nothing(self):
        got = plan(Book([], closed=[exited("AAA", None)]),
                   account([Holding(symbol="AAA", quantity=10.0)]),
                   exits=[], as_of=TODAY)
        assert got.to_close == [] and [h.symbol for h in got.unmanaged] == ["AAA"]

    def test_a_signal_for_a_name_with_no_book_row_still_sells_what_is_held(self):
        got = plan(Book([]), account([Holding(symbol="AAA", quantity=10.0)]),
                   exits=[Signal("AAA")], as_of=TODAY)
        assert [(i.symbol, i.shares) for i in got.to_close] == [("AAA", 10)]

    def test_a_position_with_no_book_record_at_all_stays_untouched(self):
        got = plan(Book([]), account([Holding(symbol="ZZZ", quantity=5.0)]),
                   exits=[], as_of=TODAY)
        assert got.to_close == [] and [h.symbol for h in got.unmanaged] == ["ZZZ"]


@pytest.mark.unit
class TestTrims:
    """A trim is an order. Read as a share gap it is never placed.

    ``review()`` books the trim against the idea the moment it instructs it, so
    the book holds the post-trim size while the venue still holds the whole
    position. That difference is this morning's own instruction; filed as
    drift it reads as "someone traded this by hand" and waits for a human who
    was never told there was anything to do.
    """

    def test_a_trim_becomes_a_partial_sell(self):
        got = plan(Book([Rec("AAA", shares=50)]),
                   account([Holding(symbol="AAA", quantity=100.0)]),
                   exits=[Signal("AAA", shares=50, action="TRIM", urgency=2)],
                   as_of=TODAY)
        assert [(i.action, i.symbol, i.shares) for i in got.to_trim] == \
            [(SELL, "AAA", 50)]
        assert got.drift == []
        assert not got.clean

    def test_a_trim_is_submitted_with_the_other_sells(self):
        got = plan(Book([Rec("AAA", shares=50)]),
                   account([Holding(symbol="AAA", quantity=100.0)]),
                   exits=[Signal("AAA", shares=50, action="TRIM")], as_of=TODAY)
        assert [i.symbol for i in got.intents] == ["AAA"]

    def test_a_trim_with_no_size_never_becomes_a_full_sell(self):
        """The one way a take-some-profit turns into an exit nobody asked for.

        A close falls back to everything held — the instruction is "get out"
        and the number is only how the book remembers the size. A trim has no
        such fallback: a partial sell whose size did not survive is not a
        licence to sell the lot.
        """
        got = plan(Book([Rec("AAA", shares=100)]),
                   account([Holding(symbol="AAA", quantity=100.0)]),
                   exits=[Signal("AAA", shares=0, action="TRIM")], as_of=TODAY)
        assert got.to_trim == []
        assert any("不下单" in n for n in got.notes)

    def test_a_trim_that_cannot_be_sized_still_reports_the_gap(self):
        """Not being able to price it is no reason to also go quiet about it."""
        got = plan(Book([Rec("AAA", shares=50)]),
                   account([Holding(symbol="AAA", quantity=100.0)]),
                   exits=[Signal("AAA", shares=0, action="TRIM")], as_of=TODAY)
        assert got.drift == [("AAA", 50, 100)]

    def test_a_trim_of_a_position_nobody_holds_is_a_note(self):
        got = plan(Book([Rec("AAA", shares=50)]), account(),
                   exits=[Signal("AAA", shares=50, action="TRIM")], as_of=TODAY)
        assert got.to_trim == [] and got.to_open == []
        assert any("没有仓位" in n for n in got.notes)

    def test_a_raise_stop_is_not_an_order(self):
        got = plan(Book([Rec("AAA")]),
                   account([Holding(symbol="AAA", quantity=100.0)]),
                   exits=[Signal("AAA", shares=0, action="RAISE_STOP")], as_of=TODAY)
        assert got.intents == [] and got.clean


@pytest.mark.unit
class TestTheCoreBook:
    """The long-term book runs on a monthly clock this module has no part in.

    Calling its positions unrecognised every single day is how that section
    becomes wallpaper, and the one day it says something new goes by with it.
    """

    def test_a_core_holding_is_not_called_unrecognised(self):
        got = plan(Book([]), account([Holding(symbol="MSFT", quantity=30.0)]),
                   exits=[], as_of=TODAY, core=["MSFT"])
        assert [h.symbol for h in got.core_held] == ["MSFT"]
        assert got.unmanaged == [] and got.intents == []

    def test_a_core_name_the_swing_book_also_holds_is_reconciled_once(self):
        got = plan(Book([Rec("MSFT", shares=30)]),
                   account([Holding(symbol="MSFT", quantity=30.0)]),
                   exits=[], as_of=TODAY, core=["MSFT"])
        assert got.matched == [("MSFT", 30)] and got.core_held == []

    def test_a_core_name_the_swing_book_also_exited_is_still_a_sell(self):
        """Membership of the core list does not cancel an exit that was taken."""
        got = plan(Book([], closed=[exited("MSFT", "2026-09-02")]),
                   account([Holding(symbol="MSFT", quantity=30.0)]),
                   exits=[Signal("MSFT")], as_of=TODAY, core=["MSFT"])
        assert [i.symbol for i in got.to_close] == ["MSFT"]
        assert got.core_held == []


@pytest.mark.unit
class TestWhatTheVenueActuallyReturns:
    """Holdings are not a clean list, and every deviation here is silent."""

    def test_a_short_is_never_read_as_a_matching_long(self):
        """The worst reading available: the account is positioned the opposite
        way and the reconciliation reports agreement."""
        got = plan(Book([Rec("AAA", shares=100)]),
                   account([Holding(symbol="AAA", quantity=100.0, side="short")]),
                   exits=[], as_of=TODAY)
        assert got.matched == [] and got.drift == []
        assert [h.symbol for h, _ in got.conflicts] == ["AAA"]
        assert not got.clean

    def test_a_direction_conflict_is_never_an_order(self):
        """Buying closes the short, selling deepens it. Both are this module
        taking a position on a trade it has no record of."""
        got = plan(Book([Rec("AAA", shares=100)]),
                   account([Holding(symbol="AAA", quantity=100.0, side="short")]),
                   exits=[Signal("AAA")], as_of=TODAY)
        assert got.intents == []

    def test_a_short_the_book_never_mentioned_is_reported_untouched(self):
        got = plan(Book([]),
                   account([Holding(symbol="ZZZ", quantity=10.0, side="short")]),
                   exits=[], as_of=TODAY)
        assert [h.symbol for h in got.unmanaged] == ["ZZZ"] and got.intents == []

    def test_two_rows_for_one_symbol_are_summed_not_overwritten(self):
        """Keyed naively the later row wins and a 150-share position reads as
        50 — a drift the account does not have."""
        got = plan(Book([Rec("AAA", shares=150)]),
                   account([Holding(symbol="AAA", quantity=100.0),
                            Holding(symbol="AAA", quantity=50.0)]),
                   exits=[], as_of=TODAY)
        assert got.matched == [("AAA", 150)] and got.drift == []

    def test_a_row_with_no_shares_is_not_a_holding(self):
        got = plan(Book([], closed=[exited("AAA", "2026-09-01")]),
                   account([Holding(symbol="AAA", quantity=0.0)]),
                   exits=[], as_of=TODAY)
        assert got.to_close == [] and got.unmanaged == []

    def test_a_book_row_with_no_share_count_says_so(self):
        """Skipped in silence it reads as "nothing to do"; it is a
        recommendation whose sizing produced nothing."""
        got = plan(Book([Rec("AAA", shares=0)]), account(), exits=[], as_of=TODAY)
        assert got.to_open == []
        assert any("没有股数" in n for n in got.notes)


@pytest.mark.unit
class TestStaleEntries:
    """An unfilled idea is priced off one close and meant for the next open."""

    def test_an_entry_older_than_the_window_is_not_proposed(self):
        got = plan(Book([Rec("AAA", issued="2026-08-27")]), account(),
                   as_of=TODAY, quote=lambda s: 100.0)
        assert got.to_open == []
        assert [i.symbol for i, *_ in got.stale] == ["AAA"]
        assert not got.clean

    def test_a_stale_entry_is_never_submitted(self):
        got = plan(Book([Rec("AAA", issued="2026-08-27")]), account(),
                   as_of=TODAY, quote=lambda s: 100.0)
        assert got.intents == []

    def test_todays_entry_is_still_fresh(self):
        got = plan(Book([Rec("AAA", issued="2026-09-02")]), account(), as_of=TODAY)
        assert [i.symbol for i in got.to_open] == ["AAA"] and got.stale == []

    def test_an_entry_whose_own_open_has_gone_by_is_stale(self):
        """Yesterday's unfilled idea, planned for today.

        ``as_of`` is the session being planned, not the session the data came
        from — and those are always one apart. Aged against the data day, every
        entry was handed one extra session in which its limit, its stop and its
        R all pointed at a price that had already moved.
        """
        got = plan(Book([Rec("AAA", issued="2026-09-01")]), account(),
                   as_of=TODAY, quote=lambda s: 100.0)
        assert got.to_open == []
        assert [(i.symbol, age) for i, age, *_ in got.stale] == [("AAA", 1)]

    def test_r_is_recomputed_against_the_stop_as_issued(self):
        """A later entry with an unchanged stop buys less upside for the same risk."""
        rec = Rec("AAA", entry=100.0, stop=95.0, target=115.0, issued="2026-08-27")
        got = plan(Book([rec]), account(), as_of=TODAY, quote=lambda s: 105.0)
        _, _, r_now, px, _ = got.stale[0]
        assert px == 105.0
        assert r_now == pytest.approx((115 - 105) / (105 - 95))   # 1.00, was 3.00
        assert r_now < rec.planned_r()

    def test_a_price_that_fell_toward_the_stop_reads_as_a_trap_not_a_bargain(self):
        """R rises as a name falls to its stop — what shrank is the room, not the risk."""
        rec = Rec("AAA", entry=100.0, stop=95.0, target=115.0, issued="2026-08-27")
        got = plan(Book([rec]), account(), as_of=TODAY, quote=lambda s: 95.8)
        _, _, r_now, _, to_stop = got.stale[0]
        assert r_now > 20                      # flattering
        assert to_stop == pytest.approx(0.0084, abs=1e-3)
        assert "止损贴脸" in execute._stale_verdict(r_now, to_stop)

    def test_a_price_already_through_the_stop_is_told_to_be_withdrawn(self):
        """Through the stop is not "close to" it. The level was named as the
        price the idea is wrong at, and it has been passed."""
        rec = Rec("AAA", entry=100.0, stop=95.0, target=115.0, issued="2026-08-27")
        got = plan(Book([rec]), account(), as_of=TODAY, quote=lambda s: 90.0)
        _, _, r_now, _, to_stop = got.stale[0]
        assert math.isnan(r_now) and to_stop < 0
        assert "作废" in execute._stale_verdict(r_now, to_stop)

    def test_a_stale_entry_below_the_minimum_is_told_to_be_withdrawn(self):
        assert "应作废" in execute._stale_verdict(1.1, 0.08)

    def test_an_unpriceable_stale_entry_says_so(self):
        got = plan(Book([Rec("AAA", issued="2026-08-27")]), account(), as_of=TODAY)
        _, _, r_now, px, _ = got.stale[0]
        assert math.isnan(r_now) and math.isnan(px)
        assert execute._stale_verdict(r_now, float("nan")) == "定不了价"

    def test_an_unreadable_issue_date_is_treated_as_fresh_not_dropped(self):
        rec = Rec("AAA")
        rec.issued_date = "not a date"
        got = plan(Book([rec]), account(), as_of=TODAY)
        assert [i.symbol for i in got.to_open] == ["AAA"]


class StubBroker:
    def __init__(self, price=100.0, ok=True, acct=None):
        self.price, self.ok, self.placed = price, ok, []
        self._acct = acct

    def account(self):
        if self._acct is None:
            raise RuntimeError("this venue cannot be re-read")
        return self._acct

    def quote(self, symbol):
        return self.price

    def place_order(self, symbol, action, quantity, order_type="Market",
                    limit_price=None, dry_run=False):
        self.placed.append((symbol, action, quantity, order_type, limit_price))
        return OrderResult(ok=self.ok, symbol=symbol, action=action,
                           quantity=quantity, message="filled" if self.ok else "rejected")


@pytest.mark.unit
class TestSubmit:
    def _sec(self, tmp_path):
        return Secretary(limits=RiskLimits(), ledger=TradeLedger(tmp_path / "ledger.json"))

    def test_every_order_goes_through_the_risk_gate(self, tmp_path, monkeypatch):
        """A second path to the venue would be a second set of risk limits."""
        seen = []
        sec = self._sec(tmp_path)
        real = sec.check

        def spy(order, acct, price, market_open=True):
            seen.append(order.symbol)
            return real(order, acct, price, market_open=market_open)

        monkeypatch.setattr(sec, "check", spy)
        rec = plan(Book([Rec("AAA", shares=10, entry=100.0)]), account(), as_of=TODAY)
        submit(rec, StubBroker(), sec, account(), market_open=True)
        assert seen == ["AAA"]

    def test_an_order_the_gate_refuses_is_not_placed(self, tmp_path, monkeypatch):
        sec = self._sec(tmp_path)
        monkeypatch.setattr(sec, "check",
                            lambda *a, **k: Verdict(ok=False, reason="kill switch"))
        b = StubBroker()
        rec = plan(Book([Rec("AAA", shares=10)]), account(), as_of=TODAY)
        out = submit(rec, b, sec, account(), market_open=True)
        assert b.placed == []
        assert out[0][1].reason == "kill switch" and out[0][2] is None

    def test_a_resized_order_is_placed_at_the_size_the_gate_allowed(self, tmp_path,
                                                                    monkeypatch):
        sec = self._sec(tmp_path)
        monkeypatch.setattr(sec, "check", lambda o, *a, **k: Verdict(
            ok=True, order=Order(symbol=o.symbol, action=o.action, quantity=3)))
        b = StubBroker()
        rec = plan(Book([Rec("AAA", shares=10)]), account(), as_of=TODAY)
        submit(rec, b, sec, account(), market_open=True)
        assert b.placed[0][2] == 3

    def test_a_broker_that_raises_costs_that_order_not_the_sweep(self, tmp_path):
        class Boom(StubBroker):
            def place_order(self, *a, **k):
                raise RuntimeError("socket closed")

        sec = self._sec(tmp_path)
        rec = plan(Book([Rec("AAA", shares=10), Rec("BBB", shares=10)]),
                   account(), as_of=TODAY)
        out = submit(rec, Boom(), sec, account(), market_open=True)
        assert len(out) == 2 and all(r is None for _, _, r in out)

    def test_a_venue_that_answers_with_nothing_is_reported(self, tmp_path):
        class Silent(StubBroker):
            def place_order(self, *a, **k):
                return None
        rec = plan(Book([Rec("AAA", shares=10, limit=None)]), account(), as_of=TODAY)
        sec = Secretary(RiskLimits(), TradeLedger(tmp_path / "l.json"))
        out = submit(rec, Silent(), sec, account(), market_open=True)
        assert "没能提交" in execute.format_results(out)

    def test_a_quote_that_raises_still_reaches_the_gate(self, tmp_path):
        class NoQuote(StubBroker):
            def quote(self, symbol):
                raise RuntimeError("no data")

        sec = self._sec(tmp_path)
        rec = plan(Book([Rec("AAA", shares=10)]), account(), as_of=TODAY)
        out = submit(rec, NoQuote(), sec, account(), market_open=True)
        assert len(out) == 1                       # priced at 0, the gate decides

    def test_fills_are_written_to_the_ledger(self, tmp_path):
        sec = self._sec(tmp_path)
        rec = plan(Book([Rec("AAA", shares=10)]), account(), as_of=TODAY)
        submit(rec, StubBroker(), sec, account(), market_open=True)
        assert sec.ledger.trades_today() >= 1


@pytest.mark.unit
class TestTheSessionGate:
    """``require_market_open`` is a limit, and this bridge switched it off.

    The guard asked ``clock`` for an ``is_market_open`` that has never existed;
    ``hasattr`` said no every time and the fallback handed the gate
    ``market_open=True``. Every test that exercised ``submit`` passed the flag
    explicitly, so the only path production takes was the one path untested.
    """

    def test_nothing_is_placed_while_the_market_is_shut(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path))
        monkeypatch.setattr(execute.clock, "market_state",
                            lambda *a, **k: SimpleNamespace(is_tradeable=False))
        rec = plan(Book([Rec("AAA", shares=10, limit=None)]), account(), as_of=TODAY)
        b = StubBroker()
        out = submit(rec, b, Secretary(RiskLimits(), TradeLedger(tmp_path / "l.json")),
                     account())
        assert b.placed == []
        assert [v.reason for _, v, _ in out] == ["market is closed"]

    def test_an_open_session_lets_the_same_order_through(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path))
        monkeypatch.setattr(execute.clock, "market_state",
                            lambda *a, **k: SimpleNamespace(is_tradeable=True))
        rec = plan(Book([Rec("AAA", shares=10, limit=None)]), account(), as_of=TODAY)
        b = StubBroker()
        submit(rec, b, Secretary(RiskLimits(), TradeLedger(tmp_path / "l.json")),
               account())
        assert [o[0] for o in b.placed] == ["AAA"]

    def test_a_clock_that_cannot_be_read_counts_as_shut(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("no tz database")
        monkeypatch.setattr(execute.clock, "market_state", boom)
        assert execute.market_is_open(log=lambda m: None) is False


@pytest.mark.unit
class TestTheAccountBetweenOrders:
    """One snapshot must not vet a whole basket.

    Each buy was measured against the same untouched cash and the same
    untouched gross exposure, so a gate that reads as strict approves more than
    the account can pay for. It only bites where this module works — placing a
    whole book at once.
    """

    def limits(self):
        return RiskLimits(max_position_weight=1.0, max_new_position_weight=1.0,
                          max_gross_exposure=1.0, max_order_value_pct=1.0,
                          symbol_cooldown_minutes=0, min_price=1.0)

    def basket(self):
        return plan(Book([Rec("AAA", shares=100, limit=None),
                          Rec("BBB", shares=100, limit=None)]),
                    account(cash=12_000.0), as_of=TODAY)

    def test_the_second_buy_is_cut_to_the_cash_the_first_one_left(self, tmp_path,
                                                                  monkeypatch):
        monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path))
        b = StubBroker()                       # account() raises: local fallback
        out = submit(self.basket(), b,
                     Secretary(self.limits(), TradeLedger(tmp_path / "l.json")),
                     account(cash=12_000.0), market_open=True)
        placed = {o[0]: o[2] for o in b.placed}
        assert placed["AAA"] == 100            # $10,000 of $12,000
        assert placed["BBB"] == 20             # what $2,000 buys, not another 100
        assert sum(q * 100.0 for q in placed.values()) <= 12_000.0
        assert "被改小" in execute.format_results(out)

    def test_the_venue_is_asked_before_the_estimate(self, tmp_path, monkeypatch):
        """The venue's own books are the truth when they can be reached."""
        monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path))
        b = StubBroker(acct=account(cash=300.0, equity=100_000.0))
        submit(self.basket(), b,
               Secretary(self.limits(), TradeLedger(tmp_path / "l.json")),
               account(cash=12_000.0), market_open=True)
        assert {o[0]: o[2] for o in b.placed}["BBB"] == 3      # $300 of re-read cash

    def test_a_buy_is_charged_to_cash_exactly_once(self):
        after = execute._applied(account(cash=10_000.0, equity=10_000.0),
                                 Order("AAA", BUY, 10), 100.0)
        assert (after.cash, after.buying_power) == (9_000.0, 9_000.0)

    def test_a_venue_reporting_no_buying_power_is_still_charged_once(self):
        """``power or cash`` is the Secretary's own reading of a zero — taken
        after cash was debited it charges the same fill twice."""
        acct = account(cash=10_000.0, equity=10_000.0)
        acct.buying_power = 0.0
        after = execute._applied(acct, Order("AAA", BUY, 10), 100.0)
        assert after.buying_power == 9_000.0

    def test_a_nonsense_account_read_does_not_replace_a_good_one(self):
        b = StubBroker(acct=account(cash=0.0, equity=0.0))
        out = execute._after(b, account(cash=12_000.0, equity=100_000.0),
                             Order("AAA", BUY, 100), 100.0, log=lambda m: None)
        assert out.account_value > 0 and out.cash == 2_000.0

    def test_a_venue_that_has_not_registered_the_fill_cannot_widen_the_room(self):
        """The case this guard exists for: the venue answers with room that is
        already committed, one order too late to matter."""
        stale = account(cash=12_000.0)
        b = StubBroker(acct=stale)                 # answers 12,000 forever
        rec = plan(Book([Rec("AAA", shares=100, limit=None),
                         Rec("BBB", shares=100, limit=None)]),
                   account(cash=12_000.0), as_of=TODAY)
        submit(rec, b, self.secretary_for(), account(cash=12_000.0),
               market_open=True, log=lambda m: None)
        spent = sum(q * 100.0 for _, _, q, _, _ in b.placed)
        assert spent <= 12_000.0, "a lagging account read let it overbuy"

    def secretary_for(self):
        import tempfile
        from pathlib import Path
        return Secretary(
            RiskLimits(max_position_weight=1.0, max_new_position_weight=1.0,
                       max_gross_exposure=1.0, max_order_value_pct=1.0,
                       symbol_cooldown_minutes=0, min_price=1.0),
            TradeLedger(Path(tempfile.mkdtemp()) / "l.json"))

    def test_a_sale_frees_shares_but_not_cash_before_it_settles(self):
        acct = account([Holding(symbol="AAA", quantity=100.0, market_value=10_000.0)],
                       cash=500.0)
        after = execute._applied(acct, Order("AAA", SELL, 100), 100.0)
        assert after.holdings == [] and after.cash == 500.0


@pytest.mark.unit
class TestTheLedger:
    def test_a_partial_fill_is_booked_at_what_filled(self, tmp_path, monkeypatch):
        """The daily budgets are spent from this row. Booked at the size that
        was asked for, 30 of 100 filled charges the day for all 100."""
        monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path))
        class Partial(StubBroker):
            def place_order(self, symbol, action, quantity, **k):
                self.placed.append((symbol, action, quantity, "Market", None))
                r = OrderResult(ok=True, symbol=symbol, action=action,
                                quantity=quantity, message="partially filled")
                r.filled_quantity = quantity * 0.3
                return r
        sec = Secretary(RiskLimits(), TradeLedger(tmp_path / "l.json"))
        rec = plan(Book([Rec("AAA", shares=100, limit=None)]), account(), as_of=TODAY)
        b = Partial()
        submit(rec, b, sec, account(), market_open=True)
        sent = b.placed[0][2]                      # after the gate resized it
        assert [e["quantity"] for e in sec.ledger.entries] == [int(sent * 0.3)]
        assert sec.ledger.entries[0]["notional"] == int(sent * 0.3) * 100.0

    def test_an_order_accepted_without_a_fill_keeps_its_full_size(self):
        """Live at the venue: the exposure is committed whether or not it has
        printed."""
        booked = execute._as_filled(
            Order("AAA", BUY, 100),
            OrderResult(ok=True, symbol="AAA", action=BUY, quantity=100,
                        status="new"))
        assert booked.quantity == 100

    def test_a_fill_is_written_once(self, tmp_path, monkeypatch):
        """``record()`` persists on its own; the extra save wrote it all twice."""
        monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path))
        saves = []
        real = TradeLedger.save
        monkeypatch.setattr(TradeLedger, "save",
                            lambda self: (saves.append(1), real(self))[1])
        rec = plan(Book([Rec("AAA", shares=10, limit=None)]), account(), as_of=TODAY)
        submit(rec, StubBroker(),
               Secretary(RiskLimits(), TradeLedger(tmp_path / "l.json")),
               account(), market_open=True)
        assert len(saves) == 1


@pytest.mark.unit
class TestRendering:
    def test_the_plan_names_what_it_will_not_touch(self):
        got = plan(Book([]), account([Holding(symbol="ZZZ", quantity=5.0,
                                              avg_cost=10.0, market_value=60.0)]),
                   as_of=TODAY)
        text = execute.format_plan(got, account(), "paper")
        assert "账本不认识" in text and "整个账户都归它管" in text

    def test_a_clean_account_says_there_is_nothing_to_do(self):
        got = plan(Book([Rec("AAA")]),
                   account([Holding(symbol="AAA", quantity=100.0)]), as_of=TODAY)
        assert "没有要下的单" in execute.format_plan(got, account(), "paper")

    def test_a_trim_is_not_printed_as_a_share_gap(self):
        got = plan(Book([Rec("AAA", shares=50)]),
                   account([Holding(symbol="AAA", quantity=100.0)]),
                   exits=[Signal("AAA", shares=50, action="TRIM")], as_of=TODAY)
        text = execute.format_plan(got, account(), "paper")
        assert "减仓 (1)" in text and "股数对不上 (" not in text

    def test_a_direction_conflict_is_printed_with_its_reason(self):
        got = plan(Book([Rec("AAA", shares=100)]),
                   account([Holding(symbol="AAA", quantity=100.0, side="short")]),
                   exits=[], as_of=TODAY)
        text = execute.format_plan(got, account(), "paper")
        assert "方向相反" in text and "不下任何单" in text

    def test_the_core_book_is_named_as_itself(self):
        got = plan(Book([]), account([Holding(symbol="MSFT", quantity=30.0)]),
                   exits=[], as_of=TODAY, core=["MSFT"])
        text = execute.format_plan(got, account(), "paper")
        assert "核心长仓" in text and "账本不认识" not in text

    def test_the_stale_section_explains_why_it_is_not_ordering(self):
        got = plan(Book([Rec("AAA", issued="2026-08-27")]), account(),
                   as_of=TODAY, quote=lambda s: 100.0)
        text = execute.format_plan(got, account(), "paper")
        assert "过期未成交" in text and "这些不会下单" in text


@pytest.mark.unit
class TestTheWholeLoop:
    """book → venue → book → venue, on the real classes.

    Everything above stubs the pieces around the bridge. This does not: a real
    :class:`RecommendationBook`, a real :class:`LocalPaperBroker`, a real
    :class:`Secretary`, and only the quotes injected. The defect worth a test
    at this level is the one that needs three steps to appear — the book exits
    a name on day two and the sell only goes missing on day three, when the
    row is no longer open and nothing is passing signals in.
    """

    D1, D2, D3 = date(2026, 9, 2), date(2026, 9, 3), date(2026, 9, 4)

    def broker(self, tmp_path, prices):
        from tradingagents.live.paper import LocalPaperBroker
        b = LocalPaperBroker(starting_cash=100_000.0, path=tmp_path / "pf.json")
        b._quotes = dict(prices)
        return b

    def book(self, tmp_path):
        from tradingagents.live.recommendations import RecommendationBook
        return RecommendationBook(tmp_path / "recs.json")

    def secretary(self, tmp_path):
        return Secretary(
            RiskLimits(max_position_weight=0.30, max_new_position_weight=0.30,
                       max_order_value_pct=0.30, symbol_cooldown_minutes=0,
                       min_price=1.0),
            TradeLedger(tmp_path / "ledger.json"))

    def test_the_desk_converges_over_four_sessions(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path))
        book = self.book(tmp_path)
        book.add("AAA", "Buy", 100, 100.0, 95.0, 115.0, issued_date=self.D1)
        book.add("BBB", "Buy", 200, 50.0, 47.0, 60.0, issued_date=self.D1)

        b = self.broker(tmp_path, {"AAA": 100.0, "BBB": 50.0})
        rc = plan(book, b.account(), exits=[], as_of=self.D1, quote=b.quote)
        assert {i.symbol for i in rc.to_open} == {"AAA", "BBB"}
        out = submit(rc, b, self.secretary(tmp_path), b.account(), market_open=True)
        assert all(getattr(r, "ok", False) for _, _, r in out)

        rc = plan(book, b.account(), exits=[], as_of=self.D2, quote=b.quote)
        assert rc.clean and not rc.unmanaged
        assert sorted(rc.matched) == [("AAA", 100), ("BBB", 200)]

        # AAA goes through its stop and the advisor writes that back.
        b2 = self.broker(tmp_path, {"AAA": 92.0, "BBB": 51.0})
        signals = book.review({"AAA": 92.0, "BBB": 51.0}, {}, as_of=self.D2)
        closing = [s for s in signals if s.closes_position]
        assert [s.symbol for s in closing] == ["AAA"]
        for s in closing:
            book.close(s.rec_id, s.price, s.exit_reason, exit_date=self.D2)
        assert [r.symbol for r in book.open_recommendations()] == ["BBB"]

        # A plain run the next session, nothing passing signals in. The sell
        # has to survive the round trip through the book on its own.
        rc = plan(book, b2.account(), exits=[], as_of=self.D3, quote=b2.quote)
        assert [i.symbol for i in rc.to_close] == ["AAA"] and rc.unmanaged == []
        out = submit(rc, b2, self.secretary(tmp_path), b2.account(), market_open=True)
        assert all(getattr(r, "ok", False) for _, _, r in out)

        b3 = self.broker(tmp_path, {"BBB": 51.0})
        rc = plan(book, b3.account(), exits=[], as_of=self.D3, quote=b3.quote)
        assert rc.clean and rc.unmanaged == [] and rc.matched == [("BBB", 200)]

    def test_a_target_hit_trims_the_venue_to_the_size_the_book_kept(self, tmp_path,
                                                                    monkeypatch):
        monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path))
        book = self.book(tmp_path)
        book.add("AAA", "Buy", 100, 100.0, 95.0, 115.0, issued_date=self.D1)

        b = self.broker(tmp_path, {"AAA": 100.0})
        rc = plan(book, b.account(), exits=[], as_of=self.D1, quote=b.quote)
        submit(rc, b, self.secretary(tmp_path), b.account(), market_open=True)
        assert b.account().position("AAA").quantity == 100

        b2 = self.broker(tmp_path, {"AAA": 118.0})
        signals = book.review({"AAA": 118.0}, {}, as_of=self.D2)
        rc = plan(book, b2.account(), exits=signals, as_of=self.D2, quote=b2.quote)
        assert [(i.symbol, i.shares) for i in rc.to_trim] == [("AAA", 50)]
        assert rc.drift == []

        submit(rc, b2, self.secretary(tmp_path), b2.account(), market_open=True)
        assert b2.account().position("AAA").quantity == 50
        assert book.open_recommendations()[0].shares == 50
