"""The bridge between the book and the account, and the two ways it can lie.

The gap this module closes is silent by construction: the advisor writes a full
book of decisions, the monitor with ``--no-llm`` never decides, and neither ever
errors — so the track record scores ideas as taken while the venue holds
something else entirely. Most of this file is about the two ways a bridge that
"just reads the book" makes that worse: it places six-day-old limits as though
they were this morning's, and it sells whatever the book does not recognise.
"""

from datetime import date

import math
import pytest

from tradingagents.live import execute
from tradingagents.live.broker import BUY, SELL, Account, Holding, OrderResult
from tradingagents.live.execute import Intent, plan, submit
from tradingagents.live.secretary import Order, RiskLimits, Secretary, TradeLedger, Verdict


class Rec:
    """The subset of Recommendation the bridge reads."""

    def __init__(self, symbol, shares=100, entry=100.0, stop=95.0, target=115.0,
                 limit=100.3, issued="2026-09-02", rid=None):
        self.id = rid or f"{symbol}-{issued.replace('-', '')}"
        self.symbol = symbol
        self.shares = shares
        self.reference_price = entry
        self.initial_stop_price = stop
        self.stop_price = stop
        self.target_price = target
        self.limit_price = limit
        self.issued_date = issued

    def planned_r(self):
        return (self.target_price - self.reference_price) / (
            self.reference_price - self.initial_stop_price)


class Book:
    def __init__(self, recs):
        self._recs = list(recs)

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
    def __init__(self, price=100.0, ok=True):
        self.price, self.ok, self.placed = price, ok, []

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

    def test_the_stale_section_explains_why_it_is_not_ordering(self):
        got = plan(Book([Rec("AAA", issued="2026-08-27")]), account(),
                   as_of=TODAY, quote=lambda s: 100.0)
        text = execute.format_plan(got, account(), "paper")
        assert "过期未成交" in text and "这些不会下单" in text
