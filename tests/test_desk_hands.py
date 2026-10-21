"""The hands and the gate: what Claude may ask the venue for, and what it gets.

Claude gives levels and reasons; ``orders.py`` gives share counts and refusals.
Everything here checks that division — the formula, every hard limit in
MANUAL.md §3, the bracket at the venue, the one cancel that exists, and the
decision log — against a fake venue. No network.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from tests.test_desk import (  # noqa: F401  (fixtures and the bars generator)
    DATA_DAY,
    MONDAY_OPEN,
    SHAPES,
    StubBook,
    frame_for,
    home,
    mkt,
    news,
    screen_of,
    shape,
)
from tradingagents.desk import orders, pack
from tradingagents.desk.orders import DeskBook, Executor, Gate, Position, size
from tradingagents.live.broker import Account, Holding, OrderResult


class FakeVenue:
    """Alpaca as the executor sees it: brackets, OCOs, replace, close."""

    def __init__(self, account, is_open=True, equity=None, last_equity=None):
        self._account = account
        self.is_open = is_open
        self.equity = equity if equity is not None else account.account_value
        self.last_equity = last_equity if last_equity is not None else account.account_value
        self.placed = []          # (kind, symbol, qty, levels)
        self.cancelled = []
        self.resting = {}         # symbol → stop price
        self.fills = []
        self.n = 0

    def account(self):
        return self._account

    def quote(self, symbol):
        return frame_for(symbol, DATA_DAY.isoformat())["Close"].iloc[-1] if symbol.upper() in SHAPES else 0.0

    def market_open(self):
        return self.is_open

    def equity_today(self):
        return self.equity, self.last_equity

    def _id(self):
        self.n += 1
        return f"o{self.n}"

    def place_bracket(self, symbol, qty, limit, stop, target, dry_run=False):
        self.placed.append(("bracket", symbol, qty, (limit, stop, target)))
        if dry_run:
            return OrderResult(ok=True, symbol=symbol, action="Buy", quantity=qty, message="DRY")
        self.resting[symbol] = stop
        return OrderResult(ok=True, symbol=symbol, action="Buy", quantity=qty, message="accepted",
                           broker_order_id=self._id(), artifact=json.dumps({"stop": "s1", "limit": "t1"}))

    def protect(self, symbol, qty, stop, target, dry_run=False):
        self.placed.append(("oco", symbol, qty, (stop, target)))
        if not dry_run:
            self.resting[symbol] = stop
        return OrderResult(ok=True, symbol=symbol, action="Sell", quantity=qty, message="accepted",
                           broker_order_id=self._id())

    def stop_leg(self, symbol):
        return {"id": "s-" + symbol, "symbol": symbol, "side": "sell", "type": "stop"} \
            if symbol in self.resting else None

    def open_orders(self):
        return [{"id": "s-" + s, "symbol": s, "side": "sell", "type": "stop", "qty": 0} for s in self.resting]

    def raise_stop(self, symbol, new_stop):
        cur = self.resting.get(symbol)
        if cur is None:
            return False, "no stop"
        if new_stop <= cur:
            return False, "not above"
        self.resting[symbol] = new_stop
        return True, f"{cur} → {new_stop}"

    def cancel(self, order_id):
        self.cancelled.append(order_id)
        self.resting.pop(order_id.replace("s-", ""), None)
        return True

    def close_now(self, symbol):
        self.placed.append(("close", symbol, 0, ()))
        self.resting.pop(symbol, None)
        return OrderResult(ok=True, symbol=symbol, action="Sell", quantity=0, message="filled")

    def place_order(self, symbol, action, quantity, order_type="Market", limit_price=None, dry_run=False):
        self.placed.append(("market", symbol, quantity, ()))
        return OrderResult(ok=True, symbol=symbol, action=action, quantity=quantity, message="filled")

    def fills_since(self, since):
        return list(self.fills)


def acct(cash=100_000.0, holdings=(), equity=None):
    hv = sum(h.market_value for h in holdings)
    return Account(account_value=equity or cash + hv, cash=cash, buying_power=cash, holdings=list(holdings))


def held(sym, qty, cost, last):
    return Holding(symbol=sym, quantity=qty, avg_cost=cost, last=last, market_value=qty * last,
                   unrealized=qty * (last - cost))


def executor(venue, home, **kw):
    return Executor(broker=venue, market=mkt(task="trade"), book=DeskBook(home / "desk" / "trade" / "book.json"),
                    now=MONDAY_OPEN, log_path=home / "desk" / "trade" / "decisions.jsonl", **kw)


def buy(sym, entry, stop, target, **kw):
    return {"action": "buy", "symbol": sym, "entry": entry, "stop": stop, "target": target,
            "thesis": "t", "invalidation": "i", "principles": ["一"], **kw}


# ---------------------------------------------------------------------------
# the formula
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSize:
    def test_the_manuals_formula(self):
        v = size(equity=100_000, entry=100.0, stop=97.0, cash=100_000)
        assert v.ok and v.shares == 100                # 1% = $1,000 ÷ $3 → 333, capped by 10% = $10,000 ÷ 100
        assert v.risk == 300.0 and v.notional == 10_000.0

    def test_cash_binds_before_the_cap(self):
        v = size(equity=100_000, entry=100.0, stop=97.0, cash=2_500)
        assert v.shares == 25

    def test_an_existing_position_counts_against_the_cap(self):
        v = size(equity=100_000, entry=100.0, stop=97.0, cash=100_000, held_value=9_500)
        assert v.shares == 5

    def test_a_stop_above_the_entry_is_not_a_size(self):
        assert not size(100_000, 100.0, 101.0, 100_000).ok


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------

def gate(home, **kw):
    defaults = dict(equity=100_000, cash=100_000, last_equity=100_000, holdings={},
                    book=DeskBook(home / "desk" / "trade" / "book.json"), today=date(2026, 8, 24))
    defaults.update(kw)
    return Gate(**defaults)


@pytest.mark.unit
class TestGate:
    def test_a_clean_buy_passes_with_the_formulas_shares(self, home):
        v = gate(home).buy("AAA", 100.0, 97.0, 107.0, price=100.5, atr_pct=0.02, sector="Technology",
                           earnings_days=30)
        assert v.ok and v.shares == 100

    @pytest.mark.parametrize("kw, text", [
        (dict(stop=99.0), "ATR"),                                  # 1% stop under a 2% ATR
        (dict(stop=85.0), "10%"),                                  # stop too far
        (dict(target=104.0), "R 1.33"),                            # under 2R
        (dict(entry=104.0, price=100.0), "from the last price"),   # chasing
        (dict(earnings_days=1), "earnings"),
        (dict(entry=4.0, stop=3.8, target=5.0, price=4.0), "under $5"),
        (dict(stop=101.0), "bracket"),
    ])
    def test_each_hard_limit_refuses(self, home, kw, text):
        args = dict(entry=100.0, stop=97.0, target=107.0, price=100.5, atr_pct=0.02,
                    sector="Technology", earnings_days=30)
        args.update(kw)
        v = gate(home).buy("AAA", args["entry"], args["stop"], args["target"], price=args["price"],
                           atr_pct=args["atr_pct"], sector=args["sector"], earnings_days=args["earnings_days"])
        assert not v.ok and text in v.reason

    def test_no_headline_buys_a_name_under_its_200_day_line(self, home):
        v = gate(home).buy("AAA", 100.0, 97.0, 107.0, price=100.5, atr_pct=0.02, sector="",
                           earnings_days=30, sma200=110.0)
        assert not v.ok and "200-day" in v.reason
        v = gate(home).buy("AAA", 100.0, 97.0, 107.0, price=100.5, atr_pct=0.02, sector="",
                           earnings_days=30, sma200=90.0)
        assert v.ok

    def test_the_executor_refuses_a_broken_name_even_with_a_thesis(self, home):
        shape("DOWN", 100.0, drift=-0.30)
        v = FakeVenue(acct())
        px = v.quote("DOWN")
        out = executor(v, home).run({"orders": [buy("DOWN", round(px, 2), round(px * 0.97, 2), round(px * 1.10, 2),
                                                     thesis="huge news")]})
        assert not out[0].ok and "200-day" in out[0].reason and v.placed == []

    def test_the_daily_new_risk_budget(self, home):
        g = gate(home, risk_committed_today=3_800)
        v = g.buy("AAA", 100.0, 97.0, 107.0, price=100.0, atr_pct=0.02, sector="", earnings_days=30)
        assert not v.ok and "4%" in v.reason

    def test_the_drawdown_halt(self, home):
        g = gate(home, equity=96_500, last_equity=100_000)
        v = g.buy("AAA", 100.0, 97.0, 107.0, price=100.0, atr_pct=0.02, sector="", earnings_days=30)
        assert not v.ok and "down 3.5%" in v.reason

    def test_position_count_sector_count_and_reentry(self, home):
        book = DeskBook(home / "desk" / "trade" / "book.json")
        for i in range(3):
            book.positions[f"T{i}"] = Position(f"T{i}", 1, 10, 9, 12, "2026-08-20", sector="Technology")
        book.closed.append(orders.Closed("AAA", 1, 10, 9, "2026-08-10", "2026-08-20", "stop: x"))
        g = gate(home, book=book)
        v = g.buy("BBB", 100.0, 97.0, 107.0, price=100.0, atr_pct=0.02, sector="Technology", earnings_days=30)
        assert not v.ok and "Technology" in v.reason
        v = g.buy("AAA", 100.0, 97.0, 107.0, price=100.0, atr_pct=0.02, sector="Energy", earnings_days=30)
        assert not v.ok and "stopped out" in v.reason
        for i in range(3, 10):
            book.positions[f"E{i}"] = Position(f"E{i}", 1, 10, 9, 12, "2026-08-20", sector="Energy")
        v = g.buy("CCC", 100.0, 97.0, 107.0, price=100.0, atr_pct=0.02, sector="Utilities", earnings_days=30)
        assert not v.ok and "10 positions" in v.reason

    def test_stops_only_move_up(self, home):
        book = DeskBook(home / "desk" / "trade" / "book.json")
        book.positions["AAA"] = Position("AAA", 10, 100, 97, 107, "2026-08-20")
        g = gate(home, book=book)
        assert not g.stop_change("AAA", 96.0).ok
        assert not g.stop_change("AAA", 97.0).ok
        assert g.stop_change("AAA", 100.0).ok
        assert not g.stop_change("ZZZ", 100.0).ok

    def test_closed_market_refuses_buys(self, home):
        v = gate(home, market_open=False).buy("AAA", 100.0, 97.0, 107.0, price=100.0, atr_pct=0.02,
                                              sector="", earnings_days=30)
        assert not v.ok and "closed" in v.reason


@pytest.mark.unit
class TestAggressiveSleeve:
    """MANUAL §10: momentum on full size, two of four triggers, no cushion."""

    def buy(self, g, **kw):
        args = dict(price=100.5, atr_pct=0.02, sector="Technology", earnings_days=30, sleeve="aggressive",
                    triggers=["volume", "catalyst"], vol_ratio=2.0)
        args.update(kw)
        return g.buy("AAA", args.pop("entry", 100.0), args.pop("stop", 97.0), args.pop("target", 105.0), **args)

    def test_a_breakout_with_two_triggers_passes_on_full_size(self, home):
        v = self.buy(gate(home))
        assert v.ok and v.shares == 80            # 1% of 100k = $1,000 ÷ $3 = 333, capped by 8% = $8,000 ÷ 100
        assert v.risk == 240.0

    def test_a_claimed_rs_trigger_is_checked_against_spy(self, home):
        v = self.buy(gate(home), triggers=["rs", "pattern"], rs_1m=0.04)
        assert not v.ok and "rs trigger" in v.reason
        assert self.buy(gate(home), triggers=["rs", "pattern"], rs_1m=0.15).ok
        assert self.buy(gate(home), triggers=["rs", "pattern"]).ok          # unknown rs: the claim stands

    def test_one_trigger_is_not_a_breakout(self, home):
        v = self.buy(gate(home), triggers=["catalyst"])
        assert not v.ok and "2 of" in v.reason

    def test_a_claimed_volume_trigger_is_checked_against_the_bar(self, home):
        v = self.buy(gate(home), triggers=["volume", "pattern"], vol_ratio=1.1)
        assert not v.ok and "volume ratio is 1.1" in v.reason
        assert self.buy(gate(home), triggers=["catalyst", "pattern"], vol_ratio=1.1).ok

    def test_the_sleeve_has_its_own_r_chase_and_stop_rules(self, home):
        assert self.buy(gate(home), target=104.0).ok                      # R 1.33 ≥ 1.2
        assert not self.buy(gate(home), target=103.4).ok                  # R 1.13
        assert self.buy(gate(home), entry=104.5, price=100.0, stop=101.4, target=109.5).ok   # 4.5% chase allowed
        assert not self.buy(gate(home), entry=106.0, price=100.0, stop=102.8, target=111.0).ok   # 6% is not
        assert self.buy(gate(home), stop=98.8, target=102.0).ok            # 1.2% stop = 0.6 ATR, allowed here
        assert not self.buy(gate(home), stop=91.0, target=115.0).ok        # 9% stop, over the 8% cap

    def test_four_sleeve_seats_and_no_double_sleeve_on_one_name(self, home):
        book = DeskBook(home / "desk" / "trade" / "book.json")
        for i in range(4):
            book.positions[f"X{i}"] = Position(f"X{i}", 1, 10, 9, 12, "2026-08-20", sleeve="aggressive")
        v = self.buy(gate(home, book=book))
        assert not v.ok and "4 aggressive" in v.reason
        book.positions.pop("X3"); book.positions["AAA"] = Position("AAA", 1, 10, 9, 12, "2026-08-20")
        assert not self.buy(gate(home, book=book)).ok                     # already a core position

    def test_core_keeps_six_seats(self, home):
        book = DeskBook(home / "desk" / "trade" / "book.json")
        for i in range(6):
            book.positions[f"C{i}"] = Position(f"C{i}", 1, 10, 9, 12, "2026-08-20", sector=f"S{i}")
        v = gate(home, book=book).buy("AAA", 100.0, 97.0, 107.0, price=100.5, atr_pct=0.02, sector="", earnings_days=30)
        assert not v.ok and "6 core" in v.reason
        assert self.buy(gate(home, book=book)).ok                          # the sleeve's seats are still free

    def test_the_executor_books_the_sleeve_with_a_20_day_horizon(self, home):
        shape("AAA", 100.0, drift=0.6)
        v = FakeVenue(acct())
        px = v.quote("AAA")
        out = executor(v, home).run({"orders": [buy("AAA", round(px, 2), round(px * 0.97, 2), round(px * 1.06, 2),
                                                     sleeve="aggressive", triggers=["catalyst", "pattern"])]})
        assert out[0].ok, out[0].reason
        p = DeskBook(home / "desk" / "trade" / "book.json").positions["AAA"]
        assert p.sleeve == "aggressive" and p.horizon_days == 20 and p.triggers == ["catalyst", "pattern"]
        line = json.loads((home / "desk" / "trade" / "decisions.jsonl").read_text().splitlines()[-1])
        assert line["sleeve"] == "aggressive" and "进攻仓" in orders.format_outcomes(out)


# ---------------------------------------------------------------------------
# the executor
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestExecutor:
    def test_a_buy_becomes_a_bracket_and_a_book_row_and_a_log_line(self, home):
        shape("AAA", 100.0, drift=0.6)
        v = FakeVenue(acct())
        px = v.quote("AAA")
        ex = executor(v, home)
        out = ex.run({"orders": [buy("AAA", round(px, 2), round(px * 0.97, 2), round(px * 1.08, 2))]})
        assert out[0].ok, out[0].reason
        kind, sym, qty, (limit, stop, target) = v.placed[0]
        assert kind == "bracket" and sym == "AAA" and qty == out[0].shares and stop < limit < target
        p = DeskBook(home / "desk" / "trade" / "book.json").positions["AAA"]
        assert p.thesis == "t" and p.invalidation == "i" and p.principles == ["一"] and p.legs["stop"] == "s1"
        lines = [json.loads(l) for l in (home / "desk" / "trade" / "decisions.jsonl").read_text().splitlines()]
        assert lines[-1]["kind"] == "order" and lines[-1]["ok"] and lines[-1]["risk"] > 0

    def test_claude_does_not_choose_the_share_count(self, home):
        shape("AAA", 100.0, drift=0.6)
        v = FakeVenue(acct())
        px = v.quote("AAA")
        out = executor(v, home).run({"orders": [buy("AAA", round(px, 2), round(px * 0.97, 2), round(px * 1.08, 2),
                                                     shares=5000)]})
        assert out[0].ok and v.placed[0][2] == out[0].shares != 5000

    def test_a_refused_buy_places_nothing_and_is_logged_with_the_reason(self, home):
        shape("AAA", 100.0, drift=0.6)
        v = FakeVenue(acct())
        px = v.quote("AAA")
        out = executor(v, home).run({"orders": [buy("AAA", round(px, 2), round(px * 0.97, 2), round(px * 1.02, 2))]})
        assert not out[0].ok and "R" in out[0].reason and v.placed == []
        line = json.loads((home / "desk" / "trade" / "decisions.jsonl").read_text().splitlines()[-1])
        assert line["ok"] is False and "R" in line["gate"]

    def test_the_second_buy_of_the_day_sees_the_first_ones_risk(self, home):
        for s in ("AAA", "BBB", "CCC", "DDD", "EEE"):
            shape(s, 100.0, drift=0.6)
        v = FakeVenue(acct(cash=1_000_000, equity=1_000_000))
        px = v.quote("AAA")
        # a 9% stop: the 10% name cap binds at ~1,000 shares, so each buy risks ~0.9%
        # of equity and the fifth would take the day past 4%
        ords = [buy(s, round(px, 2), round(px * 0.91, 2), round(px * 1.25, 2), sector=s) for s in ("AAA", "BBB", "CCC", "DDD", "EEE")]
        out = executor(v, home).run({"orders": ords})
        assert [o.ok for o in out] == [True, True, True, True, False] and "4%" in out[4].reason

    def test_sell_cancels_the_legs_and_flattens_in_one_step(self, home):
        shape("AAA", 100.0, drift=0.6)
        v = FakeVenue(acct(cash=90_000, holdings=[held("AAA", 50, 100.0, 118.0)]))
        v.resting["AAA"] = 110.0
        book = DeskBook(home / "desk" / "trade" / "book.json")
        book.positions["AAA"] = Position("AAA", 50, 100.0, 110.0, 130.0, "2026-08-10", thesis="t", invalidation="i")
        book.save()
        out = executor(v, home).run({"orders": [{"action": "sell", "symbol": "AAA", "reason": "论点失效"}]})
        assert out[0].ok and v.placed[-1][0] == "close" and "AAA" not in v.resting
        b = DeskBook(home / "desk" / "trade" / "book.json")
        assert "AAA" not in b.positions and b.closed[-1].reason.startswith("manual") and b.closed[-1].thesis == "t"

    def test_raise_stop_goes_to_the_venue_and_a_lower_one_does_not(self, home):
        shape("AAA", 100.0, drift=0.6)
        v = FakeVenue(acct(cash=90_000, holdings=[held("AAA", 50, 100.0, 118.0)]))
        v.resting["AAA"] = 97.0
        book = DeskBook(home / "desk" / "trade" / "book.json")
        book.positions["AAA"] = Position("AAA", 50, 100.0, 97.0, 130.0, "2026-08-10")
        book.save()
        out = executor(v, home).run({"orders": [{"action": "raise_stop", "symbol": "AAA", "stop": 95.0},
                                                {"action": "raise_stop", "symbol": "AAA", "stop": 100.0}]})
        assert [o.ok for o in out] == [False, True] and v.resting["AAA"] == 100.0
        assert DeskBook(home / "desk" / "trade" / "book.json").positions["AAA"].stop == 100.0

    def test_protect_puts_a_stop_under_an_unmanaged_position(self, home):
        shape("AAA", 100.0, drift=0.6)
        v = FakeVenue(acct(cash=90_000, holdings=[held("AAA", 50, 100.0, 118.0)]))
        px = v.quote("AAA")
        out = executor(v, home).run({"orders": [{"action": "protect", "symbol": "AAA",
                                                 "stop": round(px * 0.96, 2), "target": round(px * 1.10, 2)}]})
        assert out[0].ok and v.placed[0][0] == "oco" and "AAA" in v.resting
        p = DeskBook(home / "desk" / "trade" / "book.json").positions["AAA"]
        assert p.adopted and p.shares == 50

    def test_trim_sells_part_and_reprotects_the_rest(self, home):
        shape("AAA", 100.0, drift=0.6)
        v = FakeVenue(acct(cash=90_000, holdings=[held("AAA", 50, 100.0, 118.0)]))
        px = v.quote("AAA")
        v.resting["AAA"] = 97.0
        book = DeskBook(home / "desk" / "trade" / "book.json")
        book.positions["AAA"] = Position("AAA", 50, 100.0, 97.0, round(px * 1.2, 2), "2026-08-10")
        book.save()
        out = executor(v, home).run({"orders": [{"action": "trim", "symbol": "AAA", "fraction": 0.5, "reason": "到目标"}]})
        assert out[0].ok and out[0].shares == 25
        kinds = [p[0] for p in v.placed]
        assert kinds == ["market", "oco"] and v.placed[1][2] == 25
        assert DeskBook(home / "desk" / "trade" / "book.json").positions["AAA"].shares == 25

    def test_the_kill_switch_refuses_everything(self, home):
        (home / "STOP").write_text("")
        v = FakeVenue(acct())
        out = executor(v, home).run({"orders": [buy("AAA", 100, 97, 107)]})
        assert not out[0].ok and "kill switch" in out[0].reason and v.placed == []

    def test_dry_run_places_nothing_and_books_nothing(self, home):
        shape("AAA", 100.0, drift=0.6)
        v = FakeVenue(acct())
        px = v.quote("AAA")
        out = executor(v, home, dry_run=True).run({"orders": [buy("AAA", round(px, 2), round(px * 0.97, 2), round(px * 1.08, 2))]})
        assert out[0].ok and "AAA" not in v.resting
        assert DeskBook(home / "desk" / "trade" / "book.json").positions == {}

    def test_a_postmortem_marks_the_closed_row_reviewed(self, home):
        book = DeskBook(home / "desk" / "trade" / "book.json")
        book.closed.append(orders.Closed("AAA", 1, 10, 9, "2026-08-10", "2026-08-20", "stop: x"))
        book.save()
        orders.log_entry({"kind": "postmortem", "symbol": "AAA", "answer": "noise"},
                         home / "desk" / "trade" / "decisions.jsonl")
        assert DeskBook(home / "desk" / "trade" / "book.json").closed[0].reviewed


# ---------------------------------------------------------------------------
# the pack
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPack:
    def test_the_pack_has_the_account_the_candidates_and_the_limits(self, home):
        for s in ("AAA", "BBB"):
            shape(s, 100.0, drift=0.6)
        shape("HLD", 100.0, drift=0.3)
        v = FakeVenue(acct(cash=90_000, holdings=[held("HLD", 10, 90.0, 100.0)]))
        v.resting["HLD"] = 95.0
        book = DeskBook(home / "desk" / "trade" / "book.json")
        book.positions["HLD"] = Position("HLD", 10, 90.0, 95.0, 120.0, "2026-08-10", thesis="T", invalidation="I")
        book.save()
        pk = pack.Packer(broker=v, market=mkt(task="trade"), book=book,
                         screen=screen_of(("AAA", "Technology"), ("BBB", "Energy")), now=MONDAY_OPEN).run(["BBB"])
        assert [r.symbol for r in pk.held] == ["HLD"] and pk.held[0].protected and pk.held[0].thesis == "T"
        assert {r.symbol for r in pk.candidates} == {"AAA", "BBB"}
        aaa = next(r for r in pk.candidates if r.symbol == "AAA")
        assert aaa.stop < aaa.entry <= aaa.price and aaa.r > 0 and aaa.source == "screen"
        assert pk.limits["min_r"] == orders.MIN_R
        text = pack.format_pack(pk)
        assert "## 硬限制" in text and "## 持仓" in text and "## 候选" in text and "| AAA |" in text
        assert (home / "desk" / "trade" / "2026-08-24-pack.json").exists()

    def test_a_stop_that_fired_at_the_venue_closes_the_row_and_asks_for_a_postmortem(self, home):
        shape("AAA", 100.0, drift=0.6)
        v = FakeVenue(acct(cash=100_000))                       # AAA is gone from the account
        v.fills = [{"id": "x", "symbol": "AAA", "side": "sell", "qty": 10, "price": 96.5, "type": "stop",
                    "filled_at": "2026-08-21T14:00:00Z"}]
        book = DeskBook(home / "desk" / "trade" / "book.json")
        book.positions["AAA"] = Position("AAA", 10, 100.0, 97.0, 110.0, "2026-08-10", thesis="T", invalidation="I")
        book.save()
        pk = pack.Packer(broker=v, market=mkt(task="trade"), book=book, screen=screen_of(), now=MONDAY_OPEN).run()
        assert pk.postmortems and pk.postmortems[0]["symbol"] == "AAA" and pk.postmortems[0]["exit"] == 96.5
        assert pk.postmortems[0]["reason"].startswith("stop") and pk.postmortems[0]["thesis"] == "T"
        assert "需要复盘" in pack.format_pack(pk) and "AAA" not in DeskBook(home / "desk" / "trade" / "book.json").positions

    def test_an_unprotected_holding_is_flagged(self, home):
        shape("HLD", 100.0, drift=0.3)
        v = FakeVenue(acct(cash=90_000, holdings=[held("HLD", 10, 90.0, 100.0)]))
        pk = pack.Packer(broker=v, market=mkt(task="trade"), book=DeskBook(home / "desk" / "trade" / "book.json"),
                         screen=screen_of(), now=MONDAY_OPEN).run()
        assert not pk.held[0].protected and "没有止损挂在交易所" in pack.format_pack(pk)

    def test_a_lost_book_row_is_rebuilt_from_the_venues_resting_stop(self, home):
        """The morning after a machine with no state: the venue still holds the
        stop, so the pack rebuilds the row instead of calling the position unmanaged."""
        shape("HLD", 100.0, drift=0.3)
        v = FakeVenue(acct(cash=90_000, holdings=[held("HLD", 10, 90.0, 100.0)]))
        v.resting["HLD"] = 95.0
        v.open_orders = lambda: [{"id": "s-HLD", "symbol": "HLD", "side": "sell", "type": "stop",
                                  "stop_price": 95.0, "limit_price": float("nan"), "submitted_at": "2026-08-12T14:00"},
                                 {"id": "t-HLD", "symbol": "HLD", "side": "sell", "type": "limit",
                                  "stop_price": float("nan"), "limit_price": 120.0, "submitted_at": "2026-08-12T14:00"}]
        book = DeskBook(home / "desk" / "trade" / "book.json")
        pk = pack.Packer(broker=v, market=mkt(task="trade"), book=book, screen=screen_of(), now=MONDAY_OPEN).run()
        p = DeskBook(home / "desk" / "trade" / "book.json").positions["HLD"]
        assert p.stop == 95.0 and p.target == 120.0 and p.entry == 90.0 and p.opened == "2026-08-12" and p.adopted
        assert pk.held[0].protected and pk.held[0].in_book and any("恢复" in w for w in pk.warnings)

    def test_facts_for_hand_picked_names(self, home):
        shape("NEW", 100.0, drift=0.6)
        text = pack.facts_table(["NEW", "NOPE"], market=mkt(task="trade"), now=MONDAY_OPEN)
        assert "| NEW |" in text and "NOPE: 拿不到行情" in text and "```" in text


# ---------------------------------------------------------------------------
# 2026-10-13 反思：the faults the day's trading exposed, fixed and pinned
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestReflection1013:
    def test_r_is_measured_against_the_original_stop_after_a_raise(self):
        """APA: stop moved to cost, then the target filled — the pack showed "—R"."""
        p = Position(symbol="APA", shares=137, entry=42.43, stop=42.43, target=47.0,
                     opened="2026-09-25", stop0=40.53)
        assert round(p.r_at(47.0), 1) == 2.4
        legacy = Position(symbol="X", shares=1, entry=10.0, stop=10.0, target=12.0, opened="2026-09-25")
        assert legacy.r_at(12.0) != legacy.r_at(12.0)  # nan: no original stop on record

    def test_an_unfilled_buy_is_pending_and_the_pack_drops_it_without_a_postmortem(self, home):
        """PBR: a limit that never filled sat in the book and was later "closed" at +0.9R."""
        shape("AAA", 100.0, drift=0.6)
        v = FakeVenue(acct())
        v.order_fill = lambda oid: {"id": oid, "status": "new", "filled": 0.0, "price": float("nan")}
        px = v.quote("AAA")
        ex = executor(v, home)
        out = ex.run({"orders": [buy("AAA", round(px, 2), round(px * 0.97, 2), round(px * 1.08, 2))]})
        assert out[0].ok and "pending" in out[0].venue
        book = DeskBook(home / "desk" / "trade" / "book.json")
        assert book.positions["AAA"].pending
        # the account never shows the shares: the pack removes the row, no closed record
        pk = pack.Packer(broker=v, market=mkt(task="trade"), book=book, screen=screen_of(), now=MONDAY_OPEN).run()
        after = DeskBook(home / "desk" / "trade" / "book.json")
        assert "AAA" not in after.positions and not after.closed and not pk.postmortems
        assert any("未成交" in w for w in pk.warnings)

    def test_a_filled_buy_books_the_fill_price(self, home):
        shape("AAA", 100.0, drift=0.6)
        v = FakeVenue(acct())
        px = v.quote("AAA")
        v.order_fill = lambda oid: {"id": oid, "status": "filled", "filled": 10.0, "price": round(px * 0.999, 2)}
        out = executor(v, home).run({"orders": [buy("AAA", round(px, 2), round(px * 0.97, 2), round(px * 1.08, 2))]})
        assert out[0].ok and "filled @" in out[0].venue
        p = DeskBook(home / "desk" / "trade" / "book.json").positions["AAA"]
        assert p.entry == round(px * 0.999, 2) and p.stop0 == p.stop and not p.pending

    def test_a_pending_row_the_account_now_holds_is_booked_at_cost(self, home):
        shape("HLD", 100.0, drift=0.3)
        book = DeskBook(home / "desk" / "trade" / "book.json")
        book.positions["HLD"] = Position(symbol="HLD", shares=10, entry=99.0, stop=95.0, target=120.0,
                                         opened="2026-08-12", pending=True, stop0=95.0)
        book.save()
        v = FakeVenue(acct(cash=90_000, holdings=[held("HLD", 10, 98.5, 100.0)]))
        v.resting["HLD"] = 95.0
        pack.Packer(broker=v, market=mkt(task="trade"), book=book, screen=screen_of(), now=MONDAY_OPEN).run()
        p = DeskBook(home / "desk" / "trade" / "book.json").positions["HLD"]
        assert not p.pending and p.entry == 98.5

    def test_a_lost_sleeve_row_is_rebuilt_from_the_decision_log(self, home):
        """TBBB: the book row was lost to a stale snapshot; the venue held the stop,
        and the decision log still knew it was an aggressive-sleeve buy."""
        shape("HLD", 100.0, drift=0.3)
        log = home / "desk" / "trade" / "decisions.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(json.dumps({"kind": "order", "action": "buy", "symbol": "HLD", "ok": True, "date": "2026-08-12",
                                   "sleeve": "aggressive", "triggers": ["volume", "pattern"], "stop": 94.0,
                                   "thesis": "动量账", "invalidation": "跌回 94"}) + "\n", encoding="utf-8")
        v = FakeVenue(acct(cash=90_000, holdings=[held("HLD", 10, 90.0, 100.0)]))
        v.resting["HLD"] = 95.0
        v.open_orders = lambda: [{"id": "s-HLD", "symbol": "HLD", "side": "sell", "type": "stop",
                                  "stop_price": 95.0, "limit_price": float("nan"), "submitted_at": "2026-08-13T14:00"}]
        book = DeskBook(home / "desk" / "trade" / "book.json")
        pk = pack.Packer(broker=v, market=mkt(task="trade"), book=book, screen=screen_of(), now=MONDAY_OPEN).run()
        p = DeskBook(home / "desk" / "trade" / "book.json").positions["HLD"]
        assert p.sleeve == "aggressive" and p.triggers == ["volume", "pattern"] and p.thesis == "动量账"
        assert p.stop == 95.0 and p.stop0 == 94.0 and p.horizon_days == 20 and not p.adopted
        assert any("下单记录" in w for w in pk.warnings)

    def test_state_push_merges_append_only_logs_instead_of_overwriting(self):
        """2026-09-30 and 2026-10-12: a session with an older copy pushed a snapshot
        and a day's decisions vanished."""
        from tradingagents.desk.state import merge_jsonl
        remote = '{"a": 1}\n{"b": 2}\n{"c": 3}\n'
        local = '{"a": 1}\n{"b": 2}\n{"d": 4}\n'
        assert merge_jsonl(remote, local) == '{"a": 1}\n{"b": 2}\n{"c": 3}\n{"d": 4}\n'
        assert merge_jsonl("", local) == local
        assert merge_jsonl(remote, "") == remote


@pytest.mark.unit
class TestReflection1014:
    def _pending_book(self, home, opened):  # noqa: F811
        shape("AAA", 100.0, drift=0.6)
        book = DeskBook(home / "desk" / "trade" / "book.json")
        book.positions["AAA"] = Position(symbol="AAA", shares=10, entry=95.0, stop=90.0, target=110.0,
                                         opened=opened, pending=True, stop0=90.0, parent_id="p1")
        book.save()
        return book

    def test_a_fresh_pullback_limit_is_kept_while_it_rests(self, home):  # noqa: F811
        book = self._pending_book(home, MONDAY_OPEN.date().isoformat())
        v = FakeVenue(acct())
        v.open_orders = lambda: [{"id": "p1", "symbol": "AAA", "side": "buy", "type": "limit", "parent_id": ""}]
        pk = pack.Packer(broker=v, market=mkt(task="trade"), book=book, screen=screen_of(), now=MONDAY_OPEN).run()
        after = DeskBook(home / "desk" / "trade" / "book.json")
        assert after.positions["AAA"].pending and not v.cancelled and not pk.postmortems

    def test_a_stale_pullback_limit_is_cancelled_and_dropped(self, home):  # noqa: F811
        """PBR: a limit that never filled rested at the venue for weeks."""
        from datetime import timedelta
        book = self._pending_book(home, (MONDAY_OPEN.date() - timedelta(days=5)).isoformat())
        v = FakeVenue(acct())
        v.open_orders = lambda: [{"id": "p1", "symbol": "AAA", "side": "buy", "type": "limit", "parent_id": ""}]
        pk = pack.Packer(broker=v, market=mkt(task="trade"), book=book, screen=screen_of(), now=MONDAY_OPEN).run()
        after = DeskBook(home / "desk" / "trade" / "book.json")
        assert "AAA" not in after.positions and v.cancelled == ["p1"] and not after.closed
        assert any("已撤单" in w for w in pk.warnings)

    def test_a_pending_row_whose_order_is_gone_is_dropped(self, home):  # noqa: F811
        book = self._pending_book(home, MONDAY_OPEN.date().isoformat())
        v = FakeVenue(acct())
        pk = pack.Packer(broker=v, market=mkt(task="trade"), book=book, screen=screen_of(), now=MONDAY_OPEN).run()
        after = DeskBook(home / "desk" / "trade" / "book.json")
        assert "AAA" not in after.positions and not v.cancelled and any("已无此单" in w for w in pk.warnings)


@pytest.mark.unit
class TestReflection1015:
    def test_a_protected_takeover_is_booked_at_the_accounts_cost(self, home):  # noqa: F811
        """2026-09-25: takeover rows carried the price of the minute protect ran,
        so every R on them was measured from the wrong entry."""
        shape("HLD", 100.0, drift=0.3)
        v = FakeVenue(acct(cash=90_000, holdings=[held("HLD", 10, 91.5, 100.0)]))
        px = v.quote("HLD")
        out = executor(v, home).run({"orders": [{"action": "protect", "symbol": "HLD",
                                                  "stop": round(px * 0.95, 2), "target": round(px * 1.12, 2)}]})
        assert out[0].ok, out[0].reason
        p = DeskBook(home / "desk" / "trade" / "book.json").positions["HLD"]
        assert p.entry == 91.5 and p.stop0 == p.stop and p.adopted


@pytest.mark.unit
class TestReflection1016:
    def test_the_pack_reminds_about_resting_buys_and_untrimmed_1r_sleeve_rows(self, home):  # noqa: F811
        from datetime import timedelta
        shape("HLD", 100.0, drift=0.6)
        shape("AAA", 100.0, drift=0.6)
        book = DeskBook(home / "desk" / "trade" / "book.json")
        px = frame_for("HLD", DATA_DAY.isoformat())["Close"].iloc[-1]
        book.positions["HLD"] = Position(symbol="HLD", shares=10, entry=round(px * 0.9, 2), stop=round(px * 0.85, 2),
                                         target=round(px * 1.3, 2), opened="2026-08-12", sleeve="aggressive",
                                         stop0=round(px * 0.85, 2))
        book.positions["AAA"] = Position(symbol="AAA", shares=10, entry=round(px * 0.97, 2), stop=90.0, target=130.0,
                                         opened=(MONDAY_OPEN.date() - timedelta(days=3)).isoformat(), pending=True,
                                         parent_id="p1")   # the Friday before: one session old
        book.save()
        v = FakeVenue(acct(cash=90_000, holdings=[held("HLD", 10, round(px * 0.9, 2), px)]))
        v.resting["HLD"] = round(px * 0.85, 2)
        v.open_orders = lambda: [{"id": "s-HLD", "symbol": "HLD", "side": "sell", "type": "stop", "stop_price": round(px * 0.85, 2)},
                                 {"id": "p1", "symbol": "AAA", "side": "buy", "type": "limit", "parent_id": ""}]
        pk = pack.Packer(broker=v, market=mkt(task="trade"), book=book, screen=screen_of(), now=MONDAY_OPEN).run()
        text = pack.format_pack(pk)
        assert any(r.startswith("⏳ 挂单 AAA") and "第 1 天" in r for r in pk.reminders)
        assert any(r.startswith("💡 HLD") for r in pk.reminders) and "💡 HLD" in text and "⏳ 挂单 AAA" in text
        # once half is taken off, the reminder goes quiet
        book = DeskBook(home / "desk" / "trade" / "book.json")
        book.positions["HLD"].trimmed = True
        book.save()
        pk2 = pack.Packer(broker=v, market=mkt(task="trade"), book=book, screen=screen_of(), now=MONDAY_OPEN).run()
        assert not any(r.startswith("💡 HLD") for r in pk2.reminders)


@pytest.mark.unit
class TestReflection1019:
    def test_pending_age_counts_sessions_not_calendar_days(self):
        from datetime import date as d
        assert pack.sessions_since("2026-10-16", d(2026, 10, 19)) == 1      # Friday → Monday
        assert pack.sessions_since("2026-10-14", d(2026, 10, 19)) == 3
        assert pack.sessions_since("bad", d(2026, 10, 19)) == 99

    def test_the_newer_book_wins_a_state_merge(self):
        from tradingagents.desk.state import newer_book
        old = '{"saved_at": "2026-10-13T10:00:00", "positions": {"A": 1}}'
        new = '{"saved_at": "2026-10-14T15:20:00", "positions": {"A": 2}}'
        none = '{"positions": {"A": 0}}'
        assert newer_book(old, new) == new and newer_book(new, old) == new
        assert newer_book(none, old) == old and newer_book(old, none) == old
        assert newer_book("not json", new) == new

    def test_the_book_saves_a_timestamp(self, home):  # noqa: F811
        book = DeskBook(home / "desk" / "trade" / "book.json")
        book.save()
        assert json.loads((home / "desk" / "trade" / "book.json").read_text())["saved_at"][:4] == "2026"

    def test_a_book_stop_below_the_venues_is_synced_up(self, home):  # noqa: F811
        """MSFT: the venue held 514.73, a stale book copy said 502."""
        shape("HLD", 100.0, drift=0.3)
        book = DeskBook(home / "desk" / "trade" / "book.json")
        book.positions["HLD"] = Position(symbol="HLD", shares=10, entry=90.0, stop=85.0, target=120.0,
                                         opened="2026-08-12", stop0=85.0)
        book.save()
        v = FakeVenue(acct(cash=90_000, holdings=[held("HLD", 10, 90.0, 100.0)]))
        v.resting["HLD"] = 95.0
        v.open_orders = lambda: [{"id": "s-HLD", "symbol": "HLD", "side": "sell", "type": "stop", "stop_price": 95.0}]
        pk = pack.Packer(broker=v, market=mkt(task="trade"), book=book, screen=screen_of(), now=MONDAY_OPEN).run()
        assert DeskBook(home / "desk" / "trade" / "book.json").positions["HLD"].stop == 95.0
        assert any("已按交易所同步" in w for w in pk.warnings)

    def test_an_orphan_venue_buy_is_cancelled_once_stale(self, home):  # noqa: F811
        """COP: its book row was lost to a stale copy; the GTC limit kept resting."""
        shape("AAA", 100.0, drift=0.6)
        v = FakeVenue(acct())
        v.open_orders = lambda: [{"id": "b-AAA", "symbol": "AAA", "side": "buy", "type": "limit", "parent_id": "",
                                  "submitted_at": "2026-08-12T14:00"},
                                 {"id": "b-BBB", "symbol": "BBB", "side": "buy", "type": "limit", "parent_id": "",
                                  "submitted_at": MONDAY_OPEN.date().isoformat() + "T14:00"}]
        book = DeskBook(home / "desk" / "trade" / "book.json")
        pk = pack.Packer(broker=v, market=mkt(task="trade"), book=book, screen=screen_of(), now=MONDAY_OPEN).run()
        assert v.cancelled == ["b-AAA"]
        assert any(w.startswith("AAA") and "已撤单" in w for w in pk.warnings)
        assert any(w.startswith("BBB") and "会撤" in w for w in pk.warnings)


@pytest.mark.unit
class TestReflection1020:
    def test_the_volume_trigger_accepts_the_breakout_bar_before_last(self, home):  # noqa: F811
        """TALO: broke out on volume 2x, bought the next close on 0.9x."""
        g = gate(home)
        kw = {"price": 100.5, "atr_pct": 0.03, "sector": "Energy", "earnings_days": 30, "sleeve": "aggressive",
              "triggers": ["volume", "pattern"]}
        assert not g.buy("AAA", 100.0, 96.0, 112.0, vol_ratio=0.9, **kw).ok
        assert g.buy("AAA", 100.0, 96.0, 112.0, vol_ratio=0.9, vol_ratio_prev=2.0, **kw).ok
        assert g.buy("AAA", 100.0, 96.0, 112.0, vol_ratio=1.8, vol_ratio_prev=0.5, **kw).ok


@pytest.mark.unit
class TestReflection1021:
    def test_a_resting_buy_can_be_cancelled_and_replaced_the_same_day(self, home):  # noqa: F811
        shape("AAA", 100.0, drift=0.6)
        v = FakeVenue(acct())
        v.order_fill = lambda oid: {"id": oid, "status": "new", "filled": 0.0, "price": float("nan")}
        px = v.quote("AAA")
        ex = executor(v, home)
        out = ex.run({"orders": [buy("AAA", round(px * 0.98, 2), round(px * 0.94, 2), round(px * 1.08, 2))]})
        assert out[0].ok and DeskBook(home / "desk" / "trade" / "book.json").positions["AAA"].pending
        v.open_orders = lambda: [{"id": "o1", "symbol": "AAA", "side": "buy", "type": "limit", "parent_id": ""}]
        ex2 = executor(v, home)
        out = ex2.run({"orders": [{"action": "cancel_buy", "symbol": "AAA", "reason": "改价"},
                                  buy("AAA", round(px, 2), round(px * 0.97, 2), round(px * 1.08, 2))]})
        assert out[0].action == "cancel_buy" and out[0].ok and v.cancelled == ["o1"]
        assert out[1].action == "buy" and out[1].ok, out[1].reason
        assert DeskBook(home / "desk" / "trade" / "book.json").positions["AAA"].entry == round(px, 2)

    def test_cancel_buy_refuses_a_filled_position(self, home):  # noqa: F811
        shape("HLD", 100.0, drift=0.3)
        book = DeskBook(home / "desk" / "trade" / "book.json")
        book.positions["HLD"] = Position(symbol="HLD", shares=10, entry=90.0, stop=85.0, target=120.0, opened="2026-08-12")
        book.save()
        v = FakeVenue(acct(cash=90_000, holdings=[held("HLD", 10, 90.0, 100.0)]))
        out = executor(v, home).run({"orders": [{"action": "cancel_buy", "symbol": "HLD"}]})
        assert not out[0].ok and "stops are never cancelled" in out[0].reason and v.cancelled == []
