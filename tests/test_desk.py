"""The desk: three tasks that share bars and nothing else.

Nothing here touches a network, a venue or an LLM. One bars generator draws
every shape the tests need, and every task reads it through the same seam the
real run uses (``Market(bars_loader=...)``), so the numbers under test are the
numbers the pages would print.
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from tradingagents.desk import advise, report, trade, universe
from tradingagents.desk.market import Market
from tradingagents.live import clock
from tradingagents.live.broker import Account, Holding, OrderResult
from tradingagents.live.newsfeed import NewsItem
from tradingagents.live.recommendations import RecommendationBook

FRIDAY_AFTER_CLOSE = datetime(2026, 8, 21, 17, 0, tzinfo=clock.ET)
MONDAY_OPEN = datetime(2026, 8, 24, 9, 40, tzinfo=clock.ET)
DATA_DAY = date(2026, 8, 21)


# ---------------------------------------------------------------------------
# bars
# ---------------------------------------------------------------------------

SHAPES = {}


def shape(symbol, base=100.0, drift=0.30, wobble=0.01, last_move=0.0, rows=260, volume=2e6):
    """A deterministic daily history: ``drift`` over the year, a fixed wobble,
    and an optional final-day move so a test can put a name below its averages."""
    SHAPES[symbol.upper()] = {"base": base, "drift": drift, "wobble": wobble, "last_move": last_move,
                                  "rows": rows, "volume": volume}


def frame_for(symbol: str, when: str):
    spec = SHAPES.get(symbol.upper())
    if spec is None:
        return None
    rows = spec["rows"]
    end = date.fromisoformat(when)
    dates, closes, vols = [], [], []
    d = end - timedelta(days=int(rows * 7 / 5) + 10)
    i = 0
    while len(closes) < rows:
        if d.weekday() < 5:
            px = spec["base"] * (1 - spec["drift"] / 2 + spec["drift"] * i / (rows - 1))
            px *= 1 + spec["wobble"] * ((i % 7) - 3) / 3
            closes.append(round(px, 4))
            dates.append(d)
            vols.append(spec["volume"] * (1.0 + 0.1 * ((i % 5) - 2)))
            i += 1
        d += timedelta(days=1)
    if spec["last_move"]:
        closes[-1] = round(closes[-1] * (1 + spec["last_move"]), 4)
        vols[-1] *= 2.0
    highs = [c * 1.004 for c in closes]
    lows = [c * 0.996 for c in closes]
    opens = list(closes)
    return pd.DataFrame({"Date": pd.to_datetime(dates), "Open": opens, "High": highs,
                         "Low": lows, "Close": closes, "Volume": vols})


class StubNews:
    def __init__(self, items=()):
        self.items = list(items)

    def poll(self, tickers, macro=True, pause=0.0):
        return list(self.items)


class StubPolicy:
    def __init__(self, events=()):
        self.events = list(events)

    def poll(self, categories=None, pause=0.0):
        return list(self.events)


class StubBook:
    """An earnings/fundamentals book with nothing in it."""

    def get(self, symbols, *a, **k):
        return {}


def news(ticker, title, lean="bullish", materiality=5, hours_ago=2):
    pub = (datetime.now().astimezone() - timedelta(hours=hours_ago)).isoformat()
    return NewsItem(ticker=ticker, title=title, link="http://x", source="t", published=pub,
                    materiality=materiality, lean=lean, fingerprint=title)


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path))
    SHAPES.clear()
    shape("SPY", 500.0, drift=0.15, wobble=0.004)
    yield tmp_path


def mkt(task="test", news_items=(), events=()):
    return Market(task=task, bars_loader=frame_for, news=StubNews(news_items),
                  policy=StubPolicy(events), earnings=StubBook(), fundamentals=StubBook())


# ---------------------------------------------------------------------------
# market
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMarket:
    def test_indicators_come_from_the_bars(self):
        shape("AAA", 100.0, drift=0.40)
        f = mkt().facts("AAA", DATA_DAY)
        assert f.ok and f.price > f.snap.sma50 > f.snap.sma200
        assert 0 < f.snap.atr_pct < 0.05
        assert 40 <= f.snap.rsi14 <= 100
        assert f.snap.ret_3m > 0 and f.above("sma200") is True
        assert "标普500" in f.trend.relative

    def test_a_symbol_with_no_bars_is_not_ok(self):
        f = mkt().facts("NOPE", DATA_DAY)
        assert not f.ok and math.isnan(f.price)

    def test_the_macro_board_reads_rates_in_points_and_prices_in_percent(self):
        shape("^TNX", 4.0, drift=0.10, wobble=0.0)     # 4.0 → ~4.2, in yield points
        shape("^IRX", 5.0, drift=-0.10, wobble=0.0)    # inverted curve
        shape("^GSPC", 5000.0, drift=0.10, wobble=0.0)
        shape("^VIX", 15.0, drift=0.0, wobble=0.0)
        board = mkt().macro(DATA_DAY)
        ten = board.row("^TNX")
        assert ten.ok and ten.is_yield and abs(ten.m1) < 0.2      # points, not a ratio
        spx = board.row("^GSPC")
        assert spx.ok and 0 < spx.m1 < 0.05                        # a ratio
        assert board.curve_spread() < 0
        text = " ".join(board.read())
        assert "倒挂" in text and "标普500" in text and "波动平静" in text

    def test_headlines_split_by_ticker_and_drop_stale_ones(self):
        m = mkt(news_items=[news("AAA", "good"), news("", "macro"), news("AAA", "old", hours_ago=90)])
        by, macro = m.headlines(["AAA"])
        assert [n.title for n in by["AAA"]] == ["good"]
        assert [n.title for n in macro] == ["macro"]


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def names(*rows):
    return [universe.Name(symbol=s, sector=sec, source=src) for s, sec, src in rows]


@pytest.mark.unit
class TestReport:
    def test_scores_rank_the_strong_name_above_the_broken_one(self):
        shape("AAA", 100.0, drift=0.40)
        shape("BBB", 50.0, drift=-0.30)
        rep = report.Reporter(report.ReportConfig(with_pages=False), market=mkt(),
                              now=FRIDAY_AFTER_CLOSE,
                              names=names(("AAA", "Technology", "screen"), ("BBB", "Energy", "bellwether"))).run()
        assert rep.date == "2026-08-24" and rep.data_date == "2026-08-21"
        assert [i.symbol for i in rep.scored] == ["AAA", "BBB"]
        assert rep.ideas[0].symbol == "AAA" and rep.avoid[0].symbol == "BBB"
        aaa = rep.ideas[0]
        assert any("多头排列" in r or "50 日" in r for r in aaa.reasons)
        assert aaa.stop < aaa.entry <= aaa.price and aaa.target > aaa.entry
        assert any("200 日线" in c for c in rep.avoid[0].cautions)

    def test_the_report_never_reads_an_account(self, home):
        """No broker is constructed, no book is opened: the report is holdings-blind."""
        shape("AAA", 100.0, drift=0.40)
        rep = report.Reporter(report.ReportConfig(with_pages=False), market=mkt(), now=FRIDAY_AFTER_CLOSE,
                              names=names(("AAA", "Technology", "screen"))).run()
        assert not (home / "desk" / "trade").exists()
        assert not (home / "recommendations.json").exists()
        assert rep.path.endswith("2026-08-24.md")

    def test_policy_tilt_and_news_move_the_score(self):
        shape("AAA", 100.0, drift=0.40)
        shape("BBB", 100.0, drift=0.40)
        m = mkt(news_items=[news("BBB", "BBB plunges on probe", lean="bearish", materiality=8)])
        rep = report.Reporter(report.ReportConfig(with_pages=False), market=m, now=FRIDAY_AFTER_CLOSE,
                              names=names(("AAA", "Technology", "screen"), ("BBB", "Technology", "screen"))).run()
        a, b = {i.symbol: i for i in rep.scored}["AAA"], {i.symbol: i for i in rep.scored}["BBB"]
        assert a.score > b.score and any("利空" in c for c in b.cautions)

    def test_the_page_carries_charts_sectors_and_the_json_pack(self, home):
        shape("AAA", 100.0, drift=0.40)
        shape("XLK", 200.0, drift=0.20)
        rep = report.Reporter(report.ReportConfig(with_pages=True), market=mkt(), now=FRIDAY_AFTER_CLOSE,
                              names=names(("AAA", "Technology", "screen"))).run()
        text = report.format_report(rep)
        assert "## 一、宏观与利率" in text and "## 三、板块" in text and "## 四、值得关注" in text
        assert "```" in text and "MA20" in text                # the chart block
        assert "| 科技 | XLK |" in text
        assert (home / "desk" / "report" / "2026-08-24" / "AAA.md").exists()
        pack = json.loads((home / "desk" / "report" / "2026-08-24.json").read_text())
        assert pack["ideas"][0]["symbol"] == "AAA" and "macro" in pack and "sectors" in pack

    def test_the_next_report_marks_yesterdays_calls(self, home):
        """A call with its levels is a claim the market answers; the next
        pack prints the answer next to it so the review cannot be skipped."""
        shape("AAA", 100.0, drift=0.40)
        shape("BBB", 100.0, drift=0.40, last_move=-0.06)      # gaps under its stop on the last bar
        (home / "desk" / "report").mkdir(parents=True, exist_ok=True)
        (home / "desk" / "report" / "2026-08-21-final.md").write_text(
            "# 市场日报\n\n| 排名 | 代码 | 我的分 | 参考分 | 入场 | 止损 | 目标 | R | 财报 | 一句话 |\n"
            "|---:|---|---:|---:|---:|---:|---:|---:|---|---|\n"
            "| 1 | AAA | 70 | 60 | 50.0 | 48.0 | 60.0 | 5.0 | — | 等回调 |\n"
            "| 2 | BBB | 65 | 60 | 118.0 | 115.0 | 130.0 | 4.0 | — | 试试 |\n", encoding="utf-8")
        rep = report.Reporter(report.ReportConfig(with_pages=False), market=mkt(), now=FRIDAY_AFTER_CLOSE,
                              names=names(("AAA", "Technology", "screen"), ("BBB", "Energy", "screen"))).run()
        by = {c["symbol"]: c for c in rep.review}
        assert by["AAA"]["status"] == "跑远了（没等到回调）" and by["AAA"]["report"] == "2026-08-21"
        assert by["BBB"]["status"] == "入场后止损"
        assert "## 昨日复盘" in report.format_report(rep) and "| BBB | 65 |" in report.format_report(rep)

    def test_prominent_merges_bellwethers_with_screen_leaders(self, monkeypatch):
        rows = [("ZZZ", {"rank": 1, "sector": "Energy", "name": "Zed", "score": 9.0, "price": 10}),
                ("AAPL", {"rank": 2, "sector": "Technology", "name": "Apple", "score": 8.0, "price": 200})]

        class Frame:
            def iterrows(self):
                return iter(rows)

        out = universe.prominent(DATA_DAY, screen=lambda when, ex, top: Frame())
        syms = [n.symbol for n in out]
        assert "AAPL" in syms and "ZZZ" in syms and syms.count("AAPL") == 1
        assert next(n for n in out if n.symbol == "AAPL").screen_rank == 2
        assert next(n for n in out if n.symbol == "ZZZ").source == "screen"


# ---------------------------------------------------------------------------
# advise
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAdvise:
    def test_the_three_file_formats_read_the_same_portfolio(self, tmp_path):
        j = tmp_path / "p.json"
        j.write_text(json.dumps({"cash": 1000, "holdings": [{"symbol": "aaa", "shares": 10, "cost": 90}]}))
        c = tmp_path / "p.csv"
        c.write_text("symbol,shares,cost\nAAA,10,90\ncash,1000,\n")
        t = tmp_path / "p.txt"
        t.write_text("# mine\nAAA 10 90\ncash 1000\n")
        for p in (j, c, t):
            pf = advise.load_portfolio(p)
            assert pf.cash == 1000 and [(h.symbol, h.shares, h.cost) for h in pf.holdings] == [("AAA", 10, 90)]

    def test_verdicts_follow_the_rules(self):
        shape("UP", 100.0, drift=0.40)                     # intact trend, small weight → hold or add
        shape("DOWN", 100.0, drift=-0.30)                  # broken → sell
        shape("BIG", 100.0, drift=0.40)                    # 60% of the book → trim
        pf = advise.Portfolio(cash=2000, holdings=[advise.Holding("UP", 5, 90.0),
                                                   advise.Holding("DOWN", 10, 120.0),
                                                   advise.Holding("BIG", 60, 80.0)])
        adv = advise.Adviser(market=mkt(), now=FRIDAY_AFTER_CLOSE, with_pages=False).run(pf)
        by = {ln.symbol: ln for ln in adv.lines}
        assert by["DOWN"].action == advise.SELL and by["DOWN"].urgent
        assert by["BIG"].action == advise.TRIM and 0 < by["BIG"].act_shares < 60
        assert by["UP"].action in (advise.HOLD, advise.ADD)
        assert adv.lines[0].symbol == "DOWN"                # sells first
        assert any("需要今天处理" in a for a in adv.alerts)
        assert any("一个名字占了" in a for a in adv.alerts)
        text = advise.format_advice(adv)
        assert "| DOWN |" in text and "**卖出**" in text and "## 先看这里" in text

    def test_a_hard_bearish_headline_forces_an_exit_even_in_an_uptrend(self):
        shape("UP", 100.0, drift=0.40)
        m = mkt(news_items=[news("UP", "UP hit with fraud charges", lean="bearish", materiality=9)])
        pf = advise.Portfolio(cash=0, holdings=[advise.Holding("UP", 10, 90.0)])
        adv = advise.Adviser(market=m, now=FRIDAY_AFTER_CLOSE, with_pages=False).run(pf)
        assert adv.lines[0].action == advise.TRIM and adv.lines[0].urgent

    def test_the_adviser_does_not_touch_the_trade_book(self, home):
        shape("UP", 100.0, drift=0.40)
        pf = advise.Portfolio(cash=0, holdings=[advise.Holding("UP", 10, 90.0)])
        advise.Adviser(market=mkt(), now=FRIDAY_AFTER_CLOSE, with_pages=False).run(pf)
        assert not (home / "desk" / "trade").exists()


# ---------------------------------------------------------------------------
# trade
# ---------------------------------------------------------------------------

class FakeAlpaca:
    def __init__(self, account, is_open=True):
        self._account = account
        self.orders = []
        self.is_open = is_open

    def account(self):
        return self._account

    def quote(self, symbol):
        return frame_for(symbol, DATA_DAY.isoformat())["Close"].iloc[-1] if symbol.upper() in SHAPES else 0.0

    def market_open(self):
        return self.is_open

    def place_order(self, symbol, action, quantity, order_type="Market", limit_price=None, dry_run=False):
        self.orders.append((symbol, action, quantity, order_type, limit_price))
        return OrderResult(ok=True, symbol=symbol, action=action, quantity=quantity,
                           order_type=order_type, limit_price=limit_price, message="accepted")


class ScreenRows:
    def __init__(self, rows):
        self.rows = rows

    def iterrows(self):
        return iter(self.rows)


def screen_of(*syms):
    rows = [(s, {"rank": i + 1, "sector": sec, "name": s, "score": 10 - i, "price": 100})
            for i, (s, sec) in enumerate(syms)]
    return lambda when, ex, top: ScreenRows(rows)


def trader(broker, screen=None, **cfg):
    c = trade.TradeConfig(**cfg)
    return trade.Trader(c, broker=broker, market=mkt(task="trade"), screen=screen or screen_of(),
                        now=MONDAY_OPEN)


@pytest.mark.unit
class TestTrade:
    def test_a_new_position_is_opened_from_the_screen_and_booked(self, home):
        shape("AAA", 100.0, drift=0.60)
        acct = Account(account_value=100_000, cash=100_000, buying_power=100_000, holdings=[])
        b = FakeAlpaca(acct)
        note = trader(b, screen_of(("AAA", "Technology"))).run()
        assert [o[:2] for o in b.orders] == [("AAA", "Buy")]
        sym, action, qty, kind, limit = b.orders[0]
        assert kind == "Limit" and limit > 100 and 0 < qty * limit <= 10_000
        book = RecommendationBook(path=home / "desk" / "trade" / "book.json")
        rec = book.open_recommendations()[0]
        assert rec.symbol == "AAA" and rec.stop_price < rec.reference_price < rec.target_price
        text = trade.format_note(note)
        assert "## 今天的单" in text and "✓ Buy AAA" in text and "跌破" in text
        assert "分析" not in text            # the note says what, not why at length
        assert (home / "desk" / "trade" / "2026-08-24.md").exists()

    def test_a_broken_name_is_not_bought(self):
        shape("BBB", 100.0, drift=-0.30)
        acct = Account(account_value=100_000, cash=100_000, buying_power=100_000, holdings=[])
        b = FakeAlpaca(acct)
        note = trader(b, screen_of(("BBB", "Energy"))).run()
        assert b.orders == [] and any("不开新仓" in p for p in note.plan)

    def test_a_held_position_is_adopted_and_stopped_out(self, home):
        """The account holds a name the desk never opened. Day one it gets a
        stop; when the price is under that stop, the desk sells it."""
        shape("CCC", 100.0, drift=0.30, last_move=-0.15)   # a 15% gap down on the last bar
        h = Holding(symbol="CCC", quantity=50, avg_cost=110.0, last=85.0, market_value=4250, unrealized=-1250)
        acct = Account(account_value=100_000, cash=95_750, buying_power=95_750, holdings=[h])
        b = FakeAlpaca(acct)
        t = trader(b, no_entries=True)
        note = t.run()
        assert any("接管" in n for n in note.notes)
        book = RecommendationBook(path=home / "desk" / "trade" / "book.json")
        assert book.open_recommendations()[0].symbol == "CCC"
        # next session the name gaps again, through the stop the desk set on adoption
        shape("CCC", 100.0, drift=0.30, last_move=-0.25)
        note2 = trade.Trader(trade.TradeConfig(no_entries=True), broker=b, market=mkt(task="trade"),
                             book=RecommendationBook(path=home / "desk" / "trade" / "book.json"),
                             now=MONDAY_OPEN + timedelta(days=1)).run()
        sells = [o for o in b.orders if o[1] == "Sell"]
        assert sells and sells[0][0] == "CCC" and sells[0][2] == 50
        assert any(o.ok and "清仓" in o.reason for o in note2.orders)
        book = RecommendationBook(path=home / "desk" / "trade" / "book.json")
        assert book.open_recommendations() == []

    def test_adoption_sets_the_levels_from_todays_price_not_the_cost(self, home):
        """A position bought at 120 that now trades at 144 must not get a stop
        computed off 120 (under water forever) nor a target off 120 (already
        passed): levels come from where the name is."""
        shape("DDD", 100.0, drift=0.60)
        h = Holding(symbol="DDD", quantity=10, avg_cost=60.0, last=120.0, market_value=1200, unrealized=600)
        b = FakeAlpaca(Account(account_value=100_000, cash=98_800, buying_power=98_800, holdings=[h]))
        note = trader(b, no_entries=True).run()
        rec = RecommendationBook(path=home / "desk" / "trade" / "book.json").open_recommendations()[0]
        px = b.quote("DDD")
        assert rec.stop_price < px < rec.target_price and rec.reference_price == pytest.approx(px)
        assert rec.issued_date == "2026-08-24"
        assert note.positions[0].days_left == 30 and b.orders == []

    def test_dry_run_places_nothing_but_books_nothing_either(self, home):
        shape("AAA", 100.0, drift=0.60)
        acct = Account(account_value=100_000, cash=100_000, buying_power=100_000, holdings=[])
        b = FakeAlpaca(acct)
        note = trader(b, screen_of(("AAA", "Technology")), dry_run=True).run()
        assert b.orders == [] and note.orders and note.orders[0].ok and "dry run" in note.orders[0].message

    def test_the_kill_switch_stops_everything(self, home):
        (home / "STOP").write_text("")
        b = FakeAlpaca(Account(account_value=1, cash=1, buying_power=1, holdings=[]))
        note = trader(b).run()
        assert note.refused and b.orders == []

    def test_slots_cap_the_number_of_positions(self, home):
        for s in ("AAA", "BBB", "CCC"):
            shape(s, 100.0, drift=0.60)
        acct = Account(account_value=100_000, cash=100_000, buying_power=100_000, holdings=[])
        b = FakeAlpaca(acct)
        trader(b, screen_of(("AAA", "Technology"), ("BBB", "Energy"), ("CCC", "Healthcare")),
               max_positions=2).run()
        assert len(b.orders) == 2

    def test_the_cli_dispatches(self, capsys):
        from tradingagents.desk.__main__ import main
        assert main([]) == 2 and "trade" in capsys.readouterr().out
