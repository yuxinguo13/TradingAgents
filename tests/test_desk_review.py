"""Settlement and score-keeping: a call is answered in R and against SPY,
never by its own move alone; insiders are read as buyers, not as sellers;
an empty news poll is a gap, not an absence."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from tests.test_desk import (
    FRIDAY_AFTER_CLOSE,
    SHAPES,
    StubBook,
    StubNews,
    StubPolicy,
    frame_for,
    mkt,
    names,
    shape,
)
from tradingagents.desk import report, review
from tradingagents.desk.market import Market
from tradingagents.desk.orders import DeskBook, Position
from tradingagents.live import insiders
from tradingagents.live.deepdive import Bars


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path))
    SHAPES.clear()
    shape("SPY", 500.0, drift=0.15, wobble=0.004)
    yield tmp_path


def bars(symbol, closes, start=date(2026, 8, 3), spread=0.01):
    """Weekday bars with highs/lows a fixed spread around the close."""
    dates, d = [], start
    while len(dates) < len(closes):
        if d.weekday() < 5:
            dates.append(d)
        d += timedelta(days=1)
    return Bars(symbol, dates, list(closes), [c * (1 + spread) for c in closes],
                [c * (1 - spread) for c in closes], list(closes), [1e6] * len(closes))


# ---------------------------------------------------------------------------
# settling one call
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSettleCall:
    SPY = bars("SPY", [100, 100, 101, 102, 103, 104, 105, 106])

    def test_a_call_that_reaches_its_entry_and_then_its_target_scores_the_full_r(self):
        b = bars("AAA", [100, 100, 99, 101, 104, 107, 110, 112])
        # session = the 2nd bar (Aug 4): window = bars 1..5; entry 99.5 hit on bar 2 (low 98.01)
        s = review.settle_call("AAA", (72, 99.5, 97.0, 106.0, 2.6), b, self.SPY, date(2026, 8, 4))
        assert s.entered and s.outcome == "到目标"
        assert s.r == pytest.approx((106.0 - 99.5) / 2.5)
        assert s.raw == pytest.approx(107 / 100 - 1)
        assert s.spy == pytest.approx(104 / 100 - 1) and s.alpha == pytest.approx(0.07 - 0.04)

    def test_a_call_that_is_stopped_scores_minus_one(self):
        b = bars("AAA", [100, 100, 99, 96, 95, 97, 98, 99])
        s = review.settle_call("AAA", (70, 99.5, 97.0, 106.0, 2.6), b, self.SPY, date(2026, 8, 4))
        assert s.entered and s.outcome == "止损" and s.r == -1.0

    def test_a_call_never_reached_is_recorded_only_against_the_index(self):
        b = bars("AAA", [100, 101, 102, 104, 106, 108, 110, 112])
        s = review.settle_call("AAA", (70, 99.5, 97.0, 106.0, 2.6), b, self.SPY, date(2026, 8, 4))
        assert not s.entered and s.outcome == "未触发" and s.r != s.r      # NaN
        assert s.alpha == pytest.approx((108 / 100 - 1) - (104 / 100 - 1))

    def test_an_entered_call_still_open_is_marked_at_the_last_close(self):
        b = bars("AAA", [100, 100, 99, 100, 101, 102, 103, 104])
        s = review.settle_call("AAA", (70, 99.5, 97.0, 120.0, 8.0), b, self.SPY, date(2026, 8, 4))
        assert s.entered and s.outcome == "持有中" and s.r == pytest.approx((102 - 99.5) / 2.5)

    def test_a_window_that_has_not_fully_traded_is_not_settled(self):
        b = bars("AAA", [100, 100, 99, 101])
        assert review.settle_call("AAA", (70, 99.5, 97.0, 106.0, 2.6), b, self.SPY, date(2026, 8, 4)) is None

    def test_window_return_is_close_before_start_to_close_at_end(self):
        b = bars("AAA", [100, 110, 120, 130, 140])          # Aug 3..7
        assert review.window_return(b, date(2026, 8, 4), date(2026, 8, 6)) == pytest.approx(0.30)
        assert review.window_return(b, date(2026, 8, 3), date(2026, 8, 6)) != review.window_return(b, date(2026, 8, 3), date(2026, 8, 6))


# ---------------------------------------------------------------------------
# the report settles the call from five sessions ago, next to SPY
# ---------------------------------------------------------------------------

def _window(symbol="AAA"):
    """The synthetic frame's dates and closes; its last bar is the data day."""
    frame = frame_for(symbol, "2026-08-21")
    dates = frame["Date"].dt.date.tolist()
    return dates, dict(zip(dates, frame["Close"], strict=True))


def _final(home, day, rows):
    (home / "desk" / "report").mkdir(parents=True, exist_ok=True)
    body = ("# 市场日报\n\n| 排名 | 代码 | 我的分 | 参考分 | 入场 | 止损 | 目标 | R | 财报 | 一句话 |\n"
            "|---:|---|---:|---:|---:|---:|---:|---:|---|---|\n")
    for i, (sym, sc, e, st, tg) in enumerate(rows, 1):
        body += f"| {i} | {sym} | {sc} | 60 | {e} | {st} | {tg} | 2.0 | — | x |\n"
    (home / "desk" / "report" / f"{day}-final.md").write_text(body, encoding="utf-8")


@pytest.mark.unit
class TestReportSettlement:
    def test_the_report_settles_a_week_old_call_in_r_and_against_spy(self, home):
        shape("AAA", 100.0, drift=0.40)
        dates, closes = _window()
        session = dates[-5]                     # five sessions before the data day, inclusive
        # entry just above the session's low so it fills that day; target under a later
        # high so it is reached inside the window; stop well under the wobble.
        entry = round(closes[session] * 0.998, 2)
        stop = round(entry * 0.97, 2)
        target = round(max(closes[dates[-3]], closes[dates[-2]]) * 1.002, 2)
        _final(home, session.isoformat(), [("AAA", 71, entry, stop, target)])
        _final(home, "2026-08-21", [("AAA", 70, entry, stop, target)])     # too young to settle
        rep = report.Reporter(report.ReportConfig(with_pages=False), market=mkt(), now=FRIDAY_AFTER_CLOSE,
                              names=names(("AAA", "Technology", "screen"))).run()
        assert rep.settled_report == session.isoformat()
        s = rep.settled[0]
        assert s["symbol"] == "AAA" and s["entered"] and s["outcome"] == "到目标"
        assert s["r"] == pytest.approx((target - entry) / (entry - stop), rel=1e-3)
        assert s["alpha"] is not None and s["spy"] is not None
        text = report.format_report(rep)
        assert f"## 五日结算（{session}" in text and "| 到目标 |" in text
        # yesterday's review now says how the call did against the index
        assert "相对标普" in text and rep.review[0]["alpha"] is not None

    def test_no_settleable_report_means_no_section(self, home):
        shape("AAA", 100.0, drift=0.40)
        rep = report.Reporter(report.ReportConfig(with_pages=False), market=mkt(), now=FRIDAY_AFTER_CLOSE,
                              names=names(("AAA", "Technology", "screen"))).run()
        assert rep.settled == [] and "五日结算" not in report.format_report(rep)


# ---------------------------------------------------------------------------
# the aggregate
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestReviewAggregate:
    def test_closed_trades_are_scored_per_sleeve_and_principle(self, home):
        shape("AAA", 100.0, drift=0.40)
        shape("BBB", 100.0, drift=-0.20)
        book = DeskBook(home / "desk" / "trade" / "book.json")
        book.positions["AAA"] = Position("AAA", 10, 100.0, 95.0, 115.0, "2026-08-10",
                                         principles=["一·多头排列"], sleeve="core")
        book.positions["BBB"] = Position("BBB", 10, 100.0, 96.0, 110.0, "2026-08-10",
                                         principles=["一·多头排列", "十·突破"], sleeve="aggressive")
        book.close("AAA", 110.0, date(2026, 8, 18), "target: hit")
        book.close("BBB", 96.0, date(2026, 8, 18), "stop: hit")
        assert book.closed[1].sleeve == "aggressive"
        rv = review.run(mkt(), book=DeskBook(book.path), as_of=date(2026, 8, 21))
        assert rv.trades["core"].n == 1 and rv.trades["core"].hits == 1 and rv.trades["core"].mean_r == pytest.approx(2.0)
        assert rv.trades["aggressive"].mean_r == pytest.approx(-1.0)
        assert rv.principles["一·多头排列"].n == 2 and rv.principles["十·突破"].hit_rate == 0.0
        aaa = next(c for c in rv.closed if c["symbol"] == "AAA")
        assert aaa["alpha"] == aaa["alpha"] and aaa["raw"] == aaa["raw"]   # settled against SPY, not NaN
        text = review.format_review(rv)
        assert "| 主仓 | 1 |" in text and "| 进攻仓 | 1 |" in text and "十·突破" in text
        path = review.save(rv)
        assert path.name == "review-2026-08-21.md" and path.with_suffix(".json").exists()

    def test_report_calls_are_banded_by_the_score_they_were_given(self, home):
        shape("AAA", 100.0, drift=0.40)
        dates, closes = _window()
        session = dates[-5]
        entry = round(closes[session] * 0.998, 2)
        _final(home, session.isoformat(), [("AAA", 78, entry, round(entry * 0.97, 2), round(entry * 1.5, 2))])
        rv = review.run(mkt(), book=DeskBook(home / "x.json"), as_of=date(2026, 8, 21))
        assert rv.reports == 1 and rv.calls["≥75"].n == 1 and rv.settled[0]["entered"]
        assert rv.calls["≥75"].mean_r == rv.calls["≥75"].mean_r        # marked at the last close, a number
        assert "还没有平掉" in review.format_review(rv)

    def test_an_empty_desk_says_so(self, home):
        rv = review.run(mkt(), book=DeskBook(home / "x.json"), as_of=date(2026, 8, 21))
        assert rv.reports == 0 and not rv.closed and len(rv.notes) == 2


# ---------------------------------------------------------------------------
# insiders
# ---------------------------------------------------------------------------

def _frame(rows):
    return pd.DataFrame(rows, columns=["Shares", "Value", "Text", "Insider", "Position", "Transaction", "Start Date"])


@pytest.mark.unit
class TestInsiders:
    def test_only_open_market_buys_and_sales_count(self):
        assert insiders.classify("Purchase at price 12.00 per share.") == "buy"
        assert insiders.classify("Sale at price 222.19 - 223.75 per share.") == "sell"
        assert insiders.classify("Stock Award(Grant) at price 0.00 per share.") == ""
        assert insiders.classify("Stock Gift at price 0.00 per share.") == ""
        assert insiders.classify("Option Exercise at price 10 per share.") == ""
        assert insiders.classify("Sale of options") == ""

    def test_two_buyers_are_a_cluster_and_one_seller_is_not_a_warning(self):
        f = _frame([
            (1000, 12000, "", "SMITH JANE", "CEO", "Purchase at price 12.00 per share.", pd.Timestamp("2026-09-10")),
            (500, 6100, "", "DOE JOHN", "Director", "Purchase at price 12.20 per share.", pd.Timestamp("2026-09-01")),
            (20000, 250000, "", "LEE ANN", "CFO", "Sale at price 12.50 per share.", pd.Timestamp("2026-09-05")),
            (900, 9000, "", "OLD GUY", "Director", "Purchase at price 10.00 per share.", pd.Timestamp("2026-03-01")),  # outside the window
            (100, 0, "", "SMITH JANE", "CEO", "Stock Award(Grant) at price 0.00 per share.", pd.Timestamp("2026-09-12")),
        ])
        r = insiders.summarize(f, "abc", as_of=date(2026, 9, 25))
        assert r.buys == 2 and r.sells == 1 and r.buyers == ["Smith Jane", "Doe John"]
        assert r.buy_value == 18100 and r.last_buy == "2026-09-10"
        assert r.cluster_buying and not r.cluster_selling
        assert "2 位内部人" in r.read() and "Smith Jane" in r.read()

    def test_selling_is_counted_but_never_a_caution(self):
        sellers = [(1000, 1e6, "", n, "Officer", "Sale at price 50 per share.", pd.Timestamp("2026-09-10"))
                   for n in ("A A", "B B", "C C")]
        r = insiders.summarize(_frame(sellers), "abc", as_of=date(2026, 9, 25))
        assert r.sells == 3 and r.cluster_selling and "集中卖出" in r.read()
        r1 = insiders.summarize(_frame(sellers[:1]), "abc", as_of=date(2026, 9, 25))
        assert not r1.cluster_selling and r1.read() == ""

    def test_a_selling_cluster_does_not_reach_the_cautions(self, home):
        shape("AAA", 100.0, drift=0.40)
        sellers = [(1000, 1e7, "", n, "Officer", "Sale at price 50 per share.", pd.Timestamp("2026-08-18"))
                   for n in ("A A", "B B", "C C", "D D")]

        class Book:
            def get(self, symbols, **k):
                return {"AAA": insiders.summarize(_frame(sellers), "AAA", as_of=date(2026, 8, 21))}

        m = Market(task="test", bars_loader=frame_for, news=StubNews(), policy=StubPolicy(),
                   earnings=StubBook(), fundamentals=StubBook(), insiders=Book())
        rep = report.Reporter(report.ReportConfig(with_pages=False), market=m, now=FRIDAY_AFTER_CLOSE,
                              names=names(("AAA", "Technology", "screen"))).run()
        a = rep.scored[0]
        assert not any("内部人" in c for c in a.cautions)
        assert "内部人（近 90 天）：买 0 笔 $0.0M · 卖 4 笔 $40.0M" in report.format_report(rep)

    def test_the_report_prints_the_cluster_as_a_reason_without_moving_the_score(self, home):
        shape("AAA", 100.0, drift=0.40)
        shape("BBB", 100.0, drift=0.40)

        class Book:
            def get(self, symbols, **k):
                f = _frame([(1000, 12000, "", "SMITH JANE", "CEO", "Purchase at price 12 per share.", pd.Timestamp("2026-08-18")),
                            (500, 6100, "", "DOE JOHN", "Director", "Purchase at price 12 per share.", pd.Timestamp("2026-08-19"))])
                return {"AAA": insiders.summarize(f, "AAA", as_of=date(2026, 8, 21))}

        m = Market(task="test", bars_loader=frame_for, news=StubNews(), policy=StubPolicy(),
                   earnings=StubBook(), fundamentals=StubBook(), insiders=Book())
        rep = report.Reporter(report.ReportConfig(with_pages=False), market=m, now=FRIDAY_AFTER_CLOSE,
                              names=names(("AAA", "Technology", "screen"), ("BBB", "Technology", "screen"))).run()
        a, b = {i.symbol: i for i in rep.scored}["AAA"], {i.symbol: i for i in rep.scored}["BBB"]
        assert any("内部人买入" in r for r in a.reasons) and not any("内部人" in r for r in b.reasons)
        assert a.score == b.score
        assert "内部人（近 90 天）：买 2 笔" in report.format_report(rep)
        assert a.to_dict()["insiders"].startswith("内部人买入")

    def test_the_cache_serves_fresh_reads_and_refetches_stale_ones(self, tmp_path):
        calls = []

        def fetcher(sym, as_of=None, log=None):
            calls.append(sym)
            return insiders.InsiderRead(symbol=sym, fetched_at=insiders.time.time())

        book = insiders.InsidersBook(tmp_path / "i.json", fetcher=fetcher)
        book.get(["AAA", "BBB"])
        book.get(["AAA"])
        assert calls == ["AAA", "BBB"]
        again = insiders.InsidersBook(tmp_path / "i.json", fetcher=fetcher, ttl_hours=0)
        again.get(["AAA"])
        assert calls == ["AAA", "BBB", "AAA"]


# ---------------------------------------------------------------------------
# an empty poll is a gap
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCoverageGap:
    def test_an_empty_poll_across_the_batch_is_reported_as_a_gap(self, home):
        shape("AAA", 100.0, drift=0.40)
        rep = report.Reporter(report.ReportConfig(with_pages=False), market=mkt(), now=FRIDAY_AFTER_CLOSE,
                              names=names(("AAA", "Technology", "screen"))).run()
        assert any("新闻源没有返回任何条目" in w for w in rep.warnings)
        assert "不是没有消息" in report.format_report(rep)

    def test_a_poll_with_items_is_not_a_gap(self):
        from tests.test_desk import news
        m = mkt(news_items=[news("AAA", "AAA wins a contract")])
        m.headlines(["AAA"])
        assert m.errors == []
