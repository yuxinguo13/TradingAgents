"""One name on demand: the pack gathers what the report knows per name, the
history settles the desk's earlier calls on it, and the site is one page."""

from __future__ import annotations

import json

import pytest

from tests.test_desk import (
    FRIDAY_AFTER_CLOSE,
    SHAPES,
    frame_for,
    mkt,
    shape,
)
from tradingagents.desk import site, stock


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path))
    SHAPES.clear()
    shape("SPY", 500.0, drift=0.15, wobble=0.004)
    shape("XLK", 200.0, drift=0.20)
    yield tmp_path


def _final(home, day, sym, sc, e, st, tg):
    d = home / "desk" / "report"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{day}-final.md").write_text(
        "# 市场日报\n\n| 排名 | 代码 | 我的分 | 参考分 | 入场 | 止损 | 目标 | R | 财报 | 一句话 |\n"
        "|---:|---|---:|---:|---:|---:|---:|---:|---|---|\n"
        f"| 1 | {sym} | {sc} | 60 | {e} | {st} | {tg} | 2.0 | — | x |\n", encoding="utf-8")


@pytest.mark.unit
class TestStockPack:
    def test_the_pack_carries_levels_score_sector_and_history(self, home):
        shape("AAA", 100.0, drift=0.40)
        frame = frame_for("AAA", "2026-08-21")
        dates = frame["Date"].dt.date.tolist()
        closes = dict(zip(dates, frame["Close"], strict=True))
        session = dates[-5]
        entry = round(closes[session] * 0.998, 2)
        _final(home, session.isoformat(), "AAA", 72, entry, round(entry * 0.97, 2), round(entry * 1.5, 2))
        m = mkt()
        m._names = {}
        pack = stock.Stocker(market=m, now=FRIDAY_AFTER_CLOSE).run("aaa")
        assert pack.symbol == "AAA" and pack.date == "2026-08-24" and pack.data_date == "2026-08-21"
        assert pack.price > 0 and pack.stop < pack.entry <= pack.price < pack.target
        assert pack.score > 0 and any("多头排列" in r or "50 日" in r for r in pack.reasons)
        assert pack.high_52w >= pack.price and pack.low_52w <= pack.price
        assert pack.rs_3m == pytest.approx(pack.ret_3m - pack.spy_ret_3m)
        assert pack.history and pack.history[0]["report"] == session.isoformat()
        assert pack.history[0]["outcome"] in ("持有中", "到目标", "止损")
        assert pack.chart and "MA20" in "\n".join(pack.chart)
        sdir = home / "desk" / "stock"
        assert (sdir / "2026-08-24-AAA.md").exists() and (sdir / "2026-08-24-AAA.json").exists()
        text = stock.format_pack(pack)
        assert "## 二、代码的读数" in text and "## 五、这个名字的历史判断" in text and "| 2026-" in text
        data = json.loads((sdir / "2026-08-24-AAA.json").read_text(encoding="utf-8"))
        assert data["symbol"] == "AAA" and data["history"][0]["score"] == 72

    def test_a_name_without_bars_is_a_warning_not_a_crash(self, home):
        pack = stock.Stocker(market=mkt(), now=FRIDAY_AFTER_CLOSE).run("NOPE")
        assert pack.warnings and "拿不到行情" in pack.warnings[0]
        assert "拿不到行情" in stock.format_pack(pack)


@pytest.mark.unit
class TestStockSite:
    def test_the_site_is_one_page_with_my_write_up_and_the_chart(self, home):
        shape("AAA", 100.0, drift=0.40)
        m = mkt()
        m._names = {}
        pack = stock.Stocker(market=m, now=FRIDAY_AFTER_CLOSE).run("AAA")
        (home / "desk" / "stock" / "2026-08-24-AAA-final.md").write_text(
            "# AAA · 个股分析\n\n## 一句话\n\n**68 分**，等回调到 95 再看。\n\n"
            "| 排名 | 代码 | 我的分 | 参考分 | 入场 | 止损 | 目标 | R | 财报 | 一句话 |\n|---:|---|---:|---:|---:|---:|---:|---:|---|---|\n"
            "| 1 | AAA | 68 | 60 | 95.0 | 92.0 | 104.0 | 3.0 | — | 等 |\n", encoding="utf-8")
        out = site.build_stock("AAA", "2026-08-24", market=m, out_dir=home / "site-AAA")
        html = (out / "index.html").read_text(encoding="utf-8")
        assert 'data-sym="AAA"' in html and "我的分 68" in html and "等回调到 95" in html
        assert "代码的读数" in html and "个股分析（单独）" in html
        assert json.loads((out / "files.json").read_text()) == ["desk.css", "desk.js"]
        assert '"entry": 95.0' in html and '"stop": 92.0' in html          # my levels drive the chart lines
        assert not any("详情页" in w for w in pack.warnings)      # the empty stub news feed is a coverage-gap note, expected
