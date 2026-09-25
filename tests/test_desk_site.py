"""The report as a web site: Claude's scores and write-ups are read back from
the final markdown, every name gets a page, and the index links to them."""

from __future__ import annotations

import json

import pytest

from tests.test_desk import (  # noqa: F401
    FRIDAY_AFTER_CLOSE,
    SHAPES,
    home,
    mkt,
    names,
    shape,
)
from tradingagents.desk import report, site

FINAL = """# 市场日报 · 2026-08-24

## 四、值得关注：我的打分

| 排名 | 代码 | 我的分 | 参考分 | 入场 | 止损 | 目标 | R | 财报 | 一句话 |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| 1 | AAA | 73 | 60 | 92.6 | 90.9 | 100.4 | 4.5 | 10/29 | 最干净 |
| 2 | BBB | 55 | 70 | 不追 | — | — | — | 10/28 | 等回调 |

### 1. AAA 甲公司 · 科技 · 73 分

- 趋势 40：多头排列。
- **参考位：** 入场 92.6，止损 90.9。

### 观察名单

- **BBB 乙公司**：RSI 81，等回调。

## 五、走弱 / 回避

- **CCC、DDD**（可选消费）：200 日线下。

## 六、全部评分
见数据包。
"""


@pytest.mark.unit
class TestParsing:
    def test_scores_come_from_the_table_and_the_bullets(self):
        sc = site.my_scores(FINAL)
        assert sc["AAA"] == (73, 92.6, 90.9, 100.4, 4.5)
        assert sc["BBB"][0] == 55 and sc["BBB"][1] is None

    def test_notes_are_the_section_or_the_bullet(self):
        no = site.my_notes(FINAL)
        assert no["AAA"].startswith("### 我的判断：甲公司") and "参考位" in no["AAA"]
        assert "RSI 81" in no["BBB"]
        assert "200 日线下" in no["CCC"] and "200 日线下" in no["DDD"]


@pytest.mark.unit
class TestBuild:
    def test_the_site_has_an_index_and_a_page_per_name(self, home):
        shape("AAA", 100.0, drift=0.6)
        shape("BBB", 100.0, drift=0.4)
        m = mkt(task="report")
        rep = report.Reporter(report.ReportConfig(with_pages=True), market=m, now=FRIDAY_AFTER_CLOSE,
                              names=names(("AAA", "Technology", "screen"), ("BBB", "Energy", "bellwether"))).run()
        (home / "desk" / "report" / f"{rep.date}-final.md").write_text(FINAL, encoding="utf-8")
        out = site.build(rep.date, market=m)
        assert (out / "index.html").exists() and (out / "desk.js").exists() and (out / "desk.css").exists()
        files = json.loads((out / "files.json").read_text())
        assert "pages/AAA.html" in files and "pages/BBB.html" in files
        index = (out / "index.html").read_text(encoding="utf-8")
        assert 'href="pages/AAA.html"' in index and '<h2 id="s6">' in index and "我的分" in index
        page = (out / "pages" / "AAA.html").read_text(encoding="utf-8")
        assert "我的分 73" in page and "我的判断：甲公司" in page and 'data-sym="AAA"' in page
        assert "代码的读数" in page and "../index.html" in page
        data = json.loads(page.split('type="application/json">')[1].split("</script>")[0])
        assert data["stocks"]["AAA"]["stop"] == 90.9 and len(data["stocks"]["AAA"]["close"]) > 100
