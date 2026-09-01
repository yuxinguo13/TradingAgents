"""Where to go and check this yourself.

Every number in this report is derived from two feeds — Yahoo's OHLCV and a
keyword news search — and both of them are wrong sometimes. The honest response
is not to add a third feed and average them; it is to print the door to the
primary source next to the claim, so a reader who doubts a line can open it in
one click instead of retyping the ticker into five sites.

The links are split by what a reader is actually trying to do, because a list of
twelve undifferentiated URLs is a list nobody opens:

* **行情与图形** — the chart, at a resolution this text file cannot draw.
* **财务与估值** — the statements behind the fundamentals block.
* **原始文件** — SEC filings and insider transactions. The only tier with no
  intermediary between the reader and the company.
* **消息与预期** — news, the earnings calendar, analyst estimates.
* **中文渠道** — the same instrument on the platforms this reader already uses.
  Listed last and marked, because their US data is itself relayed from the
  sources above rather than being an independent check.

No key, no login and no scraping: these are addresses, not requests. Nothing in
this module makes a network call.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote_plus


@dataclass(frozen=True)
class Link:
    label: str
    url: str
    note: str = ""

    def markdown(self) -> str:
        return f"[{self.label}]({self.url})" + (f" — {self.note}" if self.note else "")


def _q(symbol: str) -> str:
    return quote_plus((symbol or "").strip().upper())


def _dotted(symbol: str) -> str:
    """``BRK-B`` → ``BRK.B``. Yahoo hyphenates share classes; most others dot them."""
    return (symbol or "").strip().upper().replace("-", ".")


def quote_links(symbol: str) -> list[Link]:
    s, d = _q(symbol), _q(_dotted(symbol))
    return [
        Link("Yahoo Finance 行情", f"https://finance.yahoo.com/quote/{s}",
             "本报告所有价格与均线的来源，先对这里"),
        Link("TradingView 图表", f"https://www.tradingview.com/chart/?symbol={s}",
             "可画线、可换周期，用来复核本页的 ASCII 图"),
        Link("Finviz 快照", f"https://finviz.com/quote.ashx?t={s}",
             "一屏看完估值、技术与同业对比"),
        Link("StockCharts 技术面", f"https://stockcharts.com/h-sc/ui?s={s}"),
        Link("Barchart 观点", f"https://www.barchart.com/stocks/quotes/{d}/opinion",
             "把十几个技术指标折算成一个多空票数"),
    ]


def financial_links(symbol: str) -> list[Link]:
    s, d = _q(symbol), _q(_dotted(symbol))
    return [
        Link("StockAnalysis 财务报表", f"https://stockanalysis.com/stocks/{s}/financials/",
             "十年利润表/资产负债表/现金流，免费且不用登录"),
        Link("Yahoo 财务", f"https://finance.yahoo.com/quote/{s}/financials",
             "本页财报数字的来源"),
        Link("Yahoo 关键统计", f"https://finance.yahoo.com/quote/{s}/key-statistics",
             "估值倍数、利润率、资产负债的完整版"),
        Link("Macrotrends 长期趋势", f"https://www.macrotrends.net/stocks/charts/{s}/x/revenue",
             "营收与利润的十年折线，看结构性变化"),
        Link("Wisesheets/Koyfin 替代：GuruFocus",
             f"https://www.gurufocus.com/stock/{s}/summary"),
    ]


def filing_links(symbol: str) -> list[Link]:
    s = _q(symbol)
    return [
        Link("SEC EDGAR 全部文件",
             f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&ticker={s}"
             f"&type=&dateb=&owner=include&count=40",
             "10-K/10-Q/8-K 原文；财报电话会与风险因素只在这里"),
        Link("SEC 全文检索",
             f"https://efts.sec.gov/LATEST/search-index?q=%22{s}%22&forms=8-K",
             "按关键词搜文件正文"),
        Link("OpenInsider 内部人交易", f"http://openinsider.com/search?q={s}",
             "高管自己在买还是在卖，是少数不靠解读的信号"),
        Link("13F 机构持仓", f"https://whalewisdom.com/stock/{s}",
             "注意：13F 滞后 45 天，不能当作买入理由"),
    ]


def news_links(symbol: str, company: str = "") -> list[Link]:
    s = _q(symbol)
    name = (company or "").strip()
    # The whole query is encoded once. Interpolating a pre-encoded phrase and
    # then appending a literal "%20stock" produced a URL mixing "+" and "%20"
    # separators, which Google News reads as one long token.
    phrase = quote_plus(f"{name} stock" if name else f"{s} stock")
    return [
        Link("Yahoo 新闻", f"https://finance.yahoo.com/quote/{s}/news"),
        Link("Google News 搜索",
             f"https://news.google.com/search?q={phrase}&hl=en-US"),
        Link("财报日历与预期", f"https://stockanalysis.com/stocks/{s}/forecast/",
             "下次财报日、市场一致预期、历史超预期记录"),
        Link("Seeking Alpha 讨论", f"https://seekingalpha.com/symbol/{s}",
             "观点密度高，但是买方与卖方混杂，当作反方意见读"),
        Link("公司投资者关系",
             f"https://www.google.com/search?q={quote_plus((name or s) + ' investor relations')}",
             "财报原始 PPT 与电话会记录"),
    ]


def chinese_links(symbol: str) -> list[Link]:
    """Chinese-language platforms carrying the same US instrument.

    Marked as relays on purpose. Their quotes and statements come from the same
    upstream vendors as everything above, so agreement between them is not
    confirmation — it is the same number printed twice.
    """
    s = _q(symbol)
    return [
        Link("雪球", f"https://xueqiu.com/S/{s}", "中文讨论与财报摘要"),
        Link("富途牛牛", f"https://www.futunn.com/stock/{s}-US", "中文行情与公告翻译"),
        Link("东方财富", f"https://quote.eastmoney.com/us/{s}.html", "中文财务报表"),
        Link("同花顺", f"http://stock.10jqka.com.cn/usstock/{s}/", "中文资讯"),
    ]


SECTIONS = (
    ("行情与图形", quote_links),
    ("财务与估值", financial_links),
    ("原始文件（无中介）", filing_links),
    ("消息与预期", news_links),
    ("中文渠道（转载源，不作独立验证）", chinese_links),
)


def all_links(symbol: str, company: str = "") -> list[tuple[str, list[Link]]]:
    out = []
    for title, fn in SECTIONS:
        try:
            links = fn(symbol, company) if fn is news_links else fn(symbol)
        except TypeError:
            links = fn(symbol)
        out.append((title, links))
    return out


def markdown_block(symbol: str, company: str = "", heading: str = "### 自己去查") -> str:
    lines = [heading, ""]
    for title, links in all_links(symbol, company):
        lines.append(f"**{title}**")
        lines += [f"- {ln.markdown()}" for ln in links]
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
