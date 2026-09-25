"""One seam over the market data, shared by all three tasks.

Every number the desk reads about a symbol comes through :class:`Market`, and
:class:`Market` reads bars through one injectable ``bars_loader``. That is the
whole testing story: a test hands in a loader that draws a shape, and every
indicator, chart, level and verdict downstream is computed from it exactly as
it would be from the vendor's bars. There is no second path.

The indicators are computed here from the bars rather than fetched again from
:func:`live.brain.snapshot`, which loads the same history a second time for
the same numbers. Same definitions (14-day Wilder ATR and RSI, 20/50/200-day
simple averages, 21/63-session returns) so the pages read identically.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

from tradingagents.live import charting
from tradingagents.live.brain import Snapshot
from tradingagents.live.deepdive import Bars, load_bars
from tradingagents.live.newsfeed import NewsItem

from . import home

logger = logging.getLogger(__name__)


def _num(v, default=float("nan")) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def _ok(v) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(v)


# ---------------------------------------------------------------------------
# indicators from bars
# ---------------------------------------------------------------------------

def rsi(closes, n: int = 14) -> float:
    """Wilder's RSI over the last ``n`` changes. NaN with too little history."""
    cs = [c for c in closes if _ok(c)]
    if len(cs) <= n:
        return float("nan")
    gains, losses = [], []
    for a, b in zip(cs[:-1], cs[1:], strict=False):
        d = b - a
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    ag = sum(gains[:n]) / n
    al = sum(losses[:n]) / n
    for g, lo in zip(gains[n:], losses[n:], strict=False):
        ag = (ag * (n - 1) + g) / n
        al = (al * (n - 1) + lo) / n
    if al == 0:
        return 100.0
    return 100.0 - 100.0 / (1.0 + ag / al)


def snapshot_from_bars(bars: Bars) -> Snapshot:
    """The cheap numeric read, from bars already in hand."""
    snap = Snapshot(symbol=bars.symbol)
    cs = [c for c in bars.closes if _ok(c)]
    if len(cs) < 2:
        snap.error = "no usable closes"
        return snap
    price, prev = cs[-1], cs[-2]
    atr = bars.atr()
    snap.price = price
    snap.prev_close = prev
    snap.change_pct = price / prev - 1.0 if prev else 0.0
    snap.atr_pct = atr / price if _ok(atr) and price else 0.0
    snap.move_atrs = (price - prev) / atr if _ok(atr) and atr else 0.0
    snap.rsi14 = rsi(cs)
    snap.sma20 = charting.sma(cs, 20)[-1] if len(cs) >= 20 else float("nan")
    snap.sma50 = charting.sma(cs, 50)[-1] if len(cs) >= 50 else float("nan")
    snap.sma200 = charting.sma(cs, 200)[-1] if len(cs) >= 200 else float("nan")
    snap.sma20 = _num(snap.sma20)
    snap.sma50 = _num(snap.sma50)
    snap.sma200 = _num(snap.sma200)
    facts = bars.facts()
    snap.vol_ratio = _num(facts.get("vol_ratio"))
    snap.ret_1m = bars.ret(21)
    snap.ret_3m = bars.ret(63)
    year = cs[-252:]
    snap.off_high_52w = price / max(year) - 1.0 if year and max(year) else float("nan")
    snap.bar_date = bars.dates[-1] if bars.dates else ""
    snap.ok = True
    return snap


# ---------------------------------------------------------------------------
# one symbol, everything known about it
# ---------------------------------------------------------------------------

@dataclass
class Facts:
    """One symbol's bars, indicators and chart read, plus what was attached."""

    symbol: str
    bars: Bars
    snap: Snapshot
    trend: charting.TrendRead
    name: str = ""
    sector: str = ""
    earnings: object = None
    fundamentals: object = None
    insiders: object = None
    news: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.snap.ok and self.bars.closes)

    @property
    def price(self) -> float:
        return _num(self.snap.price) if self.snap.ok else float("nan")

    @property
    def atr(self) -> float:
        return self.bars.atr() if self.bars else float("nan")

    def support(self) -> float:
        return _num(getattr(self.trend, "support", None))

    def resistance(self) -> float:
        return _num(getattr(self.trend, "resistance", None))

    def above(self, which: str) -> bool | None:
        """True/False against a moving average, None when it does not exist."""
        ma = _num(getattr(self.snap, which, None))
        return None if math.isnan(ma) or not self.price else self.price > ma

    def ext_200(self) -> float:
        s = _num(self.snap.sma200)
        return self.price / s - 1.0 if not math.isnan(s) and s > 0 else float("nan")

    def bullish_news(self) -> list:
        return [n for n in self.news if getattr(n, "lean", "") == "bullish"]

    def bearish_news(self) -> list:
        return [n for n in self.news if getattr(n, "lean", "") == "bearish"]


# ---------------------------------------------------------------------------
# the macro board
# ---------------------------------------------------------------------------

# (symbol, Chinese label, group). Groups render as separate tables.
INSTRUMENTS: tuple[tuple[str, str, str], ...] = (
    ("^GSPC", "标普500", "指数"),
    ("^IXIC", "纳斯达克", "指数"),
    ("^DJI", "道琼斯", "指数"),
    ("^RUT", "罗素2000", "指数"),
    ("^IRX", "3月期美债", "利率"),
    ("^FVX", "5年期美债", "利率"),
    ("^TNX", "10年期美债", "利率"),
    ("^TYX", "30年期美债", "利率"),
    ("^VIX", "VIX 恐慌指数", "风险"),
    ("DX-Y.NYB", "美元指数", "风险"),
    ("CL=F", "原油", "商品"),
    ("GC=F", "黄金", "商品"),
    ("BTC-USD", "比特币", "商品"),
)

SECTOR_ETFS: tuple[tuple[str, str, str], ...] = (
    ("XLK", "科技", "Technology"),
    ("XLC", "通信", "Communication Services"),
    ("XLY", "可选消费", "Consumer Cyclical"),
    ("XLP", "必需消费", "Consumer Defensive"),
    ("XLV", "医疗", "Healthcare"),
    ("XLF", "金融", "Financial Services"),
    ("XLI", "工业", "Industrials"),
    ("XLE", "能源", "Energy"),
    ("XLB", "材料", "Basic Materials"),
    ("XLU", "公用事业", "Utilities"),
    ("XLRE", "房地产", "Real Estate"),
)

SECTOR_ZH = {yf: zh for _, zh, yf in SECTOR_ETFS}
SECTOR_ZH["Unknown"] = "未分类"


@dataclass
class MacroRow:
    symbol: str
    label: str
    group: str
    last: float = float("nan")
    d1: float = float("nan")      # 1-session change (points for yields, % otherwise)
    w1: float = float("nan")      # 5 sessions
    m1: float = float("nan")      # 21 sessions
    above_50: bool | None = None
    above_200: bool | None = None
    spark: str = ""
    ok: bool = False

    @property
    def is_yield(self) -> bool:
        return self.group == "利率"


@dataclass
class MacroBoard:
    as_of: str = ""
    rows: list = field(default_factory=list)
    sectors: list = field(default_factory=list)      # MacroRow per sector ETF

    def row(self, symbol: str) -> MacroRow | None:
        for r in self.rows + self.sectors:
            if r.symbol == symbol:
                return r
        return None

    def curve_spread(self) -> float:
        """10y minus 3m, in percentage points. Negative is an inverted curve."""
        ten, three = self.row("^TNX"), self.row("^IRX")
        if ten and three and ten.ok and three.ok:
            return ten.last - three.last
        return float("nan")

    def read(self) -> list[str]:
        """The board as sentences, Chinese, one per fact that matters."""
        out = []
        spx = self.row("^GSPC")
        if spx and spx.ok:
            stack = ("站在 50 日和 200 日线上方" if spx.above_50 and spx.above_200
                     else "跌破 50 日线但仍在 200 日线上方" if spx.above_200
                     else "在 200 日线下方" if spx.above_200 is False else "")
            out.append(f"标普500 收 {spx.last:,.0f}，近一周 {spx.w1 * 100:+.1f}%，"
                       f"近一月 {spx.m1 * 100:+.1f}%{'，' + stack if stack else ''}")
        ten = self.row("^TNX")
        if ten and ten.ok:
            move = f"较一月前 {ten.m1:+.2f} 个百分点" if _ok(ten.m1) else ""
            out.append(f"10 年期美债收益率 {ten.last:.2f}%{'，' + move if move else ''}")
        sp = self.curve_spread()
        if _ok(sp):
            out.append("收益率曲线" + (f"倒挂 {abs(sp):.2f} 个百分点（3月期高于10年期）" if sp < 0
                                    else f"正向，10年期高于3月期 {sp:.2f} 个百分点"))
        vix = self.row("^VIX")
        if vix and vix.ok:
            mood = ("市场在恐慌区" if vix.last >= 30 else "波动偏高" if vix.last >= 20
                    else "波动平静")
            out.append(f"VIX {vix.last:.1f}，{mood}")
        dxy = self.row("DX-Y.NYB")
        if dxy and dxy.ok and _ok(dxy.m1):
            out.append(f"美元指数 {dxy.last:.1f}，近一月 {dxy.m1 * 100:+.1f}%")
        oil, gold = self.row("CL=F"), self.row("GC=F")
        bits = []
        if oil and oil.ok and _ok(oil.m1):
            bits.append(f"原油 {oil.last:.0f} 美元（月 {oil.m1 * 100:+.1f}%）")
        if gold and gold.ok and _ok(gold.m1):
            bits.append(f"黄金 {gold.last:,.0f} 美元（月 {gold.m1 * 100:+.1f}%）")
        if bits:
            out.append("，".join(bits))
        lead = sorted([s for s in self.sectors if s.ok and _ok(s.w1)],
                      key=lambda s: s.w1, reverse=True)
        if len(lead) >= 3:
            out.append(f"本周最强板块：{lead[0].label} {lead[0].w1 * 100:+.1f}%、"
                       f"{lead[1].label} {lead[1].w1 * 100:+.1f}%；"
                       f"最弱：{lead[-1].label} {lead[-1].w1 * 100:+.1f}%")
        return out


def macro_row(symbol: str, label: str, group: str, bars: Bars) -> MacroRow:
    row = MacroRow(symbol=symbol, label=label, group=group)
    cs = [c for c in bars.closes if _ok(c)]
    if len(cs) < 2:
        return row
    row.last = cs[-1]
    is_yield = group == "利率"

    def change(n):
        if len(cs) <= n or not cs[-n - 1]:
            return float("nan")
        return cs[-1] - cs[-n - 1] if is_yield else cs[-1] / cs[-n - 1] - 1.0

    row.d1, row.w1, row.m1 = change(1), change(5), change(21)
    s50 = charting.sma(cs, 50)[-1] if len(cs) >= 50 else None
    s200 = charting.sma(cs, 200)[-1] if len(cs) >= 200 else None
    row.above_50 = None if not _ok(s50) else cs[-1] > s50
    row.above_200 = None if not _ok(s200) else cs[-1] > s200
    row.spark = charting.sparkline(cs[-63:], width=16)
    row.ok = True
    return row


# ---------------------------------------------------------------------------
# the market
# ---------------------------------------------------------------------------

class Market:
    """Everything the desk knows about prices, headlines and policy.

    ``bars_loader(symbol, date) -> DataFrame`` is the one seam; the default is
    the verified, cached yfinance loader the rest of the package uses. The
    news and policy monitors, the earnings and fundamentals books and the
    company-name resolver are injectable too, and each defaults to the real
    thing with its state under ``$TRADINGAGENTS_HOME/desk/<task>``.
    """

    def __init__(self, *, task: str = "desk", bars_loader=None, news=None,
                 policy=None, earnings=None, fundamentals=None, insiders=None, names=None,
                 spy: str = "SPY"):
        self.task = task
        self._loader = bars_loader
        self._news = news
        self._policy = policy
        self._earnings = earnings
        self._fundamentals = fundamentals
        self._insiders = insiders
        self._names = names
        self.spy = spy
        self._facts: dict[tuple[str, str], Facts] = {}
        self._bars: dict[tuple[str, str], Bars] = {}
        self.errors: list[str] = []

    # --- state -------------------------------------------------------------

    def state_dir(self) -> Path:
        d = home() / self.task
        d.mkdir(parents=True, exist_ok=True)
        return d

    # --- bars and facts ----------------------------------------------------

    def bars(self, symbol: str, when: str | date) -> Bars:
        key = (symbol.upper(), str(when))
        if key not in self._bars:
            self._bars[key] = load_bars(symbol, str(when), loader=self._loader)
        return self._bars[key]

    def facts(self, symbol: str, when: str | date, *, benchmark: bool = True) -> Facts:
        key = (symbol.upper(), str(when))
        if key in self._facts:
            return self._facts[key]
        bars = self.bars(symbol, when)
        snap = snapshot_from_bars(bars) if bars.closes else Snapshot(symbol=symbol.upper(),
                                                                   error="no bars")
        excess = {}
        if benchmark and symbol.upper() != self.spy and bars.closes:
            spy = self.bars(self.spy, when)
            mine, theirs = bars.ret(21), spy.ret(21) if spy.closes else float("nan")
            if _ok(mine) and _ok(theirs):
                excess["标普500"] = mine - theirs
        trend = charting.read_trend(
            bars.symbol, bars.closes, bars.highs, bars.lows, bars.volumes,
            rsi=snap.rsi14, atr_pct=snap.atr_pct, vol_ratio=snap.vol_ratio,
            ret_1m=snap.ret_1m, ret_3m=snap.ret_3m, benchmark=excess,
        ) if bars.closes else charting.TrendRead(symbol=symbol.upper())
        f = Facts(symbol=symbol.upper(), bars=bars, snap=snap, trend=trend)
        self._facts[key] = f
        return f

    def macro(self, when: str | date) -> MacroBoard:
        board = MacroBoard(as_of=str(when))
        for sym, label, group in INSTRUMENTS:
            board.rows.append(macro_row(sym, label, group, self.bars(sym, when)))
        for sym, label, _ in SECTOR_ETFS:
            board.sectors.append(macro_row(sym, label, "板块", self.bars(sym, when)))
        return board

    # --- headlines ---------------------------------------------------------

    def news_monitor(self):
        if self._news is None:
            from tradingagents.live.newsfeed import NewsMonitor
            self._news = NewsMonitor(state_path=self.state_dir() / "news_seen.json")
        return self._news

    def headlines(self, symbols: list[str], *, macro: bool = True,
                  max_age_hours: float = 48.0) -> tuple[dict[str, list[NewsItem]], list[NewsItem]]:
        """Fresh headlines per symbol, and the macro ones. Never raises."""
        by_symbol: dict[str, list[NewsItem]] = {s.upper(): [] for s in symbols}
        macro_items: list[NewsItem] = []
        try:
            items = self.news_monitor().poll([s.upper() for s in symbols], macro=macro)
        except Exception as exc:
            self.errors.append(f"news feeds unavailable ({type(exc).__name__}: {exc})")
            return by_symbol, macro_items
        if symbols and not items:
            # Google News and the RSS feeds answer an outage with an empty
            # body, not an error. An empty poll across a whole batch is the
            # feed, not the news: say so, so a blank column is not read as
            # "nothing happened" (upstream's coverage-gap rule).
            self.errors.append("新闻源没有返回任何条目：各名字的消息栏是空白，不是没有消息")
        now = datetime.now(timezone.utc)
        for item in items:
            if item.age_hours(now) > max_age_hours:
                continue
            if item.ticker:
                by_symbol.setdefault(item.ticker.upper(), []).append(item)
            else:
                macro_items.append(item)
        return by_symbol, macro_items

    def policy_monitor(self):
        if self._policy is None:
            from tradingagents.live.policy import PolicyMonitor
            self._policy = PolicyMonitor(state_path=self.state_dir() / "policy_seen.json")
        return self._policy

    def policy(self) -> list:
        try:
            return list(self.policy_monitor().poll())
        except Exception as exc:
            self.errors.append(f"policy feeds unavailable ({type(exc).__name__}: {exc})")
            return []

    # --- slower facts ------------------------------------------------------

    def earnings(self, symbols: list[str], as_of: date) -> dict:
        if not symbols:
            return {}
        try:
            if self._earnings is None:
                from tradingagents.live.earnings import EarningsBook
                self._earnings = EarningsBook(path=self.state_dir() / "earnings.json")
            return dict(self._earnings.get(list(symbols), as_of))
        except Exception as exc:
            self.errors.append(f"earnings dates unavailable ({type(exc).__name__}: {exc})")
            return {}

    def fundamentals(self, symbols: list[str]) -> dict:
        if not symbols:
            return {}
        try:
            if self._fundamentals is None:
                from tradingagents.live.fundamentals import FundamentalsBook
                self._fundamentals = FundamentalsBook(path=self.state_dir() / "fundamentals.json")
            return dict(self._fundamentals.get(list(symbols)))
        except Exception as exc:
            self.errors.append(f"fundamentals unavailable ({type(exc).__name__}: {exc})")
            return {}

    def insiders(self, symbols: list[str], as_of: date | None = None) -> dict:
        """Open-market insider buys and sells per symbol; empty when unavailable."""
        if not symbols:
            return {}
        try:
            if self._insiders is None:
                from tradingagents.live.insiders import InsidersBook
                self._insiders = InsidersBook(path=self.state_dir() / "insiders.json")
            return dict(self._insiders.get(list(symbols), as_of=as_of))
        except Exception as exc:
            self.errors.append(f"insiders unavailable ({type(exc).__name__}: {exc})")
            return {}

    def names(self):
        if self._names is None:
            from tradingagents.live.zhnames import ZhNames
            self._names = ZhNames()
        return self._names

    def attach(self, facts: Facts, *, earnings: dict | None = None,
               fundamentals: dict | None = None, news: dict | None = None,
               insiders: dict | None = None) -> Facts:
        """Fill the slow fields from the books already fetched for the batch."""
        sym = facts.symbol
        if earnings is not None:
            facts.earnings = earnings.get(sym)
        if fundamentals is not None:
            f = fundamentals.get(sym)
            facts.fundamentals = f
            if f is not None:
                facts.sector = facts.sector or str(getattr(f, "sector", "") or "")
                facts.name = facts.name or str(getattr(f, "name", "") or "")
        if news is not None:
            facts.news = list(news.get(sym, []))
        if insiders is not None:
            facts.insiders = insiders.get(sym)
        return facts
