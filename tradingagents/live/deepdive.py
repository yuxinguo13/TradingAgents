"""One page per name: the evidence, the drawing, the arithmetic, and the doubt.

The daily report is a table, and a table can only ever assert. It says ATAI,
1082 shares, stop 7.26, and every one of those cells is a conclusion with its
reasoning stripped off. A reader who wants to disagree with the stop has
nowhere to stand.

So each name gets a page, and the page is ordered the way an argument is
ordered rather than the way a database is:

1. **图形** first, because it is the only part a reader can check in one second
   and the only part that shows *how* the name got where it is.
2. **算术** second: the stop as a percentage and as an ATR, the R, the
   break-even win rate the R implies, and what one loss costs the account. All
   of it is division the reader can redo on a phone, which is the point.
3. **财报** third — the business behind the trend, from :mod:`~.fundamentals`.
4. **消息** fourth, with links, dates and sources on every item.
5. **反方** fifth, and never omitted. :func:`risks` reads the same data the buy
   case read and reports what argues against it. A page with no bear case is a
   page that has stopped being evidence and started being an advertisement.
6. **原始数据 + 去哪儿查** last: the closing prices this page computed from, and
   the addresses of the primary sources, so nothing above has to be taken on
   trust.

Everything here is rendering. Nothing fetches, decides or sizes; the caller
passes what it already gathered for the daily report, which is why a page costs
one cached OHLCV read and no new judgement.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date as _date

from . import charting, fundamentals as fund, research
from .zhnames import ZhName

logger = logging.getLogger(__name__)

# Roles a symbol can have on a given day. A name can hold more than one — a
# core holding also appearing on the watchlist — and the page names all of them,
# because "why is this on my report" is the first question a reader has.
ROLE_LABEL = {
    "buy": "今日买入建议",
    "sell": "今日卖出/离场",
    "open": "在场的波段建议",
    "core": "核心长仓",
    "watch": "长期关注",
    "daytrade": "日内盯盘",
    "cut": "被否决的候选",
}


def _num(v, default: float = float("nan")) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def _ok(v) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(float(v))


def _cost(rec) -> str:
    """What the order costs at the price it would fill at.

    Mirrors :func:`advisor.estimated_cost` — the limit when there is one, the
    reference otherwise. Duplicated rather than imported because advisor
    imports this module.
    """
    px = _num(getattr(rec, "limit_price", None))
    if math.isnan(px):
        px = _num(getattr(rec, "reference_price", None))
    shares = _num(getattr(rec, "shares", 0), 0.0)
    return 0.0 if math.isnan(px) else px * shares


def _pct(v, digits: int = 1) -> str:
    x = _num(v)
    return "—" if math.isnan(x) else f"{x * 100:+.{digits}f}%"


def _px(v, digits: int = 2) -> str:
    x = _num(v)
    return "—" if math.isnan(x) else f"{x:,.{digits}f}"


# ---------------------------------------------------------------------------
# bars
# ---------------------------------------------------------------------------

@dataclass
class Bars:
    """A symbol's daily history, as plain lists. Look-ahead already filtered."""

    symbol: str
    dates: list = field(default_factory=list)
    opens: list = field(default_factory=list)
    highs: list = field(default_factory=list)
    lows: list = field(default_factory=list)
    closes: list = field(default_factory=list)
    volumes: list = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.closes)

    @property
    def ok(self) -> bool:
        return len(self.closes) >= 30

    def tail(self, n: int) -> "Bars":
        return Bars(self.symbol, self.dates[-n:], self.opens[-n:], self.highs[-n:],
                    self.lows[-n:], self.closes[-n:], self.volumes[-n:])

    def ret(self, sessions: int) -> float:
        """Return over ``sessions`` bars, NaN when the history is shorter."""
        c = self.closes
        if len(c) <= sessions or not c[-sessions - 1]:
            return float("nan")
        return c[-1] / c[-sessions - 1] - 1.0

    def dollar_volume(self, window: int = 50) -> float:
        pairs = [(c, v) for c, v in zip(self.closes[-window:], self.volumes[-window:])
                 if _ok(c) and _ok(v)]
        return sum(c * v for c, v in pairs) / len(pairs) if pairs else float("nan")

    def atr(self, window: int = 14) -> float:
        """Wilder's true range, averaged. Same window brain.snapshot uses."""
        n = min(len(self.closes) - 1, window)
        if n < 2:
            return float("nan")
        trs = []
        for i in range(len(self.closes) - n, len(self.closes)):
            h, l, pc = self.highs[i], self.lows[i], self.closes[i - 1]
            if not all(_ok(x) for x in (h, l, pc)):
                continue
            trs.append(max(h - l, abs(h - pc), abs(l - pc)))
        return sum(trs) / len(trs) if trs else float("nan")

    def facts(self) -> dict:
        """The mapping :mod:`~.horizons` wants. Computed once per symbol."""
        if not self.closes:
            return {}
        price = self.closes[-1]
        atr = self.atr()
        s200 = charting.sma(self.closes, 200)[-1] if len(self.closes) >= 200 else None
        year = self.closes[-252:]
        vols = [v for v in self.volumes[-21:-1] if _ok(v)]
        avg_vol = sum(vols) / len(vols) if vols else float("nan")
        return {
            "price": price,
            "sma200": s200,
            "ret_1m": self.ret(21), "ret_3m": self.ret(63),
            "ret_6m": self.ret(126), "ret_12m": self.ret(252),
            "off_high": price / max(year) - 1.0 if year and max(year) else float("nan"),
            "atr": atr,
            "atr_pct": atr / price if _ok(atr) and price else float("nan"),
            "prev_high": self.highs[-1] if self.highs else float("nan"),
            "prev_low": self.lows[-1] if self.lows else float("nan"),
            "change_pct": (price / self.closes[-2] - 1.0
                           if len(self.closes) > 1 and self.closes[-2] else float("nan")),
            "vol_ratio": (self.volumes[-1] / avg_vol
                          if _ok(avg_vol) and avg_vol and _ok(self.volumes[-1])
                          else float("nan")),
            "dollar_vol": self.dollar_volume(),
            "spark": charting.sparkline(self.closes[-63:], width=20),
        }


def load_bars(symbol: str, when: str, *, loader=None) -> Bars:
    """Cached OHLCV → :class:`Bars`. Empty on any failure; never raises.

    ``loader`` is injectable so a test never touches the network and so the
    caller can share one memoisation table across a whole report.
    """
    bars = Bars(symbol=(symbol or "").upper())
    try:
        if loader is None:
            from tradingagents.dataflows.stockstats_utils import load_ohlcv as loader_
            loader = loader_
        df = loader(symbol, when)
        if df is None or len(df) == 0:
            return bars
        import pandas as pd
        col = {c.lower(): c for c in df.columns}

        def series(name):
            key = col.get(name)
            return ([] if key is None
                    else pd.to_numeric(df[key], errors="coerce").tolist())

        bars.closes = series("close")
        bars.opens = series("open") or list(bars.closes)
        bars.highs = series("high") or list(bars.closes)
        bars.lows = series("low") or list(bars.closes)
        bars.volumes = series("volume") or [float("nan")] * len(bars.closes)
        date_col = col.get("date")
        if date_col is not None:
            bars.dates = [str(d)[:10] for d in df[date_col].tolist()]
        else:
            bars.dates = [str(i)[:10] for i in df.index.tolist()]
    except Exception as exc:
        logger.debug("bars for %s unavailable: %s", symbol, exc)
        return Bars(symbol=(symbol or "").upper())
    return bars


# ---------------------------------------------------------------------------
# one symbol's assembled evidence
# ---------------------------------------------------------------------------

@dataclass
class SymbolAnalysis:
    """Everything one page needs, gathered but not yet rendered."""

    symbol: str
    zh: ZhName | None = None
    roles: list = field(default_factory=list)
    sector: str = ""
    tag: str = ""
    bars: Bars | None = None
    snap: object = None
    trend: charting.TrendRead | None = None
    fundamentals: object = None
    earnings: object = None
    news: list = field(default_factory=list)
    rec: object = None                 # the buy Recommendation, when there is one
    open_idea: object = None           # horizons.OpenIdea, when still held
    exit_signal: object = None
    watch: object = None               # advisor.WatchRow, when watched
    core: object = None                # horizons.CoreLine, when a core holding
    daytrade: object = None
    tilt: float = 0.0
    screen_rank: float = float("nan")
    excess: dict = field(default_factory=dict)
    page: str = ""                     # relative path, set by the writer

    @property
    def name(self) -> str:
        return self.zh.full() if self.zh else self.symbol

    @property
    def short(self) -> str:
        return self.zh.label() if self.zh else self.symbol

    @property
    def price(self) -> float:
        if self.bars and self.bars.closes:
            return _num(self.bars.closes[-1])
        return _num(getattr(self.snap, "price", None))

    @property
    def spark(self) -> str:
        if self.trend and self.trend.spark:
            return self.trend.spark
        return charting.sparkline(self.bars.closes[-63:], 20) if self.bars else ""


def build_trend(bars: Bars, snap=None, excess: dict | None = None,
                atr_stop_mult: float = 2.0) -> charting.TrendRead:
    """The chart read, taking whatever the snapshot already computed."""
    g = lambda k: _num(getattr(snap, k, None)) if snap is not None else float("nan")
    return charting.read_trend(
        bars.symbol, bars.closes, bars.highs, bars.lows, bars.volumes,
        rsi=g("rsi14"), atr_pct=g("atr_pct"), vol_ratio=g("vol_ratio"),
        ret_1m=g("ret_1m"), ret_3m=g("ret_3m"),
        benchmark=excess or {}, atr_stop_mult=atr_stop_mult)


# ---------------------------------------------------------------------------
# the arithmetic, spelled out
# ---------------------------------------------------------------------------

def level_read(a: SymbolAnalysis, account_value: float = float("nan"),
               atr_stop_mult: float = 2.0) -> list[str]:
    """The trade's numbers as sentences, each one a division the reader can redo.

    Written out rather than tabulated because the table already exists on the
    daily page. What it cannot show is *why* 1082 shares — that number is the
    end of a chain (equity → risk budget → stop distance → shares) and every
    link in it is an assumption worth seeing.
    """
    rec = a.rec
    if rec is None:
        return []
    entry = _num(getattr(rec, "reference_price", None))
    stop = _num(getattr(rec, "initial_stop_price", None))
    if math.isnan(stop):
        stop = _num(getattr(rec, "stop_price", None))
    target = _num(getattr(rec, "target_price", None))
    shares = _num(getattr(rec, "shares", None), 0.0)
    out: list[str] = []

    if _ok(entry) and _ok(stop) and entry > 0:
        dist = (entry - stop) / entry
        atr_pct = _num(getattr(a.snap, "atr_pct", None))
        atr_line = (f"，也就是 {atr_stop_mult:g} 个 ATR（日均波幅 {atr_pct * 100:.1f}%）"
                    if _ok(atr_pct) and atr_pct > 0 else "")
        out.append(f"**止损** {_px(stop)}，在参考价 {_px(entry)} 下方 {dist * 100:.1f}%"
                   f"{atr_line}。这个距离不是看图挑的，是按波动率算的——"
                   f"波动大的股票止损自然要放远，代价是同样的风险预算只能买更少的股数。")
    if _ok(entry) and _ok(target) and entry > 0:
        out.append(f"**目标** {_px(target)}，比参考价高 {(target / entry - 1) * 100:.1f}%。"
                   f"它是把近三个月的斜率按 {getattr(rec, 'horizon_days', 30)} 天外推出来的，"
                   f"不是任何人的预测；趋势一旦停下来，这个数字就没有依据了。")
    r = rec.planned_r() if hasattr(rec, "planned_r") else float("nan")
    if _ok(r) and r > -1:
        out.append(f"**R = {r:.2f}**（赚 {_px(target)} − {_px(entry)} 对亏 "
                   f"{_px(entry)} − {_px(stop)}）。盈亏平衡胜率 = 1/(1+R) = "
                   f"**{1 / (1 + r) * 100:.0f}%**：这笔交易只要长期胜率高于这个数就是正期望，"
                   f"低于就是负期望。本报告不声称知道胜率，只保证把这条线画出来。")
    rps = rec.risk_per_share() if hasattr(rec, "risk_per_share") else float("nan")
    if _ok(rps) and shares > 0:
        risk = rps * shares
        share_line = (f"，占账户净值 {risk / account_value * 100:.2f}%"
                      if _ok(account_value) and account_value > 0 else "")
        out.append(f"**股数 {shares:,.0f}** ＝ 风险预算 ÷ 每股风险 {_px(rps)}。"
                   f"止损被打掉的实际损失约 **${risk:,.0f}**{share_line}。"
                   f"这是止损正常成交的情况；跳空低开会更差。")
    lim = _num(getattr(rec, "limit_price", None))
    if _ok(lim) and _ok(entry):
        out.append(f"**限价 {_px(lim)}**（参考价上方 {(lim / entry - 1) * 100:.2f}%）。"
                   f"开盘直接市价单会付掉当天第一笔成交的价格；跳空 6% 的早晨，"
                   f"那已经是另一笔交易了。限价的代价是真跳空时买不到——"
                   f"买不到可以再等，买贵了要一直扛。")
    if a.trend and _ok(a.trend.support) and _ok(stop):
        rel = "在支撑下方，属于「结构破了才认输」" if stop < a.trend.support else \
              "在支撑上方，会先于结构被打掉——这种止损更容易被洗出去"
        out.append(f"**止损与结构的关系**：最近的摆动低点在 {_px(a.trend.support)}，止损{rel}。")
    if a.trend and _ok(a.trend.resistance) and _ok(target):
        rel = ("目标在前高上方，需要突破才能到达"
               if target > a.trend.resistance else "目标在前高下方，路径上没有明显的卖压")
        out.append(f"**目标与结构的关系**：最近的摆动高点在 {_px(a.trend.resistance)}，{rel}。")
    return out


def risks(a: SymbolAnalysis) -> list[str]:
    """The bear case, read off the same data as the bull case.

    Assembled by rule rather than written, so it cannot be quietly skipped on
    the names where it is least convenient. Every item names the number it came
    from; an unsupported worry is not a risk, it is a mood.
    """
    out: list[str] = []
    t, f, s = a.trend, a.fundamentals, a.snap
    price = a.price

    if t and not t.error:
        if t.ma_stack in ("跌破 200 日线", "空头排列"):
            out.append(f"图形上是{t.ma_stack}——{t.ma_detail or '价格在长期均线下方'}。"
                       f"本报告的买入规则本身会过滤掉这种形态。")
        if t.structure == "下降结构":
            out.append("摆动高低点都在下移，属于下降结构；在出现一个更高的低点之前，"
                       "任何反弹都还只是反弹。")
        if _ok(t.price) and _ok(t.support) and t.support > 0:
            gap = t.price / t.support - 1
            if gap > 0.15:
                out.append(f"离最近的支撑 {_px(t.support)} 还有 {gap * 100:.0f}%——"
                           f"这段距离里没有明显的接盘位置。")
    if s is not None:
        atr_pct = _num(getattr(s, "atr_pct", None))
        if _ok(atr_pct) and atr_pct > 0.06:
            out.append(f"日均波幅 {atr_pct * 100:.1f}%，属于高波动：同样的仓位金额，"
                       f"这只股票的日常噪音就能触发大多数人的心理止损。")
        rsi = _num(getattr(s, "rsi14", None))
        if _ok(rsi) and rsi >= 75:
            out.append(f"RSI {rsi:.0f} 已在极端超买区，短期回撤概率高于起涨阶段。")
    if a.bars is not None:
        dv = a.bars.dollar_volume()
        if _ok(dv) and dv < 1e7:
            out.append(f"50 日日均成交额只有 ${dv / 1e6:,.1f}M，流动性偏薄；"
                       f"急跌时的滑点会明显大于本报告假设的成交价。")
    e = a.earnings
    if e is not None:
        try:
            days = e.days_to_next()
            horizon = _num(getattr(a.rec, "horizon_days", None), 30.0)
            if _ok(days) and 0 <= days <= horizon:
                out.append(f"下次财报在 {days:.0f} 天后，落在持有期内。"
                           f"财报跳空会直接穿过止损——那一晚的风险不是 1R，是没有上限的。")
        except Exception:
            pass
    if f is not None and not getattr(f, "error", ""):
        if not f.profitable:
            out.append("公司尚未盈利：估值没有盈利可锚，股价由叙事和融资环境决定，"
                       "这类名字在流动性收紧时跌得最快。")
        if fund.ok(f.free_cashflow) and f.free_cashflow < 0:
            out.append(f"自由现金流为负（{fund.money(f.free_cashflow)}），"
                       f"意味着经营本身在消耗现金，未来可能需要再融资摊薄。")
        if fund.ok(f.debt_to_equity) and f.debt_to_equity > 200:
            out.append(f"负债/权益 {f.debt_to_equity:,.0f}%，杠杆高；利率或再融资条件变化"
                       f"对它的影响会被放大。")
        if fund.ok(f.pe_trailing) and f.pe_trailing > 60:
            out.append(f"TTM 市盈率 {f.pe_trailing:,.0f} 倍，价格里已经计入了不少增长；"
                       f"增长只要减速，杀估值的幅度会大于业绩本身的降幅。")
        if fund.ok(f.short_pct_float) and f.short_pct_float > 0.15:
            out.append(f"空头占流通股 {f.short_pct_float * 100:.0f}%，是双向的："
                       f"可能轧空，也说明有人在认真做空这个逻辑。")
        up = f.upside(price)
        if fund.ok(up) and up < 0:
            out.append(f"现价已经高于卖方目标价均值 {fund.ratio(f.target_mean, 2)}"
                       f"（{_pct(up)}）——共识认为没有上行空间了。")
        if (fund.ok(f.target_high) and fund.ok(f.target_low) and fund.ok(f.target_mean)
                and f.target_mean > 0 and (f.target_high - f.target_low) / f.target_mean > 1.0):
            out.append(f"卖方目标价区间 {fund.ratio(f.target_low, 2)}–"
                       f"{fund.ratio(f.target_high, 2)}，宽度超过均值——分歧极大，"
                       f"共识在这只股票上没有信息量。")
    bearish = [n for n in a.news if getattr(n, "lean", "") == "bearish"
               and getattr(n, "materiality", 0) >= 7]
    for n in bearish[:2]:
        out.append(f"24 小时内的利空标题：{n.title}（{n.source}）")
    if a.tilt < -0.3:
        out.append(f"政策面对该板块的倾向为 {a.tilt:+.2f}（负值＝逆风），"
                   f"这只影响排序、不构成否决，但它是逆风。")
    if not out:
        out.append("按本报告掌握的数据，没有触发任何一条成文的风险规则。"
                   "这不等于没有风险——它只等于「这台机器读到的东西里没有」。")
    return out


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def _charts(a: SymbolAnalysis, levels: dict | None = None) -> list[str]:
    bars = a.bars
    if bars is None or not bars.closes:
        return ["_没有价格数据，无法作图。_", ""]
    out: list[str] = []
    long_n = min(len(bars), 126)
    long_bars = bars.tail(long_n)
    overlays = {}
    s50 = charting.sma(bars.closes, 50)[-long_n:]
    if any(x is not None for x in s50):
        overlays["SMA50"] = s50
    if len(bars) >= 200:
        overlays["SMA200"] = charting.sma(bars.closes, 200)[-long_n:]
    body = charting.line_chart(long_bars.closes, height=14, width=76,
                               overlays=overlays, levels=levels or {},
                               dates=long_bars.dates,
                               title=f"{a.symbol} · 近 {long_n} 个交易日（日线收盘）")
    if body:
        out += ["```text"] + body + ["```", ""]

    short_n = min(len(bars), 21)
    if short_n >= 5:
        near = bars.tail(short_n)
        body = charting.line_chart(near.closes, height=9, width=60,
                                   levels=levels or {}, dates=near.dates,
                                   title=f"{a.symbol} · 近 {short_n} 个交易日（放大看最近节奏）")
        if body:
            out += ["```text"] + body + ["```", ""]

    vol = [v for v in bars.volumes[-63:] if _ok(v)]
    if len(vol) >= 20:
        out += [f"成交量近 63 日： `{charting.sparkline(vol, 42)}`", ""]
    return out


def _news_block(a: SymbolAnalysis) -> list[str]:
    if not a.news:
        return ["_过去 24 小时没有抓到这只股票的新闻。没有消息本身也是一种状态："
                "本次建议的依据只有价格与财报。_", ""]
    out = ["| 时间 | 材料度 | 倾向 | 标题 | 来源 |", "|---|---:|---|---|---|"]
    lean_zh = {"bullish": "偏多", "bearish": "偏空", "neutral": "中性"}
    for n in sorted(a.news, key=lambda x: (-getattr(x, "materiality", 0),
                                           getattr(x, "age_hours", lambda: 0)()))[:12]:
        title = (n.title or "").replace("|", "\\|")
        cell = f"[{title}]({n.link})" if getattr(n, "link", "") else title
        try:
            age = f"{n.age_hours():.0f}h 前"
        except Exception:
            age = "—"
        out.append(f"| {age} | {getattr(n, 'materiality', 0)} | "
                   f"{lean_zh.get(getattr(n, 'lean', ''), '—')} | {cell} | "
                   f"{getattr(n, 'source', '') or '—'} |")
    out += ["", "_材料度是关键词打分（0–12），不是重要性的度量：机构持仓申报一类的"
            "噪音被压到 3 以下，但一条打 9 分的标题也可能只是转载。_", ""]
    return out


def _raw_block(a: SymbolAnalysis, rows: int = 15) -> list[str]:
    bars = a.bars
    if bars is None or not bars.closes:
        return []
    n = min(rows, len(bars))
    out = ["| 日期 | 开 | 高 | 低 | 收 | 涨跌 | 成交量 |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for i in range(len(bars) - n, len(bars)):
        prev = bars.closes[i - 1] if i > 0 else float("nan")
        chg = (bars.closes[i] / prev - 1.0) if _ok(prev) and prev else float("nan")
        vol = bars.volumes[i] if i < len(bars.volumes) else float("nan")
        day = bars.dates[i] if i < len(bars.dates) else ""
        out.append(f"| {day} | {_px(bars.opens[i])} | {_px(bars.highs[i])} | "
                   f"{_px(bars.lows[i])} | **{_px(bars.closes[i])}** | {_pct(chg)} | "
                   f"{f'{vol:,.0f}' if _ok(vol) else '—'} |")
    return out + [""]


def _headline(a: SymbolAnalysis) -> str:
    """The one sentence, assembled from what the page is about to show."""
    bits = []
    if a.trend and not a.trend.error:
        bits.append(a.trend.verdict)
    if a.rec is not None and hasattr(a.rec, "planned_r"):
        r = a.rec.planned_r()
        if _ok(r):
            bits.append(f"本次建议按 {r:.2f}R 定价，盈亏平衡胜率 {1 / (1 + r) * 100:.0f}%")
    if a.core is not None:
        bits.append(f"作为核心长仓，当前状态「{a.core.status}」，动作「{a.core.action}」")
    return "；".join(bits) + "。" if bits else "本页只有数据，没有足够的历史给出图形判断。"


def render_page(a: SymbolAnalysis, *, report_date: str = "", data_date: str = "",
                account_value: float = float("nan"), atr_stop_mult: float = 2.0,
                back_link: str = "") -> str:
    """One symbol's whole page as markdown."""
    zh = a.zh
    title = f"{a.symbol} · {zh.full() if zh else a.symbol}"
    out = [f"# {title}", ""]

    meta = []
    if a.roles:
        meta.append("／".join(ROLE_LABEL.get(r, r) for r in a.roles))
    if a.sector:
        meta.append(a.sector)
    if a.tag:
        meta.append(f"标签 {a.tag}")
    if _ok(a.screen_rank):
        meta.append(f"筛选排名 #{int(a.screen_rank)}")
    if data_date and report_date:
        meta.append(f"数据截至 {data_date} 收盘 → 面向 {report_date} 开盘")
    if meta:
        out += ["_" + " · ".join(meta) + "_", ""]
    if back_link:
        out += [f"[← 回到 {report_date} 当日报告]({back_link})", ""]

    out += ["> " + _headline(a), ""]

    # --- 一、图形 ---
    out += ["## 一、价格与图形", ""]
    levels = {}
    if a.rec is not None:
        for label, attr in (("止损", "initial_stop_price"), ("参考", "reference_price"),
                            ("目标", "target_price")):
            v = _num(getattr(a.rec, attr, None))
            if not _ok(v) and attr == "initial_stop_price":
                v = _num(getattr(a.rec, "stop_price", None))
            if _ok(v):
                levels[label] = v
    out += _charts(a, levels)

    if a.trend and not a.trend.error:
        out += ["### 图形读数", ""]
        out += [f"- **{k}**：{v}" for k, v in a.trend.bullets()]
        sup, res = a.trend.support, a.trend.resistance
        if _ok(sup) or _ok(res):
            out.append(f"- **支撑 / 阻力**：最近的摆动低点 {_px(sup)}，摆动高点 {_px(res)}"
                       f"（由 ±3 根 K 线的分形极值定义，不是画线画出来的）")
        out += ["", f"**我的读图结论**：{a.trend.verdict}", "",
                "_以上每一条都是对已经发生的价格的描述。均线、RSI、摆动结构全部是滞后指标，"
                "它们能说明现在处于什么状态，不能说明下一步。你自己在图上看到的东西，"
                "和这里写的不一致时，以你看到的为准——然后去查是哪个数字有问题。_", ""]
    elif a.trend and a.trend.error:
        out += [f"_{a.trend.error}_", ""]

    # --- 二、算术 ---
    lines = level_read(a, account_value, atr_stop_mult)
    if lines:
        out += ["## 二、这笔交易的算术", ""]
        rec = a.rec
        out += ["| 项目 | 数值 |", "|---|---:|",
                f"| 参考价（上一收盘） | {_px(getattr(rec, 'reference_price', None))} |",
                f"| 限价 | {_px(getattr(rec, 'limit_price', None))} |",
                f"| 止损（发出时） | {_px(getattr(rec, 'initial_stop_price', None))} |",
                f"| 目标 | {_px(getattr(rec, 'target_price', None))} |",
                f"| 股数 | {_num(getattr(rec, 'shares', 0)):,.0f} |",
                # Priced at the limit, not at the reference: that is the price
                # the instruction fills at, and it is the number the daily
                # report's COST column shows. Two pages of one report
                # disagreeing about the cost of the same order is worse than
                # either number being slightly off.
                f"| 预计成本（按成交价） | ${_cost(rec):,.2f} |",
                f"| 计划风险 | ${_num(rec.risk_per_share()) * _num(getattr(rec, 'shares', 0)):,.2f} |",
                f"| R | {_num(rec.planned_r()):.2f} |",
                f"| 持有期 | {getattr(rec, 'horizon_days', '—')} 天 |", ""]
        out += [f"- {line}" for line in lines]
        out.append("")

    if a.open_idea is not None:
        o = a.open_idea
        out += ["## 二、这笔在场建议的现状", "",
                f"- 持有 {o.days_held} 天，剩余 {o.days_left} 天到时间止损",
                f"- 现价 {_px(o.price)}，浮动盈亏 {_num(o.r_now):+.2f}R"
                f"（${_num(o.pnl):,.2f}）",
                f"- 距离止损 {_pct(o.to_stop)}，距离目标 {_pct(o.to_target)}",
                f"- 状态：**{o.status}**", ""]

    if a.exit_signal is not None:
        s = a.exit_signal
        out += ["## 二、今天为什么要卖", "",
                f"- 动作 **{s.action}** {s.shares:,} 股，紧急度 {s.urgency}/3",
                f"- 触发理由：{s.reason}",
                f"- 结算价 {_px(s.price)}，盈亏 ${_num(s.pnl):,.2f}，"
                f"实现 {_num(s.r_multiple):+.2f}R", ""]

    if a.core is not None:
        c = a.core
        out += ["## 二、核心长仓的月度状态", "",
                f"- 目标权重 {c.holding.weight * 100:.1f}%"
                + (f"，买入理由：{c.holding.thesis}" if c.holding.thesis else ""),
                f"- 距 200 日线 {_pct(c.ext_200)}，近 6 个月 {_pct(c.ret_6m)}，"
                f"近 12 个月 {_pct(c.ret_12m)}",
                f"- 状态 **{c.status}**，动作 **{c.action}**",
                f"- {c.note}", ""]

    if a.daytrade is not None:
        d = a.daytrade
        out += ["## 二、日内关键价位（不是下单指令）", "",
                f"- 昨日高 {_px(d.prev_high)} / 昨日低 {_px(d.prev_low)}，ATR {_px(d.atr)}"
                f"（{_pct(d.atr_pct).lstrip('+')}）",
                f"- 向上触发 {_px(d.long_trigger)} → 止损 {_px(d.long_stop)}"
                f"（0.5 ATR）→ 目标 {_px(d.long_target)}（1 ATR），"
                f"盈亏比约 {_num(d.rr):.2f}",
                f"- 向下触发 {_px(d.short_trigger)}（跌破昨日低点）",
                f"- 相对成交量 {_num(d.rvol):.1f} 倍，50 日日均成交额 "
                f"${_num(d.dollar_vol) / 1e6:,.0f}M", ""]

    # --- 三、财报 ---
    if a.fundamentals is not None:
        out += [fund.markdown_block(a.fundamentals, a.price, "## 三、财报与基本面")]

    if a.earnings is not None and not getattr(a.earnings, "error", ""):
        e = a.earnings
        bits = []
        if getattr(e, "next_date", ""):
            bits.append(f"下次财报 **{e.next_date}**（约 {e.days_to_next():.0f} 天后）")
        if getattr(e, "last_date", ""):
            surprise = (f"，超预期 {e.surprise_pct:+.1f}%"
                        if _ok(getattr(e, "surprise_pct", None)) else "")
            bits.append(f"上次财报 {e.last_date}{surprise}")
        if bits:
            out += ["**财报日历**：" + "；".join(bits)
                    + "。财报日是这套止损体系唯一无法覆盖的风险：跳空会直接穿过止损价。", ""]

    # --- 四、消息 ---
    out += ["## 四、消息面（过去 24 小时）", ""]
    out += _news_block(a)

    # --- 五、反方 ---
    out += ["## 五、反方观点与风险", "",
            "_以下每一条都是按规则从同一批数据里读出来的，不是为了平衡而写的场面话。_", ""]
    out += [f"- {r}" for r in risks(a)]
    out.append("")

    # --- 六、原始数据 ---
    raw = _raw_block(a)
    if raw:
        out += ["## 六、原始数据（本页所有计算的来源）", "",
                "最近 15 个交易日的 OHLCV：", ""] + raw

    # --- 七、去哪儿查 ---
    out += [research.markdown_block(a.symbol, (a.zh.english if a.zh else ""),
                                    "## 七、自己去查（本页不做独立验证）")]
    out += ["---", "",
            "_这一页是一台程序按成文规则读公开数据的结果，不是投资建议。"
            "它不知道你的资金、税务和其它持仓。图形结论来自滞后指标，财报数字来自厂商转录，"
            "新闻材料度来自关键词打分——三者都会错。把它当作一份整理好的证据，"
            "而不是一个答案。_", ""]
    return "\n".join(out)
