"""Task 2 — the market, sector by sector, with no account in view.

Every session the report reads the same things in the same order:

1. **The macro board** — indices, the Treasury curve, VIX, the dollar, oil,
   gold, bitcoin, and the eleven sector ETFs. Prose for what changed.
2. **Policy and news** — the policy monitor's events (Fed, fiscal, tariffs,
   regulation, geopolitics) summarised into a brief and a per-sector tilt,
   plus the macro headlines of the day.
3. **The names** — the bellwethers of every sector plus the leaders of the
   momentum screen (:mod:`.universe`), each read off its bars: moving-average
   stack, momentum, volume, relative strength, position in the range, the
   swing structure, the earnings date, and its own fresh headlines.
4. **A score** for every name on one published rule (:func:`score`), so the
   ranking can be argued with line by line. The top of the list are the ideas,
   the bottom are the names to stay away from, and every one carries the
   reasons that produced its number.
5. **Charts and pages** — an ASCII chart with the averages and the levels for
   every idea, and a full page per name under ``<date>/<SYM>.md``.

The report never reads a portfolio and never writes one. The structured pack
``<date>.json`` beside the page is for whoever wants to reason further —
including a Claude Code session that adds the day's context from the web.
"""

from __future__ import annotations

import contextlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date, datetime

from tradingagents.live import charting, clock
from tradingagents.live.advisor import last_completed_session, project_target, sessions_for
from tradingagents.live.deepdive import SymbolAnalysis, render_page
from tradingagents.live.policy import policy_brief, sector_pressure
from tradingagents.live.sizing import pullback_entry, structural_stop

from . import task_dir, universe
from .market import SECTOR_ETFS, SECTOR_ZH, Facts, MacroBoard, Market, _num, _ok

logger = logging.getLogger(__name__)

TASK = "report"
HORIZON_DAYS = 30


@dataclass
class ReportConfig:
    exchange: str = "all"
    screen_top: int = 60
    per_sector: int = 4
    top_ideas: int = 10
    avoid: int = 5
    news_symbols: int = 30
    with_pages: bool = True
    use_cache: bool = False


@dataclass
class Idea:
    symbol: str
    name: str = ""
    sector: str = "Unknown"
    score: float = 0.0
    price: float = float("nan")
    change_pct: float = float("nan")
    ret_1m: float = float("nan")
    ret_3m: float = float("nan")
    rsi: float = float("nan")
    vol_ratio: float = float("nan")
    ext_200: float = float("nan")
    reasons: list = field(default_factory=list)
    cautions: list = field(default_factory=list)
    entry: float = float("nan")
    stop: float = float("nan")
    target: float = float("nan")
    r: float = float("nan")
    earnings_date: str = ""
    verdict: str = ""
    spark: str = ""
    source: str = ""
    page: str = ""
    facts: Facts | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("facts", None)
        d["news"] = [{"title": n.title, "link": n.link, "source": n.source,
                      "published": n.published, "lean": n.lean, "materiality": n.materiality}
                     for n in (self.facts.news if self.facts else [])[:5]]
        return d


@dataclass
class SectorLine:
    sector: str
    zh: str
    etf: str = ""
    w1: float = float("nan")
    m1: float = float("nan")
    tilt: float = 0.0
    leaders: int = 0
    above_50: int = 0
    best: str = ""


@dataclass
class MarketReport:
    date: str
    data_date: str
    generated_at: str = ""
    macro: MacroBoard | None = None
    macro_read: list = field(default_factory=list)
    policy_events: list = field(default_factory=list)
    policy_brief: str = ""
    tilt: dict = field(default_factory=dict)
    macro_news: list = field(default_factory=list)
    sectors: list = field(default_factory=list)
    scored: list = field(default_factory=list)      # every Idea, best first
    ideas: list = field(default_factory=list)
    avoid: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    path: str = ""

    def to_dict(self) -> dict:
        m = self.macro
        return {
            "date": self.date, "data_date": self.data_date, "generated_at": self.generated_at,
            "macro": {"read": self.macro_read,
                      "rows": [asdict(r) for r in (m.rows if m else [])],
                      "sectors": [asdict(r) for r in (m.sectors if m else [])],
                      "curve_spread": m.curve_spread() if m else None},
            "policy": {"brief": self.policy_brief, "tilt": self.tilt,
                       "events": [_event_dict(e) for e in self.policy_events]},
            "macro_news": [{"title": n.title, "link": n.link, "source": n.source,
                            "published": n.published, "lean": n.lean,
                            "materiality": n.materiality} for n in self.macro_news],
            "sectors": [asdict(s) for s in self.sectors],
            "ideas": [i.to_dict() for i in self.ideas],
            "avoid": [i.to_dict() for i in self.avoid],
            "scored": [i.to_dict() for i in self.scored],
            "warnings": self.warnings, "notes": self.notes,
        }


def _event_dict(e) -> dict:
    try:
        return asdict(e)
    except TypeError:
        return {k: getattr(e, k, None) for k in ("category", "headline", "url", "published",
                                                  "severity", "direction", "sector_impact")}


# ---------------------------------------------------------------------------
# the score
# ---------------------------------------------------------------------------

def score(f: Facts, tilt: float = 0.0, spy_ret_3m: float = float("nan")) -> tuple[float, list[str], list[str]]:
    """One number per name, and the sentences that produced it.

    Weights, so the reader can redo the sum: trend up to 40, momentum ±20,
    volume ±5, relative strength ±10, policy ±10, news ±15, and deductions for
    being stretched or broken. Nothing here is fitted; it is a published
    convention, which is the point — a ranking that cannot be argued with
    line by line is not one worth printing.
    """
    s, why, warn = 0.0, [], []
    snap = f.snap
    px = f.price
    a50, a200 = f.above("sma50"), f.above("sma200")
    s50, s200 = _num(snap.sma50), _num(snap.sma200)
    if a200 is True:
        s += 20
    elif a200 is False:
        s -= 15
        warn.append(f"在 200 日线（{s200:,.2f}）下方")
    if a50 is True:
        s += 15
    elif a50 is False:
        s -= 5
    if _ok(s50) and _ok(s200) and s50 > s200:
        s += 5
    if a50 and a200 and _ok(s50) and _ok(s200) and s50 > s200:
        why.append(f"多头排列：价格 {px:,.2f} > 50 日 {s50:,.2f} > 200 日 {s200:,.2f}")
    elif a50 and a200:
        why.append("站在 50 日和 200 日线上方")

    r3 = _num(snap.ret_3m)
    if _ok(r3):
        s += max(-20.0, min(20.0, r3 * 100 / 2.5))
        if r3 > 0.10:
            why.append(f"近三月 {r3 * 100:+.0f}%")
        elif r3 < -0.10:
            warn.append(f"近三月 {r3 * 100:+.0f}%")

    vr, chg = _num(snap.vol_ratio), _num(snap.change_pct)
    if _ok(vr) and _ok(chg) and vr >= 1.5:
        s += 5 if chg > 0 else -5
        why.append(f"放量{'上涨' if chg > 0 else '下跌'}（量比 {vr:.1f}）") if chg > 0 else \
            warn.append(f"放量下跌（量比 {vr:.1f}，{chg * 100:+.1f}%）")

    if _ok(r3) and _ok(spy_ret_3m):
        rs = r3 - spy_ret_3m
        s += max(-10.0, min(10.0, rs * 100 / 2))
        if rs > 0.05:
            why.append(f"三个月跑赢标普 {rs * 100:.0f} 个百分点")
        elif rs < -0.05:
            warn.append(f"三个月跑输标普 {abs(rs) * 100:.0f} 个百分点")

    if tilt:
        s += max(-10.0, min(10.0, tilt * 10))
        (why if tilt > 0 else warn).append(f"政策面对该板块{'有利' if tilt > 0 else '不利'}（倾向 {tilt:+.2f}）")

    bull = sum(int(getattr(n, "materiality", 0) or 0) for n in f.bullish_news())
    bear = sum(int(getattr(n, "materiality", 0) or 0) for n in f.bearish_news())
    if bull:
        s += min(10.0, bull)
        why.append(f"近两天有利好消息 {len(f.bullish_news())} 条")
    if bear:
        s -= min(15.0, bear * 1.5)
        warn.append(f"近两天有利空消息 {len(f.bearish_news())} 条")

    rsi = _num(snap.rsi14)
    if _ok(rsi) and rsi >= 75:
        s -= 5
        warn.append(f"RSI {rsi:.0f}，短线超买")
    elif _ok(rsi) and rsi <= 30:
        warn.append(f"RSI {rsi:.0f}，超卖")
    ext = f.ext_200()
    if _ok(ext) and ext > 0.40:
        s -= 10
        warn.append(f"高出 200 日线 {ext * 100:.0f}%，已经拉得很远")
    off = _num(snap.off_high_52w)
    if _ok(off) and off < -0.25:
        s -= 10
        warn.append(f"距 52 周高点 {off * 100:.0f}%")
    elif _ok(off) and off > -0.03:
        why.append("在 52 周高点附近")

    e = f.earnings
    if e is not None:
        try:
            days = e.days_to_next(date.today())
            if _ok(days) and 0 <= days <= 7:
                s -= 5
                warn.append(f"{days:.0f} 天后财报，波动会放大")
        except Exception:
            pass
    return round(s, 1), why, warn


# ---------------------------------------------------------------------------
# the reporter
# ---------------------------------------------------------------------------

class Reporter:
    def __init__(self, cfg: ReportConfig | None = None, *, market: Market | None = None,
                 screen=None, now: datetime | None = None, names=None):
        self.cfg = cfg or ReportConfig()
        self.market = market or Market(task=TASK)
        self.screen = screen
        self.now = now
        self._names = names

    def run(self, when: str | date | None = None) -> MarketReport:
        now = self.now or datetime.now(clock.ET)
        try:
            order_day, data_day = sessions_for(when, now)
        except ValueError as exc:
            order_day, data_day = last_completed_session(now), last_completed_session(now)
            logger.warning("%s", exc)
        report = MarketReport(date=order_day.isoformat(), data_date=data_day.isoformat(),
                              generated_at=datetime.now().isoformat())

        # 1. macro
        report.macro = self.market.macro(data_day)
        report.macro_read = report.macro.read()

        # 2. policy and macro news
        report.policy_events = self.market.policy()
        try:
            report.policy_brief = policy_brief(report.policy_events)
            report.tilt = sector_pressure(report.policy_events) if report.policy_events else {}
        except Exception as exc:
            report.warnings.append(f"policy brief failed ({type(exc).__name__}: {exc})")

        # 3. names
        names = self._names or universe.prominent(
            data_day, exchange=self.cfg.exchange, top=self.cfg.screen_top,
            per_sector=self.cfg.per_sector, screen=self.screen, use_cache=self.cfg.use_cache)
        if not any(n.source == "screen" for n in names):
            report.notes.append("筛选没有跑出结果，今天只看各板块龙头")
        spy = self.market.facts("SPY", data_day, benchmark=False)
        spy_r3 = _num(spy.snap.ret_3m)
        facts = {n.symbol: self.market.facts(n.symbol, data_day) for n in names}
        alive = [n for n in names if facts[n.symbol].ok]
        for n in names:
            if not facts[n.symbol].ok:
                report.warnings.append(f"{n.symbol}: 拿不到行情，跳过")
        syms = [n.symbol for n in alive]
        earnings = self.market.earnings(syms, data_day)
        fundamentals = self.market.fundamentals(syms)
        by_symbol, macro_news = self.market.headlines(syms[:self.cfg.news_symbols], macro=True)
        report.macro_news = macro_news[:12]
        for n in alive:
            f = facts[n.symbol]
            f.sector = n.sector if n.sector != "Unknown" else f.sector
            f.name = n.name
            self.market.attach(f, earnings=earnings, fundamentals=fundamentals, news=by_symbol)
            if not f.sector or f.sector == "Unknown":
                f.sector = universe.sector_of(n.symbol)

        # 4. scores
        for n in alive:
            f = facts[n.symbol]
            sc, why, warn = score(f, report.tilt.get(f.sector, 0.0), spy_r3)
            idea = Idea(symbol=n.symbol, name=f.name or n.name, sector=f.sector or "Unknown",
                        score=sc, price=f.price, change_pct=_num(f.snap.change_pct),
                        ret_1m=_num(f.snap.ret_1m), ret_3m=_num(f.snap.ret_3m),
                        rsi=_num(f.snap.rsi14), vol_ratio=_num(f.snap.vol_ratio),
                        ext_200=f.ext_200(), reasons=why, cautions=warn,
                        verdict=f.trend.verdict, spark=f.trend.spark, source=n.source, facts=f)
            self.levels(idea, f)
            if f.earnings is not None:
                idea.earnings_date = str(getattr(f.earnings, "next_date", "") or "")
            report.scored.append(idea)
        report.scored.sort(key=lambda i: i.score, reverse=True)
        report.ideas = [i for i in report.scored if i.score > 0][:self.cfg.top_ideas]
        report.avoid = [i for i in reversed(report.scored) if i.score < 0][:self.cfg.avoid]
        report.sectors = self.sector_lines(report)

        # 5. pages
        if self.cfg.with_pages:
            self.write_pages(report)
        self.save(report)
        return report

    def levels(self, idea: Idea, f: Facts) -> None:
        """Where a buyer would enter, stop and aim, if they were to."""
        snap = f.snap
        entry = pullback_entry(f.price, _num(snap.sma20), f.support() if _ok(f.support()) else None,
                               sma50=_num(snap.sma50), atr_pct=_num(snap.atr_pct))
        idea.entry = _num(entry) if entry else f.price
        stop = structural_stop(idea.entry, f.support() if _ok(f.support()) else None,
                               atr_pct=_num(snap.atr_pct)) if _ok(_num(snap.atr_pct)) else None
        idea.stop = _num(stop)
        target = project_target(idea.entry, snap, HORIZON_DAYS)
        idea.target = _num(target)
        if _ok(idea.stop) and _ok(idea.target) and idea.entry > idea.stop:
            idea.r = (idea.target - idea.entry) / (idea.entry - idea.stop)

    def sector_lines(self, report: MarketReport) -> list[SectorLine]:
        rows = {r.symbol: r for r in (report.macro.sectors if report.macro else [])}
        etf_of = {yf: etf for etf, _, yf in SECTOR_ETFS}
        out = []
        by_sector: dict[str, list[Idea]] = {}
        for i in report.scored:
            by_sector.setdefault(i.sector, []).append(i)
        for _, zh, yf in SECTOR_ETFS:
            line = SectorLine(sector=yf, zh=zh, etf=etf_of[yf], tilt=report.tilt.get(yf, 0.0))
            row = rows.get(line.etf)
            if row and row.ok:
                line.w1, line.m1 = row.w1, row.m1
            ideas = by_sector.get(yf, [])
            line.leaders = len(ideas)
            line.above_50 = sum(1 for i in ideas if i.facts and i.facts.above("sma50"))
            line.best = "、".join(i.symbol for i in ideas[:3])
            out.append(line)
        if by_sector.get("Unknown"):
            u = by_sector["Unknown"]
            out.append(SectorLine(sector="Unknown", zh="未分类", leaders=len(u),
                                  above_50=sum(1 for i in u if i.facts and i.facts.above("sma50")),
                                  best="、".join(i.symbol for i in u[:3])))
        return out

    def write_pages(self, report: MarketReport) -> None:
        d = task_dir(TASK) / report.date
        d.mkdir(parents=True, exist_ok=True)
        names = self.market.names()
        for idea in report.scored:
            f = idea.facts
            if f is None:
                continue
            try:
                a = SymbolAnalysis(symbol=idea.symbol, zh=names.get(idea.symbol, idea.name),
                                   sector=SECTOR_ZH.get(idea.sector, idea.sector), bars=f.bars,
                                   snap=f.snap, trend=f.trend, fundamentals=f.fundamentals,
                                   earnings=f.earnings, news=f.news,
                                   tilt=report.tilt.get(idea.sector, 0.0))
                page = render_page(a, report_date=report.date, data_date=report.data_date,
                                   back_link=f"../{report.date}.md")
                (d / f"{idea.symbol}.md").write_text(page, encoding="utf-8")
                idea.page = f"{report.date}/{idea.symbol}.md"
            except Exception as exc:
                report.warnings.append(f"{idea.symbol}: 详情页没生成（{type(exc).__name__}: {exc}）")

    def save(self, report: MarketReport) -> None:
        try:
            d = task_dir(TASK)
            path = d / f"{report.date}.md"
            path.write_text(format_report(report), encoding="utf-8")
            (d / f"{report.date}.json").write_text(
                json.dumps(report.to_dict(), ensure_ascii=False, indent=1, default=str),
                encoding="utf-8")
            report.path = str(path)
        except OSError as exc:
            report.warnings.append(f"the report could not be saved: {exc}")


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def _pct(v: float, digits: int = 1) -> str:
    return f"{v * 100:+.{digits}f}%" if _ok(v) else "—"


def _pts(v: float) -> str:
    return f"{v:+.2f}" if _ok(v) else "—"


def _f(v: float, digits: int = 2) -> str:
    return f"{v:,.{digits}f}" if _ok(v) else "—"


def chart_for(idea: Idea, width: int = 72) -> list[str]:
    f = idea.facts
    if f is None or len(f.bars.closes) < 30:
        return []
    n = 120
    closes = f.bars.closes[-n:]
    overlays = {}
    for label, w in (("MA20", 20), ("MA50", 50)):
        if len(f.bars.closes) >= w:
            overlays[label] = charting.sma(f.bars.closes, w)[-n:]
    levels = {}
    for label, v in (("支撑", f.support()), ("阻力", f.resistance()),
                     ("止损", idea.stop), ("目标", idea.target)):
        if _ok(v):
            levels[label] = v
    return charting.line_chart(closes, overlays=overlays, levels=levels,
                               dates=f.bars.dates[-n:], width=width,
                               title=f"{idea.symbol} 近 {min(n, len(closes))} 个交易日")


def format_report(report: MarketReport) -> str:
    out = [f"# 市场日报 · {report.date}", "",
           f"数据截至 {report.data_date} 收盘。不看任何账户；每个名字都按同一把尺子打分，分数旁边是它的来由。", ""]

    out.append("## 一、宏观与利率")
    out += [f"- {line}" for line in report.macro_read] or ["- 宏观数据没拿到"]
    m = report.macro
    if m and any(r.ok for r in m.rows):
        out += ["", "| | 收盘 | 日 | 周 | 月 | 走势 |", "|---|---:|---:|---:|---:|---|"]
        for r in m.rows:
            if not r.ok:
                continue
            fmt = _pts if r.is_yield else _pct
            last = f"{r.last:.2f}%" if r.is_yield else _f(r.last, 2 if r.last < 1000 else 0)
            out.append(f"| {r.label} | {last} | {fmt(r.d1)} | {fmt(r.w1)} | {fmt(r.m1)} | `{r.spark}` |")
    out.append("")

    out.append("## 二、政策与新闻")
    if report.policy_brief:
        out += [report.policy_brief.strip(), ""]
    else:
        out += ["- 近两天没有新的政策事件", ""]
    if report.macro_news:
        out.append("宏观标题：")
        for n in report.macro_news[:10]:
            lean = {"bullish": "利好", "bearish": "利空"}.get(n.lean, "")
            out.append(f"- [{n.title}]({n.link})（{n.source}{'，' + lean if lean else ''}）")
        out.append("")

    out.append("## 三、板块")
    out += ["| 板块 | ETF | 周 | 月 | 政策倾向 | 龙头在 50 日线上 | 领先名字 |",
            "|---|---|---:|---:|---:|---:|---|"]
    for s in report.sectors:
        out.append(f"| {s.zh} | {s.etf or '—'} | {_pct(s.w1)} | {_pct(s.m1)} | {s.tilt:+.2f} "
                   f"| {s.above_50}/{s.leaders} | {s.best} |")
    out.append("")

    out.append(f"## 四、值得关注（前 {len(report.ideas)}）")
    if not report.ideas:
        out.append("今天没有得分为正的名字。")
    for i, idea in enumerate(report.ideas, 1):
        out += idea_block(i, idea, report)
    out.append("")

    out.append("## 五、走弱 / 回避")
    if not report.avoid:
        out.append("- 没有明显走坏的名字")
    for idea in report.avoid:
        why = "；".join(idea.cautions[:3]) or idea.verdict
        out.append(f"- **{idea.symbol}** {idea.name}（{SECTOR_ZH.get(idea.sector, idea.sector)}，{idea.score:+.0f} 分）：{why}")
    out.append("")

    out.append("## 六、全部评分")
    out += ["| 代码 | 板块 | 现价 | 日 | 月 | 三月 | RSI | 量比 | 距200日 | 分 | 页 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for idea in report.scored:
        page = f"[页]({idea.page})" if idea.page else ""
        out.append(f"| {idea.symbol} | {SECTOR_ZH.get(idea.sector, idea.sector)} | {_f(idea.price)} "
                   f"| {_pct(idea.change_pct)} | {_pct(idea.ret_1m)} | {_pct(idea.ret_3m)} "
                   f"| {_f(idea.rsi, 0)} | {_f(idea.vol_ratio, 1)} | {_pct(idea.ext_200, 0)} "
                   f"| {idea.score:+.0f} | {page} |")
    out.append("")

    if report.warnings or report.notes:
        out.append("## 说明")
        out += [f"- {n}" for n in report.notes] + [f"- ⚠ {w}" for w in report.warnings]
        out.append("")
    out += ["---", "分数的算法：趋势 ≤40，动量 ±20，量能 ±5，相对强弱 ±10，政策倾向 ±10，消息 ±15，"
            "拉伸/破位/临近财报扣分。参考位只是参考：入场取回调到均线或支撑的位置，止损在结构之下，目标按近期趋势外推。"
            "这不是投资建议。", ""]
    return "\n".join(out)


def idea_block(i: int, idea: Idea, report: MarketReport) -> list[str]:
    zh = SECTOR_ZH.get(idea.sector, idea.sector)
    out = ["", f"### {i}. {idea.symbol} {idea.name}  ·  {zh}  ·  {idea.score:+.0f} 分"]
    out.append(f"现价 {_f(idea.price)}（日 {_pct(idea.change_pct)}，月 {_pct(idea.ret_1m)}，三月 {_pct(idea.ret_3m)}）"
               f"　`{idea.spark}`")
    chart = chart_for(idea)
    if chart:
        out += ["", "```"] + chart + ["```"]
    if idea.verdict:
        out.append(f"- 图形：{idea.verdict}")
    out += [f"- ✓ {r}" for r in idea.reasons]
    out += [f"- ⚠ {c}" for c in idea.cautions]
    if _ok(idea.stop) and _ok(idea.target):
        out.append(f"- 参考位：入场 {_f(idea.entry)} · 止损 {_f(idea.stop)} · 目标 {_f(idea.target)}"
                   f"（{idea.r:.1f}R）" if _ok(idea.r) else
                   f"- 参考位：入场 {_f(idea.entry)} · 止损 {_f(idea.stop)} · 目标 {_f(idea.target)}")
    if idea.earnings_date:
        out.append(f"- 下次财报：{idea.earnings_date}")
    f = idea.facts
    if f and f.fundamentals is not None:
        fu = f.fundamentals
        bits = []
        pe = _num(getattr(fu, "pe_forward", None))
        if _ok(pe):
            bits.append(f"前瞻 PE {pe:.0f}")
        g = _num(getattr(fu, "revenue_growth", None))
        if _ok(g):
            bits.append(f"营收增速 {g * 100:+.0f}%")
        up = None
        with contextlib.suppress(Exception):
            up = fu.upside(idea.price)
        if _ok(_num(up)):
            bits.append(f"机构目标价空间 {_num(up) * 100:+.0f}%")
        if bits:
            out.append("- 基本面：" + "，".join(bits))
    if f and f.news:
        for n in f.news[:3]:
            lean = {"bullish": "利好", "bearish": "利空"}.get(n.lean, "")
            out.append(f"- 新闻：[{n.title}]({n.link})（{n.source}{'，' + lean if lean else ''}）")
    if idea.page:
        out.append(f"- [详情页]({idea.page})")
    return out


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="tradingagents.desk report",
                                description="the market, sector by sector, with no account in view")
    p.add_argument("--date", default=None, help="the session the report is for (default: next open)")
    p.add_argument("--top", type=int, default=None, help="ideas to print in full")
    p.add_argument("--exchange", default=None, choices=["nasdaq", "all"])
    p.add_argument("--use-cache", action="store_true", help="reuse a saved screen instead of scanning")
    p.add_argument("--no-pages", action="store_true")
    p.add_argument("-q", "--quiet", action="store_true", help="print only the path")
    args = p.parse_args(argv)
    cfg = ReportConfig()
    if args.top:
        cfg.top_ideas = args.top
    if args.exchange:
        cfg.exchange = args.exchange
    cfg.use_cache = args.use_cache
    cfg.with_pages = not args.no_pages
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from . import state
    print(state.pull())
    report = Reporter(cfg).run(args.date)
    print(report.path if args.quiet else format_report(report))
    return 0
